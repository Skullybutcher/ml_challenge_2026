"""
Pairwise feature engineering between a Source-1 record and a candidate
Source-2/3 record. Kept dependency-light (rapidfuzz + stdlib) so it runs
fast over large candidate sets on Kaggle CPU.
"""
from __future__ import annotations
import os
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

from rapidfuzz import fuzz
from normalize import tokens, rare_tokens

BASE_FEATURE_NAMES = [
    "name_levenshtein_ratio", "name_token_sort_ratio", "name_partial_ratio",
    "name_jaccard", "name_len_diff", "name_char_bigram_jaccard",
    "addr_levenshtein_ratio", "addr_token_sort_ratio",
    "addr_jaccard", "addr_len_diff",
    "country_match", "country_either_empty",
    "rare_token_overlap", "rare_token_union_size",
    "name_first_token_match",
]
E5_FEATURE_NAMES = ["e5_name_cosine", "e5_address_cosine"]
FEATURE_NAMES = list(BASE_FEATURE_NAMES)


def enable_e5_features() -> None:
    """Enable the two opt-in E5 cosine columns for this pipeline process."""
    global FEATURE_NAMES
    FEATURE_NAMES = list(BASE_FEATURE_NAMES) + list(E5_FEATURE_NAMES)


class E5EmbeddingStore:
    """Read-only combined train/test memmaps produced by generate_embeddings.py."""

    TABLE_CODES = {
        "train_s1": 0, "train_s2": 1, "train_s3": 2,
        "test_s1": 3, "test_s2": 4, "test_s3": 5,
    }
    DIMENSION = 1024

    def __init__(self, name_dir: str | os.PathLike, address_dir: str | os.PathLike):
        import numpy as np

        self.name_dir = Path(name_dir)
        self.address_dir = Path(address_dir)
        name_manifest_path = self.name_dir / "e5_embeddings_manifest.json"
        address_manifest_path = self.address_dir / "e5_embeddings_manifest.json"
        if not name_manifest_path.is_file() or not address_manifest_path.is_file():
            raise FileNotFoundError(
                "Both E5 manifests are required; embedding generation must finish first."
            )
        self.name_manifest = json.loads(name_manifest_path.read_text(encoding="utf-8"))
        self.address_manifest = json.loads(address_manifest_path.read_text(encoding="utf-8"))
        for manifest in (self.name_manifest, self.address_manifest):
            if manifest.get("prefix") != "passage: ":
                raise ValueError("E5 artifacts must use the required 'passage: ' prefix.")
            if manifest.get("pooling") != "masked_mean" or manifest.get("normalize_l2") is not True:
                raise ValueError("E5 artifacts must use masked mean pooling and L2 normalization.")
            if manifest.get("id_dtype") != "object":
                raise ValueError("E5 entity ID sidecars must use object dtype.")
        if self.name_manifest.get("tables") != self.address_manifest.get("tables"):
            raise ValueError("E5 name/address manifests have different table row counts.")
        if self.name_manifest.get("dimension") != self.DIMENSION:
            raise ValueError("E5 embedding dimension must be 1024.")
        if self.name_manifest.get("dtype") != "float16":
            raise ValueError("E5 vectors must be fp16 on disk.")
        self.counts = {key: int(value) for key, value in self.name_manifest["tables"].items()}
        self.split_tables = {
            "train": ("train_s1", "train_s2", "train_s3"),
            "test": ("test_s1", "test_s2", "test_s3"),
        }
        self.offsets = {}
        self.split_counts = {}
        for split, tables in self.split_tables.items():
            offset = 0
            for table in tables:
                self.offsets[table] = offset
                offset += self.counts[table]
            self.split_counts[split] = offset
        self._arrays = {}
        self._cached_ids_split = None
        self._cached_ids = None
        self._np = np

    def attach(self, frame, table: str):
        """Attach stable input-row indexes after verifying sidecar ID alignment."""
        if table not in self.TABLE_CODES:
            raise ValueError(f"Unknown E5 table key: {table}")
        source_ids = frame["entity_id"].astype(str).to_numpy(dtype=object)
        split = table.split("_", 1)[0]
        stem = f"{split}_names_ids.npy"
        sidecars = (
            self.name_dir / stem,
            self.address_dir / f"{split}_addrs_ids.npy",
        )
        for sidecar in sidecars:
            if not sidecar.is_file():
                raise FileNotFoundError(f"Missing E5 entity ID sidecar: {sidecar}")
        try:
            shared_ids = os.path.samefile(*sidecars)
        except OSError as exc:
            raise ValueError("Cannot verify shared E5 name/address ID sidecar.") from exc
        if not shared_ids:
            raise ValueError("E5 name/address IDs must share the generator-created sidecar.")
        # Object NPY files cannot be memory-mapped. Load the shared ID array once
        # per split and reuse it across all bounded training/inference chunks.
        if self._cached_ids_split != split:
            self._cached_ids = self._np.load(sidecars[0], allow_pickle=True)
            if self._cached_ids.dtype != self._np.dtype(object):
                raise ValueError(f"E5 ID sidecar must be object dtype: {sidecars[0]}")
            if self._cached_ids.shape != (self.split_counts[split],):
                raise ValueError(f"E5 ID sidecar has invalid shape: {sidecars[0]}")
            self._cached_ids_split = split
        start = self.offsets[table]
        expected = self._cached_ids[start:start + len(frame)]
        if (len(expected) != len(frame)
                or not self._np.array_equal(expected, source_ids)):
            raise ValueError(
                f"E5 ID order does not match {table} in {sidecars[0]}; refusing wrong-vector joins."
            )
        frame["_e5_row_idx"] = (
            self.offsets[table] + self._np.arange(len(frame), dtype=self._np.int32)
        )
        frame["_e5_table_code"] = self.TABLE_CODES[table]
        return frame

    def _vectors(self, table: str, field: str):
        if table not in self.TABLE_CODES:
            raise ValueError(f"No row-count metadata for E5 table {table}.")
        split = table.split("_", 1)[0]
        key = (split, field)
        if key not in self._arrays:
            root = self.name_dir if field == "name" else self.address_dir
            filename = f"{split}_names.npy" if field == "name" else f"{split}_addrs.npy"
            path = root / filename
            array = self._np.load(path, mmap_mode="r", allow_pickle=False)
            expected = (self.split_counts[split], self.DIMENSION)
            if array.shape != expected or array.dtype != self._np.float16:
                raise ValueError(f"Invalid E5 array {path}: {array.shape} {array.dtype}; expected {expected} float16.")
            self._arrays[(split, field)] = array
            key = (split, field)
        return self._arrays[key]

    def cosine_features(self, pairs_df, start: int, end: int):
        """Compute float32 name/address cosine scores for one bounded pair batch."""
        import numpy as np

        table1_codes = pairs_df["_e5_table1"].to_numpy(dtype=np.int8)[start:end]
        table2_codes = pairs_df["_e5_table2"].to_numpy(dtype=np.int8)[start:end]
        idx1 = pairs_df["_e5_row1"].to_numpy(dtype=np.int32)[start:end]
        idx2 = pairs_df["_e5_row2"].to_numpy(dtype=np.int32)[start:end]
        if len(table1_codes) == 0:
            return np.empty((0, len(E5_FEATURE_NAMES)), dtype=np.float32)
        unique1 = np.unique(table1_codes)
        if len(unique1) != 1:
            raise ValueError("A pair batch unexpectedly mixes Source-1 embedding tables.")
        table1 = next(key for key, value in self.TABLE_CODES.items() if value == int(unique1[0]))
        split1 = table1.split("_", 1)[0]
        output = np.empty((end - start, len(E5_FEATURE_NAMES)), dtype=np.float32)
        for code in np.unique(table2_codes):
            table2 = next(key for key, value in self.TABLE_CODES.items() if value == int(code))
            split2 = table2.split("_", 1)[0]
            if split1 != split2:
                raise ValueError("A pair batch unexpectedly mixes train/test embedding arrays.")
            local = np.flatnonzero(table2_codes == code)
            left_indices = idx1[local]
            right_indices = idx2[local]
            if (left_indices.min(initial=0) < 0 or right_indices.min(initial=0) < 0
                    or left_indices.max(initial=-1) >= self.split_counts[split1]
                    or right_indices.max(initial=-1) >= self.split_counts[split2]):
                raise IndexError(f"E5 row index out of range for {table1} -> {table2}.")
            for column, field in enumerate(("name", "address")):
                left = self._vectors(table1, field)[left_indices]
                right = self._vectors(table2, field)[right_indices]
                output[local, column] = np.einsum(
                    "ij,ij->i", left, right, dtype=np.float32,
                    casting="unsafe", optimize=True,
                )
                del left, right
        return output


def _char_bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(
    name1: str, addr1: str, country1: str,
    name2: str, addr2: str, country2: str,
) -> List[float]:
    n1_tok, n2_tok = tokens(name1), tokens(name2)
    a1_tok, a2_tok = tokens(addr1), tokens(addr2)
    r1_tok = rare_tokens(name1) | rare_tokens(addr1)
    r2_tok = rare_tokens(name2) | rare_tokens(addr2)

    rare_union = r1_tok | r2_tok
    rare_overlap = len(r1_tok & r2_tok)

    f1_tok = next(iter(sorted(n1_tok)), "")
    f2_tok = next(iter(sorted(n2_tok)), "")

    return [
        fuzz.ratio(name1, name2) / 100.0,
        fuzz.token_sort_ratio(name1, name2) / 100.0,
        fuzz.partial_ratio(name1, name2) / 100.0,
        _jaccard(n1_tok, n2_tok),
        abs(len(name1) - len(name2)),
        _jaccard(_char_bigrams(name1), _char_bigrams(name2)),
        fuzz.ratio(addr1, addr2) / 100.0,
        fuzz.token_sort_ratio(addr1, addr2) / 100.0,
        _jaccard(a1_tok, a2_tok),
        abs(len(addr1) - len(addr2)),
        1.0 if country1 and country2 and country1.lower() == country2.lower() else 0.0,
        1.0 if (not country1 or not country2) else 0.0,
        float(rare_overlap),
        float(len(rare_union)),
        1.0 if f1_tok and f1_tok == f2_tok else 0.0,
    ]


def _diagnostic_value(value) -> str:
    """Describe a failing value without writing business text into the log."""
    value_type = type(value)
    try:
        value_len = len(value)
    except Exception:
        value_len = "n/a"
    try:
        payload = repr(value).encode("utf-8", errors="backslashreplace")
        digest = hashlib.sha256(payload).hexdigest()[:12]
    except Exception:
        digest = "unavailable"
    return (f"type={value_type.__module__}.{value_type.__qualname__} "
            f"exact_str={value_type is str} len={value_len} repr_sha256={digest}")


def build_feature_frame(pairs: List[Dict]) -> "pandas.DataFrame":
    """
    pairs: list of dicts each with keys
      s1_id, other_id, name1, addr1, country1, name2, addr2, country2
    Small-sample path. For scale use build_feature_frame_vectorized().
    """
    import pandas as pd
    rows = []
    for p in pairs:
        feats = pair_features(
            p["name1"], p["addr1"], p["country1"],
            p["name2"], p["addr2"], p["country2"],
        )
        rows.append([p["s1_id"], p["other_id"]] + feats)
    return pd.DataFrame(rows, columns=["s1_id", "other_id"] + BASE_FEATURE_NAMES)


def build_feature_frame_vectorized(
    pairs_df: "pandas.DataFrame", e5_store: E5EmbeddingStore | None = None
) -> "pandas.DataFrame":
    """Vectorized-merge friendly: columns s1_id, other_id, name1/addr1/country1, name2/addr2/country2."""
    import numpy as np
    import pandas as pd
    feature_names = FEATURE_NAMES if e5_store is not None else BASE_FEATURE_NAMES
    feature_count = len(BASE_FEATURE_NAMES) + (len(E5_FEATURE_NAMES) if e5_store is not None else 0)
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=["s1_id", "other_id"] + feature_names)
    # Merge results can contain non-string scalars even though source TSVs are
    # loaded as text. The scalar feature functions require strings (for example,
    # _char_bigrams slices each value), so enforce that contract after filling
    # nulls and before converting the columns to Python lists.
    n1 = pairs_df["name1"].fillna("").astype(str).tolist()
    a1 = pairs_df["addr1"].fillna("").astype(str).tolist()
    c1 = pairs_df["country1"].fillna("").astype(str).tolist()
    n2 = pairs_df["name2"].fillna("").astype(str).tolist()
    a2 = pairs_df["addr2"].fillna("").astype(str).tolist()
    c2 = pairs_df["country2"].fillna("").astype(str).tolist()
    # Keep NumPy's nested-sequence conversion bounded when a candidate chunk
    # contains millions of pairs. This also preserves a predictable peak while
    # retaining the original scalar feature computation and ordering.
    feat_array = np.empty((len(pairs_df), feature_count), dtype=np.float32)
    batch_size = 100_000
    trace_batches = os.environ.get("AKARI_TRACE_FEATURE_BATCHES") == "1"
    for start in range(0, len(pairs_df), batch_size):
        end = min(start + batch_size, len(pairs_df))
        if trace_batches:
            print(f"[FEATURES] start rows={start}:{end} total={len(pairs_df):,}", flush=True)
        batch_inputs = zip(
            n1[start:end], a1[start:end], c1[start:end],
            n2[start:end], a2[start:end], c2[start:end])
        # zip iterators are consumed by the list comprehension, so retain only
        # the six bounded 100k slices needed to replay a failed batch.
        batch_columns = (
            n1[start:end], a1[start:end], c1[start:end],
            n2[start:end], a2[start:end], c2[start:end])
        try:
            batch = [pair_features(x1, y1, z1, x2, y2, z2)
                     for x1, y1, z1, x2, y2, z2 in batch_inputs]
        except Exception as batch_exc:
            try:
                with open(__file__, "rb") as source_file:
                    source_sha = hashlib.sha256(source_file.read()).hexdigest()
            except OSError:
                source_sha = "unavailable"
            print(
                f"[FEATURES-ERROR] batch={start}:{end} total={len(pairs_df):,} "
                f"python={sys.version.split()[0]} executable={sys.executable} "
                f"module={os.path.realpath(__file__)} source_sha256={source_sha} "
                f"pair_features_file={pair_features.__code__.co_filename} "
                f"exception={type(batch_exc).__module__}.{type(batch_exc).__qualname__}: "
                f"{batch_exc!r}",
                flush=True,
            )
            for local_offset, values in enumerate(zip(*batch_columns)):
                try:
                    pair_features(*values)
                except Exception as row_exc:
                    absolute_offset = start + local_offset
                    pair_ids = pairs_df.iloc[absolute_offset][["s1_id", "other_id"]].to_dict()
                    names = ("name1", "addr1", "country1", "name2", "addr2", "country2")
                    details = "; ".join(
                        f"{name}({_diagnostic_value(value)})"
                        for name, value in zip(names, values))
                    print(
                        f"[FEATURES-ERROR] first failing pair offset={absolute_offset} "
                        f"ids={pair_ids!r} row_exception="
                        f"{type(row_exc).__module__}.{type(row_exc).__qualname__}: "
                        f"{row_exc!r}; inputs={details}",
                        flush=True,
                    )
                    break
            else:
                print(
                    f"[FEATURES-ERROR] batch exception was not reproduced by row-wise replay "
                    f"for rows {start}:{end}.",
                    flush=True,
                )
            raise
        del batch_inputs, batch_columns
        batch_array = np.asarray(batch, dtype=np.float32)
        if batch_array.shape != (end - start, len(BASE_FEATURE_NAMES)):
            raise ValueError(
                f"Feature batch shape {batch_array.shape}; expected "
                f"({end - start}, {len(BASE_FEATURE_NAMES)})"
            )
        feat_array[start:end, :len(BASE_FEATURE_NAMES)] = batch_array
        if e5_store is not None:
            feat_array[start:end, len(BASE_FEATURE_NAMES):] = e5_store.cosine_features(
                pairs_df, start, end
            )
        if trace_batches:
            print(f"[FEATURES] done rows={start}:{end} total={len(pairs_df):,}", flush=True)
    out = pd.DataFrame(feat_array, columns=feature_names)
    out.insert(0, "s1_id", pairs_df["s1_id"].to_numpy())
    out.insert(1, "other_id", pairs_df["other_id"].to_numpy())
    return out
