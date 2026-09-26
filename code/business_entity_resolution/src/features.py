"""
Pairwise feature engineering between a Source-1 record and a candidate
Source-2/3 record. Kept dependency-light (rapidfuzz + stdlib) so it runs
fast over large candidate sets on Kaggle CPU.
"""
from __future__ import annotations
import os
import hashlib
import sys
from typing import Dict, List

from rapidfuzz import fuzz
from normalize import tokens, rare_tokens

FEATURE_NAMES = [
    "name_levenshtein_ratio", "name_token_sort_ratio", "name_partial_ratio",
    "name_jaccard", "name_len_diff", "name_char_bigram_jaccard",
    "addr_levenshtein_ratio", "addr_token_sort_ratio",
    "addr_jaccard", "addr_len_diff",
    "country_match", "country_either_empty",
    "rare_token_overlap", "rare_token_union_size",
    "name_first_token_match",
]


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
    return pd.DataFrame(rows, columns=["s1_id", "other_id"] + FEATURE_NAMES)


def build_feature_frame_vectorized(pairs_df: "pandas.DataFrame") -> "pandas.DataFrame":
    """Vectorized-merge friendly: columns s1_id, other_id, name1/addr1/country1, name2/addr2/country2."""
    import numpy as np
    import pandas as pd
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=["s1_id", "other_id"] + FEATURE_NAMES)
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
    feat_array = np.empty((len(pairs_df), len(FEATURE_NAMES)), dtype=np.float32)
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
        if batch_array.shape != (end - start, len(FEATURE_NAMES)):
            raise ValueError(
                f"Feature batch shape {batch_array.shape}; expected "
                f"({end - start}, {len(FEATURE_NAMES)})"
            )
        feat_array[start:end] = batch_array
        if trace_batches:
            print(f"[FEATURES] done rows={start}:{end} total={len(pairs_df):,}", flush=True)
    out = pd.DataFrame(feat_array, columns=FEATURE_NAMES)
    out.insert(0, "s1_id", pairs_df["s1_id"].to_numpy())
    out.insert(1, "other_id", pairs_df["other_id"].to_numpy())
    return out
