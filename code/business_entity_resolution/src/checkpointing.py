"""Atomic, fingerprinted checkpoints for completed training chunks."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
TRAIN_INPUTS = (
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def prepare_training_checkpoint(
    checkpoint_root: str | os.PathLike[str],
    data_dir: str | os.PathLike[str],
    params: dict[str, Any],
    code_paths: list[str | os.PathLike[str]],
) -> tuple[Path, str]:
    """Return a cache directory keyed by all inputs that affect chunk rows."""
    data_root = Path(data_dir) / "train"
    inputs = {}
    for name in TRAIN_INPUTS:
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required training input for checkpoint fingerprint: {path}")
        inputs[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}

    code = {}
    for path_value in code_paths:
        path = Path(path_value).resolve()
        code[str(path)] = _sha256_file(path)

    payload = {
        "schema": SCHEMA_VERSION,
        "params": params,
        "inputs": inputs,
        "code": code,
        "runtime": {
            "python": __import__("sys").version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "rapidfuzz": _package_version("rapidfuzz"),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    cache_dir = Path(checkpoint_root) / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    expected_manifest = {"fingerprint": fingerprint, "payload": payload}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            actual_manifest = json.load(f)
        if actual_manifest != expected_manifest:
            raise ValueError(f"Checkpoint manifest mismatch in {manifest_path}")
    else:
        _atomic_json(manifest_path, expected_manifest)
    return cache_dir, fingerprint


def s1_chunk_digest(s1_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for entity_id in s1_ids:
        digest.update(str(entity_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def chunk_checkpoint_path(cache_dir: Path, chunk_index: int) -> Path:
    return cache_dir / f"train_chunk_{chunk_index:04d}.npz"


def load_training_chunk(
    path: Path,
    fingerprint: str,
    metadata: dict[str, Any],
    feature_names: list[str],
) -> tuple[pd.DataFrame, np.ndarray, int, int] | None:
    """Load a complete cache entry; return None for missing/corrupt/stale files."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["fingerprint"].item()) != fingerprint:
                return None
            for key, expected in metadata.items():
                if str(saved[key].item()) != str(expected):
                    return None
            s1_ids = saved["s1_ids"]
            other_ids = saved["other_ids"]
            features = saved["features"]
            labels = saved["labels"].astype(np.int64, copy=False)
            n_pre = int(saved["n_pre"].item())
            n_pos_pre = int(saved["n_pos_pre"].item())
            post_pos = int(saved["post_pos"].item())

        if s1_ids.dtype.kind not in "US" or other_ids.dtype.kind not in "US":
            return None
        if features.dtype != np.float32 or features.ndim != 2:
            return None
        if features.shape != (len(s1_ids), len(feature_names)):
            return None
        if len(other_ids) != len(s1_ids) or len(labels) != len(s1_ids):
            return None
        if labels.sum(dtype=np.int64) != post_pos or not (0 <= n_pos_pre <= n_pre):
            return None

        feat_df = pd.DataFrame(features, columns=feature_names, copy=False)
        feat_df.insert(0, "s1_id", s1_ids.astype(str, copy=False))
        feat_df.insert(1, "other_id", other_ids.astype(str, copy=False))
        return feat_df, labels, n_pre, n_pos_pre
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def save_training_chunk(
    path: Path,
    fingerprint: str,
    metadata: dict[str, Any],
    feat_df: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str],
    n_pre: int,
    n_pos_pre: int,
) -> None:
    """Write one self-contained chunk, promoting it only after the write closes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(
                f,
                fingerprint=np.asarray(fingerprint),
                **{key: np.asarray(value) for key, value in metadata.items()},
                s1_ids=feat_df["s1_id"].astype(str).to_numpy(dtype="U"),
                other_ids=feat_df["other_id"].astype(str).to_numpy(dtype="U"),
                features=feat_df.loc[:, feature_names].to_numpy(dtype=np.float32, copy=False),
                labels=np.asarray(labels, dtype=np.uint8),
                n_pre=np.asarray(n_pre, dtype=np.int64),
                n_pos_pre=np.asarray(n_pos_pre, dtype=np.int64),
                post_pos=np.asarray(np.asarray(labels).sum(), dtype=np.int64),
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
