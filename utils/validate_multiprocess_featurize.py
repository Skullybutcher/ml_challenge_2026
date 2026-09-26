"""Exact parity check for the experimental multiprocessing featurizer.

Run only after the active 100k Run 1 has completed. This recreates the
deterministic 5k-S1 rare-channel sample, then compares serial and multiprocess
features on the same ordered pair frame at each original 2,500-S1 chunk.
It deliberately does not train a model or write submission files.
"""
from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

from features import FEATURE_NAMES, build_feature_frame_vectorized  # noqa: E402
from features_multiprocess_experiment import (  # noqa: E402
    build_feature_frame_multiprocess_experimental,
)


def _sample_training_data(data_dir: Path, sample_s1: int, seed: int):
    from io_utils import (load_ground_truth, load_source,
                          validate_ground_truth_refs)
    from pipeline import add_norm_columns

    s1 = load_source(str(data_dir / "train" / "train_source1.tsv"), "S1-")
    s2 = load_source(str(data_dir / "train" / "train_source2.tsv"), "S2-")
    s3 = load_source(str(data_dir / "train" / "train_source3.tsv"), "S3-")
    gt = load_ground_truth(str(data_dir / "train" / "train_ground_truth.tsv"))
    validate_ground_truth_refs(gt, set(s1.entity_id), set(s2.entity_id), set(s3.entity_id))

    if sample_s1 < len(s1):
        rng = np.random.default_rng(seed)
        keep_s1 = set(rng.choice(s1["entity_id"].to_numpy(), size=sample_s1, replace=False))
        s1 = s1[s1["entity_id"].isin(keep_s1)].reset_index(drop=True)
        gt = {key: value for key, value in gt.items() if key in keep_s1}
        needed = set().union(*gt.values()) if gt else set()
        s2_extra = set(rng.choice(s2["entity_id"].to_numpy(),
                                  size=min(len(s2), sample_s1 * 3), replace=False))
        s3_extra = set(rng.choice(s3["entity_id"].to_numpy(),
                                  size=min(len(s3), sample_s1 * 3), replace=False))
        s2 = s2[s2["entity_id"].isin(needed | s2_extra)].reset_index(drop=True)
        s3 = s3[s3["entity_id"].isin(needed | s3_extra)].reset_index(drop=True)
    return (add_norm_columns(s1), add_norm_columns(s2), add_norm_columns(s3), gt)


def _block(s1: pd.DataFrame, other: pd.DataFrame):
    from blocking import (generate_candidates_prefix, generate_candidates_rare,
                          generate_candidates_token, union_candidates)

    token = generate_candidates_token(s1, other, max_df=300)
    prefix = generate_candidates_prefix(
        s1, other, prefix_len=4, max_pairs_per_key=200_000)
    rare = generate_candidates_rare(
        s1, other, max_df=300, min_len=6, max_df_long=2_000)
    return union_candidates(token, prefix, rare)


def _labels(features: pd.DataFrame, gt: dict[str, set[str]]) -> np.ndarray:
    return np.asarray([
        1 if other_id in gt.get(s1_id, set()) else 0
        for s1_id, other_id in zip(features["s1_id"].to_numpy(),
                                   features["other_id"].to_numpy())
    ], dtype=np.int64)


def _check_ids(actual: pd.DataFrame, pairs: pd.DataFrame, label: str) -> None:
    for column in ("s1_id", "other_id"):
        if not np.array_equal(actual[column].to_numpy(), pairs[column].to_numpy()):
            raise AssertionError(f"{label}: {column} values/order differ from input pairs")


def _compare_feature_blocks(reference: np.ndarray, actual: np.ndarray,
                            chunk_size: int = 100_000) -> None:
    if reference.shape != actual.shape:
        raise AssertionError(f"Feature shapes differ: {reference.shape} vs {actual.shape}")
    if reference.dtype != np.float32 or actual.dtype != np.float32:
        raise AssertionError(f"Expected float32, got {reference.dtype} and {actual.dtype}")
    for lo in range(0, len(reference), chunk_size):
        hi = min(lo + chunk_size, len(reference))
        ref_bits = reference[lo:hi].view(np.uint32)
        actual_bits = actual[lo:hi].view(np.uint32)
        if not np.array_equal(ref_bits, actual_bits):
            bad = np.argwhere(ref_bits != actual_bits)[0]
            raise AssertionError(
                f"Feature mismatch at row {lo + int(bad[0])}, column {int(bad[1])}"
            )


def _compare_sampled(reference: pd.DataFrame, reference_labels: np.ndarray,
                     actual: pd.DataFrame, actual_labels: np.ndarray) -> None:
    if list(reference.columns) != list(actual.columns):
        raise AssertionError("Post-subsample feature columns differ")
    for column in ("s1_id", "other_id"):
        if not np.array_equal(reference[column].to_numpy(), actual[column].to_numpy()):
            raise AssertionError(f"Post-subsample {column} values/order differ")
    _compare_feature_blocks(
        reference[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False),
        actual[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False),
    )
    if not np.array_equal(reference_labels, actual_labels):
        raise AssertionError("Post-subsample labels differ")


def _compare_chunk(pairs: pd.DataFrame, gt: dict[str, set[str]], seed: int,
                   executor: ProcessPoolExecutor, workers: int,
                   batch_size: int) -> tuple[int, int]:
    from pipeline import subsample_negatives_incremental

    started = time.perf_counter()
    serial = build_feature_frame_vectorized(pairs)
    if list(serial.columns) != ["s1_id", "other_id"] + FEATURE_NAMES:
        raise AssertionError("Serial output columns do not match the feature contract")
    _check_ids(serial, pairs, "serial")
    serial_labels = _labels(serial, gt)
    serial_features = serial[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False)

    # Keep the full serial reference on disk, while retaining only the small
    # post-subsample result in memory for exact comparison after multiprocessing.
    with tempfile.TemporaryDirectory(prefix="akari_mp_equal_") as tmp:
        memmap_path = Path(tmp) / "serial_features.npy"
        reference = None
        try:
            reference = np.lib.format.open_memmap(
                memmap_path, mode="w+", dtype=np.float32, shape=serial_features.shape)
            reference[:] = serial_features
            reference.flush()
            serial_sub, serial_sub_labels = subsample_negatives_incremental(
                serial, serial_labels, neg_per_pos_cap=15, seed=seed)
            del serial_features, serial
            gc.collect()

            parallel = build_feature_frame_multiprocess_experimental(
                pairs, executor, batch_size=batch_size,
                max_in_flight=workers,
            )
            if list(parallel.columns) != ["s1_id", "other_id"] + FEATURE_NAMES:
                raise AssertionError("Multiprocess output columns do not match the feature contract")
            _check_ids(parallel, pairs, "multiprocess")
            parallel_labels = _labels(parallel, gt)
            parallel_features = parallel[FEATURE_NAMES].to_numpy(dtype=np.float32, copy=False)
            _compare_feature_blocks(reference, parallel_features)
            if not np.array_equal(serial_labels, parallel_labels):
                raise AssertionError("Pre-subsample labels differ")

            parallel_sub, parallel_sub_labels = subsample_negatives_incremental(
                parallel, parallel_labels, neg_per_pos_cap=15, seed=seed)
            _compare_sampled(
                serial_sub, serial_sub_labels, parallel_sub, parallel_sub_labels)
            rows = len(pairs)
            sampled = len(serial_sub)
            del parallel_features, parallel, parallel_sub
            gc.collect()
        finally:
            if reference is not None:
                reference.flush()
                reference._mmap.close()
                del reference
                gc.collect()

    elapsed = time.perf_counter() - started
    print(f"PASS rows={rows:,} sampled={sampled:,} seconds={elapsed:.1f}", flush=True)
    return rows, sampled


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--sample-s1", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-chunk-size", type=int, default=2_500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=25_000)
    args = parser.parse_args()
    if args.sample_s1 != 5_000:
        parser.error("This parity harness is intentionally fixed to the 5k-S1 check")
    if args.train_chunk_size <= 0:
        parser.error("--train-chunk-size must be positive")
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")

    from pipeline import build_pair_frame

    s1, s2, s3, gt = _sample_training_data(args.data_dir, args.sample_s1, args.seed)
    print(f"sampled S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}", flush=True)

    print("building rare-channel candidates (max_df=300, rare_max_df=2000)...", flush=True)
    candidates_s2 = _block(s1, s2)
    candidates_s3 = _block(s1, s3)
    s1_ids = s1["entity_id"].tolist()
    other_lists = [list(candidates_s2.get(eid, ())) + list(candidates_s3.get(eid, ()))
                   for eid in s1_ids]

    total_rows = 0
    total_sampled = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for chunk_index, lo in enumerate(range(0, len(s1_ids), args.train_chunk_size), start=1):
            hi = min(lo + args.train_chunk_size, len(s1_ids))
            ids_chunk = s1_ids[lo:hi]
            cands_chunk = {eid: set(values) for eid, values in zip(ids_chunk, other_lists[lo:hi])}
            chunk_s2 = {eid: cands_chunk[eid].intersection(candidates_s2.get(eid, ()))
                        for eid in ids_chunk if candidates_s2.get(eid)}
            chunk_s3 = {eid: cands_chunk[eid].intersection(candidates_s3.get(eid, ()))
                        for eid in ids_chunk if candidates_s3.get(eid)}
            pairs = pd.concat([
                build_pair_frame(s1.iloc[lo:hi], s2, chunk_s2),
                build_pair_frame(s1.iloc[lo:hi], s3, chunk_s3),
            ], ignore_index=True)
            del chunk_s2, chunk_s3, cands_chunk
            print(f"chunk {chunk_index}: pairs={len(pairs):,}", flush=True)
            rows, sampled = _compare_chunk(
                pairs, gt, args.seed, executor, args.workers, args.batch_size)
            total_rows += rows
            total_sampled += sampled
            for eid in ids_chunk:
                candidates_s2.pop(eid, None)
                candidates_s3.pop(eid, None)
            del pairs
            gc.collect()

    print(f"PASS 5k-S1 exact parity rows={total_rows:,} sampled={total_sampled:,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
