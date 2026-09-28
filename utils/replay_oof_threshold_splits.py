#!/usr/bin/env python
"""Replay grouped OOF predictions from Run 1 feature checkpoints and score
single-variable threshold policies for source/singleton segments.

This utility never modifies the pipeline or the training checkpoints. It checks
that the reconstructed pair order and labels exactly match the saved OOF
artifacts before training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temp, path)


def f05_from_counts(tp: np.ndarray, predicted: np.ndarray, truth: np.ndarray) -> float:
    fp = predicted.astype(np.float64) - tp
    fn = truth.astype(np.float64) - tp
    denom = 1.25 * tp + fp + 0.25 * fn
    scores = np.ones(len(truth), dtype=np.float64)
    has_truth_or_pred = (truth > 0) | (predicted > 0)
    scores[has_truth_or_pred] = 0.0
    valid = denom > 0
    scores[valid] = 1.25 * tp[valid] / denom[valid]
    return float(scores.mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-s1", type=int, default=100_000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=0,
                        help="LightGBM threads; zero preserves the original default.")
    args = parser.parse_args()

    repo = args.repo_dir.resolve()
    data_dir = (args.data_dir or (repo.parent / "dataset")).resolve()
    out_dir = (args.out_dir or (repo / "out_100k")).resolve()
    checkpoint_root = out_dir / ".train_chunk_checkpoints"
    manifest_path = out_dir / "oof_artifact_manifest.json"
    pair_path = out_dir / "oof_pair_ids.npz"
    labels_path = out_dir / "oof_labels.npy"
    for path in (manifest_path, pair_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Run 1 artifact is missing: {path}")

    src = repo / "code" / "business_entity_resolution" / "src"
    sys.path.insert(0, str(src))
    from features import FEATURE_NAMES
    from io_utils import load_ground_truth, load_source
    from model import calibrate_oof, train_oof

    artifact_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_artifact = np.load(pair_path, allow_pickle=False)["pair_ids"]
    label_artifact = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    expected_rows = int(artifact_meta["rows"])
    if len(pair_artifact) != expected_rows or len(label_artifact) != expected_rows:
        raise ValueError("OOF artifact row counts do not match the manifest.")

    candidates: dict[Path, list[Path]] = {}
    for file in checkpoint_root.rglob("train_chunk_*.npz"):
        candidates.setdefault(file.parent, []).append(file)
    valid_dirs = []
    for directory, files in candidates.items():
        files = sorted(files)
        if len(files) != int(artifact_meta.get("chunks", 40)):
            continue
        indices = [int(path.stem.rsplit("_", 1)[1]) for path in files]
        if indices == list(range(len(files))):
            valid_dirs.append((directory, files))
    if len(valid_dirs) != 1:
        raise RuntimeError(
            f"Expected one complete checkpoint set for {expected_rows:,} OOF rows; "
            f"found {len(valid_dirs)} candidates: {[str(x[0]) for x in valid_dirs]}"
        )

    chunk_files = valid_dirs[0][1]
    x = np.empty((expected_rows, len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.empty(expected_rows, dtype=np.uint8)
    s1_ids = np.empty(expected_rows, dtype=pair_artifact.dtype)
    other_ids = np.empty(expected_rows, dtype=pair_artifact.dtype)
    cursor = 0
    fingerprint = None
    for index, path in enumerate(chunk_files):
        with np.load(path, allow_pickle=False) as chunk:
            if int(chunk["chunk_index"]) != index:
                raise ValueError(f"Unexpected chunk index in {path}")
            current_fingerprint = str(chunk["fingerprint"].item())
            if fingerprint is None:
                fingerprint = current_fingerprint
            elif current_fingerprint != fingerprint:
                raise ValueError("Checkpoint fingerprints differ within the selected run.")
            feats = chunk["features"]
            n = len(feats)
            if feats.shape != (n, len(FEATURE_NAMES)):
                raise ValueError(f"Feature matrix has unexpected shape in {path}: {feats.shape}")
            if cursor + n > expected_rows:
                raise ValueError("Checkpoint rows exceed the OOF manifest.")
            x[cursor:cursor+n] = feats
            labels[cursor:cursor+n] = chunk["labels"]
            s1_ids[cursor:cursor+n] = chunk["s1_ids"]
            other_ids[cursor:cursor+n] = chunk["other_ids"]
            cursor += n
        print(f"loaded cached train chunk {index + 1}/{len(chunk_files)} "
              f"({cursor:,}/{expected_rows:,} rows)", flush=True)
    if cursor != expected_rows:
        raise ValueError(f"Checkpoint rows {cursor:,} != manifest rows {expected_rows:,}")

    reconstructed_pairs = np.empty_like(pair_artifact)
    reconstructed_pairs[:, 0] = s1_ids
    reconstructed_pairs[:, 1] = other_ids
    if not np.array_equal(reconstructed_pairs, pair_artifact):
        raise ValueError("Checkpoint pair order does not match saved OOF pair IDs.")
    if not np.array_equal(labels, label_artifact):
        raise ValueError("Checkpoint labels do not match saved OOF labels.")
    if int(labels.sum()) != int(artifact_meta["positives"]):
        raise ValueError("Checkpoint positive count does not match the OOF manifest.")
    print("Checkpoint rows, pair IDs, and labels match the Run 1 OOF artifacts.", flush=True)

    frame = pd.DataFrame(x, columns=FEATURE_NAMES, copy=False)
    frame.insert(0, "s1_id", s1_ids)
    frame.insert(1, "other_id", other_ids)
    del x
    print(f"Replaying {args.n_splits}-fold grouped LightGBM OOF with seed={args.seed} "
          f"on {len(frame):,} checkpoint rows...", flush=True)
    started = time.time()
    lgb_params = {"num_threads": args.threads} if args.threads > 0 else None
    raw_probs, models = train_oof(
        frame, labels, n_splits=args.n_splits, lgb_params=lgb_params, seed=args.seed
    )
    del models, frame
    print(f"OOF model replay finished in {(time.time() - started) / 60:.1f} minutes.", flush=True)
    calibrated = calibrate_oof(raw_probs, labels).predict(raw_probs).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(out_dir / "oof_predictions.npy", raw_probs.astype(np.float32))
    atomic_save_npy(out_dir / "oof_calibrated_predictions.npy", calibrated)
    del raw_probs

    print("Recreating the exact 100k Run 1 S1 sample (seeded sampling, original order)...", flush=True)
    s1 = load_source(str(data_dir / "train" / "train_source1.tsv"), "S1-")
    rng = np.random.default_rng(args.seed)
    if args.sample_s1 < len(s1):
        keep_s1 = set(rng.choice(s1["entity_id"].to_numpy(),
                                 size=args.sample_s1, replace=False))
        sampled_s1 = s1.loc[s1["entity_id"].isin(keep_s1), "entity_id"].tolist()
    else:
        sampled_s1 = s1["entity_id"].tolist()
    gt_all = load_ground_truth(str(data_dir / "train" / "train_ground_truth.tsv"))
    gt = {key: value for key, value in gt_all.items() if key in set(sampled_s1)}
    del gt_all, s1
    if len(gt) != len(sampled_s1):
        raise ValueError("Recreated S1 sample and filtered ground truth do not align.")

    # Pair IDs map directly to sampled S1 positions; -1 would indicate corrupt data.
    row_group = pd.Index(sampled_s1).get_indexer(s1_ids)
    if np.any(row_group < 0):
        raise ValueError("A checkpoint pair references an S1 outside the sampled train set.")
    truth_counts = np.fromiter((len(gt.get(entity_id, set())) for entity_id in sampled_s1),
                               dtype=np.int32, count=len(sampled_s1))
    source2 = np.char.startswith(other_ids, "S2-")
    source3 = np.char.startswith(other_ids, "S3-")
    if np.any(~(source2 | source3)):
        raise ValueError("Found a candidate ID outside the S2/S3 source prefixes.")
    singleton_rows = truth_counts[row_group] == 0

    grid = np.linspace(0.05, 0.95, 37)
    curve_rows = []
    best_score = -1.0
    for threshold in grid:
        keep = calibrated >= threshold
        pred_counts = np.bincount(row_group[keep], minlength=len(sampled_s1))
        tp_counts = np.bincount(row_group[keep], weights=labels[keep],
                                minlength=len(sampled_s1))
        score = f05_from_counts(tp_counts, pred_counts, truth_counts)
        curve_rows.append({"threshold": float(threshold), "macro_f05": score})
        best_score = max(best_score, score)
    plateau = [row for row in curve_rows if row["macro_f05"] >= best_score - 0.001]
    base_threshold = max(row["threshold"] for row in plateau)
    base_score = next(row["macro_f05"] for row in curve_rows
                      if row["threshold"] == base_threshold)
    print("Full Run 1 OOF threshold curve (37 points):", flush=True)
    for row in curve_rows:
        print(f"t={row['threshold']:.3f} macro_f05={row['macro_f05']:.6f}", flush=True)
    print(f"Global best={best_score:.6f}; plateau-rule pick={base_threshold:.3f} "
          f"({base_score:.6f}).", flush=True)

    base_keep = calibrated >= base_threshold
    split_masks = {
        "S2 matches only": source2,
        "S3 matches only": source3,
        "non-singleton only": ~singleton_rows,
        "singleton only": singleton_rows,
        "S2 non-singleton": source2 & ~singleton_rows,
        "S3 non-singleton": source3 & ~singleton_rows,
    }
    split_rows = []
    for name, active in split_masks.items():
        if not np.any(active):
            continue
        other = ~active
        best = {"threshold": base_threshold, "macro_f05": base_score}
        for threshold in grid:
            keep = (base_keep & other) | ((calibrated >= threshold) & active)
            pred_counts = np.bincount(row_group[keep], minlength=len(sampled_s1))
            tp_counts = np.bincount(row_group[keep], weights=labels[keep],
                                    minlength=len(sampled_s1))
            score = f05_from_counts(tp_counts, pred_counts, truth_counts)
            if score > best["macro_f05"]:
                best = {"threshold": float(threshold), "macro_f05": score}
        split_rows.append({
            "split": name,
            "threshold": best["threshold"],
            "macro_f05": best["macro_f05"],
            "delta_vs_global_best": best["macro_f05"] - best_score,
            "delta_vs_plateau_policy": best["macro_f05"] - base_score,
        })
    curve_df = pd.DataFrame(curve_rows)
    split_df = pd.DataFrame(split_rows)
    curve_path = out_dir / "threshold_curve.csv"
    split_path = out_dir / "threshold_split_results.csv"
    curve_temp = curve_path.with_name(curve_path.name + ".tmp")
    split_temp = split_path.with_name(split_path.name + ".tmp")
    curve_df.to_csv(curve_temp, index=False)
    split_df.to_csv(split_temp, index=False)
    os.replace(curve_temp, curve_path)
    os.replace(split_temp, split_path)
    print("Split threshold results (each split is varied alone against the global plateau policy):",
          flush=True)
    print(split_df.to_string(index=False), flush=True)
    manifest = {
        "source": "out_100k/.train_chunk_checkpoints",
        "rows": expected_rows,
        "positives": int(labels.sum()),
        "pair_order_verified": True,
        "labels_verified": True,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "lightgbm_threads": args.threads or "original_default",
        "global_best_macro_f05": best_score,
        "plateau_threshold": base_threshold,
        "plateau_macro_f05": base_score,
        "threshold_grid_points": len(grid),
        "france_train_s1_rows": 0,
        "notes": "France has no labelled train S1 rows; no France threshold can be tuned from this OOF sample.",
    }
    manifest_temp = out_dir / "oof_replay_manifest.json.tmp"
    manifest_temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_temp, out_dir / "oof_replay_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
