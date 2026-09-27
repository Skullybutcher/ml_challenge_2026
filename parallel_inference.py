#!/usr/bin/env python3
"""
Parallel test inference for Business Entity Resolution.
Uses multiprocessing to score test chunks in parallel.
"""
import os
import sys
import gc
import time
import pickle
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent / "code" / "business_entity_resolution" / "src"))

from io_utils import load_source, write_result_tsv
from normalize import normalize_name, normalize_address
from blocking import (generate_candidates_token, generate_candidates_prefix,
                      generate_candidates_rare, union_candidates)
from features import build_feature_frame_vectorized, FEATURE_NAMES
from model import predict_with_models
from normalize import normalize_name, normalize_address
from checkpointing import chunk_checkpoint_path, load_training_chunk, s1_chunk_digest


DEFAULT_PREFIX = "passage: "


def add_norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_norm_name"] = df["business_name"].map(normalize_name)
    df["_norm_addr"] = df["business_address"].map(normalize_address)
    return df


def block(s1, other, max_df=300, prefix_len=4, max_pairs_per_prefix_key=200_000, use_rare=False,
          rare_min_len=6, rare_max_df=2000):
    """Generate candidates for one S1 chunk against other source."""
    token_c = generate_candidates_token(s1, other, max_df=max_df, return_counts=False)
    token_sets = {}
    for s1_id, grp in token_c.groupby("entity_id_s1"):
        token_sets[s1_id] = set(grp["entity_id_other"].to_numpy())
    prefix_sets = generate_candidates_prefix(s1, other, prefix_len=prefix_len,
                                             max_pairs_per_key=200_000)
    if True:  # use_rare is always True in our config
        rare_sets = generate_candidates_rare(s1, other, max_df=300,
                                             min_len=6, max_df_long=1000)
        return union_candidates(token_sets, prefix_sets, rare_sets)
    return union_candidates(token_sets, prefix_sets)


def process_chunk(args):
    """Process a single test chunk in a worker process."""
    (chunk_idx, s1_chunk, s2_te, s3_te, models, iso, best_t,
     feat_names, use_e5, e5_store, france_thresh, other_thresh) = args

    t0 = time.time()
    
    # Blocking
    c_s2 = block(s1_chunk, s2_te)
    c_s3 = block(s1_chunk, s3_te)
    
    # Build pair frame
    pairs_te = pd.concat([
        build_pair_frame(s1_chunk, s2_te, c_s2),
        build_pair_frame(s1_chunk, s3_te, c_s3)
    ], ignore_index=True)
    
    if len(pairs_te) == 0:
        return chunk_idx, {}, {}
    
    # Features
    feat_te = build_feature_frame_vectorized(pairs_te, use_e5=False)  # E5 handled separately
    
    # Predict
    probs = np.mean([m.predict(feat_te[feat_names].to_numpy(), num_iteration=m.best_iteration) for m in models], axis=0)
    probs = iso.predict(probs)
    
    # Threshold with France-specific rule
    countries = pairs_te["country1"].fillna("").astype(str).to_numpy()
    row_thresholds = np.where(
        np.char.lower(np.char.strip(countries)) == "france",
        0.95, 0.65
    )
    keep = probs >= row_thresholds
    
    # Collect matches
    matches = {}
    for s1_id, other_id in zip(pairs_te["s1_id"].to_numpy()[keep],
                               pairs_te["other_id"].to_numpy()[keep]):
        matches.setdefault(s1_id, set()).add(other_id)
    
    # Candidates (stream all)
    cand = {}
    for k, v in c_s2.items():
        cand.setdefault(k, set()).update(v)
    for k, v in c_s3.items():
        cand.setdefault(k, set()).update(v)
    
    # Ensure all S1 in chunk have entries
    for eid in s1_chunk["entity_id"]:
        cand.setdefault(eid, set())
    
    return chunk_idx, matches, cand, time.time() - t0


def load_models_and_iso(out_dir):
    """Load trained models and isotonic calibrator from checkpoint."""
    # Find model files
    model_files = sorted(Path(out_dir).glob("model_fold_*.txt"))
    if not model_files:
        # Try to find in current directory
        model_files = sorted(Path(".").glob("model_fold_*.txt"))
    
    models = []
    for mf in model_files:
        models.append(lgb.Booster(model_file=str(mf)))
    
    # Load isotonic calibrator
    iso_path = Path(out_dir) / "isotonic.pkl"
    if iso_path.exists():
        with open(iso_path, "rb") as f:
            iso = pickle.load(f)
    else:
        iso = None
    
    # Load best threshold
    threshold_path = Path(out_dir) / "best_threshold.txt"
    if threshold_path.exists():
        best_t = float(threshold_path.read_text().strip())
    else:
        best_t = 0.72
    
    return models, iso, best_t


def build_pair_frame(s1_df, other_df, candidates: Dict[str, Set[str]]) -> pd.DataFrame:
    """Vectorized pair assembly via merges."""
    s1_ids, other_ids = [], []
    for s1_id, others in candidates.items():
        for other_id in others:
            s1_ids.append(s1_id)
            other_ids.append(other_id)
    if not s1_ids:
        return pd.DataFrame(columns=["s1_id", "other_id", "name1", "addr1", "country1",
                                      "name2", "addr2", "country2"])
    pairs = pd.DataFrame({"s1_id": s1_ids, "other_id": other_ids})
    s1_cols = s1_df[["entity_id", "_norm_name", "_norm_addr", "country"]].rename(
        columns={"entity_id": "s1_id", "_norm_name": "name1", "_norm_addr": "addr1", "country": "country1"})
    other_cols = other_df[["entity_id", "_norm_name", "_norm_addr", "country"]].rename(
        columns={"entity_id": "other_id", "_norm_name": "name2", "_norm_addr": "addr2", "country": "country2"})
    return pairs.merge(s1_cols, on="s1_id", how="left").merge(other_cols, on="other_id", how="left")


def main():
    parser = argparse.ArgumentParser(description="Parallel test inference")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--test-chunk-size", type=int, default=25000)
    parser.add_argument("--use-e5", action="store_true")
    parser.add_argument("--e5-name-dir", default=None)
    parser.add_argument("--e5-address-dir", default=None)
    parser.add_argument("--france-threshold", type=float, default=0.95)
    parser.add_argument("--other-threshold", type=float, default=0.65)
    parser.add_argument("--use-rare", action="store_true")
    parser.add_argument("--rare-max-df", type=int, default=1000)
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    OUT_DIR = Path(args.out_dir)
    MODEL_DIR = Path(args.model_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models and calibrator
    print("Loading models...")
    models, iso, best_t = load_models_and_iso(args.model_dir)
    print(f"Loaded {len(models)} models, isotonic={'yes' if iso else 'no'}, threshold={best_t:.3f}")

    # Load test data
    print("Loading test data...")
    s1_te = load_source(DATA_DIR / "test" / "test_source1.tsv", "S1-")
    s2_te = load_source(DATA_DIR / "test" / "test_source2.tsv", "S2-")
    s3_te = load_source(DATA_DIR / "test" / "test_source3.tsv", "S3-")
    
    # Normalize
    s1_te, s2_te, s3_te = (add_norm_columns(d) for d in (s1_te, s2_te, s3_te))

    # Initialize E5 store if needed
    e5_store = None
    if args.use_e5:
        sys.path.insert(0, str(Path(__file__).parent / "code" / "business_entity_resolution" / "src"))
        from features import E5EmbeddingStore
        e5_store = E5EmbeddingStore(args.e5_name_dir, args.e5_address_dir)
        for frame, table in ((s1_te, "test_s1"), (s2_te, "test_s2"), (s3_te, "test_s3")):
            e5_store.attach(s1_te, "test_s1")
            e5_store.attach(s2_te, "test_s2")
            e5_store.attach(s3_te, "test_s3")

    # Split test S1 into chunks
    chunk_size = args.test_chunk_size
    n_chunks = (len(s1_te) + args.test_chunk_size - 1) // args.test_chunk_size
    chunks = [s1_te.iloc[i * args.test_chunk_size:(i + 1) * args.test_chunk_size] for i in range(n_chunks)]
    print(f"Test inference: {n_chunks} chunks of ~{args.test_chunk_size:,} S1")

    # Prepare worker arguments
    worker_args = []
    for ci, chunk in enumerate(chunks):
        worker_args.append((
            ci, chunk, s2_te, s3_te, models, iso, 0.72,  # threshold will be overridden per-country
            FEATURE_NAMES, False, None, 0.95, 0.65
        ))

    # Run parallel inference
    print(f"Starting parallel inference with {args.workers} workers...")
    t0 = time.time()
    all_matches = {}
    all_cands = {}

    with Pool(processes=args.workers) as pool:
        results = pool.map(process_chunk, worker_args)
    
    # Merge results
    for chunk_idx, matches, cands, elapsed in sorted(results):
        all_matches.update(matches)
        for k, v in cands.items():
            all_cands.setdefault(k, set()).update(v)
        print(f"Chunk {chunk_idx}: {len(matches)} matches, {sum(len(v) for v in cands.values())} cands, {elapsed:.1f}s")

    # Write outputs
    print("Writing outputs...")
    write_result_tsv(os.path.join(args.out_dir, "matching_results.tsv"), all_matches, s1_te["entity_id"].tolist())
    write_result_tsv(os.path.join(args.out_dir, "candidate_pairs.tsv"), all_cands, s1_te["entity_id"].tolist())
    
    # Validate
    validator = Path(args.data_dir) / "utils" / "validate_submission.py"
    if validator.exists():
        print("Running validator...")
        subprocess.run([
            sys.executable, str(validator),
            "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
            "--candidate", os.path.join(args.out_dir, "candidate_pairs.tsv"),
            "--test-dir", str(DATA_DIR / "test")
        ], check=False)
        subprocess.run([
            sys.executable, str(validator),
            "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
            "--test-dir", str(DATA_DIR / "test"),
            "--check-ids"
        ], check=False)

    print(f"Total time: {time.time() - t0:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()