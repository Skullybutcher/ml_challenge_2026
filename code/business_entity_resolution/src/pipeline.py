"""
End-to-end pipeline: load -> validate -> normalize -> block -> featurize ->
train (grouped OOF) -> calibrate -> tune threshold -> run test inference ->
write candidate_pairs.tsv + matching_results.tsv.

Memory-safe variant (code_ber_chunked):
  - Task 1: train featurization is done in chunks of --train-chunk-size
    (default 5000 S1 entities). Per chunk: pair frame -> features -> labels ->
    subsample_negatives; only the subsampled frames are concatenated.
  - Task 2: --skip-test skips loading test data and steps 8-9 (sample runs).
  - Task 3: candidate_pairs.tsv is streamed per test chunk instead of
    accumulating a dict-of-sets for the full test set.

Usage (from code/business_entity_resolution/src/):
    python pipeline.py --data-dir /kaggle/input/<your-dataset> \
                        --out-dir /kaggle/working/output
"""
from __future__ import annotations
import argparse
import gc
import os
import subprocess
import sys
import threading
import time
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from io_utils import load_source, load_ground_truth, validate_ground_truth_refs, write_result_tsv
from normalize import normalize_name, normalize_address
from blocking import (generate_candidates, generate_candidates_token,
                      generate_candidates_prefix, generate_candidates_rare,
                      union_candidates)


def cap_candidates_per_s1(cand_s2: Dict[str, Set[str]], cand_s3: Dict[str, Set[str]],
                          counted_s2, counted_s3, s1_ids: List[str],
                          max_pairs: int) -> tuple:
    """Per-S1 cap on the S2+S3 candidate union (mutates the dicts in place).

    Ranking per S1, by construction of the channels: the union = token-channel
    pairs (which have a shared-token count from the counted frame) plus
    prefix-EXCLUSIVE pairs (in the union, absent from the counted frame).
    Sort key per candidate:
        (-effective_score, counted_flag, other_id)
      effective_score = shared-token count if counted else 1 (a shared name
        prefix is worth one shared token),
      counted_flag = 0 for prefix-exclusive pairs, 1 for counted pairs.
    Property guaranteed: prefix-exclusive pairs (score 0 in raw terms) are
    NEVER ranked below token-weak (count-1) pairs — equal effective score,
    better tiebreak — so they are not silently dropped first; they only lose
    to pairs sharing >= 2 tokens. Verified by unit check before the 50k runs.

    counted_s2/counted_s3: optional (entity_id_s1, entity_id_other,
    shared_tokens) DataFrames from generate_candidates_token(return_counts=True).
    None (e.g. --use-tfidf) => every pair treated as count 1, tiebreak decides.

    Returns (per_s1_counts dict keyed by s1_id after capping).
    """
    per_s1_counts: Dict[str, int] = {}
    for eid in s1_ids:
        s2_set = cand_s2.get(eid) or set()
        s3_set = cand_s3.get(eid) or set()
        union = s2_set | s3_set
        per_s1_counts[eid] = len(union)
    if max_pairs <= 0:
        return per_s1_counts

    grouped = {}
    for name, df in (("s2", counted_s2), ("s3", counted_s3)):
        if df is not None and len(df) > 0:
            grouped[name] = df.groupby("entity_id_s1")

    for eid in s1_ids:
        s2_set = cand_s2.get(eid) or set()
        s3_set = cand_s3.get(eid) or set()
        union = s2_set | s3_set
        if len(union) <= max_pairs:
            continue
        token_scores: Dict[str, int] = {}
        for name in ("s2", "s3"):
            g = grouped.get(name)
            if g is None or eid not in g.groups:
                continue
            grp = g.get_group(eid)
            token_scores.update(zip(grp["entity_id_other"].to_numpy(),
                                    grp["shared_tokens"].to_numpy().tolist()))

        def rank_key(oid):
            cnt = token_scores.get(oid, 0)
            return (-(cnt if cnt > 0 else 1), 1 if cnt > 0 else 0, oid)

        keep_set = set(sorted(union, key=rank_key)[:max_pairs])
        cand_s2[eid] = keep_set & s2_set
        cand_s3[eid] = keep_set & s3_set
        per_s1_counts[eid] = len(keep_set)
    return per_s1_counts
from features import build_feature_frame_vectorized, FEATURE_NAMES
from model import train_oof, calibrate_oof, tune_threshold, predict_with_models


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class PeakRSSSampler:
    """Sample this process's RSS every `interval` seconds in a daemon thread."""

    def __init__(self, interval: float = 30.0):
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import psutil
        except ImportError:
            log("peak-RSS sampling disabled (psutil not installed)")
            return
        self._proc = psutil.Process()

        def _sample() -> None:
            while not self._stop.is_set():
                try:
                    self.peak_bytes = max(self.peak_bytes, self._proc.memory_info().rss)
                except Exception:
                    pass
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=_sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2)
        return self.peak_bytes / (1024 ** 3)


def add_norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_norm_name"] = df["business_name"].map(normalize_name)
    df["_norm_addr"] = df["business_address"].map(normalize_address)
    return df


def build_pair_frame(s1_df, other_df, candidates: Dict[str, Set[str]]) -> pd.DataFrame:
    """Vectorized pair assembly via merges (no per-row .loc)."""
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


def subsample_negatives(feat_df: pd.DataFrame, labels: np.ndarray, neg_per_pos_cap: int,
                        seed: int = 42) -> tuple:
    """Keep all positives; cap negatives per S1 entity. Training-only, not recall."""
    if len(feat_df) == 0:
        return feat_df, labels
    rng = np.random.default_rng(seed)
    df = feat_df.copy()
    df["_label"] = labels
    keep = []
    for _, grp in df.groupby("s1_id", sort=False):
        pos = grp.index[grp["_label"] == 1].to_numpy()
        neg = grp.index[grp["_label"] == 0].to_numpy()
        keep.extend(pos.tolist())
        cap = max(neg_per_pos_cap * max(len(pos), 1), 5)
        if len(neg) > cap:
            neg = rng.choice(neg, size=cap, replace=False)
        keep.extend(neg.tolist())
    keep = np.array(sorted(keep), dtype=np.int64)
    return feat_df.loc[keep].reset_index(drop=True), labels[keep]


def subsample_negatives_incremental(feat_df: pd.DataFrame, labels: np.ndarray,
                                    neg_per_pos_cap: int, seed: int = 42) -> tuple:
    """Memory-lean subsample_negatives: drops the feat_df.copy() and the string
    groupby (per-group frames come from positional slicing), then selects rows
    into a preallocated label array. Same row-selection semantics: all positives
    kept; negatives capped at max(neg_per_pos_cap * max(n_pos, 1), 5) per S1,
    sampled without replacement with a fixed RNG stream (same seed sequence per
    chunk, so per-S1 results are independent of chunk composition)."""
    if len(feat_df) == 0:
        return feat_df, labels
    rng = np.random.default_rng(seed)
    order = np.argsort(feat_df["s1_id"].to_numpy(), kind="stable")
    sorted_s1 = feat_df["s1_id"].to_numpy()[order]
    # String-safe group boundaries (np.diff is undefined for string dtypes).
    boundaries = np.flatnonzero(sorted_s1[1:] != sorted_s1[:-1]) + 1
    group_slices = np.split(order, boundaries)  # positional indices per S1 entity
    labels_arr = np.asarray(labels)
    keep_mask = np.zeros(len(feat_df), dtype=bool)
    for idx in group_slices:
        lab = labels_arr[idx]
        pos = idx[lab == 1]
        neg = idx[lab == 0]
        keep_mask[pos] = True
        cap = max(neg_per_pos_cap * max(len(pos), 1), 5)
        if len(neg) > cap:
            neg = rng.choice(neg, size=cap, replace=False)
            keep_mask[neg] = True
        else:
            keep_mask[neg] = True
    keep = np.flatnonzero(keep_mask)
    return feat_df.iloc[keep].reset_index(drop=True), labels_arr[keep]


def candidate_recall_from_lists(gt: Dict[str, Set[str]], s1_ids: List[str],
                                other_lists: List[List[str]]) -> float:
    """candidate_recall computed over per-S1 candidate LISTS (no union dict)."""
    total, found = 0, 0
    for s1_id, others in zip(s1_ids, other_lists):
        true_matches = gt.get(s1_id)
        if not true_matches:
            continue
        total += len(true_matches)
        found += len(true_matches.intersection(others))
    return found / total if total > 0 else 1.0


def write_candidate_tsv_row(f, s1_id: str, other_ids) -> None:
    """Write one candidate_pairs.tsv row in submission format (one row per S1,
    comma-joined sorted candidate ids, empty string when no candidates)."""
    f.write(f"{s1_id}\t{','.join(sorted(other_ids))}\n")


def run(data_dir: str, out_dir: str, n_splits: int = 5, seed: int = 42,
        sample_s1: int | None = None, max_df: int = 300, prefix_len: int = 4,
        max_pairs_per_prefix_key: int = 200_000, neg_per_pos_cap: int = 15,
        use_tfidf: bool = False, test_chunk_size: int | None = 100_000,
        train_chunk_size: int | None = 5000, skip_test: bool = False,
        max_pairs_per_s1: int = 0, use_rare: bool = False,
        rare_min_len: int = 6, rare_max_df: int = 2000) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rss = PeakRSSSampler(interval=30.0)
    rss.start()

    # ---------- 1. Load + validate ----------
    log("Loading training sources...")
    s1_tr = load_source(os.path.join(data_dir, "train", "train_source1.tsv"), "S1-")
    s2_tr = load_source(os.path.join(data_dir, "train", "train_source2.tsv"), "S2-")
    s3_tr = load_source(os.path.join(data_dir, "train", "train_source3.tsv"), "S3-")
    gt = load_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    validate_ground_truth_refs(gt, set(s1_tr.entity_id), set(s2_tr.entity_id), set(s3_tr.entity_id))
    log(f"Train sizes: S1={len(s1_tr)} S2={len(s2_tr)} S3={len(s3_tr)} | "
        f"GT rows={len(gt)} | singleton rate={sum(1 for v in gt.values() if not v)/len(gt):.3f}")

    if sample_s1 is not None and sample_s1 < len(s1_tr):
        log(f"--sample-s1: subsampling to {sample_s1} S1 (matched S2/S3 kept, distractors proportional).")
        rng = np.random.default_rng(seed)
        keep_s1 = set(rng.choice(s1_tr['entity_id'].to_numpy(), size=sample_s1, replace=False))
        s1_tr = s1_tr[s1_tr['entity_id'].isin(keep_s1)].reset_index(drop=True)
        gt = {k: v for k, v in gt.items() if k in keep_s1}
        needed = set().union(*gt.values()) if gt else set()
        s2_extra = set(rng.choice(s2_tr['entity_id'].to_numpy(), size=min(len(s2_tr), sample_s1 * 3), replace=False))
        s3_extra = set(rng.choice(s3_tr['entity_id'].to_numpy(), size=min(len(s3_tr), sample_s1 * 3), replace=False))
        s2_tr = s2_tr[s2_tr['entity_id'].isin(needed | s2_extra)].reset_index(drop=True)
        s3_tr = s3_tr[s3_tr['entity_id'].isin(needed | s3_extra)].reset_index(drop=True)
        log(f"Sampled train: S1={len(s1_tr)} S2={len(s2_tr)} S3={len(s3_tr)}")

    if skip_test:
        log("--skip-test: not loading test data; steps 8-9 (inference + outputs) will be skipped.")
        s1_te = s2_te = s3_te = None
    else:
        log("Loading test sources...")
        s1_te = load_source(os.path.join(data_dir, "test", "test_source1.tsv"), "S1-")
        s2_te = load_source(os.path.join(data_dir, "test", "test_source2.tsv"), "S2-")
        s3_te = load_source(os.path.join(data_dir, "test", "test_source3.tsv"), "S3-")
        log(f"Test sizes: S1={len(s1_te)} S2={len(s2_te)} S3={len(s3_te)} | "
            f"test countries={sorted(s1_te.country.unique())}")

    # ---------- 2. Normalize ----------
    log("Normalizing text fields...")
    s1_tr, s2_tr, s3_tr = (add_norm_columns(d) for d in (s1_tr, s2_tr, s3_tr))
    if not skip_test:
        s1_te, s2_te, s3_te = (add_norm_columns(d) for d in (s1_te, s2_te, s3_te))

    def block(s1, other, return_counts: bool = False):
        if use_tfidf:
            cands = generate_candidates(s1, other)
            if return_counts:
                return cands, None
            return cands
        token_c = generate_candidates_token(s1, other, max_df=max_df, return_counts=True)
        # Sets for the union channel come from the SAME merge as the counted
        # frame (counted rows = union of that merge), so the union is exact.
        token_sets: Dict[str, Set[str]] = {}
        for s1_id, grp in token_c.groupby("entity_id_s1"):
            token_sets[s1_id] = set(grp["entity_id_other"].to_numpy())
        prefix_sets = generate_candidates_prefix(s1, other, prefix_len=prefix_len,
                                                 max_pairs_per_key=max_pairs_per_prefix_key)
        if use_rare:
            rare_sets = generate_candidates_rare(s1, other, max_df=max_df,
                                                 min_len=rare_min_len,
                                                 max_df_long=rare_max_df)
            if return_counts:
                return union_candidates(token_sets, prefix_sets, rare_sets), token_c
            return union_candidates(token_sets, prefix_sets, rare_sets)
        if return_counts:
            return union_candidates(token_sets, prefix_sets), token_c
        return union_candidates(token_sets, prefix_sets)

    # ---------- 3. Blocking on TRAIN ----------
    t_block0 = time.time()
    log(f"Blocking train S1 vs S2 (max_df={max_df})...")
    cand_tr_s2, counted_s2 = block(s1_tr, s2_tr, return_counts=True)
    log(f"Blocking train S1 vs S3 (max_df={max_df})...")
    cand_tr_s3, counted_s3 = block(s1_tr, s3_tr, return_counts=True)
    block_min = (time.time() - t_block0) / 60.0
    log(f"[TIMING] blocking (both sources): {block_min:.1f} min")

    s1_ids_all = s1_tr["entity_id"].tolist()
    country_all = s1_tr["country"].fillna("").astype(str).str.upper().to_numpy()

    if max_pairs_per_s1 > 0:
        pre_cap_total = sum(len(v) for v in cand_tr_s2.values()) + \
            sum(len(v) for v in cand_tr_s3.values())
        log(f"Per-S1 cap {max_pairs_per_s1}: pre-cap union pairs={pre_cap_total:,}")
        cap_candidates_per_s1(cand_tr_s2, cand_tr_s3, counted_s2, counted_s3,
                              s1_ids_all, max_pairs_per_s1)
        del counted_s2, counted_s3
        gc.collect()

    # Recall gate on the union, computed without materializing a third
    # union dict (a full-candidate union costs ~1GB at 50k S1).
    other_lists_all = [
        list(cand_tr_s2.get(eid, ())) + list(cand_tr_s3.get(eid, ()))
        for eid in s1_ids_all
    ]
    cand_tr_total = sum(len(v) for v in cand_tr_s2.values()) + \
        sum(len(v) for v in cand_tr_s3.values())
    recall = candidate_recall_from_lists(gt, s1_ids_all, other_lists_all)
    log(f"Train candidate pairs (S2+S3 union): {cand_tr_total:,}")
    log(f"Train candidate recall: {recall:.4f}")

    # Per-country recall split (overall recall gate stays on the full set).
    _us_idx = [i for i, c in enumerate(country_all) if c == "US"]
    _in_idx = [i for i, c in enumerate(country_all) if c == "IN"]
    for _name, _idx in (("US", _us_idx), ("IN", _in_idx)):
        _ids = [s1_ids_all[i] for i in _idx]
        _lists = [other_lists_all[i] for i in _idx]
        _r = candidate_recall_from_lists(gt, _ids, _lists)
        log(f"Train candidate recall [{_name}] (n={len(_ids):,}): {_r:.4f}")

    cnts = np.array([len(v) for v in other_lists_all], dtype=np.int64)
    log(f"Per-S1 candidate counts: p50={int(np.percentile(cnts, 50)):,} "
        f"p99={int(np.percentile(cnts, 99)):,} max={int(cnts.max()):,} "
        f"mean={cnts.mean():.1f} | at-cap S1: {int((cnts >= max_pairs_per_s1).sum()):,}"
        if max_pairs_per_s1 > 0 else
        f"Per-S1 candidate counts: p50={int(np.percentile(cnts, 50)):,} "
        f"p99={int(np.percentile(cnts, 99)):,} max={int(cnts.max()):,} mean={cnts.mean():.1f}")
    if recall < 0.90:
        log("WARNING: recall < 0.90 — lower --max-df / --prefix-len before trusting matcher.")

    # ---------- 4. Build labeled pairs (chunked featurization) ----------
    # Full train set: keep per-S1 candidate lists for recall gating (cheap),
    # then featurize in chunks of --train-chunk-size S1 entities. Per chunk:
    # build_pair_frame -> build_feature_frame_vectorized -> labels ->
    # subsample_negatives; only the subsampled frames are concatenated.
    chunk = train_chunk_size or len(s1_tr)
    n_tr_chunks = (len(s1_ids_all) + chunk - 1) // chunk
    t_feat0 = time.time()
    if n_tr_chunks == 1:
        log(f"Featurizing train in 1 chunk ({len(s1_ids_all):,} S1)...")
    else:
        log(f"Featurizing train in {n_tr_chunks} chunk(s) of ~{chunk:,} S1...")

    feat_parts: List[pd.DataFrame] = []
    label_parts: List[np.ndarray] = []
    pairs_pre_subsample = 0
    pos_pre_subsample = 0
    pairs_post_subsample = 0

    for ci in range(n_tr_chunks):
        lo, hi = ci * chunk, (ci + 1) * chunk
        s1_ids_chunk = s1_ids_all[lo:hi]
        lists_chunk = other_lists_all[lo:hi]
        cands_chunk = {eid: set(lst) for eid, lst in zip(s1_ids_chunk, lists_chunk)}
        # Split per source for pair assembly, then drop this chunk's entries so
        # at most one chunk's worth of candidate strings is alive at a time.
        chunk_s2 = {eid: cands_chunk[eid].intersection(cand_tr_s2.get(eid, ()))
                    for eid in s1_ids_chunk if cand_tr_s2.get(eid)}
        chunk_s3 = {eid: cands_chunk[eid].intersection(cand_tr_s3.get(eid, ()))
                    for eid in s1_ids_chunk if cand_tr_s3.get(eid)}
        for eid in s1_ids_chunk:
            cand_tr_s2.pop(eid, None)
            cand_tr_s3.pop(eid, None)
        cands_chunk.clear()
        del lists_chunk

        s1_chunk = s1_tr.iloc[lo:hi]
        pairs_tr = pd.concat([build_pair_frame(s1_chunk, s2_tr, chunk_s2),
                              build_pair_frame(s1_chunk, s3_tr, chunk_s3)],
                             ignore_index=True)
        del chunk_s2, chunk_s3
        n_pre = len(pairs_tr)
        feat_df = build_feature_frame_vectorized(pairs_tr)
        del pairs_tr
        labels = np.array([1 if o in gt.get(s, set()) else 0
                           for s, o in zip(feat_df["s1_id"].to_numpy(),
                                           feat_df["other_id"].to_numpy())])
        pairs_pre_subsample += n_pre
        pos_pre_subsample += int(labels.sum())
        if neg_per_pos_cap > 0:
            feat_df, labels = subsample_negatives_incremental(
                feat_df, labels, neg_per_pos_cap, seed=seed)
        pairs_post_subsample += len(feat_df)
        feat_parts.append(feat_df)
        label_parts.append(labels)
        log(f"train chunk {ci + 1}/{n_tr_chunks}: pairs={n_pre:,} -> "
            f"subsampled={len(feat_df):,} (pos in chunk: {int(labels.sum()):,})")

    del cand_tr_s2, cand_tr_s3, other_lists_all
    gc.collect()
    feat_df = pd.concat(feat_parts, ignore_index=True)
    labels = np.concatenate(label_parts) if label_parts else np.array([], dtype=np.int64)
    del feat_parts, label_parts
    gc.collect()
    log(f"Pairs before subsample: {pairs_pre_subsample:,} | pos: {pos_pre_subsample:,}")
    log(f"Pairs after subsample (cap={neg_per_pos_cap}x): {pairs_post_subsample:,} | pos: {int(labels.sum()):,}")
    log(f"[TIMING] featurize (all chunks): {(time.time() - t_feat0) / 60.0:.1f} min")

    # ---------- 5. Train grouped-OOF GBDT ----------
    log("Training GBDT with GroupKFold (grouped by S1 entity)...")
    oof_raw, models = train_oof(feat_df, labels, n_splits=n_splits, seed=seed)

    # ---------- 6. Calibrate ----------
    log("Calibrating OOF probabilities (isotonic)...")
    iso = calibrate_oof(oof_raw, labels)
    oof_calibrated = iso.predict(oof_raw)

    # ---------- 7. Tune threshold against macro F0.5 ----------
    log("Tuning decision threshold against macro F0.5...")
    all_s1_ids = s1_ids_all
    best_t, best_score = tune_threshold(feat_df, oof_calibrated, gt, all_s1_ids)
    log(f"Best threshold={best_t:.3f} -> OOF macro F0.5={best_score:.4f}")

    # sanity: all-singleton baseline, for comparison
    empty_preds = {eid: set() for eid in all_s1_ids}
    from metric import macro_f05
    baseline = macro_f05(gt, empty_preds, entity_ids=all_s1_ids)
    log(f"All-singleton baseline macro F0.5 on train: {baseline:.4f} (your real edge over the "
        f"leaderboard is measured above this floor, not from 0)")

    peak_gb = rss.stop()
    log(f"[RAM] peak RSS so far: {peak_gb:.2f} GB")

    # F4: free train-stage frames before test inference (peak moment).
    # Kept: models, iso, best_t, s1_te/s2_te/s3_te, all_s1_ids, rss.
    del feat_df, oof_raw, labels, s1_tr, s2_tr, s3_tr, gt
    gc.collect()

    if skip_test:
        log("--skip-test: skipping steps 8-9 (test inference + outputs).")
        return

    # ---------- 8. Inference: blocking on TEST (chunked to bound peak RAM) ----------
    # The full test join (1.7M S1 x ~10M S2/S3) never fits in RAM at once, and
    # candidate_pairs.tsv is streamed per chunk (Task 3) so nothing accumulates.
    all_test_s1_ids = s1_te["entity_id"].tolist()
    matches: Dict[str, Set[str]] = {eid: set() for eid in all_test_s1_ids}
    cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
    chunk = test_chunk_size or len(s1_te)
    n_chunks = (len(s1_te) + chunk - 1) // chunk
    log(f"Test inference in {n_chunks} chunk(s) of ~{chunk:,} S1 (test S1={len(s1_te):,})...")
    with open(cand_path, "w", encoding="utf-8", newline="\n") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for ci in range(n_chunks):
            s1_chunk = s1_te.iloc[ci * chunk:(ci + 1) * chunk]
            t0 = time.time()
            c_s2 = block(s1_chunk, s2_te)
            c_s3 = block(s1_chunk, s3_te)
            pairs_te = pd.concat([build_pair_frame(s1_chunk, s2_te, c_s2),
                                  build_pair_frame(s1_chunk, s3_te, c_s3)], ignore_index=True)
            feat_te = build_feature_frame_vectorized(pairs_te)
            del pairs_te
            if len(feat_te) > 0:
                probs = iso.predict(predict_with_models(models, feat_te[FEATURE_NAMES].to_numpy()))
                keep = probs >= best_t
                for s1_id, other_id in zip(feat_te["s1_id"].to_numpy()[keep],
                                           feat_te["other_id"].to_numpy()[keep]):
                    matches[s1_id].add(other_id)
            # Task 3: stream candidates straight to disk, one row per S1.
            s1_chunk_ids = s1_chunk["entity_id"].tolist()
            s1_chunk_set = set(s1_chunk_ids)
            for k, v in c_s2.items():
                c_s3.setdefault(k, set()).update(v)
            del c_s2
            for eid in s1_chunk_ids:
                write_candidate_tsv_row(f_cand, eid, c_s3.pop(eid, ()))
            c_s3.clear()  # keys outside this chunk cannot exist; drop defensively
            log(f"chunk {ci + 1}/{n_chunks}: pairs={len(feat_te):,} "
                f"matched_so_far={sum(1 for v in matches.values() if v):,} ({time.time() - t0:.1f}s)")
            del c_s3, feat_te
            gc.collect()

    # ---------- 9. Write outputs ----------
    write_result_tsv(os.path.join(out_dir, "matching_results.tsv"), matches, all_test_s1_ids)
    log(f"Wrote matching_results.tsv (and streamed candidate_pairs.tsv) to {out_dir}")

    n_matched = sum(1 for v in matches.values() if v)
    log(f"Predicted non-singletons: {n_matched}/{len(all_test_s1_ids)} "
        f"({n_matched/len(all_test_s1_ids):.3f})")

    peak_gb = rss.stop()
    log(f"[RAM] peak RSS (whole run): {peak_gb:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir containing train/ and test/ subfolders")
    ap.add_argument("--out-dir", required=True, help="dir to write matching_results.tsv / candidate_pairs.tsv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-s1", type=int, default=None)
    ap.add_argument("--max-df", type=int, default=300)
    ap.add_argument("--prefix-len", type=int, default=4)
    ap.add_argument("--max-pairs-per-prefix-key", type=int, default=200_000)
    ap.add_argument("--neg-per-pos-cap", type=int, default=15)
    ap.add_argument("--use-tfidf", action="store_true", help="small-sample TF-IDF blocking only")
    ap.add_argument("--test-chunk-size", type=int, default=100_000,
                    help="test S1 rows per inference chunk; bounds peak RAM (0 = no chunking)")
    ap.add_argument("--train-chunk-size", type=int, default=5000,
                    help="train S1 rows per featurization chunk; bounds peak RAM (0 = no chunking)")
    ap.add_argument("--max-pairs-per-s1", type=int, default=0,
                    help="rank candidates per S1 (shared-token count desc, prefix-exclusive "
                         "pairs survive ties) and keep top-N; 0 = no cap (default)")
    ap.add_argument("--skip-test", action="store_true",
                    help="skip test load + inference/outputs (steps 8-9); for sample runs")
    ap.add_argument("--use-rare", action="store_true",
                    help="add long-token rescue channel (F6a) to blocking; off by default")
    ap.add_argument("--rare-min-len", type=int, default=6,
                    help="min token length for the rescue channel (default 6)")
    ap.add_argument("--rare-max-df", type=int, default=2000,
                    help="upper df bound of the rescue band (default 2000)")
    ap.add_argument("--validate", action="store_true",
                    help="also run utils/validate_submission.py if found alongside --data-dir")
    args = ap.parse_args()

    run(args.data_dir, args.out_dir, n_splits=args.n_splits, seed=args.seed,
        sample_s1=args.sample_s1, max_df=args.max_df, prefix_len=args.prefix_len,
        max_pairs_per_prefix_key=args.max_pairs_per_prefix_key,
        neg_per_pos_cap=args.neg_per_pos_cap, use_tfidf=args.use_tfidf,
        test_chunk_size=(args.test_chunk_size or None),
        train_chunk_size=(args.train_chunk_size or None),
        skip_test=args.skip_test,
        max_pairs_per_s1=args.max_pairs_per_s1,
        use_rare=args.use_rare, rare_min_len=args.rare_min_len,
        rare_max_df=args.rare_max_df)

    if args.validate:
        validator = os.path.join(args.data_dir, "utils", "validate_submission.py")
        if os.path.exists(validator):
            log("Running official validator...")
            subprocess.run([
                sys.executable, validator,
                "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
                "--candidate", os.path.join(args.out_dir, "candidate_pairs.tsv"),
                "--test-dir", os.path.join(args.data_dir, "test"),
            ], check=False)
        else:
            log(f"Validator not found at {validator}, skipping.")


if __name__ == "__main__":
    main()
