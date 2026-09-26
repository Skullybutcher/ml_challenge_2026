"""Run the frozen pipeline with the requested plateau threshold selection.

This wrapper leaves pipeline/model/metric code and checkpoint fingerprints
unchanged. It evaluates the pipeline's same 37-point OOF threshold grid, then
uses the highest threshold whose macro F0.5 is within 0.001 of the grid best.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))

import pipeline  # noqa: E402
from metric import macro_f05  # noqa: E402
from model import _predictions_at_threshold  # noqa: E402


PLATEAU_TOLERANCE = 0.001


def tune_threshold_plateau(feat_df, calibrated_probs, gt, all_s1_ids):
    grid = np.linspace(0.05, 0.95, 37)
    s1_arr = feat_df["s1_id"].to_numpy()
    other_arr = feat_df["other_id"].to_numpy()
    curve = []
    for threshold in grid:
        predictions = _predictions_at_threshold(
            s1_arr, other_arr, calibrated_probs, float(threshold)
        )
        score = macro_f05(gt, predictions, entity_ids=all_s1_ids)
        curve.append((score, float(threshold)))

    best_score = max(score for score, _ in curve)
    eligible = [
        (score, threshold)
        for score, threshold in curve
        if score >= best_score - PLATEAU_TOLERANCE
    ]
    selected_score, selected_threshold = max(eligible, key=lambda item: item[1])
    top = sorted(curve, reverse=True)[:8]
    print(
        "Threshold curve (top 8): "
        + ", ".join(f"t={threshold:.2f}:{score:.4f}" for score, threshold in top),
        flush=True,
    )
    print(
        f"Threshold plateau (tolerance={PLATEAU_TOLERANCE:.3f}): "
        f"best={best_score:.4f}, selected={selected_threshold:.3f}, "
        f"selected_score={selected_score:.4f}",
        flush=True,
    )
    return selected_threshold, selected_score


pipeline.tune_threshold = tune_threshold_plateau
sys.argv = [str(SRC / "pipeline.py"), *sys.argv[1:]]
pipeline.main()
