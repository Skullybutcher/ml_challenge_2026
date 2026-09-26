"""Experimental bounded multiprocessing version of the scalar featurizer.

This module is deliberately separate from the production pipeline. The
post-Run-1 equality check compares it with features.build_feature_frame_vectorized
before any use in a run.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Executor, wait
from typing import Dict

import numpy as np
import pandas as pd

from features import FEATURE_NAMES, pair_features


_TEXT_COLUMNS = ("name1", "addr1", "country1", "name2", "addr2", "country2")


def _feature_batch(columns: tuple[list[str], ...]) -> np.ndarray:
    """Worker entry point; receives only one bounded batch of text columns."""
    name1, addr1, country1, name2, addr2, country2 = columns
    values = [pair_features(n1, a1, c1, n2, a2, c2)
              for n1, a1, c1, n2, a2, c2 in zip(
                  name1, addr1, country1, name2, addr2, country2)]
    result = np.asarray(values, dtype=np.float32)
    expected = (len(name1), len(FEATURE_NAMES))
    if result.shape != expected:
        raise ValueError(f"Worker returned shape {result.shape}; expected {expected}")
    return result


def build_feature_frame_multiprocess_experimental(
    pairs_df: pd.DataFrame,
    executor: Executor,
    *,
    batch_size: int = 25_000,
    max_in_flight: int = 6,
) -> pd.DataFrame:
    """Compute the existing scalar features in bounded, offset-tagged batches.

    The caller's pair-frame order is preserved exactly. At most
    ``max_in_flight`` text batches are submitted at once, and workers never
    receive the full pair frame.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=["s1_id", "other_id"] + FEATURE_NAMES)
    if not 1 <= max_in_flight <= 8:
        raise ValueError("max_in_flight must be between 1 and 8")

    # Retain lightweight Series views, converting only the bounded slice that
    # is about to be submitted. This avoids six additional full-length Python
    # lists alongside the pair frame.
    columns = [pairs_df[name] for name in _TEXT_COLUMNS]
    features = np.empty((len(pairs_df), len(FEATURE_NAMES)), dtype=np.float32)
    pending: Dict[object, tuple[int, int]] = {}
    next_start = 0

    while next_start < len(pairs_df) or pending:
        while next_start < len(pairs_df) and len(pending) < max_in_flight:
            start = next_start
            end = min(start + batch_size, len(pairs_df))
            payload = tuple(
                column.iloc[start:end].fillna("").astype(str).tolist()
                for column in columns
            )
            future = executor.submit(_feature_batch, payload)
            del payload
            pending[future] = (start, end)
            next_start = end

        completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in completed:
            start, end = pending.pop(future)
            batch = future.result()
            expected = (end - start, len(FEATURE_NAMES))
            if batch.shape != expected:
                raise ValueError(f"Batch returned shape {batch.shape}; expected {expected}")
            features[start:end] = batch
            del batch

    result = pd.DataFrame(features, columns=FEATURE_NAMES)
    result.insert(0, "s1_id", pairs_df["s1_id"].to_numpy())
    result.insert(1, "other_id", pairs_df["other_id"].to_numpy())
    return result
