#!/usr/bin/env python3
"""Resumable writer for the exact combined E5 artifact contract.

The train/test vectors are written directly to their final combined memmaps;
this avoids materializing a second full copy with vstack/concatenate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from generate_embeddings import (
    DEFAULT_PREFIX, DIMENSION, ID_DTYPE, MODEL_NAME, MODEL_REVISION,
    TABLES, VECTOR_DTYPE, atomic_json, embed_batch, enforce_resource_limits,
    table_rows,
)


def _split_tables(split: str) -> tuple[str, ...]:
    return ("train_s1", "train_s2", "train_s3") if split == "train" else (
        "test_s1", "test_s2", "test_s3")


def _config(split: str, field: str, tables: tuple[str, ...], counts: dict,
            paths: dict, args) -> dict:
    return {
        "split": split, "field": field, "rows_by_table": {t: counts[t] for t in tables},
        "inputs": {t: {"path": str(paths[t]), "size": paths[t].stat().st_size,
                        "mtime_ns": paths[t].stat().st_mtime_ns} for t in tables},
        "model": MODEL_NAME, "revision": MODEL_REVISION, "prefix": args.prefix,
        "max_length": args.max_length, "dimension": DIMENSION,
        "dtype": "S16" if field == "ids" else "float16",
        "pooling": "source_ids" if field == "ids" else "masked_mean",
        "normalize_l2": field != "ids", "row_order": list(tables),
    }


def _open_array(root: Path, stem: str, shape: tuple[int, ...], dtype, config: dict):
    partial, final = root / f"{stem}.partial.npy", root / f"{stem}.npy"
    state_path = root / f"{stem}.state.json"
    if final.exists():
        if not state_path.is_file():
            raise RuntimeError(f"Output exists without state; refusing overwrite: {final}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete") and state.get("config") == config:
            return None, state, state_path, partial, final
        raise RuntimeError(f"Output config mismatch; preserving existing file: {final}")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != config or not partial.exists():
            raise RuntimeError(f"Resume config mismatch/missing partial for {stem}; preserve files")
        array = np.lib.format.open_memmap(partial, mode="r+", dtype=dtype, shape=shape)
        return array, state, state_path, partial, final
    if partial.exists():
        raise RuntimeError(f"Partial has no state metadata; preserve and inspect: {partial}")
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)
    state = {"config": config,
             "rows_completed_by_table": {t: 0 for t in config["row_order"]},
             "complete": False}
    atomic_json(state_path, state)
    return array, state, state_path, partial, final


def _checkpoint(array, state: dict, state_path: Path, table: str, rows: int) -> None:
    array.flush()
    state["rows_completed_by_table"][table] = int(rows)
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--name-out", type=Path, required=True,
                        help="directory for train/test name memmaps and ID sidecars")
    parser.add_argument("--address-out", type=Path, required=True,
                        help="directory for train/test address memmaps and ID sidecars")
    parser.add_argument("--model-cache", type=Path, default=Path(r"D:\mlc_e5_cache"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--read-chunk-rows", type=int, default=32_768)
    parser.add_argument("--limit-per-table", type=int, default=0,
                        help="bounded pilot only; zero means all rows")
    parser.add_argument("--prefix", choices=(DEFAULT_PREFIX,), default=DEFAULT_PREFIX)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.checkpoint_every < 1 or args.max_length < 8:
        parser.error("batch, checkpoint interval, and max length must be positive")

    data_dir = args.data_dir.resolve()
    name_root, address_root = args.name_out.resolve(), args.address_out.resolve()
    cache_root = args.model_cache.resolve()
    for root in (name_root, address_root, cache_root):
        root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    counts, paths = {}, {}
    print("Counting rows and validating input tables...", flush=True)
    for table, relative in TABLES:
        path = data_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        counts[table] = args.limit_per_table or table_rows(path, args.read_chunk_rows)
        paths[table] = path
        print(f"  {table}: {counts[table]:,}", flush=True)

    expected_by_root = {name_root: [], address_root: []}
    for split in ("train", "test"):
        tables = _split_tables(split)
        rows = sum(counts[t] for t in tables)
        for field, root, stem, dtype, shape in (
            ("name", name_root, f"{split}_names", VECTOR_DTYPE, (rows, DIMENSION)),
            ("ids", name_root, f"{split}_names_ids", ID_DTYPE, (rows,)),
            ("address", address_root, f"{split}_addrs", VECTOR_DTYPE, (rows, DIMENSION)),
            ("ids", address_root, f"{split}_addrs_ids", ID_DTYPE, (rows,)),
        ):
            expected_by_root[root].append((stem, int(np.prod(shape) * np.dtype(dtype).itemsize)))

    for root, expected in expected_by_root.items():
        required = sum(size for _, size in expected)
        occupied = 0
        for stem, _ in expected:
            partial, final = root / f"{stem}.partial.npy", root / f"{stem}.npy"
            occupied += (final.stat().st_size if final.exists() else
                         partial.stat().st_size if partial.exists() else 0)
        needed = max(0, required - occupied)
        free = shutil.disk_usage(root).free
        print(f"[DISK] {root.drive}: free={free/1e9:.1f}GB new_required={needed/1e9:.1f}GB",
              flush=True)
        if free < needed * 1.05:
            raise RuntimeError(f"Insufficient space at {root}; need {needed/1e9:.1f}GB plus margin.")

    arrays = {}
    for split in ("train", "test"):
        tables = _split_tables(split)
        rows = sum(counts[t] for t in tables)
        for field, root, stem, dtype, shape in (
            ("name", name_root, f"{split}_names", VECTOR_DTYPE, (rows, DIMENSION)),
            ("ids", name_root, f"{split}_names_ids", ID_DTYPE, (rows,)),
            ("address", address_root, f"{split}_addrs", VECTOR_DTYPE, (rows, DIMENSION)),
            ("ids", address_root, f"{split}_addrs_ids", ID_DTYPE, (rows,)),
        ):
            config = _config(split, field, tables, counts, paths, args)
            arrays[(split, field, root)] = _open_array(root, stem, shape, dtype, config)

    import psutil
    import torch
    from transformers import AutoModel, AutoTokenizer
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Loading {MODEL_NAME}@{MODEL_REVISION}; device={device}; fp16={device == 'cuda'}; "
          f"batch={args.batch_size}; prefix={args.prefix!r}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_safetensors=True)
    model = model.to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    model.eval()
    print(f"Model ready: gpu={torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}",
          flush=True)

    offsets = {}
    for split in ("train", "test"):
        off = 0
        for table in _split_tables(split):
            offsets[table] = off
            off += counts[table]

    last_vitals, current_batch = 0.0, args.batch_size
    for table, _ in TABLES:
        split, n, base = table.split("_", 1)[0], counts[table], offsets[table]
        print(f"Embedding {table}: {n:,} rows", flush=True)
        reader = pd.read_csv(paths[table], sep="\t",
            usecols=["entity_id", "business_name", "business_address"], dtype=str,
            keep_default_na=False, chunksize=args.read_chunk_rows,
            nrows=args.limit_per_table or None)
        cursor = 0
        for frame in reader:
            length = min(len(frame), n - cursor)
            if length <= 0:
                break
            frame = frame.iloc[:length]
            end = cursor + length
            ids = frame["entity_id"].to_numpy(dtype=ID_DTYPE)
            if np.any(np.char.str_len(ids) > np.dtype(ID_DTYPE).itemsize):
                raise ValueError(f"Entity ID exceeds fixed-width S16 in {table}")
            for root in (name_root, address_root):
                arr, state, state_path, _, _ = arrays[(split, "ids", root)]
                done = int(state["rows_completed_by_table"].get(table, 0))
                if arr is not None and done < end:
                    start = max(cursor, done)
                    arr[base + start:base + end] = ids[start - cursor:]
                    _checkpoint(arr, state, state_path, table, end)

            for field, column, root in (("name", "business_name", name_root),
                                        ("address", "business_address", address_root)):
                arr, state, state_path, _, _ = arrays[(split, field, root)]
                done = int(state["rows_completed_by_table"].get(table, 0))
                if arr is None or done >= end:
                    continue
                pos = max(cursor, done)
                while pos < end:
                    local = pos - cursor
                    next_checkpoint = ((pos // args.checkpoint_every) + 1) * args.checkpoint_every
                    take = min(current_batch, end - pos, next_checkpoint - pos)
                    texts = frame[column].iloc[local:local + take].fillna("").astype(str).tolist()
                    last_vitals = enforce_resource_limits(psutil, torch, last_vitals)
                    vectors, used_batch = embed_batch(texts, tokenizer, model, torch, device,
                                                      args.max_length, current_batch)
                    current_batch = min(current_batch, used_batch)
                    arr[base + pos:base + pos + len(vectors)] = vectors
                    pos += len(vectors)
                    if pos % args.checkpoint_every == 0 or pos == n:
                        _checkpoint(arr, state, state_path, table, pos)
                        print(f"  {table} {column}: {pos:,}/{n:,}; batch={current_batch}",
                              flush=True)
                    last_vitals = enforce_resource_limits(psutil, torch, last_vitals)
                    del texts, vectors
            cursor += length
            if cursor >= n:
                break
        if cursor != n:
            raise RuntimeError(f"Read {cursor:,} rows for {table}; expected {n:,}")
        for root in (name_root, address_root):
            for field in ("ids", "name", "address"):
                if field == "name" and root != name_root:
                    continue
                if field == "address" and root != address_root:
                    continue
                arr, state, state_path, _, _ = arrays[(split, field, root)]
                if arr is not None:
                    _checkpoint(arr, state, state_path, table, n)
        del frame, reader
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  completed {table}", flush=True)

    for (split, field, root), (arr, state, state_path, partial, final) in arrays.items():
        if arr is None:
            continue
        tables = _split_tables(split)
        state["complete"] = all(
            int(state["rows_completed_by_table"].get(t, 0)) == counts[t] for t in tables
        )
        if not state["complete"]:
            raise RuntimeError(f"Refusing to finalize incomplete array: {partial}")
        arr.flush()
        atomic_json(state_path, state)
        arr._mmap.close()
        os.replace(partial, final)

    manifest = {
        "model": MODEL_NAME, "model_revision": MODEL_REVISION, "prefix": args.prefix,
        "dimension": DIMENSION, "dtype": "float16", "pooling": "masked_mean",
        "normalize_l2": True, "id_dtype": "S16", "max_length": args.max_length,
        "batch_size_requested": args.batch_size, "tables": counts,
        "row_order": {split: list(_split_tables(split)) for split in ("train", "test")},
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(name_root / "e5_embeddings_manifest.json", manifest)
    atomic_json(address_root / "e5_embeddings_manifest.json", manifest)
    print("Combined E5 artifacts complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
