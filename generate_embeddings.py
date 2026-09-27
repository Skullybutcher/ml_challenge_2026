#!/usr/bin/env python3
"""Resumable E5-large-v2 embedding writer for the Akari entity tables.

Writes one fp16 NumPy memmap per split/source/text field, plus fixed-width ID
sidecars. Each table is processed in CSV chunks and each field checkpoints
independently; interrupted batches are safely rewritten on resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "intfloat/e5-large-v2"
MODEL_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"
VECTOR_DTYPE = np.float16
DIMENSION = 1024
ID_DTYPE = "S16"
DEFAULT_PREFIX = "query: "  # symmetric similarity: same query prefix for both records
TABLES = (
    ("train_s1", "train/train_source1.tsv"),
    ("train_s2", "train/train_source2.tsv"),
    ("train_s3", "train/train_source3.tsv"),
    ("test_s1", "test/test_source1.tsv"),
    ("test_s2", "test/test_source2.tsv"),
    ("test_s3", "test/test_source3.tsv"),
)


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def table_rows(path: Path, chunksize: int) -> int:
    count = 0
    for chunk in pd.read_csv(path, sep="\t", usecols=["entity_id"], dtype=str,
                             keep_default_na=False, chunksize=chunksize):
        count += len(chunk)
    return count


def safe_config(data_path: Path, rows: int, args) -> dict:
    stat = data_path.stat()
    return {
        "input_path": str(data_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "rows": int(rows),
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "prefix": args.prefix,
        "max_length": args.max_length,
        "dimension": DIMENSION,
        "dtype": "float16",
        "normalize_l2": True,
    }


def open_or_create_memmap(partial_path: Path, final_path: Path, state_path: Path,
                          shape: tuple[int, ...], dtype, config: dict):
    if final_path.exists():
        if not state_path.exists():
            raise RuntimeError(f"Output exists without state metadata; refusing overwrite: {final_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("complete") and state.get("config") == config:
            return None, state
        raise RuntimeError(f"Completed output config mismatch; preserve and choose another directory: {final_path}")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config") != config:
            raise RuntimeError(f"Resume config/input mismatch; preserve partial output: {partial_path}")
        if not partial_path.exists():
            raise RuntimeError(f"State exists but partial output is missing: {partial_path}")
        array = np.lib.format.open_memmap(partial_path, mode="r+", dtype=dtype, shape=shape)
        return array, state
    if partial_path.exists():
        raise RuntimeError(f"Partial output has no state metadata; preserve and inspect: {partial_path}")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(partial_path, mode="w+", dtype=dtype, shape=shape)
    state = {"config": config, "rows_completed": 0, "complete": False}
    atomic_json(state_path, state)
    return array, state


def commit_progress(array, state_path: Path, state: dict, rows_completed: int,
                    partial_path: Path, final_path: Path) -> tuple[object | None, dict]:
    array.flush()
    state["rows_completed"] = int(rows_completed)
    state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if rows_completed == int(state["config"]["rows"]):
        state["complete"] = True
        atomic_json(state_path, state)
        # Windows will reject os.replace while the NPY memmap handle is open.
        array._mmap.close()
        del array
        os.replace(partial_path, final_path)
        return None, state
    atomic_json(state_path, state)
    return array, state


def get_gpu_temperature() -> int | None:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip().splitlines()[0]
        return int(raw)
    except Exception:
        return None


def enforce_resource_limits(psutil, torch, last_check: float) -> float:
    now = time.monotonic()
    if now - last_check < 5:
        return last_check

    def sample_vitals() -> int | None:
        proc = psutil.Process()
        procs = [proc] + proc.children(recursive=True)
        tree_gb = 0.0
        for child in procs:
            try:
                tree_gb += child.memory_info().rss
            except psutil.Error:
                pass
        tree_gb /= 1024 ** 3
        used_gb = psutil.virtual_memory().used / 1024 ** 3
        temperature = get_gpu_temperature() if torch.cuda.is_available() else None
        print(f"[VITALS] process_tree={tree_gb:.2f}GB system_used={used_gb:.2f}GB "
              f"gpu_temp={temperature if temperature is not None else 'unavailable'}C", flush=True)
        if tree_gb >= 44:
            raise RuntimeError(
                f"Process-tree memory watchdog reached: {tree_gb:.2f} GB (limit 44 GB)."
            )
        if used_gb >= 47:
            raise RuntimeError(
                f"System memory watchdog reached: {used_gb:.2f} GB (limit 47 GB)."
            )
        if temperature is not None and temperature >= 82:
            raise RuntimeError(f"GPU thermal stop reached: {temperature} C (limit 82 C).")
        return temperature

    temperature = sample_vitals()
    if temperature is not None and temperature >= 78:
        print(f"[THERMAL-PAUSE] GPU at {temperature} C; pausing embeddings until <=72 C.",
              flush=True)
        while temperature > 72:
            time.sleep(5)
            temperature = sample_vitals()
        print(f"[THERMAL-RESUME] GPU cooled to {temperature} C; continuing.", flush=True)
    return time.monotonic()


def embed_batch(texts: list[str], tokenizer, model, torch, device: str,
                max_length: int, initial_batch_size: int) -> tuple[np.ndarray, int]:
    batch_size = min(len(texts), initial_batch_size)
    while True:
        try:
            outputs = []
            for start in range(0, len(texts), batch_size):
                batch = [DEFAULT_PREFIX + value for value in texts[start:start + batch_size]]
                tokenized = tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length,
                )
                # Convert the already-padded integer lists directly. This
                # avoids the tensor-conversion path in newer Transformers
                # releases, which raised on a later mixed-length address batch.
                encoded = {
                    key: torch.from_numpy(np.asarray(value, dtype=np.int64)).to(device)
                    for key, value in tokenized.items()
                    if key in {"input_ids", "attention_mask", "token_type_ids"}
                }
                with torch.inference_mode():
                    output = model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1).to(output.dtype)
                    pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                outputs.append(pooled.to(dtype=torch.float16).cpu().numpy())
                del batch, tokenized, encoded, output, mask, pooled
            return np.concatenate(outputs, axis=0), batch_size
        except torch.cuda.OutOfMemoryError:
            if device != "cuda" or batch_size <= 32:
                raise
            batch_size //= 2
            print(f"[OOM-RECOVERY] clearing CUDA cache and retrying with batch={batch_size}.",
                  flush=True)
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path,
                        help="dataset directory containing train/ and test/ subdirectories")
    parser.add_argument("--name-out", type=Path, required=True,
                        help="root for name-vector tables and entity ID sidecars")
    parser.add_argument("--address-out", type=Path, required=True,
                        help="root for address-vector tables")
    parser.add_argument("--model-cache", type=Path, default=Path(r"D:\mlc_e5_cache"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--read-chunk-rows", type=int, default=32_768)
    parser.add_argument("--limit-per-table", type=int, default=0,
                        help="pilot only: embed at most this many rows from each table")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_length < 8 or args.checkpoint_every < 1:
        parser.error("batch size, max length, and checkpoint interval must be positive")
    data_dir = args.data_dir.resolve()
    name_root = args.name_out.resolve()
    address_root = args.address_out.resolve()
    cache_root = args.model_cache.resolve()
    name_root.mkdir(parents=True, exist_ok=True)
    address_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    import psutil
    import torch
    from transformers import AutoModel, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Loading {MODEL_NAME} on {device}; fp16={device == 'cuda'}; "
          f"batch={args.batch_size}; max_length={args.max_length}; prefix={args.prefix!r}",
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, use_fast=True
    )
    model = AutoModel.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, use_safetensors=True
    )
    if device == "cuda":
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device=device, dtype=torch.float32)
    model.eval()
    print(f"Model ready: parameters={sum(p.numel() for p in model.parameters()):,}; "
          f"gpu={torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}", flush=True)

    counts: dict[str, int] = {}
    paths: dict[str, Path] = {}
    total_rows = 0
    print("Counting rows and checking table paths...", flush=True)
    for table, relative in TABLES:
        path = data_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        # The bounded pilot skips a full-table count pass. The actual entity
        # tables all exceed the pilot limit; the streaming loop still checks
        # that it reads exactly the number of rows allocated here.
        rows = (args.limit_per_table if args.limit_per_table
                else table_rows(path, args.read_chunk_rows))
        counts[table] = rows
        paths[table] = path
        total_rows += rows
        print(f"  {table}: {rows:,} rows", flush=True)

    one_field_bytes = total_rows * DIMENSION * np.dtype(VECTOR_DTYPE).itemsize
    name_bytes = one_field_bytes + total_rows * np.dtype(ID_DTYPE).itemsize
    for path, required in ((name_root, name_bytes), (address_root, one_field_bytes)):
        free = shutil.disk_usage(path).free
        print(f"[DISK] {path.drive} free={free / 1e9:.1f}GB required~{required / 1e9:.1f}GB",
              flush=True)
        if free < required * 1.05:
            raise RuntimeError(f"Insufficient free space at {path}: need about "
                               f"{required / 1e9:.1f}GB plus safety margin.")

    table_specs: dict[str, dict] = {}
    for table, path in paths.items():
        n = counts[table]
        id_final = name_root / f"{table}_ids.npy"
        id_partial = name_root / f"{table}_ids.partial.npy"
        id_state_path = name_root / f"{table}_ids.state.json"
        id_config = safe_config(path, n, args)
        ids, id_state = open_or_create_memmap(
            id_partial, id_final, id_state_path, (n,), ID_DTYPE, id_config
        )
        name_final = name_root / f"{table}_name.npy"
        name_partial = name_root / f"{table}_name.partial.npy"
        name_state_path = name_root / f"{table}_name.state.json"
        name_config = safe_config(path, n, args)
        name_config["field"] = "business_name"
        name_vec, name_state = open_or_create_memmap(
            name_partial, name_final, name_state_path, (n, DIMENSION), VECTOR_DTYPE, name_config
        )
        addr_final = address_root / f"{table}_address.npy"
        addr_partial = address_root / f"{table}_address.partial.npy"
        addr_state_path = address_root / f"{table}_address.state.json"
        addr_config = safe_config(path, n, args)
        addr_config["field"] = "business_address"
        addr_vec, addr_state = open_or_create_memmap(
            addr_partial, addr_final, addr_state_path, (n, DIMENSION), VECTOR_DTYPE, addr_config
        )
        table_specs[table] = {
            "ids": (ids, id_state, id_state_path, id_partial, id_final),
            "name": (name_vec, name_state, name_state_path, name_partial, name_final),
            "address": (addr_vec, addr_state, addr_state_path, addr_partial, addr_final),
        }

    last_vitals = 0.0
    total_started = time.monotonic()
    current_batch = args.batch_size
    for table, path in paths.items():
        n = counts[table]
        specs = table_specs[table]
        print(f"Embedding {table} ({n:,} rows)...", flush=True)
        cursor = 0
        reader = pd.read_csv(
            path, sep="\t", usecols=["entity_id", "business_name", "business_address"],
            dtype=str, keep_default_na=False, chunksize=args.read_chunk_rows,
            nrows=args.limit_per_table or None,
        )
        for frame in reader:
            chunk_len = min(len(frame), n - cursor)
            if chunk_len <= 0:
                break
            frame = frame.iloc[:chunk_len]
            id_array, id_state, id_state_path, id_partial, id_final = specs["ids"]
            id_done = int(id_state["rows_completed"])
            if id_done < cursor + chunk_len:
                start_in_chunk = max(0, id_done - cursor)
                ids = frame["entity_id"].iloc[start_in_chunk:].to_numpy(dtype=ID_DTYPE)
                if np.any(np.char.str_len(ids) > np.dtype(ID_DTYPE).itemsize):
                    raise ValueError(f"Entity ID exceeds {ID_DTYPE} in {table}.")
                id_array[id_done:cursor+chunk_len] = ids
                id_array, id_state = commit_progress(
                    id_array, id_state_path, id_state, cursor + chunk_len,
                    id_partial, id_final,
                )
                specs["ids"] = (id_array, id_state, id_state_path, id_partial, id_final)

            for field_name, column in (("name", "business_name"),
                                       ("address", "business_address")):
                array, state, state_path, partial, final = specs[field_name]
                done = int(state["rows_completed"])
                end = cursor + chunk_len
                if array is not None and done < end:
                    position = max(cursor, done)
                    while position < end:
                        take = min(current_batch, end - position)
                        begin_local = position - cursor
                        texts = frame[column].iloc[begin_local:begin_local+take].fillna("").astype(str).tolist()
                        vectors, used_batch = embed_batch(
                            texts, tokenizer, model, torch, device,
                            args.max_length, current_batch,
                        )
                        current_batch = min(current_batch, used_batch)
                        array[position:position+len(vectors)] = vectors
                        position += len(vectors)
                        if position % max(args.checkpoint_every, 1) < len(vectors) or position == n:
                            array, state = commit_progress(
                                array, state_path, state, position, partial, final
                            )
                            specs[field_name] = (array, state, state_path, partial, final)
                            print(f"  {table} {column}: {position:,}/{n:,} "
                                  f"({100*position/max(n,1):.1f}%), batch={current_batch}",
                                  flush=True)
                        last_vitals = enforce_resource_limits(psutil, torch, last_vitals)
                        del texts, vectors
            cursor += chunk_len
            if cursor >= n:
                break
        if cursor != n:
            raise RuntimeError(f"Read {cursor:,} rows for {table}, expected {n:,}.")
        for field_name in ("ids", "name", "address"):
            array, state, state_path, partial, final = specs[field_name]
            if array is not None:
                array, state = commit_progress(array, state_path, state, n, partial, final)
                specs[field_name] = (array, state, state_path, partial, final)
        print(f"  completed {table}", flush=True)
        del reader, frame
        if device == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "prefix": args.prefix,
        "dimension": DIMENSION,
        "dtype": "float16",
        "max_length": args.max_length,
        "device": device,
        "batch_size_final": current_batch,
        "tables": counts,
        "name_output": str(name_root),
        "address_output": str(address_root),
        "elapsed_seconds": time.monotonic() - total_started,
        "note": "Each NPY vector file is row-aligned with its corresponding *_ids.npy sidecar.",
    }
    atomic_json(name_root / "e5_embeddings_manifest.json", manifest)
    atomic_json(address_root / "e5_embeddings_manifest.json", manifest)
    print(f"All tables complete in {manifest['elapsed_seconds']/3600:.2f} hours.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
