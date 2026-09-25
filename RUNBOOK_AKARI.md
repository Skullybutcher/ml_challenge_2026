# Runbook — full-scale run on Jini's PC (for Akari)

Checked-in code: `code/business_entity_resolution/src/` (vendored from `code_ber` + chunked-scaling patch; submodule link removed so a plain `git pull` is enough — no `submodule init` needed).

## 0. Setup (once)

```powershell
git pull origin feature/chunked-fullrun
pip install "pandas>=2.0" "numpy>=1.24" "scipy>=1.10" "scikit-learn>=1.3" "lightgbm>=4.0" "rapidfuzz>=3.0"
```

Place the dataset so `<DATA>` contains `train/` (train_source1/2/3.tsv + train_ground_truth.tsv) and `test/` (test_source1/2/3.tsv). All commands run from the repo root; set `$DATA` once:

```powershell
$DATA = "<DATA>"   # <-- the ONLY path to set, e.g. D:\data\dataset
$SRC  = "code/business_entity_resolution/src"
```

CPU-only. 64GB RAM removes all memory constraints seen on 16GB (peaks were ≤8.3GB at 50k). No GPU needed. Run one job at a time.

## 1. Run 1 — validation (~30-60 min). REPORT, then WAIT for Aman's go.

```powershell
python $SRC/pipeline.py --data-dir $DATA --out-dir out_100k --sample-s1 100000 --skip-test --n-splits 5 --train-chunk-size 5000 > run_100k.log 2>&1
```

Send Aman the last 15 lines of `run_100k.log`. Required numbers: `Train candidate recall`, `Pairs before/after subsample`, `OOF macro F0.5`, `Best threshold`, wall time. Gate: recall ≥ 0.90 (expect ~0.977), OOF ≈ 0.976-0.986.

## 2. Run 2 — full run, ONLY on Aman's explicit go (~8-14h, overnight OK).

```powershell
python $SRC/pipeline.py --data-dir $DATA --out-dir out_full --sample-s1 100000 --n-splits 5 --train-chunk-size 5000 > run_full.log 2>&1
```

Trains on the 100k sample, blocks + scores all 1,732,544 test S1 in 100k chunks, streams both TSVs. Poll with `Get-Content run_full.log -Tail 3`. First test chunk: check the `pairs=` count; if any chunk exceeds ~50M pairs, STOP and tell Aman (merge-explosion guard).

## 3. Validate (before anything is sent anywhere).

```powershell
python utils/validate_submission.py --matching out_full/matching_results.tsv --candidate out_full/candidate_pairs.tsv --test-dir $DATA/test
python utils/validate_submission.py --matching out_full/matching_results.tsv --test-dir $DATA/test --check-ids
```

Need `PASS` on the first. If `--check-ids` OOMs, drop `--candidate` and re-run (documented in the validator). Also confirm `matching_results.tsv` has exactly 1,732,544 data rows.

## 4. Hand back

Send Aman `out_full/matching_results.tsv` + the `Best threshold` / OOF lines + validator output. Keep `candidate_pairs.tsv` on local disk (needed for the final zip; ~15GB, transfer separately).

## Do NOT

Change code or flags, tune anything, run parallel jobs, or commit. On ANY crash: send the last 20 lines of the log verbatim, do not debug.
