# Current status — 2026-09-27 17:06 IST

This live snapshot supersedes older status summaries below; detailed prior entries are retained as history.

## Run outcomes

- **Run 1 completed:** OOF macro F0.5 **0.9825**. Logged top-eight threshold scores are 0.62:0.9825, 0.65:0.9825, 0.57:0.9825, 0.60:0.9825, 0.68:0.9824, 0.70:0.9824, 0.55:0.9823, 0.72:0.9822. The selected global plateau threshold is **0.72**.
- **E4 source/singleton threshold comparison is not established.** Run 1 saved aligned labels and pair IDs, but did not save raw OOF probabilities or the full 37-point curve. There are no France S1 examples in the training slice, so the France threshold has no direct OOF validation here.
- **Run 2 was stopped at Aman’s direction at 16:03 IST.** Four of 278 inference chunks had completed; the partial candidate file was preserved locally. All 41 training checkpoint files remain. OOF macro F0.5 was **0.9761** at 0.800; this is below Run 1’s 0.9825. No Run 2 output passed validation, and there is no active Run 2 process.

## E5 implementation and pilot

- Installed an isolated Python 3.11/CUDA runtime under D:/mlc_e5_runtime; downloaded and loaded intfloat/e5-large-v2 at pinned revision f169b11e22de13617baa190a028a32f3493550b6. The RTX 5070 Ti is visible to CUDA.
- Added resumable, per-table fp16 embedding generation in generate_embeddings.py, plus opt-in name/address cosine features and country-threshold options in the pipeline. The first full attempt (PID 14232) stopped safely at the 82°C GPU limit before any vector row checkpoint was committed; 98,304 train_s1 ID rows were committed. All partial NPY memmaps and state files are retained. The exact stdout, stderr, and final 47 stdout lines are archived outside Git under C:/mlc_model/generate_embeddings_attempt_20260927_170030.*. The job resumed at 17:05 IST as PID 20956 with batch 256; resume preflight found about 50.01 GB of name output and 49.62 GB of address output already allocated, so no vector files were reallocated. Row counting passed and train_s1 embedding began. At the latest check it was alive with about 1.58 GB process RSS, 15.04 GB system use, and GPU 68°C (thermal pause threshold).
- Completed a pilot across all six train/test source tables: 8,192 rows per table, name and address vectors shaped (8,192, 1,024), fp16, with aligned ID sidecars. Pilot checks passed for shape, dtype, finite values, unit norms, two-row feature construction, and first-8,192-ID alignment against all six source-table prefixes.
- The pilot’s GPU cooldown guard paused at 78°C, resumed after cooling to 48°C, and completed; observed process-tree memory was about 1.48 GB. A feature-path defect that deleted the pair frame before reading country labels was fixed. The four changed Python files compile under the existing Python 3.11 project environment. To avoid another thermal stop, generation now pauses at 68°C and resumes below 56°C, checks vitals every second, and defaults to batch 256. Disk preflight now credits the already allocated partial arrays so a resume does not falsely report that their full size must be free again.
- Approximate full output is 49.6 GB for names and 49.6 GB for addresses, split across C: and E:. At the last disk check, C: had 93.8 GB free and E: 76.6 GB free at launch. The generator requires about 50 GB per output volume plus a 5% margin and will stop if that preflight fails. Continue monitoring free space and memory during generation.
- France threshold 0.95 / other-country threshold 0.65 remains an **unvalidated experiment**, not a submission decision. It must be evaluated against OOF evidence where available; no score gain is claimed. E5 retraining, test inference, validation, or submission staging have not happened.

## Immediate next steps

1. E5 code/status are pushed to exp/run1-2500-a467-20260926 (d3d9a0c and 0efad8d). This thermal-guard and resumable-disk-preflight fix is being committed before restart. Datasets, embeddings, model weights, run logs, and training checkpoints remain out of Git.
2. Continue monitoring resumed PID 20956 and checkpoint progress under the separate C:/E: output roots. Monitor GPU temperature, memory, and disk space.
3. Retrain/evaluate with --use-e5 only after all required vectors are complete. Keep outputs provisional until OOF/threshold evaluation and the documented validator pass; never upload to the leaderboard.

---

# Latest status summary — 2026-09-27, 09:47 IST

## Run 1 (completed; launch gates PASS)

- Candidate recall: **0.9834 overall**, **0.9966 US** (n=60,078); India is excluded because n=0.
- OOF macro F0.5: **0.9825**; 311,199,888 candidate pairs before subsampling and 5,509,855 retained training pairs.
- Logged top-eight curve: **0.62:0.9825, 0.65:0.9825, 0.57:0.9825, 0.60:0.9825, 0.68:0.9824, 0.70:0.9824, 0.55:0.9823, 0.72:0.9822**. Global plateau choice remains **0.72**, highest displayed point within 0.001; the complete grid was not persisted.
- Launch gates passed. Peak was **31.49 GB process tree / 45.66 GB system**.

## Run 1 E4 threshold splits — pending required probabilities

- The existing Run 1 output does **not** contain raw OOF probabilities or the full 37-point curve; its log only prints the top eight scores. No source-specific or singleton-specific threshold scores can be computed from labels and pair IDs alone.
- Commit `0d4575d` adds the 5,509,855 aligned labels and `(s1_id, other_id)` pairs exported from saved chunks. The artifact manifest records this limitation. No split threshold is claimed or adopted yet.
- After Run 2 completes, regenerate raw OOF probabilities from the saved Run 1 feature checkpoints, compute the full grid and requested source/singleton splits, then record the E4 result here and in `SUBMISSIONS.md` before choosing a submission threshold.

## Run 2 (retry active; last verified 09:47 IST)

- The first Run 2 attempt completed training featurization **40/40 chunks** at 07:12:07 and finished OOF/threshold selection. Its 100k-S1 first test chunk began at 07:16:17 but had not logged completion when the system-memory watchdog stopped the run at 09:13:15: system use reached **47.00 GB**, with process-tree RSS **33.81 GB**. No test `pairs=` count had been emitted, so the 50M pair tripwire did not trigger. `run_full.exit` is absent because the watchdog terminated the launcher tree. The closest playbook row is RAM pressure at startup; this incident specifically occurred during test inference, so the direct diagnosis is system memory limit, not a code exception.
- The exact last 50 `run_full.log` lines and the full run, monitor, thermal, and tripwire evidence are preserved as `run_full_attempt_system_memory_watchdog_20260927_091315.*`. All **40/40 sampled training checkpoint NPZs** remain in `out_full/.train_chunk_checkpoints` (126,446,125 bytes total). Closest playbook entry: “Segfault / OpenBLAS alloc fail at startup” → machine RAM pressure → “Close everything else, confirm free RAM, rerun. NOT a code bug.” It is a nearby resource-pressure remedy; this run instead hit the system watchdog during inference.
- The retry started at 09:32:29 with unchanged training/model flags, `--resume-train-chunks`, and test-only `--test-chunk-size 25000` (reduced from 100,000 to bound per-chunk memory and pair counts). It found the identical training checkpoint key `8a332660fb9a`; all **40/40 chunks logged `resumed checkpoint` by 09:47:22**, with the same 147,631,526 pre-subsample and 5,434,674 retained pair totals. OOF model fitting is running again. New memory, GPU, and all-test-chunk tripwire watchers are active.
- Candidate recall: **0.9752 overall**, **0.9911 US** (n=60,078); India n=0 excluded. OOF macro F0.5: **0.9761** (passes 0.970 gate, below 0.988 decision threshold and Run 1's 0.9825). The printed top-eight curve: **0.65:0.9771, 0.62:0.9771, 0.60:0.9771, 0.68:0.9771, 0.70:0.9770, 0.57:0.9770, 0.55:0.9770, 0.72:0.9768**. Wrapper-selected Run 2 plateau threshold: **0.800**, score **0.9761**.
- Submission threshold decision is **pending E4**: per the current rule, because Run 2 OOF <0.988, adopt an E4 split threshold only if its gain is ≥0.001; otherwise use Run 1's 0.72. Run 2 inference is currently using 0.800, so its outputs are provisional until the threshold decision is resolved.
- First-attempt peaks: **33.81 GB process-tree RSS / 47.00 GB system use** (system watchdog stopped the run); GPU remained **41°C**. The CPU-package sensor is unavailable. After termination, system use returned to **12.79 GB** with 50.98 GB free. Retry peak as of 09:47:34: **18.01 GB process tree / 31.02 GB system**; current use was **8.91 / 21.92 GB** and GPU was **43°C**.
- `launch_run_full_monitored.ps1` now starts the all-test-chunk watcher. The current retry has fresh process-tree, GPU, and all-chunk tripwire watchers; PIDs are recorded in the runtime PID files.
Detailed chronological Run 1 and Run 2 history follows.

# Status — Akari business entity resolution

**Snapshot:** 2026-09-26 11:30 Asia/Kolkata
**Repository:** https://github.com/Skullybutcher/ml_challenge_2026
**Active branch:** `exp/run1-2500-a467-20260926`
**Base commit:** `a467bb7c79417e084b7e2fbd873191581c6d5a5b`
**Goal status:** Run 1 retry is active under Python 3.11.13 after a confirmed Python 3.12 access violation; Run 2 gates are not complete.

This file records the work and instructions received since the initial task prompt. It is a point-in-time log: refresh the live process, log tail, and branch state before acting on it.

## Current Run 1

The active retry started at 11:23. Launcher PID 1844 runs Python 3.11.13 from `..\venv_py311` with Pandas 2.3.3, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, LightGBM 4.7.0, and RapidFuzz 3.14.6. It uses the same 100k sample, `--use-rare`, five folds, `--skip-test`, and 2,500-S1 chunks. `faulthandler` is enabled and feature-batch start/end markers are enabled via `AKARI_TRACE_FEATURE_BATCHES=1`; stderr is captured separately. Watchdog PID 3660 samples the process tree every five seconds and stops at 47 GB process-tree RSS or 58 GB system use. Initial tree RSS was 2.62 GB; C: had 99.74 GB free. The pipeline reports its internal psutil sampler is unavailable in this environment, so use the tree watchdog's peak.

The preceding retry ended with a Windows APPCRASH at 11:10:57. Faulting process PID 34060, runtime `E:\anaconda\python.exe` 3.12.7; faulting module `python312.dll`, exception `0xc0000005` (access violation). It used isolated `venv_pd2` with Pandas 2.3.3 and NumPy 2.5.2. The 47 GB process-tree / 58 GB system-use watchdog did not trigger: peak process-tree RSS was 22.59 GB and peak system use was 35.65 GB. After the crash, free physical memory was 51.56 GB. `out_100k` was empty. The full log, stderr, and monitor records are preserved under `run_100k_attempt_a467_python312_access_violation_after_chunk2_20260926.*`.

Command:

~~~powershell
& ..\venv_py311\Scripts\python.exe code\business_entity_resolution\src\pipeline.py --data-dir $DATA --out-dir out_100k --sample-s1 100000 --skip-test --n-splits 5 --train-chunk-size 2500 --use-rare
~~~

`$DATA` is the extracted dataset root, `..\dataset`, containing `train/` and `test/`. The active interpreter is `..\venv_py311\Scripts\python.exe` (Python 3.11.13). The C: drive had 99.74 GB free at restart (11:23), above the 40 GB requirement.

Current Python 3.11 retry: candidate generation again reports 311,199,888 pairs, overall recall 0.9834, and US recall 0.9966. Feature chunk 1/40 began at 11:29:42 with 7,705,746 pairs; the first 100k-row feature batch is running. The tree watchdog has sampled a 31.30 GB peak so far, during blocking; it is now around 23 GB.

The previous attempt (PID 18208) stopped unexpectedly after train chunk 7/40. Its last log evidence at 09:55:23 was:

- Candidate pair union: 311,199,888.
- Overall candidate recall: 0.9834.
- US recall: 0.9966 (n=60,078).
- India-only measurement is dropped: the country field is blank for about 40% of train S1, and the printed `n=0` line is not a valid slice. Report overall and US recall only.
- Per-S1 candidate count p50=2,813; p99=9,252; max=20,383; mean=3,112.0.
- Train chunks: 7/40 complete. The seven chunks each processed about 7.7–7.9 million raw pairs and took about 10.5 minutes apiece.
- OOF, threshold curve, final peak RSS, and completion time were not reached. No traceback or explicit exit reason is in the log; do not label it a diagnosed code crash or OOM.

The stopped-at-7/40 log is preserved as [run_100k_attempt_a467_chunk7_stopped_20260926.log](run_100k_attempt_a467_chunk7_stopped_20260926.log); `out_100k` was empty before restart, so no resume checkpoint existed. A prior 2,500-chunk attempt under Pandas 3 was manually stopped at 48.48 GB RSS during blocking and is preserved as [run_100k_attempt2500_pandas3_stopped.log](run_100k_attempt2500_pandas3_stopped.log). The latest attempt log is [run_100k_attempt_a467_python312_access_violation_after_chunk2_20260926.log](run_100k_attempt_a467_python312_access_violation_after_chunk2_20260926.log).

The Python 3.12 retry repeated overall candidate recall 0.9834 and US recall 0.9966. Train chunk 1/40 completed at 10:55:10 after 10m46s (7,705,746 pairs → 137,233 sampled); chunk 2/40 completed at 11:06:05 after 10m55s (7,938,205 → 136,025 sampled). The serial path projected roughly 7h15m for featurization before it crashed in chunk 3; the initial 1.5–2.5h estimate is not holding. It used about one logical CPU core, with no competing pipeline job observed.

The failure table's closest row is “Segfault / OpenBLAS alloc fail at startup”: close other applications, confirm at least 10 GB free RAM, and rerun. This access violation occurred during chunk 3, not startup, and memory evidence does not support pressure (22.59 GB process-tree peak; 51.56 GB free after exit). The crash log ends without a Python traceback. Treat it as a native runtime crash; retry with Python 3.11.13, enable faulthandler, and trace feature-batch boundaries on the exp branch.

Verbatim last 20 lines (the log contains 17 lines total):

~~~text
[10:37:42] Loading training sources...
[10:38:45] Train sizes: S1=2206821 S2=5034616 S3=5285603 | GT rows=2206821 | singleton rate=0.056
[10:38:45] --sample-s1: subsampling to 100000 S1 (matched S2/S3 kept, distractors proportional).
[10:38:48] Sampled train: S1=100000 S2=457177 S3=468311
[10:38:48] --skip-test: not loading test data; steps 8-9 (inference + outputs) will be skipped.
[10:38:48] Normalizing text fields...
[10:39:05] Blocking train S1 vs S2 (max_df=300)...
[10:41:03] Blocking train S1 vs S3 (max_df=300)...
[10:43:50] [TIMING] blocking (both sources): 4.8 min
[10:44:20] Train candidate pairs (S2+S3 union): 311,199,888
[10:44:20] Train candidate recall: 0.9834
[10:44:24] Train candidate recall [US] (n=60,078): 0.9966
[10:44:24] Train candidate recall [IN] (n=0): 1.0000
[10:44:24] Per-S1 candidate counts: p50=2,813 p99=9,252 max=20,383 mean=3112.0
[10:44:24] Featurizing train in 40 chunk(s) of ~2,500 S1...
[10:55:10] train chunk 1/40: pairs=7,705,746 -> subsampled=137,233 (pos in chunk: 8,455)
[11:06:05] train chunk 2/40: pairs=7,938,205 -> subsampled=136,025 (pos in chunk: 8,380)
~~~

## Instructions and handoff history

The initial prompt asked to pull `feature/chunked-fullrun` at `db1cfe1`, read `NIGHT_HANDOFF.md`, `RUNBOOK_AKARI.md`, `AKARI_PLAYBOOK.md`, then the context gates, France notes, error taxonomy, and cap-sweep evidence in that order. It specified dataset extraction under `<DATA>/train/` and `<DATA>/test/`, dependency setup from `code/business_entity_resolution/requirements.txt`, at least 40 GB free for outputs, and a 100k sample Run 1 with `--skip-test`, five folds, and 5,000-S1 chunks. It required per-country recall review and an exact 1,732,544-row submission validation.

The next handoff corrected the base to `ccfa813` and made `--use-rare` required. It reported a separate 50k rare-channel result of recall 0.9892 / OOF 0.9858 and requested the 100k rare-channel Run 1. User-reported comparison values were 100k without rare: recall 0.9611 / OOF 0.9680 (NO-GO). Debugging was later explicitly authorized.

The debugging instruction asked to keep both fixes on an `exp/*` branch: (a) avoid creating counted-token frames when no per-S1 cap is enabled, delete `counted_s2` / `counted_s3` after the cap block, and run `gc.collect()`; (b) convert feature rows in bounded batches into a preallocated NumPy array. Both fixes are in the working tree used by the current run. Aman's latest ruling confirms the counted-frame deletion is already in `a467bb7` and addresses E1's 33 GB crash; E1 used pre-fix code, so do not revert or take further action on E1.

The latest handoff superseded the requested source commit with `a467bb7`, confirmed Aman's machine was off, and set these decisions:

1. Run 2 uses `--rare-max-df 1000`; a measured 50k run was reported at recall 0.9857 with 38% fewer pairs, with projected runtime about 22 hours versus 36 hours for the default band.
2. `process.cdist` is rejected for aligned pairs. The replacement speed experiment is independent-chunk multiprocessing for featurization with 6–8 workers, within a 48 GB combined-memory cap; first validate exact-output equality on 5k.
3. Do not attempt India-only recall: the country field is blank for about 40% of train S1, making the slice invalid. Report overall and US recall.
4. The GitHub push 403 is acknowledged. Stop trying to push; retain code, status, logs, and outputs locally. Transfer outputs manually at staging.
5. Run 1 proceeds unchanged; Run 2 follows its gates after replacing the invalid India-only check with overall + US reporting.

The docs were read in the requested order on the successive handoffs. The latest handoff’s exact Run 2 band, gates, plateau rule, failure table, experiment rules, and submission restrictions remain authoritative.

## Branch and code state

The frozen branch `feature/chunked-fullrun` has not been modified. Work is on `exp/run1-2500-a467-20260926`, based on local HEAD `7d9038139aea6eeba3dda063d227fa7803f17a16` and base commit `a467bb7`. The commit added the rare-band flags / handoff decisions and counted-frame deletion. The working tree also contains the batched conversion fix:

- `features.py`: retains scalar feature semantics and ordering while converting at most 100,000 rows at a time to float32, with a shape check for each batch.
- `pipeline.py`: skips the shared-token count-frame path on the uncapped run and releases optional count frames after the cap block. The same deletion is present in `a467bb7`.

The source fix, current logs, and refreshed status remain local. The status document itself was committed separately as 577120b. The push attempt failed with HTTP 403 for the configured Git identity; no remote exp branch was created. Per Aman's latest ruling, do not retry any push; keep updated status and artifacts local for manual transfer at staging.

A code review and Aman's ruling confirm the counted-frame deletion is in `a467bb7`; the active pipeline also includes the uncapped-path optimization. Do not reopen the E1 attribution or revert the fix.

## Earlier attempts and evidence

- Initial 100k attempts 1–3 using the bundled/repo Python environment exited before producing metrics. The preserved logs include Windows process errors `-1073741819` and `-1073740791`.
- The earlier 100k attempt 4 without the rare channel reached blocking and was stopped after observed RSS reached 49.50 GB, over the 48 GB cap; it produced no recall/OOF. Its log is `run_100k_attempt4_no_rare.log` in the sibling original worktree.
- Later debugging exposed a return-contract error in a trial fix and an Arrow-backed Pandas 3 `dtype=str` allocation failure. An isolated environment with Pandas 2.3.3 was created; the active run uses it.
- The failed Pandas 3 retry at 2,500 chunks reached 48.48 GB during the S1-vs-S3 blocking stage and was stopped before recall.
- E1, 50k with rare tokens: recall 0.9892, US 0.9989 (n=30,157), India n=0, 137,937,334 candidate pairs. It failed during feature-array creation before pair subsampling or OOF; about 33.77 GB RSS was observed before the crash. Aman's ruling explains that E1 ran pre-del-fix code; the counted-frame deletion in `a467bb7` is the fix. No revert or further E1 action.
- E2, 50k with `max_df=500`: recall 0.9898, US 0.9990 (n=30,157), India n=0, 142,701,028 pairs. It crashed during the first featurization chunk; no OOF or threshold result. The log does not prove whether `--use-rare` was enabled, so keep the result caveated and do not treat it as an adopted configuration.
- The 50k rare result 0.9892 / 0.9858 cited by the handoff is a user-reported benchmark; the local E1 crash only independently verifies its candidate recall.
- Raw logs and `RUN1_REPORT.md` from the earlier `ccfa813` worktree remain in the sibling local worktree. The older report predates some later attempts; where its attempt count or E2–E7 statement conflicts with later raw logs, use the raw logs and this updated chronology.

Repository baseline table from `NIGHT_HANDOFF.md` / `AKARI_PLAYBOOK.md`:

| Run | Candidate recall | Pairs before → after subsampling | OOF macro F0.5 | Time | Peak RSS |
|---|---:|---:|---:|---:|---:|
| 5k baseline | 0.9920 | 2.65M → 272k | 0.9856 | ~3m | 5.1 GB |
| 20k baseline | 0.9773 | 5.48M → 1.05M | 0.9759 | ~4m | 5.2 GB |
| 50k baseline, no rare channel | 0.9774 | 35.7M → 2.70M | 0.9773 | ~22m | 7.7 GB |

The local failed E1/E2 runs exceed the 60M raw-pair adoption bar despite high recall. Historical cap-sweep evidence remains in `context/cap_sweep.md`; the handoff forbids per-S1 caps because they materially reduce recall.

## Remaining gates and next actions

1. Let the fresh Run 1 finish under the 48 GB combined cap. Preserve the final 20 log lines verbatim if it stops or crashes and apply the closest `AKARI_PLAYBOOK.md` failure-table response. Report overall + US recall, pre/post-subsample pairs, OOF macro F0.5, top-8 threshold curve, selected plateau threshold, wall time, and measured peak RSS against the baseline table.
2. Do not launch Run 2 until overall recall ≥0.95 and OOF ≥0.970 with the top-8 threshold curve/plateau evidence. India-only measurement is dropped by user ruling because about 40% of train S1 have blank country; report overall and US recall instead. Preserve the code-change gate and first-chunk >50M-pair tripwire.
3. After Run 1, validate the replacement multiprocessing featurization on 5k with exact equality of IDs, feature values, labels, and sampled rows against the scalar implementation. Keep each existing S1 chunk as an outer partition, but process those partitions sequentially; use a persistent six-worker pool for bounded feature-row batches within the current chunk, tag results with source offsets, and restore row order before labels/subsampling. Record wall time and process-tree memory; only use it if exact and within 48 GB. Do not send whole pair frames to multiple workers. `cdist` is not viable for aligned pairs.
4. If gates hold, Run 2 uses the 100k command with test inference enabled, `--use-rare --rare-max-df 1000`, and output directory `out_full`. Recheck free space and use the handoff/runbook values for remaining flags/chunk size. Stop immediately if the first full-run chunk crosses the >50M raw-pair tripwire.
5. Validate with `python utils/validate_submission.py --matching out_full/matching_results.tsv --candidate out_full/candidate_pairs.tsv --test-dir $DATA/test` and add `--check-ids` if feasible. Matching must have exactly 1,732,544 rows.
6. Apply the documented threshold plateau rule and stage READY-TO-SUBMIT rows in `SUBMISSIONS.md`. Never upload to the leaderboard; Aman alone submits.
7. Work E1–E6 only within their playbook rules, with at most one ≤50k job alongside Run 2 and a 48 GB combined memory cap. New E7 designs must meet the same guardrails. Do not use ensembles, neural models, dense retrieval, full-2.2M training, per-S1 caps, or changes to metrics, writer, or CV.

No Run 2, submission validation, threshold staging, leaderboard upload, or 5k multiprocessing equality experiment has occurred yet.

## Restart-safe Run 1 setup

After the second Python 3.11 `TypeError`, the old Run 1 PID was confirmed inactive and `out_100k` was empty. Read-only review found that the reported context-manager exception is inconsistent with ordinary built-in strings and integers at `_char_bigrams`; it is not a memory tripwire. The next attempt adds exception-only diagnostics that capture the interpreter/module/source identity, exact input types and lengths, pair IDs, and value hashes without copying business text into the log. It replays only the failing 100k batch after an exception, then re-raises the original error.

To avoid repeating completed training featurization chunks after a later crash, local experimental code now supports `--resume-train-chunks`. Each completed chunk is saved after labels and deterministic negative subsampling, as a compressed NumPy archive written to a temporary file and atomically promoted. The cache key fingerprints all four training files, relevant source-code contents, feature schema, runtime versions, and all chunk-affecting flags. A cache miss, invalid archive, or mismatched key causes that chunk to be recomputed. The same output directory, data, arguments, and `PYTHONHASHSEED=42` must be used on restart. Loading, sampling, normalization, blocking, recall gating, and OOF fitting still run again; completed featurization chunks are reused. The held-out test inference chunks are not checkpointed.

The retry launcher is `start_run_100k_checkpointed.ps1` and uses Python 3.11, `--sample-s1 100000 --skip-test --n-splits 5 --train-chunk-size 2500 --use-rare --resume-train-chunks`. It writes combined output to `run_100k.log`; the fail-closed process-tree monitor writes `run_100k.monitor.log`, with a 44 GB tree / 47 GB system stop threshold under the 48 GB combined cap. Before launch, C: had about 99.4 GB free, the Python 3.11 environment dependencies imported successfully, and the changed Python files passed compilation plus `git diff --check`. The experiment remains on `exp/run1-2500-a467-20260926`; the frozen branch has not been changed or pushed.

## Latest Run 1 retry failure and fix

The Python 3.11 retry reached first-chunk featurization but stopped with `TypeError: 'int' object is not subscriptable` in `_char_bigrams(name2)`. The failing feature batch was rows 5,300,000:5,400,000 of the first outer chunk (7,705,746 candidate pairs). This attempt is archived as `run_100k_attempt_a467_py311_numeric_text_typeerror_20260926.log` with matching `.stderr.log` and `.monitor.log`; `out_100k` is empty. The exact final 20 log lines were:

```text
    run(args.data_dir, args.out_dir, n_splits=args.n_splits, seed=args.seed,
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\pipeline.py", line 425, in run
    feat_df = build_feature_frame_vectorized(pairs_tr)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 109, in build_feature_frame_vectorized
    batch = [pair_features(x1, y1, z1, x2, y2, z2)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 109, in <listcomp>
    batch = [pair_features(x1, y1, z1, x2, y2, z2)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 57, in pair_features
    _jaccard(_char_bigrams(name1), _char_bigrams(name2)),
                                   ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 25, in _char_bigrams
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 25, in <setcomp>
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()
            ~^^^^^^^^^
TypeError: 'int' object is not subscriptable
```

The exception identifies a non-string integer in `name2`; the pair frame was not persisted, so its source row/value was unavailable after process exit. The root has not been attributed to a particular source record. The local fix now enforces the feature input contract on all six text columns with `fillna("").astype(str)` before batching. This preserves null-as-empty handling and lets numeric scalar values be processed as their text representation. Both earlier memory fixes remain in place. The next action is to cycle Run 1 from the beginning and record recall, OOF, wall time, and peak memory; no run-2 gate can be evaluated until that completes.

The retry started at 11:43 local time on Python 3.11 with `--sample-s1 100000 --skip-test --n-splits 5 --train-chunk-size 2500 --use-rare`; it writes to `run_100k.log` and `out_100k`. The 2,500-S1 chunk size is retained because the prior post-fix attempt reached 43.23 GB system use, above the 40 GB trip point in the earlier instruction. A separate process-tree watcher samples every 10 seconds and stops the run at 47 GB tree RSS or 58 GB system use; its evidence is `run_100k.monitor.log`. The retry began loading training sources normally. `AKARI_TRACE_FEATURE_BATCHES=1` and Python faulthandler are enabled. No peak-RSS sampler is available inside the isolated venv, so the external watcher supplies peak measurements.

At 11:49, candidate blocking completed with 311,199,888 pairs, overall recall 0.9834, and US recall 0.9966 (n=60,078). The printed India line has n=0 and is ignored under Aman's ruling. Featurization began in 40 chunks of 2,500 S1 entities. At that point the external watcher measured 31.40 GB peak process-tree RSS and 43.51 GB peak system use; no watchdog limit was reached. OOF and threshold results are still pending.

The first chunk has now passed the prior failure location: feature rows 5,300,000:5,400,000 completed successfully after the text cast, and processing continued past 6.3M / 7.7M rows with no exception. Peak process-tree RSS remains 31.40 GB. This supports the boundary fix for the observed crash, though the run and upstream numeric source diagnosis are not yet complete.

At 11:59, train chunk 1/40 completed: 7,705,746 raw pairs became 137,233 rows after negative subsampling, with 8,455 positives. Chunk 2 has started. The current run remains active; 39 chunks plus grouped OOF, calibration, and threshold selection remain.

At 12:10, train chunk 2/40 completed: 7,938,205 raw pairs became 136,025 rows after negative subsampling, with 8,380 positives. Chunk 3 is next. The retry has not reproduced the feature type error, and peak process-tree RSS remains 31.40 GB.

At 12:20, train chunk 3/40 completed: 7,696,103 raw pairs became 137,917 rows after negative subsampling, with 8,511 positives. It passed the prior 5.3M-row failure point. Chunk 4 started; current tree RSS was about 18.7 GB and peak remained 31.40 GB.

At 12:30, train chunk 4/40 completed: 7,745,447 raw pairs became 137,948 rows after negative subsampling, with 8,500 positives. The post-run multiprocessing equality check will compare ordered pair IDs, float32 feature values, labels, and the final sampled training rows against the serial path; it will not run alongside Run 1.

Observed featurization time is roughly 10 minutes per 2,500-S1 chunk. With 40 chunks, the actual Run 1 duration is pacing around 6–7 hours, materially longer than the runbook's 1.5–2.5-hour estimate. This is a measured wall-clock projection from the first three chunks, not a final runtime.

## Multiprocessing equality check prepared (not run)

To honor the replacement speed experiment without competing with Run 1, a separate local experimental helper and runner are prepared:

- `code/business_entity_resolution/src/features_multiprocess_experiment.py` uses six workers by default, 25,000-pair payloads, and at most six batches in flight. Workers receive only bounded text batches; offsets restore original row order. It calls the unchanged scalar `pair_features` and emits float32 features.
- `utils/validate_multiprocess_featurize.py` rebuilds the deterministic 5k-S1 `--use-rare` sample with the same max-df/prefix/rare settings, then checks both original 2,500-S1 chunks against serial on the same in-memory ordered pair frames. It compares IDs/order, exact float32 bit patterns in row windows, labels, and post-subsample IDs/features/labels. The serial feature reference is held in a temporary memmap to avoid holding two full feature frames at once.
- `utils/watch_process_tree.ps1` is the dedicated watcher for this later run; its default process-tree and total-system limits are 47 GB, leaving margin under the 48 GB ceiling.

These files have not been executed or committed. Run the equality check only after Run 1 completes. The active run remains the only compute job.

## Second Python 3.11 retry failure

The retry stopped at 12:44 local time in train chunk 6/40, while featurizing rows 2,300,000:2,400,000 of a 7,794,211-pair chunk. It did not reach OOF or threshold selection. This is archived as `run_100k_attempt_a467_py311_typeerror_contextmanager_chunk6_20260926.log` plus matching `.stderr.log` and `.monitor.log`. `out_100k` remained empty. The process-tree peak was 31.40 GB and peak system use was 43.51 GB; neither watchdog threshold caused the stop. There is no live Run 1 process.

The traceback again enters `_char_bigrams(name2)`, but its final exception differs from the first retry: `TypeError: 'int' object does not support the context manager protocol`. The six input columns are now cast after null fill, so the source value/type is still unexplained. The playbook failure table has no matching generic Python `TypeError` entry; this was not a memory or candidate-recall failure. Under Aman's explicit debugging authorization, next add exception-only batch replay that scans rows in the failing 100k batch and logs the exact pair IDs, Python type names, and short reprs of the six already-cast inputs, then cycle Run 1 again. This diagnostic runs only after a batch exception and does not change feature values, metrics, writer, or CV.

Exact last 20 lines from the archived stderr log:

```text
    run(args.data_dir, args.out_dir, n_splits=args.n_splits, seed=args.seed,
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\pipeline.py", line 425, in run
    feat_df = build_feature_frame_vectorized(pairs_tr)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 113, in build_feature_frame_vectorized
    batch = [pair_features(x1, y1, z1, x2, y2, z2)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 113, in <listcomp>
    batch = [pair_features(x1, y1, z1, x2, y2, z2)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 57, in pair_features
    _jaccard(_char_bigrams(name1), _char_bigrams(name2)),
                                   ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 25, in _char_bigrams
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py", line 25, in <setcomp>
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()
                ~~^~~
TypeError: 'int' object does not support the context manager protocol
```

## Status document publishing

The status.md file is committed locally. The initial documentation commit is 577120b28719f257612c4e17d8e5d04ffbb09c2f; the current local branch tip includes a follow-up documenting publication failure. At 09:55, git push -u origin exp/run1-2500-a467-20260926 was rejected with HTTP 403 (permission denied to Stakeylock). A GitHub connector request to create the same branch at local commit b3f83e8b3c9fe90e57fdd5a777f4a1d59c029d48 was also rejected with HTTP 403 (Resource not accessible by integration). Neither method created the remote exp branch or pushed files; the local branch has no upstream. Aman's latest ruling: stop trying to push; keep the status, code, and outputs local and transfer outputs manually at staging.

## Checkpointed Run 1 retry (2026-09-26)

A fresh Python 3.11 run started at 13:44 local on `exp/run1-2500-a467-20260926`, with `--sample-s1 100000 --skip-test --n-splits 5 --train-chunk-size 2500 --use-rare --resume-train-chunks` and `PYTHONHASHSEED=42`. Its checkpoint namespace is `out_100k/.train_chunk_checkpoints/bf9a23743095f160b91c3a531a99cde52f61062d1b8a52b7fce03e1352b0867b`.

By 13:50, blocking completed: 311,199,888 union candidate pairs, overall candidate recall 0.9834 and US recall 0.9966. The India slice has n=0 and is excluded per Aman’s ruling. Featurization entered 40 chunks of 2,500 S1 rows. At 13:56 it was still computing chunk 1, so no completed chunk archive had been written yet; tree RSS peaked at 31.48 GB and system use at 43.06 GB, below the 44/47 GB watchdog limits.

Completed sampled train chunks are atomically cached and reused only if inputs, code, flags, runtime and hash seed match. A retry repeats loading, sampling, normalization, blocking and the recall gate; a chunk interrupted before its archive is committed must be recomputed. OOF fitting is rerun after cached feature chunks are assembled. This Run 1 skips test inference, and the current checkpoint implementation does not cover full-run test inference chunks.

The launcher now archives any previous `run_100k.log` and exit code before a retry and refuses to start a second process for the same output directory. Wait for the current process to exit before retrying; then rerun `start_run_100k_checkpointed.ps1` with the same data, code and output directory.

## Checkpointed retry failure and recovery evidence

The checkpointed Python 3.11 attempt ended around 14:04 local in train chunk 2/40. It raised `TypeError: 'range_iterator' object is not subscriptable` in feature batch 2,700,000:2,800,000 of 7,938,205 pairs. Replaying the same 100k input batch row by row did not reproduce the exception. No OOF score or threshold curve was reached. This is an unresolved runtime/feature exception, not a diagnosed bad input or OOM: the monitor ended normally with tree peak 31.48 GB and system peak 43.06 GB, below its thresholds.

The failure table has no matching generic `TypeError` row. Under Aman's explicit debugging authorization, the response is to preserve the failure evidence, repair native stderr collection in the launcher, enable Python faulthandler and retry with unchanged Python source, data, flags and hash seed. The launcher previously used PowerShell's `Stop` error preference during the native process; native stderr can therefore abort the wrapper before a complete traceback/exit-code report is collected. The next launch uses `Continue` while collecting Python output and restores `Stop` afterward. This is a logging correction; the original exception remains unexplained.

Chunk 1 completed at 14:00:56: 7,705,746 pairs -> 137,233 sampled rows, including 8,455 positives. Its atomic checkpoint is 3,197,672 bytes. A readback check successfully restored those exact counts, confirmed finite features and binary labels, confirmed the current input/code fingerprint still matches, and rejected a wrong fingerprint. The next retry can reuse chunk 1; chunk 2 has no completed archive. Run 2 remains gated on a completed Run 1 OOF result. No leaderboard upload occurred.

The user has now requested another Git push attempt for the experimental branch. Source, diagnostics, recovery scripts, the prepared multiprocessing check and this status document are the intended versioned artifacts; dataset, generated checkpoint archives and runtime logs stay local.

Exact last 20 lines of the failed attempt's combined log:

```text
[13:44:26] Fingerprinting training inputs and code for chunk resume...
[13:44:27] Train chunk resume enabled: C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\out_100k\.train_chunk_checkpoints\bf9a23743095f160b91c3a531a99cde52f61062d1b8a52b7fce03e1352b0867b (key=bf9a23743095)
[13:44:27] Loading training sources...
[13:45:25] Train sizes: S1=2206821 S2=5034616 S3=5285603 | GT rows=2206821 | singleton rate=0.056
[13:45:25] --sample-s1: subsampling to 100000 S1 (matched S2/S3 kept, distractors proportional).
[13:45:28] Sampled train: S1=100000 S2=457177 S3=468311
[13:45:28] --skip-test: not loading test data; steps 8-9 (inference + outputs) will be skipped.
[13:45:28] Normalizing text fields...
[13:45:43] Blocking train S1 vs S2 (max_df=300)...
[13:47:30] Blocking train S1 vs S3 (max_df=300)...
[13:50:16] [TIMING] blocking (both sources): 4.6 min
[13:50:45] Train candidate pairs (S2+S3 union): 311,199,888
[13:50:45] Train candidate recall: 0.9834
[13:50:48] Train candidate recall [US] (n=60,078): 0.9966
[13:50:48] Train candidate recall [IN] (n=0): 1.0000
[13:50:48] Per-S1 candidate counts: p50=2,813 p99=9,252 max=20,383 mean=3112.0
[13:50:48] Featurizing train in 40 chunk(s) of ~2,500 S1...
[14:00:56] train chunk 1/40: pairs=7,705,746 -> subsampled=137,233 (pos in chunk: 8,455)
[FEATURES-ERROR] batch=2700000:2800000 total=7,938,205 python=3.11.13 executable=C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\venv_py311\Scripts\python.exe module=C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py source_sha256=7bb331e3df894e5cee0fa56d3d7021d1ba013c129d95df0d29ef4cf479b17e6d pair_features_file=C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\code\business_entity_resolution\src\features.py exception=builtins.TypeError: TypeError("'range_iterator' object is not subscriptable")
[FEATURES-ERROR] batch exception was not reproduced by row-wise replay for rows 2700000:2800000.
```

## Git publication succeeded and Run 1 resumed

The renewed push succeeded. The remote branch is `exp/run1-2500-a467-20260926` in `Skullybutcher/ml_challenge_2026`. Checkpointing and failure recovery are commit `4eecdbd`; the prepared, isolated multiprocessing equality harness is commit `c87374e`. The remote head was verified against local commit `c87374eb51f069c3a89d946e01d341877ed3189e`. The frozen feature branch was not modified. Dataset files, generated chunk archives and runtime logs were not staged.

After confirming that every checkpointed Python source file still has its original content hash, Run 1 was relaunched with the same parameters and hash seed, the corrected native stderr logging and faulthandler. Initial free disk was 92.31 GB and system use 12.83 GB. The new wrapper PID is 30768 and watcher PID 14184; the monitor again enforces 44 GB tree / 47 GB system limits. The existing chunk 1 checkpoint is retained for reuse after startup loading/blocking; this restart has not yet reached OOF. Previous attempt logs are archived locally before reuse of the current log names.

## CPU/runtime investigation after the 16:55 failure

The 16:44 retry restored chunk 1 from the saved checkpoint at 16:50:30, confirming the live resume path, then exited with code 1 at 16:55 in chunk 2. The original batch error was `TypeError: 'cell' object is not subscriptable`; replay logged exact built-in strings for all six inputs and a CPython internal set insertion `SystemError` on pair S1-426427615 / S2-653212516. The tree/system peaks were 31.39/43.91 GB, below watchdog thresholds. OOF and Run 2 have not been reached.

A standalone stdlib-only reproduction (`utils/probe_bigrams_runtime.py`) repeats the original character-bigram expression on literal strings and checks the resulting sets against known expected outputs. It failed with `TypeError: 'int' object is not subscriptable` after 32.2 seconds / 26,434,649 checks without importing pandas, NumPy or RapidFuzz and without reading the dataset. Data fields and those extensions are therefore not necessary for the failure. Windows also recorded WHEA-Logger event 19 at 16:48:57: Processor Core / Corrected Machine Check / Internal parity error / APIC ID 41. The machine reports an Intel i9-14900KS. These observations make hardware/runtime instability the current lead, but do not prove a specific CPU defect or microcode issue.

The playbook has no generic TypeError row. Under the explicit debugging authorization, the response is a bounded runtime/core isolation probe, preserving the original feature definition and withholding an OOF/Run 2 decision until a stable run completes. The first probe pinned to verified efficiency-core logical CPU 16 passed 56,142,000 checks over 120 seconds. This is diagnostic evidence, not certification of the whole machine. No BIOS, firmware, voltage or global power settings have been changed. Windows topology was queried through GetLogicalProcessorInformationEx: performance logical CPUs 0-15; efficiency CPUs 16-31. APIC IDs have not been equated to Windows logical indices.

GPU check: NVIDIA RTX 5070 Ti, 16,303 MiB VRAM. The installed LightGBM 4.7.0 build rejected a two-round toy GPU smoke test with 'GPU Tree Learner was not enabled in this build'. GPU acceleration would apply to later tree training, not the current Python string feature stage.

Exact last 20 lines of the failed pipeline log:

```text
de\business_entity_resolution\src\features.py", line 140, in build_feature_frame_vectorized
    batch = [pair_features(x1, y1, z1, x2, y2, z2)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\co
de\business_entity_resolution\src\features.py", line 140, in <listcomp>
    batch = [pair_features(x1, y1, z1, x2, y2, z2)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\co
de\business_entity_resolution\src\features.py", line 59, in pair_features
    _jaccard(_char_bigrams(name1), _char_bigrams(name2)),
                                   ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\co
de\business_entity_resolution\src\features.py", line 27, in _char_bigrams
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\JINITANGSU\Documents\Codex\2026-09-26\akari-pull-and-inspect-first-then\work\ml_challenge_2026_a467\co
de\business_entity_resolution\src\features.py", line 27, in <setcomp>
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()
            ~^^^^^^^^^
TypeError: 'cell' object is not subscriptable
```


## Process affinity workaround and GPU findings

The controlled CPU placement comparison produced these results with identical standalone Python code:

| Process CPU selection | Checks | Duration | Result |
|---|---:|---:|---|
| Unrestricted | 26,434,649 | 32.2 s | TypeError: int object is not subscriptable |
| Performance cores, mask 0x0000ffff | 38,409,371 | 46.2 s | TypeError: unsupported operand types for +: NoneType and int |
| Efficiency core 16, mask 0x00010000 | 56,142,000 | 120 s | All checks passed |
| Efficiency cores, mask 0xffff0000 | 56,490,000 | 120 s | All checks passed |

On the efficiency cores, all 137,233 saved chunk 1 rows were reconstructed from the original source TSVs and their frozen normalization/feature functions: all 2,058,495 feature values matched the saved float32 values exactly, and all 137,233 labels matched ground truth exactly. The full cache verification took 33.5 seconds. This validates reuse of this checkpoint; passing bounded probes does not establish a permanent hardware repair or diagnose Vmin Shift.

Run 1 restarted at 17:12 local with unchanged data, Python source, training flags and hash seed. `utils/run_with_cpu_affinity.py` sets only the Python process CPU mask before imports; the existing launcher now requests the locally verified E-core mask 0xffff0000. The running interpreter's actual mask was verified as 0xffff0000. Its original checkpoint fingerprint still matches bf9a23743095, so chunk 1 remains reusable. Wrapper/watcher PIDs are 28172/20700. Initial free disk/system use were 92.31/12.66 GB. No model, CV, metric, writer, BIOS, firmware or global power setting was changed. The mask is specific to this host's queried topology and must not be assumed valid on another machine.

The RTX 5070 Ti has 16,303 MiB VRAM, but the installed LightGBM 4.7.0 library rejected a two-round toy GPU check because GPU Tree Learner was not enabled. LightGBM's Windows GPU option uses OpenCL; its CUDA implementation is not supported on Windows. GPU tree training would not accelerate the current Python string feature phase. The prepared multiprocessing feature equality experiment remains pending; no speedup or equality result has been claimed for it.

Primary references: [LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html), [Microsoft CPU topology fields](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-processor_relationship), [Intel stability guidance](https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html). Hardware instability is an inference from the reproduction, placement comparison and WHEA event; the specific cause is not established.

## Run 1 progress and featurization speed research (2026-09-26, 20:16 IST)

Run 1 remains active on `exp/run1-2500-a467-20260926`, launched at 17:12 local with Python 3.11, `--sample-s1 100000 --skip-test --n-splits 5 --train-chunk-size 2500 --use-rare --resume-train-chunks`, seed/hash settings unchanged, and verified E-core mask `0xffff0000`. The latest completed item is train chunk 9/40 at 20:10:42; chunk 10 is underway. Checkpoints `train_chunk_0000.npz` through `train_chunk_0008.npz` exist, with chunk 1 reused from an earlier attempt. Chunk 9 processed 7,844,465 pairs and retained 137,375 after negative subsampling. No OOF, threshold curve, final recall, test inference, or submission file is available yet.

Candidate generation produced 311,199,888 pairs with recall 0.9834 overall and 0.9966 for US (the India slice has no labeled rows). Recent chunks take about 20 minutes each. Linear extrapolation puts the remaining 31 chunks at roughly 10 hours 20 minutes of additional feature time, before OOF/model stages; this is a rough pace estimate. At the latest sample, Python RSS was about 15.5 GB, tree peak 31.49 GB, and system peak 44.21 GB, below watchdog limits 44/47 GB. The monitor and process remained alive; no exit marker was present.

### New speed candidate: aligned RapidFuzz `process.cpdist`

Research identified RapidFuzz `process.cpdist` as a separate API from all-pairs `cdist`: `cpdist` scores corresponding elements in two equal-length collections and supports `workers` for C-API scorers that release the GIL. It was added in RapidFuzz 3.8.0; the current environment has 3.14.6. Official references: [RapidFuzz process API](https://rapidfuzz.github.io/RapidFuzz/Usage/process.html#cpdist) and [RapidFuzz changelog](https://github.com/rapidfuzz/RapidFuzz/blob/main/CHANGELOG.rst).

A tiny eight-pair check with two workers found bit-identical float32 values versus the existing scalar calls for `ratio`, `token_sort_ratio`, and `partial_ratio`, using float64 cpdist output, division by 100, then float32 conversion. This is a formula-level smoke check only; it is not a large-sample parity check or speed benchmark. The five existing RapidFuzz similarity features (three name scores and two address scores) are candidates for aligned batched scoring. A separate experiment should compare cpdist to the prepared multiprocessing featurizer on the deterministic 5k-S1 sample, requiring exact full-feature/label/subsample equality and recording wall time and peak RSS. The repository requirement currently allows `rapidfuzz>=3.0`, so an adopted cpdist path must add a version guard or raise the minimum to 3.8.

A second exact-preserving direction is chunk-bounded caching/precomputation of repeated entity-side tokens, rare tokens, lengths, first tokens, and character bigrams. This may reduce Python work but needs an explicit peak-memory check. No cpdist/cache code has been connected to the active pipeline, which remains unchanged. The prepared multiprocessing implementation and 5k parity harness remain available as the comparison baseline. The approved Run 2 `--rare-max-df 1000` choice remains separate: measured 50k recall 0.9857 vs 0.9892 full band with 38% fewer pairs; no further cap or metric/CV edits are proposed.

## Latest Run 1 checkpoint (2026-09-26, 23:46 IST)

The active checkpointed Run 1 is still running under the Python 3.11 launcher (PowerShell PID 28136; Python PIDs 16336 and 2740; watchdog PID 35484). It started at 21:32:28 IST with `--sample-s1 100000 --skip-test --n-splits 5 --train-chunk-size 2500 --use-rare --resume-train-chunks`, using the existing matching checkpoint fingerprint. Do not stop it while healthy. Chunk 20/40 completed at 23:44:28: 7,881,192 candidate pairs and 139,902 rows retained after subsampling. There are 20 valid checkpoint files (`train_chunk_0000.npz` through `train_chunk_0019.npz`); `run_100k.exit` is absent.

Candidate-generation recall remains 0.9834 overall and 0.9966 for US; the India slice is invalid (`n=0`) per Aman's later ruling. OOF, threshold curve, final model recall, and full Run 2 gates remain pending. At 23:45:44 the process-tree RSS was 14.56 GB and system use was 29.81 GB; observed peaks remain 31.49 GB for the process tree and 45.66 GB system-wide, below the active watchdog limits (44/47 GB). No new speed experiment is running alongside Run 1. No leaderboard upload or Git push has occurred.

The prepared 5k multiprocessing-featurization parity harness was statically reviewed. It compares input ID order, every float32 feature bit-for-bit, pre-subsample labels, and post-subsample IDs/features/labels against the serial implementation, while avoiding model training and submission output. It has not been executed; defer execution until Run 1 finishes so it cannot contend with the active job. The Python 3.11 executable, harness, and dataset `train/` and `test/` paths were verified to exist. Ready command from the repo root after Run 1 completes: `& ..\venv_py311\Scripts\python.exe utils\validate_multiprocess_featurize.py --data-dir ..\dataset --sample-s1 5000 --train-chunk-size 2500 --workers 6 --batch-size 25000`.

At a read-only CPU snapshot around 23:39 IST, the training Python process was using about 96% of one logical CPU; the next-highest visible processes were each around 5%, and the other Python gateway was at 0%. This points to serial CPU-bound featurization rather than a competing heavy Python job at that instant; it is a spot sample, not a sustained utilization benchmark.

## Latest Run 1 checkpoint (2026-09-27, 00:22 IST)

The same checkpointed Run 1 remains active. The latest verified completion is chunk 22/40 at 00:14:05 IST: 7,792,810 candidate pairs, 138,245 rows retained after subsampling, and checkpoint `train_chunk_0021.npz`. The launcher and Python worker are alive; no `run_100k.exit` marker was present at the 00:21 check. The active command still uses Python 3.11, `--sample-s1 100000 --skip-test --n-splits 5 --train-chunk-size 2500 --use-rare --resume-train-chunks`.

Candidate recall remains 0.9834 overall and 0.9966 for US. The India-only slice is excluded because it has no labeled rows (`n=0`). OOF and the top-8 threshold curve have not printed, so the Run 2 gates are not yet complete. The Run 2 cap is pre-decided as `--rare-max-df 1000`; do not launch Run 2 until the OOF ≥0.970 and threshold-curve gates are verified.

At the latest monitor sample (00:21:56 IST), the process tree used 10.50 GB and system memory use was 26.70 GB; observed peaks remain 31.49 GB for the tree and 45.66 GB system-wide, below watchdog limits of 44/47 GB. C: had 95.10 GB free at the 00:20 check, exceeding the 40 GB output-space requirement. The five-thousand-row multiprocessing featurization equality harness remains prepared but unrun to avoid competing with Run 1. No leaderboard upload or Git push has occurred.

## Latest Run 1 checkpoint (2026-09-27, 00:33 IST)

Train chunk 23/40 completed at 00:28:19: 7,792,821 candidate pairs and 136,528 rows retained after subsampling. Checkpoint `train_chunk_0022.npz` is present (23 chunk checkpoints total). The launcher, Python worker, and watchdog are alive, and `run_100k.exit` is absent. Latest monitor sample at 00:32:16: process-tree RSS 9.90 GB, system use 26.13 GB; peaks remain 31.49 GB tree / 45.66 GB system, under the 44/47 GB watchdog limits.

The command, candidate recall, and Run 2 gate state are unchanged from the preceding checkpoint. No OOF or threshold curve yet; do not launch Run 2 until those gates are verified. No leaderboard upload or Git push has occurred.

## Latest Run 1 checkpoint and hardware telemetry (2026-09-27, 00:55 IST)

Train chunk 24/40 completed at 00:42:26: 7,993,910 candidate pairs and 140,064 rows after subsampling. The checkpoint exists; the launcher, Python worker, and memory watchdog remain alive, and `run_100k.exit` is absent. At 00:54:43, the process tree was using 9.72 GB and system use was 26.24 GB; recorded peaks remain 31.49 GB tree and 45.66 GB system-wide, below the 44/47 GB watchdog limits.

Thermal checks: `nvidia-smi` reads the RTX 5070 Ti at 45°C (20% utilization, about 25 W). NVIDIA lists 88°C as the reference maximum, so 82°C is the proposed GPU stop threshold. The CPU is an i9-14900KS (Intel lists Tjunction 100°C); 90°C is the proposed CPU package stop threshold. A reliable CPU package temperature reader is not yet available: the installed XTU service has no XTUCLI binary, `root\wmi` ACPI temperature query is denied, and the Windows ACPI zones report 301 K and 290 K with no passive limit applied. These zones are not being mislabeled as CPU package temperature. No Kernel-Power thermal-throttle event appeared in the last six hours. Until a reliable CPU sensor path is established, only GPU temperature and ACPI system-zone/thermal-limit telemetry can be actively monitored.

Run 1 candidate recall is unchanged at 0.9834 overall / 0.9966 US; OOF and the threshold curve remain pending. The CPU/GPU thermal thresholds are proposals pending an active CPU sensor watcher; memory watchdog remains active. No leaderboard upload has occurred. Push attempts resume only under Aman's latest explicit authorization and stay on the experiment branch.

## Run 1 update (2026-09-27, 01:12 IST)

Train chunk 26/40 completed at 01:08:49 IST: 7,796,347 pairs and 135,222 retained rows (8,347 positives). Checkpoint `train_chunk_0025.npz` is present; 26 chunk checkpoints are complete. Run 1's launcher, Python worker, and memory watchdog remain alive; no `run_100k.exit` marker. The training command, data, and E-core affinity remain unchanged.

At 01:11:44, process-tree RSS was 8.95 GB, system use 26.44 GB, peak tree/system 31.49/45.66 GB (watchdog limits 44/47 GB). The RTX 5070 Ti was 46°C at 17% utilization and 25.35 W. A separate four-second CPU sample showed the training worker consumed 5.11 CPU-seconds (about 1.3 logical CPUs over that interval); 16 E-core logical CPUs are assigned, 32 logical CPUs exist in the host, and whole-machine CPU use averaged 8.6%. This confirms the present feature stage is predominantly serial and does not saturate the available CPU. The installed LightGBM binary has no GPU tree learner, and this string-featurization stage is CPU-bound.

Aman approved using a signed monitor to obtain CPU package temperature. The latest LibreHardwareMonitor release artifact was downloaded only for inspection and SHA-256 matched the official release metadata, but its application binary is not Authenticode-signed, so it was not run. We are checking for a fully signed, supported sensor path; CPU package telemetry remains unavailable. GPU temperature and the existing memory watchdog continue to be monitored. The proposed thermal stops remain CPU 90°C and GPU 82°C once supported live telemetry is available.

Candidate recall remains 0.9834 overall / 0.9966 US; India is excluded because n=0. OOF and the top-8 threshold curve are still pending, so Run 2 gates are not evaluated. The approved 5k featurization parity experiment is queued for a parallel slot only after Run 2 gates pass, per the experiment rule; the current run was not interrupted or modified.

## Thermal-monitoring setup check (2026-09-27, 01:21 IST)

At 01:20:26 the checkpointed Run 1 was healthy at 26/40 chunks; chunk 26 remains the latest completion (01:08:49). The process-tree RSS was 9.27 GB; process/system peaks remain 31.49/45.66 GB under the 44/47 GB watchdog limits. The RTX 5070 Ti was 48°C at 20% utilization and 27.03 W. No exit marker exists.

The signed HWiNFO 8.52 portable binary was obtained from the SAC mirror linked on HWiNFO's official download page and Authenticode verified as `REALiX, s.r.o.`. The current desktop control surface exposes only the in-app browser and no native Windows app windows, so its UI alerts cannot be configured from this task. No CPU package sensor is active. A request is pending to run the official hash-verified LibreHardwareMonitor app; the package's Windows application binary is not Authenticode-signed, so it has not been launched. Until a supported CPU reading and alert action are configured, CPU 90°C cannot be actively enforced; GPU telemetry and the memory watchdog remain available. Do not treat the generic ACPI zones as CPU temperatures.

## Run 1 update (2026-09-27, 01:36 IST)

Train chunks 27 and 28 completed at 01:21:06 and 01:32:47 IST. Chunk 27 produced 7,831,732 candidate pairs and retained 138,555 rows (8,542 positives); chunk 28 produced 7,856,883 candidate pairs and retained 137,315 rows (8,469 positives). Checkpoints `train_chunk_0026.npz` and `train_chunk_0027.npz` are present. Run 1 is healthy at 28/40 chunks (70%); launcher, Python workers, and memory watchdog are alive, and `run_100k.exit` is absent.

At 01:36:12, process-tree RSS was 7.88 GB and system use 24.45 GB. Peaks remain 31.49 GB process tree / 45.66 GB system-wide, below 44/47 GB watchdog limits. The RTX 5070 Ti was 44°C at 4% utilization and 22.33 W. CPU package temperature is still unavailable from a supported active sensor; generic ACPI readings are not used as a proxy. Candidate recall remains 0.9834 overall / 0.9966 US, India n=0 excluded. OOF and the top-8 threshold curve have not printed, so Run 2 gates are still pending. No run changes or leaderboard uploads.

## Run 1 update (2026-09-27, 01:50 IST)

Train chunk 29/40 completed at 01:43:37 IST: 7,647,448 candidate pairs and 137,847 retained rows (8,491 positives). Checkpoint `train_chunk_0028.npz` is present; 29 chunks are complete. The launcher, Python workers, and memory watchdog remain alive, with no `run_100k.exit` marker.

At 01:50:14, process-tree RSS was 7.70 GB and system use was 23.34 GB. Peaks remain 31.49 GB process tree / 45.66 GB system, below the 44/47 GB limits. GPU was 42°C at 4% utilization and 22.34 W. CPU package temperature remains unavailable from a supported active sensor; generic ACPI readings are not a substitute. Candidate recall remains 0.9834 overall / 0.9966 US; India n=0 excluded. OOF and the top-8 threshold curve remain pending, so Run 2 gates are not met. No run changes or leaderboard uploads.

## Run 1 update (2026-09-27, 02:43 IST)

Train chunks 30-35 completed: chunk 30 at 01:53:50 (7,739,385 pairs; 136,566 retained; 8,434 positives), chunk 31 at 02:03:31 (7,687,964; 138,527; 8,540), chunk 32 at 02:12:35 (7,580,095; 139,421; 8,601), chunk 33 at 02:21:34 (7,935,218; 138,427; 8,549), chunk 34 at 02:29:43 (7,708,419; 136,299; 8,382), and chunk 35 at 02:37:02 (7,559,538; 136,317; 8,405). Their chunk checkpoints are present. Run 1 is healthy at 35/40; launcher, Python workers, and memory watchdog remain alive, and no `run_100k.exit` marker exists.

At 02:43:42, process-tree RSS was 5.40 GB and system use 20.93 GB. Peaks remain 31.49 GB tree / 45.66 GB system, below the 44/47 GB watchdog limits. GPU was 43°C at 3% utilization and 22.40 W. A supported CPU package temperature reading remains unavailable; generic ACPI zones are not used as a proxy. Candidate recall remains 0.9834 overall / 0.9966 US; India n=0 excluded. OOF and the top-8 threshold curve have not yet printed, so Run 2 gates are pending. No run changes or leaderboard uploads.

## Run 1 update (2026-09-27, 02:45 IST)

Train chunk 36/40 completed at 02:43:57 IST: 7,738,283 candidate pairs and 140,595 retained rows (8,725 positives). Checkpoint `train_chunk_0035.npz` is present. Run 1 remains healthy; launcher, Python workers, and memory watchdog are alive, and no `run_100k.exit` marker exists.

At 02:45:16, process-tree RSS was 4.67 GB and system use was 20.27 GB; peaks remain 31.49/45.66 GB, under the 44/47 GB watchdog limits. GPU was 43°C at 3% utilization and 22.90 W. CPU package temperature remains unavailable from a supported sensor. Candidate recall remains 0.9834 overall / 0.9966 US; India n=0 excluded. OOF and the top-8 threshold curve remain pending; Run 2 gates are not yet confirmed. No leaderboard upload.

## Run 1 update (2026-09-27, 03:00 IST)

Train chunks 37 and 38 completed at 02:50:15 and 02:56:00 IST. Chunk 37 produced 7,796,136 candidate pairs and retained 137,666 rows (8,500 positives); chunk 38 produced 7,958,390 pairs and retained 138,945 rows (8,573 positives). Checkpoints `train_chunk_0036.npz` and `train_chunk_0037.npz` are present. Run 1 is healthy at 38/40; launcher, Python workers, and memory watchdog remain alive, with no exit marker.

At 03:00:17, process-tree RSS was 4.02 GB and system use was 19.36 GB. Peaks remain 31.49 GB process tree / 45.66 GB system, below the 44/47 GB watchdog limits. GPU was 42°C at 4% utilization and 22.10 W. CPU package temperature remains unavailable from a supported live sensor; generic ACPI zones are not a proxy. Candidate recall remains 0.9834 overall / 0.9966 US; India n=0 excluded. OOF and the threshold curve are pending. No leaderboard upload.
## Run 1 final report (2026-09-27 IST)

Run 1 completed successfully with exit code 0 at 03:10:18 after starting at 21:32:31 (about 5 h 38 min). All 40/40 training chunks completed and were checkpointed. The run used the authorized 100k sample, rare channel, 2,500-S1 train chunks, five folds, and `--skip-test`.

| Sample | Candidate recall | OOF macro F0.5 | Pairs before → after subsampling |
|---|---:|---:|---:|
| 5k baseline | 0.9920 | 0.9856 | 2.65M → 272k |
| 20k baseline | 0.9773 | 0.9759 | 5.48M → 1.05M |
| 50k baseline | 0.9774 | 0.9773 | 35.7M → 2.70M |
| Run 1, 100k | **0.9834** | **0.9825** | **311,199,888 → 5,509,855** |

Run 1 is +0.0060 recall vs the 50k baseline and +0.0052 OOF F0.5. The US slice was 0.9966 recall (`n=60,078`). India had `n=0`, so its displayed 1.0000 is excluded. Positives remained 339,918 after subsampling. The threshold curve was: `0.62:0.9825, 0.65:0.9825, 0.57:0.9825, 0.60:0.9825, 0.68:0.9824, 0.70:0.9824, 0.55:0.9823, 0.72:0.9822`. The model's best point was threshold 0.625 / OOF 0.9825. The highest displayed top-eight threshold within 0.001 of best is 0.72 (score 0.9822, 0.0003 below best); OOF predictions and the full 37-point curve were not persisted, so this is the highest qualifying displayed point rather than a verified global maximum.

**Run 1 gates: PASS.** Recall ≥0.95, OOF ≥0.970, and the top-eight curve printed. The India-only slice is inapplicable at `n=0`. Peak memory was 31.49 GB process-tree RSS and 45.66 GB system used, under the 44/47 GB active watchdog limits. The pipeline's own RSS sampler was unavailable because `psutil` is not installed; the external process-tree monitor supplied the peak. At Run 2 preflight, C: had 94.7 GB free (≥40 GB required), free RAM was 49.7 GB, the RTX 5070 Ti was 43°C, and no Run 2 process was active. A supported CPU-package temperature reading is still unavailable; generic ACPI zones are not treated as CPU temperature.

Run 2 launch settings: same dataset and Python 3.11 environment, `PYTHONHASHSEED=42`, CPU affinity `0xffff0000`, 100k sample, five folds, 2,500-S1 train chunks, `--use-rare --rare-max-df 1000 --resume-train-chunks`, and no `--skip-test`. The requested plateau policy is applied by `utils/run_pipeline_plateau.py` to Run 2's own 37-point OOF grid; it chooses the highest threshold within 0.001 of the grid best, without editing the frozen pipeline or metric code. Memory monitoring is 44 GB process-tree / 47 GB system; GPU monitoring stops at 82°C; the first test chunk pair-count tripwire stops the run above 50,000,000 pairs. The 48 GB combined cap remains in force.
## Run 2 launch (2026-09-27, 04:09 IST)

Run 2 started on the authorized experiment branch after Run 1 gates passed. Launcher PID 32000; memory watchdog PID 22988; first-test-chunk pair tripwire PID 38256; GPU thermal watchdog PID 30584. Initial log confirms affinity `0xffff0000`, fixed hash seed 42, and a new resumable checkpoint fingerprint `8a332660fb9a`. At the first monitor sample the process tree used 1.25 GB and system memory 15.29 GB; GPU was 42°C. No failure/exit marker is present. Training-source loading has begun; test inference has not yet begun, so the >50M first-test-chunk tripwire remains armed.
## Run 2 progress (2026-09-27, 04:42 IST)

Training chunks 1-3 completed at 04:28:07, 04:34:00, and 04:39:43. Respectively they contained 3,706,796 / 3,793,193 / 3,636,619 candidate pairs before subsampling and retained 135,311 / 134,877 / 136,080 rows (8,389 / 8,333 / 8,444 positives). Three resumable chunk checkpoints are present; the Run 2 launcher is alive and run_full.exit is absent.

Run 2 candidate recall is 0.9752 overall and 0.9911 for US (n=60,078); India remains n=0 and is excluded. At 04:42, process-tree RSS was 15.08 GB with peak 17.83 GB; system use was 29.12 GB with peak 31.88 GB. GPU was 41°C. The 44/47 GB memory watchdogs and 82°C GPU monitor remain active. No failure, tripwire, or test inference has occurred yet.

## Run 2 progress (2026-09-27, 04:57 IST)

Training chunks 4-6 completed at 04:45:35, 04:51:09, and 04:56:40. They contained 3,720,283 / 3,671,432 / 3,718,471 candidate pairs before subsampling and retained 135,741 / 136,848 / 136,930 rows (8,425 / 8,474 / 8,534 positives). Six resumable checkpoints are present; the launcher is alive and run_full.exit is absent.

Run 2 candidate recall remains 0.9752 overall and 0.9911 for US (n=60,078); India n=0 excluded. At 04:57, process-tree RSS was 14.62 GB and system use was 28.64 GB; peaks remain 17.83 / 31.88 GB. GPU was 41°C. No failure or watchdog/tripwire event occurred; test inference has not begun.

## Run 2 progress (2026-09-27, 05:12 IST)

Training chunks 7-8 completed at 05:02:16 and 05:07:39. Chunk 7 had 3,704,129 candidate pairs and retained 134,480 rows (8,340 positives); chunk 8 had 3,752,034 pairs and retained 137,548 rows (8,525 positives). Eight resumable checkpoints are present; the launcher is alive and run_full.exit is absent.

Run 2 candidate recall remains 0.9752 overall and 0.9911 for US (n=60,078); India n=0 excluded. At 05:12, process-tree RSS was 14.49 GB and system use was 28.51 GB; peaks remain 17.83 / 31.88 GB. GPU was 41°C. No failure or watchdog/tripwire event occurred; test inference has not begun.

## Run 2 progress (2026-09-27, 05:27 IST)

Training chunks 9-11 completed at 05:13:04, 05:18:32, and 05:23:43. Chunk 9 had 3,709,998 candidate pairs and retained 135,697 rows (8,413 positives); chunk 10 had 3,730,444 pairs and retained 139,130 rows (8,635 positives); chunk 11 had 3,684,865 pairs and retained 137,586 rows (8,533 positives). Eleven resumable checkpoints are present; the launcher is alive and run_full.exit is absent.

Run 2 candidate recall remains 0.9752 overall and 0.9911 for US (n=60,078); India n=0 excluded. At 05:27, process-tree RSS was 14.52 GB and system use was 28.58 GB; peaks remain 17.83 / 31.88 GB. GPU was 41°C. No failure or watchdog/tripwire event occurred; test inference has not begun.

## Run 2 progress (2026-09-27, 05:42 IST)

Training chunks 12-14 completed at 05:28:58, 05:34:06, and 05:39:08. Chunk 12 had 3,667,347 candidate pairs and retained 135,740 rows (8,392 positives); chunk 13 had 3,657,105 pairs and retained 135,677 rows (8,411 positives); chunk 14 had 3,735,126 pairs and retained 133,575 rows (8,249 positives). Fourteen resumable checkpoints are present; the launcher is alive and run_full.exit is absent.

Run 2 candidate recall remains 0.9752 overall and 0.9911 for US (n=60,078); India n=0 excluded. At 05:42, process-tree RSS was 13.51 GB and system use was 27.58 GB; peaks remain 17.83 / 31.88 GB. GPU was 41°C. No failure or watchdog/tripwire event occurred; test inference has not begun.

## Run 2 progress (2026-09-27, 05:57 IST)

Training chunks 15-17 completed at 05:44:15, 05:49:04, and 05:53:46. Chunk 15 had 3,778,111 candidate pairs and retained 135,951 rows (8,438 positives); chunk 16 had 3,690,429 pairs and retained 134,810 rows (8,364 positives); chunk 17 had 3,657,122 pairs and retained 134,468 rows (8,362 positives). Seventeen resumable checkpoints are present; the launcher is alive and run_full.exit is absent.

Run 2 candidate recall remains 0.9752 overall and 0.9911 for US (n=60,078); India n=0 excluded. At 05:57, process-tree RSS was 13.61 GB and system use was 27.87 GB; peaks remain 17.83 / 31.88 GB. GPU was 42°C. No failure or watchdog/tripwire event occurred; test inference has not begun.

## Run 2 progress (2026-09-27, 06:13 IST)

Training chunks 18-21 completed at 05:58:30, 06:03:07, 06:07:42, and 06:12:09. They contained 3,703,612 / 3,707,528 / 3,848,939 / 3,694,685 candidate pairs before subsampling and retained 133,758 / 136,679 / 137,713 / 135,148 rows (8,305 / 8,469 / 8,578 / 8,382 positives). Twenty-one resumable checkpoints are present; the launcher is alive and run_full.exit is absent.

Run 2 candidate recall remains 0.9752 overall and 0.9911 for US (n=60,078); India n=0 excluded. At 06:13, process-tree RSS was 12.88 GB and system use was 27.24 GB; peaks remain 17.83 / 31.88 GB. GPU was 42°C. No failure or watchdog/tripwire event occurred; test inference has not begun.

## Run 2 progress (2026-09-27, 06:28 IST)

Training chunks 22-24 completed at 06:16:27, 06:20:37, and 06:24:41. Chunk 22 had 3,698,092 candidate pairs and retained 136,518 rows (8,470 positives); chunk 23 had 3,663,280 pairs and retained 134,649 rows (8,349 positives); chunk 24 had 3,708,683 pairs and retained 137,830 rows (8,542 positives). Twenty-four resumable checkpoints are present; the launcher is alive and run_full.exit is absent.

Run 2 candidate recall remains 0.9752 overall and 0.9911 for US (n=60,078); India n=0 excluded. At 06:28, process-tree RSS was 12.19 GB and system use was 26.32 GB; peaks remain 17.83 / 31.88 GB. GPU was 41°C. No failure or watchdog/tripwire event occurred; test inference has not begun.

## Run 2 progress (2026-09-27, 06:40 IST)

Training chunks 25-28 completed at 06:28:37, 06:32:19, 06:36:00, and 06:39:41. They contained 3,722,963 / 3,635,714 / 3,633,875 / 3,753,422 candidate pairs and retained 132,910 / 133,544 / 136,771 / 135,774 rows (8,233 / 8,286 / 8,470 / 8,408 positives). Twenty-eight resumable checkpoints are present; the launcher is alive and run_full.exit is absent.

Run 2 candidate recall remains 0.9752 overall and 0.9911 for US (n=60,078); India n=0 excluded. At 06:40, process-tree RSS was 11.67 GB and system use was 26.02 GB; peaks remain 17.83 / 31.88 GB. GPU was 42°C. No failure or watchdog/tripwire event occurred; test inference has not begun.


## Run 2 inference recovery and status (2026-09-27, 10:22 IST)

The first inference attempt was stopped by the system-memory watchdog at 09:13:15 IST when total system RAM reached 47.00 GB; process-tree RSS was 33.81 GB, below its 44 GB limit. The exact final 50 log lines and monitor evidence are preserved in `run_full_attempt_system_memory_watchdog_20260927_091315.*`. The documented RAM-pressure failure-table guidance is to reduce competing memory pressure and rerun; the minimum run-specific adjustment was lowering only test inference chunk size from 100,000 to 25,000 S1s. The training/model flags and saved checkpoints were left unchanged. The all-test-chunk >50M pair watcher is active.

The retry started at 09:32:29 IST and resumed all 40/40 training chunks from fingerprint `8a332660fb9a`; 41 checkpoint files remain in `out_full/.train_chunk_checkpoints` (40 training chunks plus metadata). Run 2 recomputed OOF 0.9761 and printed its threshold curve; candidate recall is 0.9752 overall / 0.9911 US, with India `n=0` excluded. The wrapper selected threshold 0.800. The recall, OOF, and curve gates pass. Test inference began at 09:51:31 in 70 chunks of about 25,000 S1s. At 10:21:30 IST the process and all watchdogs were alive, but the first inference chunk had not completed: `candidate_pairs.tsv` remains 0 bytes and no test-pair total has been logged. No tripwire or thermal marker is present.

At 10:21:30, process-tree RSS was 20.81 GB (peak 21.70 GB); system use was 33.89 GB (peak 34.68 GB), below the 44 GB tree, 47 GB system, and 48 GB combined limits. The RTX 5070 Ti was 42°C at 4% utilization. A supported CPU-package sensor is unavailable, so no CPU-package reading is reported. No validator was run because test outputs are not complete. No leaderboard upload was made.

## Run 2 inference monitor (2026-09-27, 10:51 IST)

The checkpointed Run 2 process and its memory, all-test-chunk tripwire, and GPU watchers remain alive. The inference log has not advanced beyond the 09:51:31 launch of 70 test chunks at about 25,000 S1s each; no chunk completion/pair count is recorded yet, and `candidate_pairs.tsv` is still empty. The training checkpoints remain intact (41 files under the Run 2 checkpoint fingerprint). The worker CPU time continues to increase, so the process is active rather than exited.

At 10:51:25 IST, process-tree RSS was 23.14 GB (peak 23.14 GB) and system memory use was 36.29 GB (peak 36.29 GB), below the 44/47 GB watchdogs and 48 GB cap. GPU was 42°C. No tripwire or exit marker is present. CPU-package temperature remains unavailable from a supported sensor. No change to Run 2 settings or outputs was made; validator is still deferred until inference writes its results.

## Run 2 pair-tripwire recovery (2026-09-27, 11:36 IST)

At 11:14:21 IST, inference chunk 1/70 (25,000 S1s) completed with 73,126,904 pairs and 24,146 matched S1s in 4,970.2 seconds. The all-chunk tripwire stopped the process at 11:14:23 because the chunk exceeded 50 million pairs; this was a merge explosion, not a memory or thermal stop. The closest playbook row is: `Any test chunk logs pairs= > 50M | Merge explosion vs 10M pool | STOP the run, report. Approved fix (only with Aman's go): --max-pairs-per-prefix-key 200000→50000 for the test stage.` The tripwire report, archived marker, and exact final 50 Run 2 log lines are preserved in `run_full.tripwire.log`, `run_full_tripwire_attempt_20260927_111423.marker`, and `run_full_tripwire_chunk1_73126904_last50_20260927.txt`. The incomplete 943 MB candidate TSV is retained locally as `out_full/candidate_pairs_tripwire_partial_20260927_111423.tsv`; it was not staged or validated.

The authorized recovery uses the smallest test-only chunk-size change: 25,000→12,500 S1s, keeping the training/model flags and checkpoint fingerprint unchanged. Both monitored-launcher defaults now use 12,500 so a restart cannot silently revert to 100,000. Retry PID 31544 started at 11:27:30 with `--resume-train-chunks` and checkpoint key `8a332660fb9a`; all 41 training checkpoint files remain. At 11:35 it was still normalizing the loaded test data, before the checkpoint-reuse lines; the run log and watchers were active. Latest resources were 7.44 GB process tree / 20.58 GB system, GPU 43°C. CPU package sensor remains unavailable. No submission upload occurred.

## Run 2 recovery verified (2026-09-27, 11:47 IST)

The retry confirmed checkpoint reuse: all 40/40 training chunks logged `resumed checkpoint` from the unchanged key `8a332660fb9a`; no training chunk was recomputed. The repeated training/OOF stage completed with the same candidate recall 0.9752, OOF macro F0.5 0.9761, top-eight curve, and selected threshold 0.800. Test inference restarted at 11:46:37 in 139 chunks of approximately 12,500 S1s. The first smaller test chunk is still running, so there is no new pair count or completed candidate output yet. At 11:47, the process tree was 12.66 GB (peak 18.01 GB), system use 25.98 GB (peak 31.33 GB), and GPU 46°C. Memory and thermal watchers plus the 50M all-chunk tripwire are active; no exit or tripwire marker is present.

## Run 2 inference monitor (2026-09-27, 12:47 IST)

Run 2 remains in test inference after the verified 40/40 training-checkpoint resume. The current attempt started test inference at 11:46:37 IST with 139 chunks of approximately 12,500 S1s. As of 12:46 IST, test chunk 1 is still running; `run_full.log` has no chunk-completion or pair-count line, `candidate_pairs.tsv` remains empty, and neither `run_full.exit` nor a new tripwire marker exists. All 41 training checkpoint files remain present.

At 12:46 IST, process-tree RSS was 17.75 GB (peak 18.41 GB) and system memory was 32.35 GB (peak 33.09 GB), below the 44/47 GB watchdogs and 48 GB cap. RTX 5070 Ti was 46°C. No supported CPU-package sensor is available. The inference worker and memory/GPU/tripwire watchers are alive; no settings or outputs were changed, and validation remains deferred until inference finishes.

## Run 2 tripwire recovery #2 (2026-09-27, 13:03 IST)

At 12:47:17 IST, test chunk 1/139 (12,500 S1s) completed with 55,474,641 pairs and 12,278 matched S1s in 3,640.3 seconds. The all-chunk 50M tripwire stopped Run 2; the closest playbook row is `Any test chunk logs pairs= > 50M | Merge explosion vs 10M pool | STOP the run, report. Approved fix (only with Aman's go): --max-pairs-per-prefix-key 200000→50000 for the test stage.` This was a pair-count stop, not a memory or thermal stop: peak tree RSS 20.28 GB, peak system use 34.89 GB, GPU 47°C. The exact final 50 lines are preserved in `run_full_tripwire_chunk1_55474641_last50_20260927.txt`; the tripwire marker is archived as `run_full_tripwire_attempt_20260927_124717.marker`; the 715,170,277-byte partial candidate TSV is retained locally at `out_full/candidate_pairs_tripwire_partial_20260927_124717.tsv` and is not staged.

Following the active recovery instruction, only test inference chunk size was halved from 12,500 to 6,250 S1; model/training flags remain unchanged. Both launcher defaults now use 6,250. All 41 Run 2 checkpoint files remain. Retry PID 36264 started at 12:52:52 with `--resume-train-chunks` and the same checkpoint fingerprint `8a332660fb9a`. At 13:03 it was in train blocking after loading sources; checkpoint reuse still needed confirmation in the log. The new leader score target reported by Aman is 0.99047; Run 2's existing OOF 0.9761 is below that target. No leaderboard upload occurred.

## Run 2 checkpoint resume confirmed (2026-09-27, 13:08 IST)

The 6,250-S1 inference-chunk retry passed the resume check: all 40/40 training chunks logged `resumed checkpoint` from the unchanged fingerprint `8a332660fb9a`; none were retrained. The retry reproduced 147,631,526 candidate pairs and 337,081 positives after blocking/subsampling, then entered GBDT training. At 13:08 IST the process and watchers were alive, with process-tree RSS 8.91 GB (peak 18.02 GB), system use 22.83 GB (peak 31.93 GB), and GPU 45°C. Test inference and its 50M tripwire remain active; no new inference pair count has been produced yet.

## Run 2 test inference chunk 1 passed (2026-09-27, 13:57 IST)

The 6,250-S1 retry completed test chunk 1/278 at 13:47:41 IST with 33,919,675 pairs, 6,199 matched S1s, and 2,136.8 seconds runtime; the 50M tripwire passed. This confirms the smaller inference chunk is below the per-chunk guardrail. Run 2 remains active in inference, with the same training/OOF result (candidate recall 0.9752; OOF macro F0.5 0.9761; selected threshold 0.800) and all 41 training checkpoints preserved. The 0.99047 leaderboard target has not been beaten by any verified result; inference outputs remain incomplete and unvalidated.

At 13:56 IST, current process-tree RSS was 13.48 GB (peak 18.02 GB), system memory 27.65 GB (peak 31.93 GB), and RTX 5070 Ti temperature 44°C. No exit or tripwire marker is present. CPU-package sensor remains unavailable. No leaderboard upload occurred.

## Run 2 test inference chunk 2 completed (2026-09-27, 14:27 IST)

Test chunk 2/278 completed at 14:23:14 IST with 33,651,377 pairs and 2,132.1 seconds runtime; cumulative matched S1 count is 12,392. It passed the 50M all-chunk tripwire. Run 2 remains active; chunks 1-2 are complete, with 276 test chunks remaining. The 41 training checkpoints remain unchanged; the candidate output is incomplete and has not been validated.

At 14:26 IST, process-tree RSS was 15.93 GB (peak 18.02 GB), system memory was 30.17 GB (peak 31.93 GB), and RTX 5070 Ti temperature was 45°C. The process, memory watchdog, all-chunk tripwire, and GPU watcher are active; no exit or tripwire marker is present. The current OOF result remains 0.9761 at threshold 0.800, below Aman's 0.99047 leaderboard target. No leaderboard upload occurred.

## Run 2 test inference chunk 3 completed (2026-09-27, 15:11 IST)

Test chunk 3/278 completed at 14:59:40 IST with 34,371,585 pairs, 6,193 newly matched S1s, and 2,185.1 seconds runtime; cumulative matched S1 count is 18,585. It passed the 50M all-chunk tripwire. Chunk 4 is now in progress; 275 chunks remain. Candidate output advanced to 1,314,172,352 bytes at the chunk completion time and remains incomplete and unvalidated. All 41 Run 2 training checkpoint files remain intact.

At 15:11 IST, the process and memory/GPU/tripwire watchers were alive. Process-tree RSS was 13.87 GB (peak 18.02 GB), system memory use was 28.47 GB (peak 31.93 GB), and RTX 5070 Ti temperature was 44°C. No exit, tripwire, or thermal marker is present. CPU-package temperature remains unavailable from a supported sensor. Run 2's OOF macro F0.5 remains 0.9761 at selected threshold 0.800, below the reported 0.99047 leaderboard target; test inference is incomplete, so no validated score exists. No leaderboard upload occurred.

## Deadline parallel-inference review (2026-09-27, 15:30 IST)

Aman reports a leaderboard top score of 0.9918. Run 2's measured OOF macro F0.5 is 0.9761; this is a cross-validation metric, not a verified test/LB score, and the 0.0157 difference is not a direct estimate of the final score gap. Test chunks 1-3 are complete; chunk 4 is running. At 15:29 IST, process-tree RSS was 14.52 GB (peak 18.02 GB), system use 27.75 GB (peak 31.93 GB), and GPU temperature 46°C; no exit or tripwire marker was present.

The proposed 8-worker script is not executable as supplied: its `main()` exits after reporting that no model-loading path exists; the current `--skip-test` path does not save models; it has no completed worker dispatch/output-merge implementation; it hardcodes threshold 0.72 instead of Run 2's selected 0.800; and it has no 5k exact-output parity result. Eight workers would also exceed the active six-worker and 48 GB experiment guardrails without a demonstrated memory profile. The current test writer opens `candidate_pairs.tsv` in write mode and has no chunk-resume checkpoint, so killing this healthy process would discard the partial inference output and require replay from chunk 1. An 18x speedup is not demonstrated by this plan, and even a successful speedup would not change the measured OOF metric.

Decision: keep the healthy checkpointed Run 2 inference and its memory, thermal, and 50M-pair tripwire watchers running. Do not kill/retrain, launch the incomplete 8-worker script, change the threshold/model, or add heuristic post-processing. Continue to report only verified outputs; no leaderboard upload has occurred.

## France-specific threshold proposal review (2026-09-27, 15:35 IST)

Aman proposed a France S1 threshold of 0.95 based on the claim that French entities are predominantly singletons. That claim and the expected score gain are not established by the current outputs. `out_full` contains candidate-ID rows and the 41 training-feature checkpoint files, but no serialized Run 2 models, OOF probability array, or test candidate probabilities; the log has only the global OOF curve. A country-specific F0.5 comparison therefore cannot be measured from the available artifacts. The proposed code is also not a drop-in edit: no `parallel_inference.py` exists, and its general threshold of 0.72 conflicts with Run 2's selected global threshold 0.800. Applying either new threshold would require restarting inference; the current writer truncates `candidate_pairs.tsv` on start and does not resume completed test chunks.

Decision: do not apply the unvalidated France threshold or stop the healthy run. Continue the existing Run 2 at the frozen 0.800 threshold under the pair-count and resource watchdogs. Revisit any France-specific threshold only with aligned OOF probabilities and a measured plateau gain; no submission threshold or score claim is changed.

## Run 2 test inference chunk 4 completed (2026-09-27, 15:36 IST)

Test chunk 4/278 completed at 15:35:33 IST with 33,675,713 pairs, 6,188 newly matched S1s, and 2,150.9 seconds runtime; cumulative matched S1 count is 24,773. It passed the 50M all-chunk tripwire. Chunk 5 is now in progress; 274 chunks remain. The 41 training checkpoint files are intact. The France-specific 0.95 proposal remains unvalidated and is not applied; inference continues at the frozen 0.800 threshold.

At 15:35 IST, process-tree RSS was 12.44 GB (peak 18.02 GB), system use was 25.79 GB (peak 31.93 GB), and RTX 5070 Ti temperature was 48°C. The process and memory, GPU, and pair-count watchers were alive; no exit or tripwire marker is present. A supported CPU-package sensor remains unavailable. No matching output has been validated and no leaderboard upload occurred.

## Review of the proposed E5 / parallel-inference plan (2026-09-27, 15:53 IST)

The active Run 2 inference was left running. As of 15:52:51 IST, chunks 1-4 of 278 had completed; chunk 5 had not logged completion yet. The process tree was 14.44 GB (peak 18.02 GB), system use 27.77 GB (peak 31.93 GB), and GPU temperature 47°C. The 50M pair tripwire, memory watchdog, and GPU watcher were active; no exit or tripwire marker was present. `candidate_pairs.tsv` was 1,748,297,525 bytes from chunk 4 and remains incomplete.

The new E5 feature plan was not launched. The current Python 3.11 environment has neither PyTorch nor Transformers installed, and the E5 model weights are not cached. The plan's embedding volume also undercounts its stated data: 12.5M train entities plus 11.7M test entities, with separate name and address vectors, is about 48.4M vectors. At 1,024 dimensions and fp16, that is about 99 GB of vector data, exceeding the 96.3 GB free on C: before model weights, temporary files, or output overhead. Adding E5 features would create a new feature/checkpoint fingerprint and require retraining; the current training checkpoints cannot be reused for that model. Its eight-worker inference script does not exist, and replacing the current inference writer would discard/replay the partial output. The claimed two-minute chunk timing is contradicted by the measured ~35.6-minute chunks in the active run. There is not enough deadline time to install/download the stack, generate embeddings, retrain, implement and validate a new inference engine, and finish test output safely.

The proposed France 0.95 threshold also remains unsupported: the training data has no France S1 examples for a France-specific OOF comparison, and no per-pair probabilities or serialized model are saved for the current run. The assertion that it adds 0.01-0.02 F0.5 is unmeasured. No threshold or feature changes were made. Continue the current run with its selected 0.800 threshold and existing safeguards; outputs stay provisional until complete and validated. No leaderboard upload occurred.

## Run 2 inference stopped at Aman's direction (2026-09-27, 16:03 IST)

Aman explicitly requested stopping the active inference because it would not finish before the deadline or produce the needed score. I terminated the Run 2 Python inference process at 16:03:29 IST; its launcher recorded exit code `-1`. This was an intentional user-directed stop, not a pipeline error, so no failure-table fix or checkpoint resume was triggered. The process-tree, memory, thermal, and pair-tripwire watchers then exited normally.

Four test chunks had completed and passed the 50M tripwire; chunk 5 was still being computed and did not complete. The partial 1,748,297,525-byte `candidate_pairs.tsv` was renamed to `out_full/candidate_pairs_partial_user_stopped_20260927_160355.tsv` and is retained locally. Run 2's 41 training checkpoint files (40 chunks plus metadata) remain intact. The full test run is incomplete: there is no `matching_results.tsv`, validator PASS, READY-TO-SUBMIT entry, or leaderboard upload. The stopped log and its exact last 50 lines are saved as `run_full_user_stopped_20260927_160355.log` and `run_full_user_stopped_20260927_160355.last50.txt`.

Before stopping, peak process-tree RSS was 18.02 GB, peak system use 31.93 GB, and the last GPU reading was 46°C. CPU-package temperature could not be read from a supported sensor. No replacement run was launched; an E5 model/retrain/inference stack cannot complete the remaining work within the measured deadline and lacks its required installed/cached model stack.
