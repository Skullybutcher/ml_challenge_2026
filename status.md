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
