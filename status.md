# Status — Akari business entity resolution

**Snapshot:** 2026-09-26 09:58 Asia/Kolkata
**Repository:** https://github.com/Skullybutcher/ml_challenge_2026
**Active branch:** `exp/run1-2500-a467-20260926`
**Base commit:** `a467bb7c79417e084b7e2fbd873191581c6d5a5b`
**Goal status:** Run 1 is active; the Run 2 gates are not yet complete.

This file records the work and instructions received since the initial task prompt. It is a point-in-time log: refresh the live process, log tail, and branch state before acting on it.

## Current Run 1

The active process is PID 18208, started at approximately 08:35 local time. It is using the isolated `venv_pd2` environment with Pandas 2.3.3 and writing to `out_100k` and `run_100k.log` in this repository worktree. The run uses a 2,500-S1 chunk after earlier runs exceeded the 40 GB checkpoint; the process has stayed near 21 GB RSS so far. A monitor is active and will stop the process if sampled RSS reaches 47 GB, preserving the log tail.

Command:

~~~powershell
& ..\venv_pd2\Scripts\python.exe code\business_entity_resolution\src\pipeline.py --data-dir $DATA --out-dir out_100k --sample-s1 100000 --skip-test --n-splits 5 --train-chunk-size 2500 --use-rare > run_100k.log 2>&1
~~~

`$DATA` was set once to the extracted dataset root, `..\dataset`, containing `train/` and `test/`. The output drive had 91.9 GB free at the last check (09:55).

Run 1 has completed candidate generation and is in train featurization. Last log evidence at 09:55:23:

- Candidate pair union: 311,199,888.
- Overall candidate recall: 0.9834.
- US recall: 0.9966 (n=60,078).
- India line: 1.0000 (n=0); this is not a meaningful India estimate because the sampled run has no labeled India examples.
- Per-S1 candidate count p50=2,813; p99=9,252; max=20,383; mean=3,112.0.
- Train chunks: 7/40 complete. The seven chunks each processed about 7.7–7.9 million raw pairs and took about 10.5 minutes apiece.
- OOF, threshold curve, peak-RSS final value, and completion time are not available yet.

The active log is [run_100k.log](run_100k.log). A prior 2,500-chunk attempt under Pandas 3 was manually stopped at 48.48 GB RSS during blocking and preserved as [run_100k_attempt2500_pandas3_stopped.log](run_100k_attempt2500_pandas3_stopped.log).

## Instructions and handoff history

The initial prompt asked to pull `feature/chunked-fullrun` at `db1cfe1`, read `NIGHT_HANDOFF.md`, `RUNBOOK_AKARI.md`, `AKARI_PLAYBOOK.md`, then the context gates, France notes, error taxonomy, and cap-sweep evidence in that order. It specified dataset extraction under `<DATA>/train/` and `<DATA>/test/`, dependency setup from `code/business_entity_resolution/requirements.txt`, at least 40 GB free for outputs, and a 100k sample Run 1 with `--skip-test`, five folds, and 5,000-S1 chunks. It required per-country recall review and an exact 1,732,544-row submission validation.

The next handoff corrected the base to `ccfa813` and made `--use-rare` required. It reported a separate 50k rare-channel result of recall 0.9892 / OOF 0.9858 and requested the 100k rare-channel Run 1. User-reported comparison values were 100k without rare: recall 0.9611 / OOF 0.9680 (NO-GO). Debugging was later explicitly authorized.

The debugging instruction asked to keep both fixes on an `exp/*` branch: (a) avoid creating counted-token frames when no per-S1 cap is enabled, delete `counted_s2` / `counted_s3` after the cap block when applicable, and run `gc.collect()`; (b) convert feature rows in bounded batches into a preallocated NumPy array. It directed a rerun at the normal chunk size and a reduction to 2,500 if peak remained above 40 GB.

The latest handoff superseded the requested source commit with `a467bb7`, confirmed Aman’s machine was off, and set these decisions:

1. Run 2 uses `--rare-max-df 1000`; a measured 50k run was reported at recall 0.9857 with 38% fewer pairs, with projected runtime about 22 hours versus 36 hours for the default band.
2. The highest-value experiment is RapidFuzz batched pairwise `process.cpdist`. Validate exact feature-output equality on 5k first; only adopt if exact.
3. Run 1 proceeds unchanged, with Run 2 launched only when the four pre-auth gates in `NIGHT_HANDOFF.md` hold.

The docs were read in the requested order on the successive handoffs. The latest handoff’s exact Run 2 band, gates, plateau rule, failure table, experiment rules, and submission restrictions remain authoritative.

## Branch and code state

The frozen branch `feature/chunked-fullrun` has not been modified. Work is on `exp/run1-2500-a467-20260926`, based on commit `a467bb7`. The commit added the rare-band flags / handoff decisions. The working tree also contains the two requested local fixes:

- `features.py`: retains scalar feature semantics and ordering while converting at most 100,000 rows at a time to float32, with a shape check for each batch.
- `pipeline.py`: skips the shared-token count-frame path on the uncapped run and releases optional count frames after the cap block.

The two source fixes and current run logs remain uncommitted. The status document itself was committed separately as 577120b. The push attempt failed with HTTP 403 for the configured Git identity; no remote exp branch was created.

A code review of the fetched `a467bb7` source found the rare-band flags there. The counted-frame release used by this run is present in the local working-tree fix; it should not be described as already committed unless a fresh `git show` confirms that. The active pipeline has the fix loaded.

## Earlier attempts and evidence

- Initial 100k attempts 1–3 using the bundled/repo Python environment exited before producing metrics. The preserved logs include Windows process errors `-1073741819` and `-1073740791`.
- The earlier 100k attempt 4 without the rare channel reached blocking and was stopped after observed RSS reached 49.50 GB, over the 48 GB cap; it produced no recall/OOF. Its log is `run_100k_attempt4_no_rare.log` in the sibling original worktree.
- Later debugging exposed a return-contract error in a trial fix and an Arrow-backed Pandas 3 `dtype=str` allocation failure. An isolated environment with Pandas 2.3.3 was created; the active run uses it.
- The failed Pandas 3 retry at 2,500 chunks reached 48.48 GB during the S1-vs-S3 blocking stage and was stopped before recall.
- E1, 50k with rare tokens: recall 0.9892, US 0.9989 (n=30,157), India n=0, 137,937,334 candidate pairs. It failed during feature-array creation before pair subsampling or OOF; about 33.77 GB RSS was observed before the crash. Reject for full-run adoption because the playbook’s candidate-pair bar is 60M.
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

1. Let the active Run 1 finish under the 48 GB total cap. Preserve the final log tail verbatim if it crashes and apply the matching `AKARI_PLAYBOOK.md` failure-table response. Report candidate recall, per-country counts/recall, pre/post-subsample pairs, OOF macro F0.5, top-8 threshold curve, selected plateau threshold, wall time, and final sampled/pipeline peak RSS against the baseline table.
2. Do not launch Run 2 until all four `NIGHT_HANDOFF.md` gates are satisfied: overall recall ≥0.95; India recall reported (this sample’s n=0 must remain explicitly caveated); OOF ≥0.970; and the top-8 threshold curve/plateau evidence available. Also preserve the handoff’s code-change gate and first-chunk >50M-pair tripwire.
3. Before Run 2, run the exact-output comparison on 5k for the batched RapidFuzz `process.cpdist` design. `process.cdist` is an all-pairs matrix and is not the aligned-pair operation. Compare IDs and all final feature values exactly after the intended float32 cast; record timing. Adopt only if equality holds.
4. If gates hold, Run 2 uses the 100k command with test inference enabled, `--use-rare --rare-max-df 1000`, and output directory `out_full`. Recheck free space and use the handoff/runbook values for remaining flags/chunk size. Stop immediately if the first full-run chunk crosses the >50M raw-pair tripwire.
5. Validate with `python utils/validate_submission.py --matching out_full/matching_results.tsv --candidate out_full/candidate_pairs.tsv --test-dir $DATA/test` and add `--check-ids` if feasible. Matching must have exactly 1,732,544 rows.
6. Apply the documented threshold plateau rule and stage READY-TO-SUBMIT rows in `SUBMISSIONS.md`. Never upload to the leaderboard; Aman alone submits.
7. Work E1–E6 only within their playbook rules, with at most one ≤50k job alongside Run 2 and a 48 GB combined memory cap. New E7 designs must meet the same guardrails. Do not use ensembles, neural models, dense retrieval, full-2.2M training, per-S1 caps, or changes to metrics, writer, or CV.

No Run 2, submission validation, threshold staging, leaderboard upload, or completed 5k `cpdist` equality experiment has occurred yet.

## Status document publishing

The local status document commit is 577120b28719f257612c4e17d8e5d04ffbb09c2f. Publishing the exp branch was attempted with git push -u origin exp/run1-2500-a467-20260926. GitHub rejected it with HTTP 403: permission denied to Stakeylock. The branch has no upstream configured and no changes were pushed. The local commit and status.md are ready; publication requires a GitHub identity with write access to this repository.
