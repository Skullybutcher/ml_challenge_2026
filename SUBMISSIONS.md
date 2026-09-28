# Submissions log (Akari-maintained)

| # | Time (IST) | Run / config | Threshold | OOF F0.5 | Validator | LB score | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Policy: Aman uploads — Akari stages READY-TO-SUBMIT rows only (see NIGHT_HANDOFF.md). A later run beating the staged OOF by ≥ 0.003 supersedes it; keep both files.

## Latest status — 2026-09-28 (research results; no submission staged)

- The matched 50k CPU-only OOF comparison completed successfully for baseline and E5, with 20/20 training chunks, empty stderr, and exit code 0. Both used the same sample, folds, rare-channel settings, and skipped test inference.
- Baseline: candidate recall **0.9857 overall / 0.9983 US** (US n=30,157); **84,765,735** pairs before subsampling, **2,753,208** retained; OOF macro F0.5 **0.9839**, best threshold **0.675**. Top-eight curve: `0.70:0.9839, 0.68:0.9839, 0.72:0.9839, 0.65:0.9839, 0.62:0.9839, 0.60:0.9838, 0.57:0.9838, 0.75:0.9837`. Wall time ~1h30; peak tree RSS 11.64 GB / system RAM 25.99 GB.
- E5: same blocking recall and pair counts; OOF macro F0.5 **0.9854**, best threshold **0.675**, paired delta **+0.0015**. Top-eight curve: `0.72:0.9854, 0.70:0.9854, 0.68:0.9854, 0.65:0.9854, 0.62:0.9854, 0.75:0.9853, 0.60:0.9853, 0.57:0.9853`. Wall time 1h52m46s; peak tree RSS 17.05 GB / system RAM 31.31 GB. These are OOF results, not test or leaderboard scores.
- The matching-only heuristic artifact is stored in Git as ordered gzip LFS parts with a restore utility; its README records passing formatting, candidate-containment, and ID-existence checks for exactly **1,732,544 rows** and **9,969,589 valid S2/S3 IDs**. It is unscored. `candidate_pairs.tsv` remains local. This is separate from Run 2 model inference, which Aman stopped after 4/278 chunks; its partial output and 41 checkpoints remain preserved locally.
- No READY-TO-SUBMIT row is staged and no leaderboard upload occurred. E5 vectors, weights, checkpoints, and runtime logs remain outside Git. Full metric details are in `status.md`.

## E4 source/singleton analysis — pending

Run 1's global baseline is OOF macro F0.5 **0.9825** with the current global plateau choice **0.72**. The original Run 1 output did not save raw OOF probabilities or the full 37-point threshold grid, so source-specific (S2/S3) and singleton/non-singleton F0.5 values are not computable from the available labels and pair IDs. Commit `0d4575d` exports the 5,509,855 aligned labels and `(s1_id, other_id)` pairs from the saved feature checkpoints. After Run 2 completes, regenerate Run 1 raw OOF probabilities from those checkpoints, calculate the full curve and requested splits, and record whether any split gains at least 0.001. No split threshold is adopted yet.
## Run 2 OOF decision (not READY-TO-SUBMIT)

Candidate recall is **0.9752** overall (US **0.9911**, n=60,078; India n=0 excluded). Run 2 OOF macro F0.5 is **0.9761**; this passes the 0.970 run gate but is below the 0.988 threshold in the current submission decision rule and below Run 1's 0.9825. The printed top-eight curve is `0.65:0.9771, 0.62:0.9771, 0.60:0.9771, 0.68:0.9771, 0.70:0.9770, 0.57:0.9770, 0.55:0.9770, 0.72:0.9768`; the wrapper selected threshold **0.800** with score **0.9761**. Since Run 2 OOF <0.988, the submission threshold depends on E4: adopt an E4 split only if it gains ≥0.001; otherwise use Run 1's **0.72**. E4 is pending because Run 1 raw OOF probabilities/full curve were not saved. Run 2's test inference currently uses 0.800, so its outputs remain provisional until the threshold is resolved. No validation or READY-TO-SUBMIT row has been recorded yet.
## Run 1 plateau selection (analysis only; no submission staged)

Run 1's best grid point was threshold 0.625 at OOF macro F0.5 0.9825. Its displayed top-eight curve is recorded in `status.md`. The highest displayed threshold within 0.001 of best is 0.72 (0.9822, 0.0003 below best). The full OOF curve was not saved, so this is the highest qualifying point visible in the published top eight. Run 2 applies the requested 0.001 plateau rule to its own full OOF grid, since its rare-channel band changes the training candidate distribution. No READY-TO-SUBMIT row is added until Run 2 passes the validator and exact row-count check.

## Deadline score note (2026-09-27)

Aman reports the current leaderboard top as **0.9918**. Run 2's OOF macro F0.5 is **0.9761**, which is a cross-validation metric and not a verified test or leaderboard score. Test inference is incomplete (chunks 1-3 of 278 complete at the latest check); there is no validator PASS, exact 1,732,544-row output, READY-TO-SUBMIT entry, or leaderboard upload. The proposed 8-worker shortcut was not used: its supplied script lacks model loading, worker dispatch/output merging, and exact-output parity, conflicts with the six-worker/48 GB experiment limits, and would restart inference from chunk 1 if it replaced the current writer.

## France threshold proposal (not adopted)

A France-specific 0.95 threshold was proposed, with an asserted singleton-rate rationale and score gain. The current artifacts do not contain Run 2 OOF probabilities, serialized models, or test candidate probabilities needed to validate that claim; `candidate_pairs.tsv` contains candidate IDs only. The suggested code is not a drop-in script, and it uses 0.72 for non-France S1s rather than Run 2's selected 0.800. No France threshold or heuristic post-processing has been applied. Run 2 inference remains on its frozen 0.800 threshold; no submission decision is changed without measured OOF evidence.

## E5 proposal review (2026-09-27)

The proposed E5 embedding/retrain/parallel-inference path was not adopted. The current environment lacks PyTorch, Transformers, and cached E5 weights; the specified train+test name/address vectors require about 99 GB at fp16 before temporary space, while C: has about 96 GB free. The new features would invalidate the current training checkpoint fingerprint, and the proposed parallel inference program has not been implemented or parity-checked. The live inference chunks have measured about 35.6 minutes each, so the plan's two-minute estimate is not supported. No new threshold or feature is staged; the only valid submission decision remains pending complete outputs and validator PASS.

## Run 2 stopped before validation (2026-09-27)

At Aman's explicit direction, Run 2 inference was stopped after four of 278 test chunks because the measured pace could not finish before the deadline. The incomplete candidate TSV is preserved locally as `out_full/candidate_pairs_partial_user_stopped_20260927_160355.tsv`; all 41 training checkpoints remain. Exit code `-1` reflects the requested process termination, not a model or pipeline result. No `matching_results.tsv` or validator PASS exists, so no READY-TO-SUBMIT variant is staged and no upload was made.
