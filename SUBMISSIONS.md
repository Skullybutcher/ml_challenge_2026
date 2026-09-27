# Submissions log (Akari-maintained)

| # | Time (IST) | Run / config | Threshold | OOF F0.5 | Validator | LB score | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Policy: Aman uploads — Akari stages READY-TO-SUBMIT rows only (see NIGHT_HANDOFF.md). A later run beating the staged OOF by ≥ 0.003 supersedes it; keep both files.
## E4 source/singleton analysis — pending

Run 1's global baseline is OOF macro F0.5 **0.9825** with the current global plateau choice **0.72**. The original Run 1 output did not save raw OOF probabilities or the full 37-point threshold grid, so source-specific (S2/S3) and singleton/non-singleton F0.5 values are not computable from the available labels and pair IDs. Commit `0d4575d` exports the 5,509,855 aligned labels and `(s1_id, other_id)` pairs from the saved feature checkpoints. After Run 2 completes, regenerate Run 1 raw OOF probabilities from those checkpoints, calculate the full curve and requested splits, and record whether any split gains at least 0.001. No split threshold is adopted yet.
## Run 2 OOF decision (not READY-TO-SUBMIT)

Candidate recall is **0.9752** overall (US **0.9911**, n=60,078; India n=0 excluded). Run 2 OOF macro F0.5 is **0.9761**; this passes the 0.970 run gate but is below the 0.988 threshold in the current submission decision rule and below Run 1's 0.9825. The printed top-eight curve is `0.65:0.9771, 0.62:0.9771, 0.60:0.9771, 0.68:0.9771, 0.70:0.9770, 0.57:0.9770, 0.55:0.9770, 0.72:0.9768`; the wrapper selected threshold **0.800** with score **0.9761**. Since Run 2 OOF <0.988, the submission threshold depends on E4: adopt an E4 split only if it gains ≥0.001; otherwise use Run 1's **0.72**. E4 is pending because Run 1 raw OOF probabilities/full curve were not saved. Run 2's test inference currently uses 0.800, so its outputs remain provisional until the threshold is resolved. No validation or READY-TO-SUBMIT row has been recorded yet.
## Run 1 plateau selection (analysis only; no submission staged)

Run 1's best grid point was threshold 0.625 at OOF macro F0.5 0.9825. Its displayed top-eight curve is recorded in `status.md`. The highest displayed threshold within 0.001 of best is 0.72 (0.9822, 0.0003 below best). The full OOF curve was not saved, so this is the highest qualifying point visible in the published top eight. Run 2 applies the requested 0.001 plateau rule to its own full OOF grid, since its rare-channel band changes the training candidate distribution. No READY-TO-SUBMIT row is added until Run 2 passes the validator and exact row-count check.

## Deadline score note (2026-09-27)

Aman reports the current leaderboard top as **0.9918**. Run 2's OOF macro F0.5 is **0.9761**, which is a cross-validation metric and not a verified test or leaderboard score. Test inference is incomplete (chunks 1-3 of 278 complete at the latest check); there is no validator PASS, exact 1,732,544-row output, READY-TO-SUBMIT entry, or leaderboard upload. The proposed 8-worker shortcut was not used: its supplied script lacks model loading, worker dispatch/output merging, and exact-output parity, conflicts with the six-worker/48 GB experiment limits, and would restart inference from chunk 1 if it replaced the current writer.
