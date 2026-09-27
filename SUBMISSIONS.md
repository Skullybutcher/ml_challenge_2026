# Submissions log (Akari-maintained)

| # | Time (IST) | Run / config | Threshold | OOF F0.5 | Validator | LB score | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Policy: Aman uploads — Akari stages READY-TO-SUBMIT rows only (see NIGHT_HANDOFF.md). A later run beating the staged OOF by ≥ 0.003 supersedes it; keep both files.
## E4 source/singleton analysis — pending

Run 1's global baseline is OOF macro F0.5 **0.9825** with the current global plateau choice **0.72**. The original Run 1 output did not save raw OOF probabilities or the full 37-point threshold grid, so source-specific (S2/S3) and singleton/non-singleton F0.5 values are not computable from the available labels and pair IDs. Commit `0d4575d` exports the 5,509,855 aligned labels and `(s1_id, other_id)` pairs from the saved feature checkpoints. After Run 2 completes, regenerate Run 1 raw OOF probabilities from those checkpoints, calculate the full curve and requested splits, and record whether any split gains at least 0.001. No split threshold is adopted yet.
## Run 1 plateau selection (analysis only; no submission staged)

Run 1's best grid point was threshold 0.625 at OOF macro F0.5 0.9825. Its displayed top-eight curve is recorded in `status.md`. The highest displayed threshold within 0.001 of best is 0.72 (0.9822, 0.0003 below best). The full OOF curve was not saved, so this is the highest qualifying point visible in the published top eight. Run 2 applies the requested 0.001 plateau rule to its own full OOF grid, since its rare-channel band changes the training candidate distribution. No READY-TO-SUBMIT row is added until Run 2 passes the validator and exact row-count check.
