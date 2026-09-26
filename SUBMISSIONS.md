# Submissions log (Akari-maintained)

| # | Time (IST) | Run / config | Threshold | OOF F0.5 | Validator | LB score | Notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Policy: Aman uploads — Akari stages READY-TO-SUBMIT rows only (see NIGHT_HANDOFF.md). A later run beating the staged OOF by ≥ 0.003 supersedes it; keep both files.
## Run 1 plateau selection (analysis only; no submission staged)

Run 1's best grid point was threshold 0.625 at OOF macro F0.5 0.9825. Its displayed top-eight curve is recorded in `status.md`. The highest displayed threshold within 0.001 of best is 0.72 (0.9822, 0.0003 below best). The full OOF curve was not saved, so this is the highest qualifying point visible in the published top eight. Run 2 applies the requested 0.001 plateau rule to its own full OOF grid, since its rare-channel band changes the training candidate distribution. No READY-TO-SUBMIT row is added until Run 2 passes the validator and exact row-count check.
