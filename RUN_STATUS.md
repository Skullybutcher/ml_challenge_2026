# Run status — Amazon ML Challenge 2026 (2026-09-25)

## Execution method (as used)
- Kaggle session started in browser; VSCode connected via the Kaggle VSCode-compatible link (existing-server kernel).
- Work done in the notebook (`kaggle_tun.ipynb`); cells executed one by one from VSCode.
- Code via `!git clone https://github.com/Skullybutcher/code_ber.git` (+ `git pull` for `bench_blocking.py` / `bench_train.py`).
- Accelerator: none/CPU (GPU discussed, rejected for Phases 1-4 — pipeline is CPU-only).
- The downloaded `notebook8590409bbd.ipynb` contains only setup cells (clone/ls/pull); phase outputs below are from chat-reported runs.

## Status by phase
| Phase | State | Measured |
|---|---|---|
| 1 audit | PASS | S1 2206821 / S2 5034616 / S3 5285603; singleton 5.58%; floor F0.5 0.0558 |
| 2 blocking sample (5k, max-df 300) | PASS | load ~148-183s; block ~4-6s; pairs 2,653,273; recall 0.9920 |
| 3 sampled train (5k, 3-fold) | PASS | pairs 2.65M → train 272,145 (pos 6.31%); OOF F0.5 0.9844-0.9847 @ t 0.55-0.65; baseline 0.0554 |
| 4 pipeline (`--sample-s1 50000 --n-splits 3`) | STOPPED by user (RAM 100% in featurize) | 31+ min in; CPU 100%+, RAM 100%; last log: building pairwise feature frame |
| 4b pipeline (`--sample-s1 20000 --max-df 100 --n-splits 3`) | TRAIN DONE, test inference died | train recall 0.9773, OOF 0.9766 @ t=0.65; test S1=1732544/S2=4887273/S3=5082316; log ends at test blocking — no TSVs written (writes happen at end) |
| 4c pipeline (same + `--test-chunk-size 100000`) | READY TO RUN | chunked test inference bounds peak RAM; chunk-equivalence verified |
| 5 validate/submit | NOT STARTED | — |

## Estimate to complete
- Phase 4b @20k/max-df 100: ~1-2h total (recall expected ~0.97-0.99 given 0.992 headroom at 5k).
- Phase 5 validator (default, no `--check-ids`): minutes.
- Full 2.2M run: not attempted; ~1B pairs — requires chunked test inference, 8-16h+. Do only after 50k passes.

## Score outlook
- Board top: 0.980473. Sample OOF 0.9847 already above it (sample optimism applies).
- Realistic: 50k model 0.975-0.985 OOF → public LB in that band if blocking recall holds and threshold (~0.55-0.65) is retuned per run.
- Timeline: 50k pass + validate today; first submission from 50k model; full/chunked run only if LB trails OOF.

## Next action
If Phase 4 dies: stop, rerun the 20k command from chat. If it passes: validator → submit → compare LB vs OOF before any GPU reranker talk.
