# ml_challenge_2026

Amazon ML Challenge 2026 — business entity resolution (Source 1 → Sources 2/3, macro F0.5).

- `code/` — pipeline submodule ([code_ber](https://github.com/Skullybutcher/code_ber)): normalize → token/prefix blocking → LightGBM matcher → calibrate → threshold-tune → `matching_results.tsv` + `candidate_pairs.tsv`.
- `utils/` — official `validate_submission.py`.
- `kaggle_run.ipynb` — Kaggle runbook (audit → blocking sample → sampled train → full run → validate).
- `TODO.md` / `HANDOVER.md` / `RUN_STATUS.md` — plan, Claude handover, live run status.
- `dataset/` and `output/` are git-ignored (large TSVs live on Kaggle / locally only).
