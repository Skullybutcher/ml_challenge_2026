# TODO — Amazon ML Challenge 2026 (Entity Resolution)

Source: `claude_desktop_data_mlChallenge.json` (10 msgs) + local audit output.

Real data (from `audit.py`): S1=2206821, S2=5034616, S3=5285603. Singleton rate 0.0558, all-singleton F0.5 floor 0.0558. Degree peaks at 3-4. S2-only 143029, S3-only 164498, both 1776047. S2/S3 multi-owned = 0.

## Status
- [x] v1 pipeline exists (`code/business_entity_resolution/src/`, 8 files) — tested only on 200-entity synthetic
- [x] `audit.py` run on real data — numbers above
- [x] v2 scale rewrite ported locally (token/prefix blocking, vectorized pairs, neg subsampling, `--sample-s1`)
- [ ] Blocking recall benchmark on train sample running locally (`bench_blocking.py`)
- [x] Blocking recall 0.9920 @ 5k sample, max_df 300 (28s block, 885s load) — proceed to matcher
- [ ] Sampled matcher train (`bench_train.py`) — timed out locally at 120s foreground; run on Kaggle
- [ ] Kaggle notebook ready (`kaggle_run.ipynb`) — user to upload + attach dataset
- [ ] No full pipeline run yet (correct — wait for recall >= 0.90)
- [ ] `output/` empty, no submission yet

## Phase split (CPU here vs Kaggle)
- Here (CPU, no GPU): audit (done), blocking recall tuning, feature/GBDT train on `--sample-s1`, calibration + threshold tune, validator
- Kaggle (scale + optional GPU): full-data blocking/inference, LightGBM full train if local RAM/time insufficient, cross-encoder reranker only (<=8B, MIT/Apache)

## Next (in order, small steps)
- [ ] 1. `git init` + `git checkout -b feature/scale-blocking` (no git repo currently)
- [ ] 2. Port v2 `blocking.py`: `generate_candidates_token` (capped merge, `max_df`), `generate_candidates_prefix` (pair-product cap `count_s1*count_other <= max_pairs_per_prefix_key`), keep TF-IDF path only for small samples
- [ ] 3. Port v2 `pipeline.py`: `build_pair_frame` (merges, not `.loc` loop), `subsample_negatives`, `--sample-s1`, `--max-df`, `--max-pairs-per-prefix-key`, `--neg-per-pos-cap`
- [ ] 4. Port v2 `features.py`: `build_feature_frame_vectorized` + empty-frame guard
- [ ] 5. Smoke test: small synthetic, then `--sample-s1 30000` on real data; record blocking recall + wall time; require recall >= 0.90 before training
- [ ] 6. Full-train: GroupKFold LightGBM + isotonic + threshold tune on macro F0.5; run `utils/validate_submission.py` before any upload
- [ ] 7. Submit once; only then consider hard-negatives / reranker / France slice checks

## Do not do
- Do not run v1 `pipeline.py` on full data (`blocking.py:69-70` brute-force NN, `pipeline.py:39-56` per-row `.loc` — infeasible at 2.2M x 10M).
- Do not tune threshold on AUC/accuracy; metric is macro F0.5 (`metric.py:13-28`).
- Do not hard-code country; test has France, train does not.
