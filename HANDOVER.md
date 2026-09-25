# Handover — Claude → OpenCode (2026-09-25)

## What Claude did
1.压缩了 68-section master plan into Day1-3 phased plan (Kaggle, 5 subs/day, LB top 0.980473).
2. Built v1 pipeline (`audit, metric, io_utils, normalize, blocking, features, model, pipeline` + README/requirements), smoke-tested on 200-entity synthetic: OOF F0.5 0.82 vs 0.37 singleton floor, outputs structurally valid.
3. User ran `audit.py` on real data — key result: only 5.58% singletons (dense matching, not abstain-heavy), avg degree ~3-4, 80.5% entities match both S2+S3.
4. Claude concluded v1 cannot scale and started v2 rewrite (token-index blocking via merges, prefix channel, vectorized pair frames, negative subsampling, `--sample-s1`), but hit OOM at 80k synthetic (token `max_df` + prefix pair-product blowup), was still debugging when user asked for files + `handover.md`. That handover was never delivered — this file fills the gap.

## Local state gap (verified)
Local `code/business_entity_resolution/src/` is still v1: `blocking.py:24-35,54-86` brute-force TF-IDF NN, `pipeline.py:39-56` per-pair `.loc`, `features.py:69-84` dict-loop only, no `--sample-s1`/`--max-df`/subsampling. v2 code exists only in Claude's unsent edits. `output/` empty.

## Immediate plan
Follow `TODO.md` steps 1-7. First unblock: port v2 blocking + pair-frame + subsampling, then sampled recall test. Full details + audit numbers in `TODO.md`.

## Open questions for user
- Kaggle vs local: where should the sampled (`--sample-s1 30000`) trial run — here or Kaggle?
- Confirm test S1 row count for submission sizing (validator: `utils/validate_submission.py`).
