# Playbook — Akari (autonomous work during + after the runs)

You have Aman’s go to work independently within THIS file’s bounds. Anything outside it: stop and ask Aman. Line refs are for commit `584309e`; use function names if lines drifted.

## 0. Resources + standing rules

- Machine: 64GB RAM. RAM budget rule: combined jobs must stay under 48GB. The full run holds ~10GB → you may run ONE extra job ≤ 8GB alongside it (i.e., samples ≤ 50k only). Nothing bigger during Run 2.
- Branches: `feature/chunked-fullrun` is FROZEN (the validated config). All experiments on `exp/<name>` branches. Never commit to the frozen branch, never rewrite its history.
- NEVER submit to the leaderboard portal. Aman owns the 5/day budget. You report numbers; he decides submissions.
- Every experiment ends with a numbers report to Aman, even on failure (especially on failure: last 20 log lines verbatim).

## 1. Codebase map (15-second orientation)

- `code/business_entity_resolution/src/pipeline.py` — `run()` stages: 1 load → 2 normalize → 3 train blocking → 4 chunked featurize+subsample → 5 GroupKFold LightGBM (`train_oof`) → 6 isotonic calibrate → 7 threshold sweep → 8 chunked test inference (streams `candidate_pairs.tsv`) → 9 write `matching_results.tsv`.
- `src/blocking.py` — `generate_candidates_token` (shared-word join, `max_df` cap), `generate_candidates_prefix` (country+4-letter buckets, per-bucket product cap), `generate_candidates` (TF-IDF + rare-token, currently UNUSED in default path), `union_candidates`, `candidate_recall`.
- `src/features.py` — 15 pairwise features (`FEATURE_NAMES`). `src/model.py` — `train_oof`, `calibrate_oof`, `tune_threshold` (prints top-8 curve). `src/metric.py` — macro F0.5 (frozen, do not touch). `src/normalize.py` — cleanup + accent fold. `src/io_utils.py` — output writers (LF-forced, frozen).

## 2. Numbers to beat (verified on 16GB box, your machine should match or better)

| sample | recall | pairs pre→post | OOF F0.5 | time | peak |
|---|---|---|---|---|---|
| 5k | 0.9920 | 2.65M→272k | 0.9856 | ~3m | 5.1GB |
| 20k | 0.9773 | 5.48M→1.05M | 0.9759 | ~4m | 5.2GB |
| 50k | 0.9774 | 35.7M→2.70M | 0.9773 | ~22m | 7.7GB |

LB top to beat: 0.9805. OOF is mildly optimistic (~0.001–0.003: calibrator + threshold both fit on OOF).

## 3. Failure playbook (diagnose → fix → report; do not improvise outside this table)

| Symptom | Likely cause | Approved response |
|---|---|---|
| Segfault / OpenBLAS alloc fail at startup | Machine RAM pressure (needs 10GB+ free even before data loads) | Close everything else, confirm free RAM, rerun. NOT a code bug. |
| `Train candidate recall < 0.90` gate | Blocking caps too tight for the sample | Report + stop. Do not train on <0.90 recall. Try `--max-df 500` once as a diagnostic. |
| Any test chunk logs `pairs=` > 50M | Merge explosion vs 10M pool | STOP the run, report. Approved fix (only with Aman's go): `--max-pairs-per-prefix-key 200000→50000` for the test stage. |
| Validator FAIL | Read the numbered issue; usual suspects: dup S1 rows, header typo, missing S1 (must be exactly 1,732,544 data rows), S1- IDs in match lists | Fix outputs, never hand-edit TSVs silently — report the issue first. CRLF corruption is already fixed; if `\r` reappears, the writer regressed — report, don't patch. |
| OOF < 0.970 at 100k | Something regressed vs the table above | Stop. Send full log + the threshold curve. Do not launch Run 2. |
| Slow (2× the table pace) | Contended CPU/disk | Check for parallel jobs; test-chunk-size stays 100k unless Aman says otherwise. |

## 4. Experiment backlog (ranked; run only in gaps: alongside Run 2 if ≤50k, otherwise after)

**E1 — rare-token channel in `block()` (biggest recall lever, +0.005–0.015 hoped).**
Union `generate_candidates` (blocking.py, TF-IDF + rare-token path) into `block()` alongside token+prefix. Re-run 50k `--skip-test`. Adopt for any future full run ONLY if recall ≥ 0.982 with pairs ≤ 60M and time ≤ 45min. If it OOMs or recall barely moves, kill it and report — negative result still counts.

**E2 — cap sweep at 50k: `--max-df 300→500`, `--prefix-len 4→5` (one flag at a time).**
Report recall / pairs / time deltas. Adopt if recall gain ≥ 0.003 per ≤50% pair growth.

**E3 — `--neg-per-pos-cap 15→8` at 50k.**
Report OOF delta + train-time saved. Adopt if OOF within 0.001 (pure speedup for future trains).

**E4 — threshold analysis (no rerun needed, uses Run 1 OOF).**
From the top-8 curve: compare argmax vs plateau-start; split OOF F0.5 by S2-matches vs S3-matches (source-specific thresholds?) and by singleton vs non-singleton. Report the table; Aman picks.

**E5 — France probe (report only, no modeling).**
From Run 2 logs/outputs: candidates-per-S1 for FR vs US vs IN; match-rate per country. Question: does France get systematically fewer/weaker candidates? One paragraph + 3 numbers.

**E6 — LightGBM micro-grid (only if 100k OOF < 0.975).**
`num_leaves 31→63`, `learning_rate 0.05→0.03`, one at a time at 50k. Adopt if OOF +0.002.

**BANNED (do not start):** ensembles, cross-encoders/neural rerankers, dense retrieval, training the matcher on full 2.2M (≈120M rows), per-S1 pair caps (proven: even 500 costs −0.072 recall), portal submissions, touching `metric.py` / writers / GroupKFold grouping.

## 5. Report format (every task ends with this)

```
Task: <E# or Run#>
Command: <exact>
Result table: recall / recall_US / recall_IN / pairs pre→post / OOF / best_t + top-8 curve / wall / peak GB
vs baseline: <delta vs table in §2>
Verdict: <adopt / reject / needs-Aman-decision> + one line why
```

Run 1's report additionally gates Run 2 — end it with an explicit GO / NO-GO recommendation against the §2 table.
