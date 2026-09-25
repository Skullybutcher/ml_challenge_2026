# Night handoff — state of play at takeover (2026-09-25 ~23:30)

## Where we are
Pipeline verified at 5k/20k/50k on a 16GB box. Nothing left to prove locally — the work moves to Jini's 64GB PC tonight: Run 1 (100k validation) → conditional Run 2 (full 1.73M test, ~12h) → experiments in the gaps. Aman sleeps; Akari owns the night. Teammate (original author) is out.

## Verified numbers (trust these; re-verify only if code changes)
| sample | recall | pairs pre→post | OOF F0.5 | time | peak |
|---|---|---|---|---|---|
| 5k | 0.9920 | 2.65M→272k | 0.9856 | ~3m | 5.1GB |
| 20k | 0.9773 | 5.48M→1.05M | 0.9759 | ~4m | 5.2GB |
| 50k | 0.9774 | 35.7M→2.70M | 0.9773 | ~22m | 7.7GB |
LB top to beat: 0.9805. Threshold keeps landing ~0.55-0.65. OOF mildly optimistic (~0.002).

## Decisions + why (do not relitigate without new evidence)
1. **Chunked training, caps OFF.** 50k/20k OOMs were structural (string-heavy pair frames + full-copy subsample), fixed by per-5k-S1 chunking. A per-S1 pair-cap sweep then FAILED: even cap500 cost −0.072 recall because 63-88% of S1 sit at-cap — true matches (transliterations, word-swaps, Hindi-script names per our 57-pair human taxonomy) share ≤1 counted token and rank past any cap. No count-based cap is safe.
2. **Accent fold + char map** (`normalize.py`): café→cafe; explicit œ→oe, æ→ae, ß→ss, ø→o, đ→d, ł→l, ð→d, þ→th (plain ASCII-strip DELETES these — verified). Justified by in-train accent variants, not just France.
3. **France verdict A:** 20-record human eyeball — no special handling, single global threshold. Only residual risk: France gets weaker features (noisier); threshold choice absorbs it.
4. **Threshold:** pick plateau-start from the top-8 curve, not argmax (calibration noise). Aman decides in the morning from Run 1's curve — report it, don't pick it.
5. **Known unknowns, ranked:** (a) India recall never measured in isolation (47% of test; 50k samples were US-heavy by seed luck) — Run 1's per-country logging covers this, read it first; (b) first-test-chunk merge size vs 10M pool (tripwire: >50M pairs in any chunk → stop); (c) OOF sits ~0.003 below LB top — the remaining edge is threshold + E1, not blocking.
6. **Deferred, not dropped:** E1 rare-token channel (best recall lever, needs 50k re-measure first). Everything else (ensembles, neural, dense, full-2.2M training) is banned on this clock.

## Akari's ordered night
1. Setup per RUNBOOK (disk precheck ≥40GB, deps, dataset from Aman).
2. Run 1 (100k, ~1h) → report with GO/NO-GO vs the table above. Read India recall first.
3. Run 2 ONLY if pre-auth gates pass (below). ~12h. Watch chunk 1 pairs.
4. During Run 2 (one ≤50k job allowed): E1 first, then E2/E3.
5. Morning report for Aman: Run 1 numbers + curve, Run 2 status/progress, experiment results, validator state, exact next decision needed (threshold pick).

## Run 2 pre-authorization (Aman-approved in advance, all must hold)
- Run 1 recall ≥ 0.95 overall AND India recall measured (any value, but reported — if IN recall < 0.90, flag prominently and still proceed, it informs threshold, not blocking).
- Run 1 OOF ≥ 0.970 and threshold curve top-8 printed.
- No code changes since this commit except Aman-approved ones.
- If ANY gate fails: no Run 2. Report + wait for Aman.
