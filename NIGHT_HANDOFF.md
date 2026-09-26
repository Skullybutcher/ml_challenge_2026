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

## Goal (Aman will not intervene — act on this until he asks for status)
Ship the best-scoring VALID submission before the deadline. Burn at most 2 leaderboard submissions unasked (policy below). With remaining time: keep improving (playbook E1–E6), keep reporting in-repo. Do not wait for Aman at any step — every decision rule is specified here. Status only when he asks.

## Run 2 launch authority (no waiting)
Launch Run 2 as soon as ALL hold: Run 1 recall ≥ 0.95 overall; India recall reported (any value — if < 0.90, note it, still proceed); Run 1 OOF ≥ 0.970 with top-8 curve printed; no code changes since this commit except Aman-approved ones. If any fail: fix per playbook failure table and re-run Run 1 — still no waiting, just keep cycling until gates pass or Aman's status check.

## Threshold pick (yours, no waiting)
From Run 1's top-8 curve: pick the HIGHEST t with score ≥ best − 0.001 (plateau-start favors precision under F0.5). If curve is flat/ambiguous, take argmax. Record the pick + curve in SUBMISSIONS.md. Use it for Run 2.

## Submission policy (Aman submits — Akari never uploads)
- Akari makes ZERO leaderboard submissions. When a full run finishes + validator PASS (default AND --check-ids if feasible) + matching row count exactly 1,732,544: stage everything submission-ready and log a READY-TO-SUBMIT row in SUBMISSIONS.md with exact upload steps. Then keep improving locally (playbook E1–E6) until Aman checks status.
- If a later run/model beats the staged one's OOF by ≥ 0.003 with the same gates: it becomes the new staged submission (log supersedes, keep both files). Aman picks what actually uploads.

## Update 2026-09-26 AM — Aman's box is OFF, Akari is sole compute
- Local 16GB machine goes offline now. All local runs stop. Nothing will come from Aman's side until he returns — do not wait for anything local.
- Included in this commit: (a) `del counted_s2, counted_s3 + gc.collect()` after the recall gate when caps are off — the counted frames (~20GB at 100k-rare) were the 49.5GB blowup, bigger than the asarray conversion; keep your batched-conversion fix too, both stack; (b) `--rare-min-len` / `--rare-max-df` flags (defaults 6 / 2000 = current behavior, validated config untouched).
- Run 2 band DECIDED: `--rare-max-df 1000` (measured locally at 50k: recall 0.9857 vs 0.9892 full-band, pairs −38%). Default band projects ~4.5B pairs / ~36h at measured test density (~2600/S1 over 2×50k test chunks vs the real 10M pool); band-1000 projects ~2.9B / ~22h. Append the flags to the Run 2 command: `--use-rare --rare-max-df 1000`.
- Speed is now the binding constraint, not memory or recall. Nominated E-series experiment: batched featurize (rapidfuzz `process.cdist` over pair columns instead of the per-pair Python loop, ~35k pairs/s today) — 3-5× would take the full run to ~6h. Validate by exact-output equality on 5k first, then adopt. This is your highest-value experiment after Run 1.
- Run 1 (100k, default band) proceeds unchanged — it validates rescue-at-scale + OOF/threshold. The band decision affects Run 2 only.
