# Per-S1 pair-cap sweep (50k sample, --skip-test, default caps) — cap stays OFF

| cap | recall_all | total pairs | p50/p99/max per S1 | featurize | peak GB | OOF F0.5 |
|---|---|---|---|---|---|---|
| 200 | 0.8585 | 9.43M | 200/200/200 | 6.5m | 7.74 | 0.9064 |
| 300 | 0.8741 | 13.63M | 300/300/300 | 5.7m | 7.52 | 0.9166 |
| 500 | 0.9054 | 20.80M | 500/500/500 | 9.1m | 7.36 | 0.9376 |
| 0 (off) | 0.9774 | 35.67M | 646/1955/5354 | 16.4m | 8.22 | 0.9771 |

- At-cap S1: 88% / 80% / 63% at 200/300/500. True matches sit deep in the shared-token-count tail — ranking cannot save any fixed cap. Even 500 costs −0.072 (36× the ≤0.002 budget) to save ~5 of 12 full-scale hours.
- 5k regression at cap 500: 0.9835 vs 0.9920 (−0.0085). Loss steepens with pool density, so full scale would be worse than measured here.
- Full-scale projection (cap 0): ~1.57B pairs, ~12h featurize at 36.2k pairs/s. Projections are upper bounds (denser pools bind caps harder).
- Ranking detail: key `(-(count if count>0 else 1), counted-flag, other_id)` — prefix-exclusive pairs never dropped first, only lose to ≥2-shared-token pairs. Didn't help; the signal itself is weak for ambiguous names.
- Verdict: caps OFF for all runs. Machinery retained (`--max-pairs-per-s1`, default 0).
