# Error taxonomy (human-classified, 2026-09-25 night)

Source: 57 labeled TRUE-match pairs (30 fold-0 S1 x up to 2 matches), worksheet `taxonomy_worksheet.tsv`. Labels consolidated from free-text into 7 groups:

| # | Pattern | Count | What it means for us |
|---|---------|-------|----------------------|
| 1 | Case/punct-only (ALL CAPS, `##` before numbers, extra hyphens/parens) | ~18 | Already eaten by normalization; model sees near-identical strings. Easy pairs, high scores. |
| 2 | Missing/extra words (dropped/added name words, shorter addresses) | ~13 | Word-overlap features degrade gracefully; mid scores. Bulk of the threshold battleground. |
| 3 | Abbrev/substitution (Ltd->abrv, LP->Inc/LLC, one word replaced, same address) | ~8 | Same-address + different-word pairs are a precision trap: address features say "match", name says "maybe". Supports a cautious threshold. |
| 4 | Transliteration/Hindi-script + accent variants ("Krishna"-types, e->é swaps) | ~6 | Confirmed IN TRAIN, not just France. Direct evidence for the accent-fold patch. Likely over-represented among cap experiment losers (share <=1 counted token). |
| 5 | Typo (1-3 letters) | ~4 | Char/rapidfuzz features catch these; mid-high scores. |
| 6 | Reorder (word/address component order) | ~5 | Token-set features immune; sequence features blind. Fine. |
| 7 | Missing/placeholder data (`?`, NaN, website-as-name, no street number) | ~7 | Missingness features + singleton path carry these. ~12% of pairs. |

## Decisions driven
- Threshold: lean cautious (groups 3+7 punish aggressive cutoffs; group 1 guarantees a high-score floor).
- Accent-fold patch: UPGRADED from France-only to train-recall justification (group 4 in-train).
- Cap failure mechanism explained: groups 4+3 pairs share few counted tokens -> rank past any fixed cap -> cap200's -0.119 recall loss. No count-based cap is safe; cap stays OFF.
- Hard negatives: revisit only if OOF errors concentrate in group 3 (false merges on same-address pairs).
