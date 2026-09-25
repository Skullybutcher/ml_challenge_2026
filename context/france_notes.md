# France eyeball notes

## Sample
40 test S1 rows: 20 France + 10 US + 10 India (`.agent_comms/shared_context/france_eyeball.tsv`).

## Findings (human eyeball, 2026-09-25 night)
1. **Accents**: France-only in sample (8 rows contain non-ASCII). Verified in code: `normalize.py` PUNCT_RE (`[^\w\s]`, UNICODE) preserves accented letters on both sides — no corruption. Residual risk is accent *variants* ("café" vs "cafe") missing on that token.
2. **Suffixes**: SARL/SAS/SA/EURL all appear (no SCI in sample), glued normally, e.g. "Recherche College SAS". All already in LEGAL_SUFFIXES — stripped correctly.
3. **Name structure**: nothing breaking prefix (country + first-4-letters) grouping.
4. **Addresses**: no 5-digit postals in sample; Rue/Avenue spelled out (no abbrev handling needed); no "chez" landmarks (cleaner than India); mostly street-before-number with a few number-first exceptions — token blocking is order-insensitive, fine.
5. **Missingness**: no empty addresses in sample (n=40, weak signal).
6. **Gut check**: no fragile cases found; shared-word blocking should catch typo variants via surviving words.

## Verdict: A — no special France handling, plus one cheap patch
- No separate threshold, no country-specific logic.
- **Recommended patch** (safe, France-only effect since US/India have no accents): accent-fold in `_base_clean` (`normalize.py:31-38`) via `unicodedata.normalize('NFKD', t).encode('ascii','ignore').decode()` so "café" and "cafe" match. Give to teammate with the chunked-patch handoff.
