# Matching-only Git handoff

Only the matching result is included in Git; `candidate_pairs.tsv` stays local. Git LFS rejects the full file above the host's per-object size cap, so the matching result is stored as ordered gzip parts under `matching_results_parts/`.

Install Git LFS and run `git lfs pull`, then run `python restore_submission.py` from this directory. It reassembles and decompresses the parts into the exact `matching_results.tsv`, verifies the gzip stream, and refuses to overwrite an existing file.

Full formatting, candidate-containment, and ID-existence checks passed on the local matching and candidate files using a memory-safe streaming copy of the repository validator: 1,732,544 rows and 9,969,589 valid S2/S3 IDs. The heuristic is unscored; no score has been measured.
