# Executed source snapshot

This directory preserves the exact Python files whose SHA-256 hashes are stored
in `experiments/final_rmrp_v50_validation_ledger_20260824/provenance_ledger.json`.
They generated the frozen IVCNZ/PCM validation evidence before the method was
renamed TRACE-R.

The release-level files in `models/` and `tools/` expose the same routing policy
with clearer TRACE-R names, equation comments, and backward-compatible aliases.
The router-equivalence test covers fallback, motion, defocus, low-light, and
mixed routes. Exact numerical reproduction should use this snapshot; new use
should import `TRACERExpertFusion` and `TRACERPolicy` from `models/tracer.py`.
