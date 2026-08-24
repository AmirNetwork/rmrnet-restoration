# RMR-P: Road Metadata-aware Restoration for Pavement Inspection

Author: Amir Ghorbani (`amir.ghorbani@rmit.edu.au`)

This repository accompanies the Automation in Construction manuscript on
capture-aware restoration for pavement inspection. The current audited release
is in [`RMRP_v50_code/`](RMRP_v50_code/README.md).

## Audited evidence

The source of truth is
[`RMRP_v50_code/experiments/final_rmrp_v50_validation_ledger_20260824/provenance_ledger.json`](RMRP_v50_code/experiments/final_rmrp_v50_validation_ledger_20260824/provenance_ledger.json).
It is explicitly marked `FROZEN_VALIDATION_ONLY` with
`test_split_used=false`; previously opened IVCNZ/PCM test identities are not
used in the reported v50 claims.

| Method | IVCNZ mAP50 | PCM mAP50 | Joint mean |
| --- | ---: | ---: | ---: |
| NAFNet | 0.399 | 0.140 | 0.270 |
| FiLM-NAFNet | 0.400 | 0.146 | 0.273 |
| InstructIR | 0.503 | 0.240 | 0.372 |
| DFPIR | 0.503 | 0.270 | 0.387 |
| DeMoE | 0.517 | 0.284 | 0.400 |
| **RMR-P** | **0.525** | **0.291** | **0.408** |

RMR-P accepts an observable 82-value camera, IMU, and vehicle packet. It does
not use dataset names, scenario labels, clean targets, test labels, or hidden
rendering kernels at inference. The release includes implementation, tests,
configuration, expected checkpoint hashes, machine-readable evidence, and the
scripts that generate the manuscript tables and figures.

## Start here

```bat
cd RMRP_v50_code
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-windows-gpu.txt
pip install -r requirements-experiments.txt
python -m pytest -q tests\test_rmrp_expert_fusion.py tests\test_practical_metadata.py
```

See [`RMRP_V50_REPRODUCIBILITY.md`](RMRP_v50_code/docs/RMRP_V50_REPRODUCIBILITY.md)
for datasets, matched adaptation, third-party checkpoint provenance, hashes,
validation controls, and paper-asset generation.

Earlier RMR-Net files remain in the repository solely as historical artifacts;
they are not the implementation or evidence used by the RMR-P v50 manuscript.
