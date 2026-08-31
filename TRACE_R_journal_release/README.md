# TRACE-R

**Telemetry-Conditioned Restoration for Road-Defect Detection**

Author: Amir Ghorbani (`amir.ghorbani@rmit.edu.au`)

TRACE-R is a single-output image restorer for vehicle-based pavement inspection.
It combines a restoration substrate with an exposure-aligned camera, IMU, and
vehicle record. The controlled instance uses DeMoE and learns distributed
sensor-conditioned adapters. The CRID field instance follows a conservative
field transfer without paired clean images: validation selects the matched DFPIR substrate, the
feature adapters remain at identity, and measured packet reliability controls a
bounded output correction. The inference path does not combine restorers or
detector outputs.

The Python identifier `rmrp` and checkpoint class `RMRPMetadataDeMoE` are retained
only for backward-compatible loading of the arXiv precursor. All user-facing
outputs and the accepted checkpoint identify the method as TRACE-R.

## Frozen result sources

- `provenance/controlled/final_provenance_ledger.json`: one-time IVCNZ/PCM test.
- `provenance/metadata_controls/metadata_control_summary.csv`: validation-only
  aligned, unavailable, and wrong-packet intervention.
- `provenance/crid/`: CRID-320 annotation freeze, detector selection, matched
  adaptation audits, validation policies, and one-time sealed test.
- `provenance/paper/asset_manifest.json`: hashes of generated paper assets.

The selected checkpoint hash is
`a79e2a775e576f17cfe78688484985830e89de7fbe582eca10d43cc4e0cf59db`.
Large datasets, detector weights, restoration checkpoints, and third-party
weights are not redistributed. See `DATA_AND_WEIGHTS.md` for sources and the
expected directory layout.

`CODE_PROVENANCE.md` distinguishes original TRACE-R components from
project-written compatibility adapters and separately licensed upstream code.

## Environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The audited workstation used Python 3.12, CUDA-enabled PyTorch 2.11.0+cu128,
Ultralytics 8.4.53, and an NVIDIA GeForce RTX 3050 6 GB.

## Verification

```powershell
python -m compileall models rcadnet baselines tools train_matched_restorer.py
python -m pytest -q tests\test_losses.py tests\test_practical_metadata.py `
  tests\test_trace_state_loss.py tests\test_tracer_sensor_adapter.py `
  tests\test_matched_detector_objective.py tests\test_nafnet_official.py `
  tests\test_dfpir_official.py
```

## Reproduction map

- `models/tracer_sensor_adapter.py`: hierarchy-wide low-rank adapter equations.
- `models/rmrp_metadata_demoe.py`: image--sensor posterior and route equations.
- `rcadnet/practical_metadata.py`: packet construction, physical map, and reliability.
- `rcadnet/losses.py`: Charbonnier and gradient fidelity terms.
- `rcadnet/task_losses.py`: frozen detector features and detector supervision.
- `train_matched_restorer.py`: complete accepted optimization objective.
- `baselines/nafnet_road.py`: faithful dependency-free official NAFNet architecture.
- `tools/run_tracer_locked_confirmatory.py`: frozen controlled test.
- `tools/run_tracer_metadata_controls.py`: packet intervention.
- `tools/train_crid320_detector.py`: training-block field detector adaptation.
- `tools/train_crid320_restorer.py`: field fine-tuning from reviewed boxes without paired clean images.
- `tools/build_crid320_staged_trace_init.py`: exact selected-DFPIR-to-TRACE initialization.
- `tools/evaluate_crid320_validation.py`: checkpoint and residual selection.
- `tools/freeze_crid320_trace_field_policy.py`: pre-test TRACE-R policy freeze.
- `tools/run_crid320_sealed_test.py`: one-time 80-frame confirmatory test.
- `tools/refresh_trace_r_asset_manifest.py`: source-ledger and paper-asset hashes.
- `tools/build_crid320_paper_assets.py`: sealed-ledger CRID table and figures.

The CRID collection contains 4,134 native frames and synchronized telemetry.
The paper reports 320 reviewed frames under a frozen 180/60/80 temporal split.
Its de-identified data release will be maintained in this repository; see
`DATA_AND_WEIGHTS.md` for the privacy boundary and current availability.

Detailed commands and split safeguards are in `REPRODUCIBILITY.md`.
