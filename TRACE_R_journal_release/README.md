# TRACE-R

**Telemetry-Conditioned Restoration for Road-Defect Detection**

Author: Amir Ghorbani (`amir.ghorbani@rmit.edu.au`)

TRACE-R is a single-output image restorer for vehicle-based pavement inspection.
It combines one DeMoE backbone with an exposure-aligned camera, IMU, and vehicle
record. The record informs an internal restoration route and identity-initialized
low-rank feature adapters at several scales. The released inference path does not
combine restorers, native/restored detector boxes, or detector outputs.

The Python identifier `rmrp` and checkpoint class `RMRPMetadataDeMoE` are retained
only for backward-compatible loading of the arXiv precursor. All user-facing
outputs and the accepted checkpoint identify the method as TRACE-R.

## Frozen result sources

- `provenance/controlled/final_provenance_ledger.json`: one-time IVCNZ/PCM test.
- `provenance/metadata_controls/metadata_control_summary.csv`: validation-only
  aligned, unavailable, and wrong-packet intervention.
- `provenance/crid/`: native-resolution CRID validation freeze and later temporal
  evaluation.
- `provenance/paper/asset_manifest.json`: hashes of generated paper assets.

The selected checkpoint hash is
`a79e2a775e576f17cfe78688484985830e89de7fbe582eca10d43cc4e0cf59db`.
Large datasets, detector weights, restoration checkpoints, and third-party
weights are not redistributed. See `DATA_AND_WEIGHTS.md` for sources and the
expected directory layout.

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
  tests\test_matched_detector_objective.py tests\test_nafnet_official.py
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
- `tools/run_crid46_sequence_disjoint_comparison.py`: CRID temporal protocol.
- `tools/build_tracer_journal_assets.py`: ledger-to-table/vector-figure build.
- `tools/build_tracer_qualitative_panels.py`: frozen qualitative selection.

Detailed commands and split safeguards are in `REPRODUCIBILITY.md`.
