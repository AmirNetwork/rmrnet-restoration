# TRACE-R: Telemetry-Routed Adaptive Corruption-Expert Restoration

Author: Amir Ghorbani (`amir.ghorbani@rmit.edu.au`)

TRACE-R is a capture-aware restoration front end for pavement inspection. It
converts an observable camera, IMU, and vehicle packet into physical motion,
focus, illumination, and reliability evidence, then routes an image to matched
DFPIR, InstructIR, or DeMoE experts. Missing sensor groups disable only the
routes they support. A label-free field policy retains the native detector view
unless restored evidence improves by a validation-fixed margin.

The journal method extends the earlier RMR-P conference model. RMR-P used one
metadata-conditioned network; TRACE-R uses a deterministic physical router,
matched expert checkpoints, explicit partial-metadata handling, packet-content
interventions, and a native-image detector-evidence guard. Historical `rmrp`
identifiers are retained only where changing them would invalidate archived
checkpoint and provenance hashes.

## Frozen evidence

The controlled source of truth is:

`experiments/final_rmrp_v50_validation_ledger_20260824/provenance_ledger.json`

The ledger has `status=FROZEN_VALIDATION_ONLY` and `test_split_used=false`.
Previously opened IVCNZ/PCM test identities are excluded from these claims.

| Method | IVCNZ mAP50 | PCM mAP50 | Joint mean |
| --- | ---: | ---: | ---: |
| NAFNet | 0.399 | 0.140 | 0.270 |
| FiLM-NAFNet | 0.400 | 0.146 | 0.273 |
| InstructIR | 0.503 | 0.240 | 0.372 |
| DFPIR | 0.503 | 0.270 | 0.387 |
| DeMoE | 0.517 | 0.284 | 0.400 |
| **TRACE-R** | **0.525** | **0.291** | **0.408** |

Aligned packets score 0.408 jointly, versus 0.351 when every sensor group is
unavailable and 0.332 with a wrong-condition packet. TRACE-R also has the
highest aggregate PSNR and SSIM on both controlled validation partitions.

## Environment

The audited workstation used Windows 11, Python 3.12, CUDA-enabled PyTorch
2.11.0+cu128, Ultralytics 8.4.53, and an NVIDIA GeForce RTX 3050 6 GB.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-windows-gpu.txt
pip install -r requirements-experiments.txt
pip install -r requirements-detection-extra.txt
pip install -r requirements-demoe-extra.txt
pip install -r requirements-dfpir-extra.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Code-to-paper map

- `models/tracer.py`: public TRACE-R API.
- `models/rmrp_expert_fusion.py`: deterministic router and expert bank. The
  historical filename preserves frozen experiment imports.
- `rcadnet/practical_metadata.py`: 82-value packet, exposure integration,
  modality masks, and physical state conversion (paper Eqs. 1-3).
- `train_matched_restorer.py`: equal-budget expert adaptation and the common
  objective (paper Eq. 6). The historical `rmrp` option reproduces the
  conference precursor and is not the TRACE-R router.
- `models/tracer_detection_policy.py`: native/restored evidence decision and
  class-family fusion (paper Eqs. 7-8).
- `tools/restore_yolo_split.py`: TRACE-R and standalone-restorer inference.
- `tools/validate_tracer_expert_fusion.py`: restartable validation-only audit.
- `tools/build_tracer_journal_assets.py`: frozen-ledger tables and PDF figures.

## Sensor packet

The normalized packet has 82 values:

- 11 three-axis gyroscope samples (33 values);
- 11 three-axis acceleration samples (33 values); and
- 16 camera, vehicle, timing, reliability, and availability fields.

The physical map implements
`Delta_theta = T_exposure * trapz(omega)`. Camera reliability supports focus
and illumination routes; IMU reliability supports motion. Camera, IMU, and
vehicle groups can be absent independently. Dataset names, scenario names,
evaluation labels, clean targets, and hidden rendering kernels are never router
inputs.

## Matched adaptation

NAFNet, FiLM-NAFNet, InstructIR, DFPIR, and DeMoE receive the same training
identities, five-condition balanced crop stream, optimizer schedule, 70-epoch
budget, 4,096 optimizer steps, and seed. The reported continuation uses:

`L = L_charbonnier + 0.15 L_gradient + 0.20 L_TDP + 0.10 L_detector`.

The detector is frozen. Clean-target features use stop-gradient; restored-image
features remain differentiable. TRACE-R adds no private expert training budget:
it composes the same selected epoch-70 DFPIR, InstructIR, and DeMoE checkpoints
reported as standalone baselines.

## Controlled validation

After preparing IVCNZ and PCM and the third-party checkpoints described in
`docs/TRACE_R_REPRODUCIBILITY.md`, run:

```bat
.venv\Scripts\activate
python tools\validate_tracer_expert_fusion.py ^
  --control correct ^
  --control unavailable ^
  --control cross_condition_shuffled ^
  --out E:\TRACE_R_experiments\controlled_validation
```

The validator is restartable and refuses test-labelled input paths. Selection
uses the unweighted joint IVCNZ/PCM validation mAP50 only.

## Build paper assets

```bat
.venv\Scripts\activate
python tools\build_tracer_journal_assets.py
cd paper_ieee_tits_trace_r
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
pdflatex supplementary.tex
pdflatex supplementary.tex
```

## Tests

```bat
.venv\Scripts\activate
python -m compileall -q models rcadnet tools train_matched_restorer.py
python -m pytest -q tests\test_tracer_public_api.py tests\test_tracer_detection_policy.py tests\test_rmrp_expert_fusion.py tests\test_practical_metadata.py
```

Large datasets and third-party checkpoints are not redistributed. Their
licenses, download sources, expected locations, and retained hashes are listed
in `THIRD_PARTY_LICENSES.md` and `docs/TRACE_R_REPRODUCIBILITY.md`.
