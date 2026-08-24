# RMR-P: Road Metadata-aware Restoration for Pavement Inspection

Author: Amir Ghorbani (`amir.ghorbani@rmit.edu.au`)

This repository contains the PyTorch implementation and audited validation
evidence for RMR-P. The reported controlled system converts an observable
82-value camera, IMU, and vehicle packet into a reliability-qualified cause,
then selects or blends matched DFPIR, InstructIR, and DeMoE restorers. Missing
modalities disable only the causes they support. Dataset names, scenario names,
test labels, clean targets, and hidden rendering kernels are not inference
inputs.

## Reported controlled evidence

The immutable source of truth is:

`experiments/final_rmrp_v50_validation_ledger_20260824/provenance_ledger.json`

Its status is `FROZEN_VALIDATION_ONLY` and `test_split_used=false`. Previously
opened IVCNZ/PCM test identities are excluded from all v50 claims.

| Method | IVCNZ mean mAP50 | PCM mean mAP50 | Joint mean |
| --- | ---: | ---: | ---: |
| NAFNet | 0.399 | 0.140 | 0.270 |
| FiLM-NAFNet | 0.400 | 0.146 | 0.273 |
| InstructIR | 0.503 | 0.240 | 0.372 |
| DFPIR | 0.503 | 0.270 | 0.387 |
| DeMoE | 0.517 | 0.284 | 0.400 |
| **RMR-P** | **0.525** | **0.291** | **0.408** |

Aligned packets score 0.408 jointly, compared with 0.351 when every sensor
group is unavailable and 0.332 when images receive packets from the wrong
corruption family. RMR-P also has the highest aggregate PSNR and SSIM on both
controlled validation partitions.

KITTI and CRID use separately adapted, domain-specific RMR-P instances to test
the same observable packet interface with measured OXTS or EXIF/SBG records.
They are not mixed with the controlled expert-bank checkpoint.

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

## Core implementation

- `models/rmrp_expert_fusion.py`: deployed physical router and expert policy.
- `rcadnet/practical_metadata.py`: 82-value packet, physical conversion,
  modality masking, and reliability semantics.
- `train_matched_restorer.py`: shared target-domain adaptation trainer.
- `rcadnet/task_losses.py`: frozen-detector feature and supervised losses.
- `tools/run_matched_training_suite.py`: restartable matched training driver.
- `tools/restore_yolo_split.py`: restoration and metadata-control inference.
- `tools/eval_yolo_suite.py`: fixed-detector evaluation.
- `tools/validate_rmrp_expert_fusion.py`: restartable v50 validation.
- `tools/freeze_rmrp_v50_validation_ledger.py`: immutable ledger builder.
- `tools/evaluate_rmrp_v50_validation_fidelity.py`: paired PSNR/SSIM audit.
- `tools/build_rmrp_v50_paper_assets.py`: ledger-to-paper tables and figures.
- `tools/build_rmrp_v50_validation_qualitative.py`: auditable qualitative scan.

## Packet contract

The public packet has 82 normalized values:

- 11 three-axis angular-rate samples (33 values);
- 11 three-axis acceleration samples (33 values); and
- 16 camera, vehicle, synchronization, reliability, and availability fields.

The physical map implements the paper relation

`Delta_theta = T_exposure * trapz(omega)`.

Camera reliability supports defocus and illumination decisions; IMU
reliability supports motion. Camera, IMU, and vehicle groups can be missing
independently. Exact synthetic renderer parameters can supervise data
preparation, but they are not present in public validation sidecars and are not
read by the deployed policy.

## Matched adaptation

NAFNet, FiLM-NAFNet, InstructIR, DFPIR, and DeMoE receive the same IVCNZ/PCM
training identities, five-condition sampling stream, 128-pixel crop
distribution, 4,096 optimizer steps, effective batch four, and seed 2026. The
common objective is:

`L = L_charbonnier + 0.15 L_gradient + 0.20 L_TDP + 0.10 L_detector`.

The detector is frozen. Clean-target detector features are stop-gradient;
restored-image features remain differentiable. RMR-P composes the same epoch-70
DFPIR, InstructIR, and DeMoE checkpoints reported as standalone baselines and
receives no extra private training budget.

## Reproduce the controlled validation

Prepare the IVCNZ and PCM train/validation directories documented in
`docs/RMRP_V50_REPRODUCIBILITY.md`, place the third-party weights at their
declared paths, then run:

```bat
.venv\Scripts\activate
python tools\validate_rmrp_expert_fusion.py ^
  --control correct ^
  --control unavailable ^
  --control cross_condition_shuffled ^
  --out E:\RMRP_experiments\rmrp_expert_fusion_v50_validation_20260822

python tools\evaluate_rmrp_v50_validation_fidelity.py
python tools\freeze_rmrp_v50_validation_ledger.py
python tools\build_rmrp_v50_paper_assets.py
python tools\build_rmrp_v50_validation_qualitative.py
```

The validator is restartable and refuses test-labelled inputs. The freeze
script performs no training, inference, selection, or metric calculation.

## Verification

```bat
python -m compileall -q models rcadnet tools
python -m pytest -q tests\test_rmrp_expert_fusion.py tests\test_practical_metadata.py
```

The paper and supplementary material are in one LaTeX source:
`paper_automation_in_construction_rmrnet/manuscript.tex`.

## Third-party code and weights

DFPIR, InstructIR, DeMoE, NAFNet, and Ultralytics retain their upstream
licenses. Large datasets and third-party checkpoints are not redistributed in
the code archive. See `THIRD_PARTY_LICENSES.md`, `CITATION.cff`, and the
reproducibility guide for provenance and expected hashes.
