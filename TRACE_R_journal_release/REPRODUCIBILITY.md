# Reproducibility protocol

## 1. Directory layout

Run commands from the repository root. Prepare these external artifacts:

```text
data/
  pothole_restoration_practical_sensor_calibrated_v2_train/
  pcm_restoration_practical_sensor_calibrated_v2_train/
datasets/
  pothole_yolo_sequence_disjoint_v1/
  road_damage_pcm_yolo_sequence_disjoint_v1/
runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/
  pothole_clean_80ep/weights/best.pt
  pcm_clean_80ep/weights/best.pt
third_party/
  DeMoE-main/
  DFPIR-main/
  InstructIR-main/
weights/
  nafnet/NAFNet-GoPro-width32.pth
```

Each paired restoration sample has a degraded image, clean target, YOLO label,
and metadata sidecar. The sidecar contains `practical_sensor_packet` in the
82-field order declared by `rcadnet/practical_metadata.py`. Scenario names and
hidden renderer kernels are not model inputs.

## 2. Accepted TRACE-R continuation

The exact executed arguments are retained in
`configs/trace_r_training_audit_config.json`. The equivalent command is:

```powershell
python train_matched_restorer.py `
  --model rmrp `
  --data-root ivcnz=data\pothole_restoration_practical_sensor_calibrated_v2_train `
  --data-root pcm=data\pcm_restoration_practical_sensor_calibrated_v2_train `
  --detector ivcnz=runs\detect\runs\yolo11s_sequence_disjoint_v1_20260716\pothole_clean_80ep\weights\best.pt `
  --detector pcm=runs\detect\runs\yolo11s_sequence_disjoint_v1_20260716\pcm_clean_80ep\weights\best.pt `
  --init-weights PATH_TO_TRACE_IDENTITY_INITIALIZATION.pth `
  --out E:\TRACE_R_experiments\trace_training `
  --epochs 32 --samples-per-epoch 512 --patch-size 128 `
  --effective-batch-size 4 --micro-batch-size 2 `
  --lr 1e-5 --weight-decay 1e-4 --new-module-lr-multiplier 5 `
  --base-weight 1 --edge-weight 0.15 --tdp-weight 0.20 `
  --detector-supervised-weight 0.10 --state-weight 0.10 --physical-weight 0.05 `
  --metadata-dropout 0.10 --metadata-noise 0.005 `
  --metadata-mismatch-probability 0.05 `
  --metadata-curriculum-epochs 4 --metadata-curriculum-ramp-epochs 4 `
  --rmrp-backbone-route-mode sensor_task `
  --rmrp-sensor-task-thresholds 0.18 0.20 0.385 `
  --defect-label-root ivcnz=datasets\pothole_yolo_sequence_disjoint_v1\labels\train `
  --defect-label-root pcm=datasets\road_damage_pcm_yolo_sequence_disjoint_v1\labels\train `
  --defect-crop-probability 0.60 --save-every 4 --seed 2026 --device cuda
```

The `rmrp` CLI key is checkpoint-compatible shorthand for TRACE-R. The selected
checkpoint is epoch 8 of the complete 32-epoch continuation and is selected by
joint validation mAP50 only.

## 3. Controlled validation and test

Baseline and TRACE-R checkpoint paths in the runners are examples. Set them to
the paths listed in the frozen ledger, then run validation selection without a
test-labelled path. Once all policies are frozen:

```powershell
python tools\run_tracer_metadata_controls.py `
  --out E:\TRACE_R_experiments\trace_metadata_controls

python tools\run_tracer_locked_confirmatory.py `
  --out E:\TRACE_R_experiments\trace_locked_confirmatory
```

The confirmatory runner writes a provenance ledger before opening test data and
refuses to change the checkpoint after that freeze. Every method produces one
restored image. DeMoE-oracle is a non-deployable upper bound. InstructIR and
DFPIR are condition-informed controls in the synthetic study and are identified
as such in the paper.

## 4. CRID-320 field evaluation at full resolution

CRID has no sharp/degraded pair. The 320 reviewed frames are frozen by time as
180 adaptation, 60 validation, and 80 test images. Each restorer starts from its
matched controlled-road checkpoint. Its output is passed through the same frozen
detector; reviewed boxes define the detector loss, whose gradient updates only
the restorer. Boxes are never restorer inputs. NAFNet, InstructIR, DFPIR, and
DeMoE use the same training images, defect crops, detector objective,
identity/edge anchors, 20-epoch candidate trajectory, and 7,200-update budget.
The trainable subset is architecture-specific and is recorded for every run.
Validation selects NAFNet epoch 20 and epoch 8 for InstructIR, DFPIR, and DeMoE,
with residual blends 1.0, 0.5, 0.5, and 0.75, respectively.

DFPIR has the highest primary validation mAP50, so the CRID TRACE-R policy is
composed from its selected epoch-8 state. That state descends from the DFPIR
epoch-70 controlled-road checkpoint, not directly from an untouched public
weight. The direct DFPIR and TRACE-R rows therefore share the same image
substrate, providing a direct control for the telemetry policy.
Its feature adapters retain their identity initialization; no additional
gradient update is represented by the reported field checkpoint. The per-frame
82-value Sony/SBG packet controls an automatic reliability gate around a bounded
full-resolution correction selected on the 60-frame validation block. The frozen
policy is tested once on the 80-frame block. This differs from the controlled
DeMoE-backed TRACE-R checkpoint, where distributed sensor adapters are learned
from paired data.

```powershell
python tools\train_crid320_detector.py
powershell -ExecutionPolicy Bypass -File tools\run_crid320_matched_adaptation.ps1
powershell -ExecutionPolicy Bypass -File tools\run_crid320_validation_sweep.ps1
python tools\build_crid320_staged_trace_init.py --help
python tools\freeze_crid320_trace_field_policy.py
python tools\run_crid320_sealed_test.py
python tools\build_crid320_paper_assets.py --confidence 0.10 --options 10
```

The detector and every restorer are selected on the 60-frame validation block.
`freeze_crid320_trace_field_policy.py` records the checkpoint, residual policy,
sensor thresholds, and SHA-256 identities before the test manifest is opened.
The sealed runner then evaluates each fixed method once on the 80-frame test
block. Native 4752x3168 coordinates are preserved and detector outputs are not
fused. See `provenance/crid/` for the exact executed arguments and ledgers.

## 5. Paper assets

The controlled tables and figures in the paper archive are fixed by
`provenance/controlled/final_provenance_ledger.json` and their hashes are listed
in `provenance/paper/asset_manifest.json`. The CRID assets can be regenerated
directly from the completed sealed-test ledger:

```powershell
python tools\build_crid320_paper_assets.py --paper paper_ieee_tits_trace_r
cd paper_ieee_tits_trace_r
pdflatex -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

The manuscript and S-numbered supplementary material are in the same
`manuscript.tex` and PDF.
