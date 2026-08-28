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
  DeMoE/
  DFPIR/
  InstructIR/
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

## 4. CRID temporal evaluation

Construct 82-field sidecars from synchronized Sony EXIF and 200-Hz SBG records,
then run validation before the later temporal block:

```powershell
python tools\run_crid46_sequence_disjoint_comparison.py `
  --out E:\TRACE_R_experiments\trace_crid `
  --detector PATH_TO_YOLO26_ROAD_DAMAGE.pt `
  --rmr-checkpoint PATH_TO_TRACE_R.pth `
  --metadata-root PATH_TO_CRID_METADATA

python tools\run_crid46_sequence_disjoint_comparison.py `
  --out E:\TRACE_R_experiments\trace_crid --run-test
```

The first command freezes the residual strength on the earlier 12-frame block.
The second applies that fixed choice once to the later 13-frame block. Native
4752x3168 coordinates are preserved and detector outputs are not fused.

## 5. Paper assets

After copying the frozen ledgers to the paths expected by the builders:

```powershell
python tools\build_tracer_journal_assets.py --paper paper_ieee_tits_trace_r
python tools\build_tracer_qualitative_panels.py --paper paper_ieee_tits_trace_r
cd paper_ieee_tits_trace_r
pdflatex -interaction=nonstopmode manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

The manuscript and integrated appendix are in the same `manuscript.tex` and PDF.
