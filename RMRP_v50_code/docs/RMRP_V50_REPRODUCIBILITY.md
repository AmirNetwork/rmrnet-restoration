# RMR-P v50 Reproducibility Guide

## Evidence boundary

The controlled evidence is sequence/source-disjoint validation. The available
IVCNZ and PCM test identities were opened in earlier development, so they are
not reused as new confirmatory evidence. The release ledger must contain:

```json
{
  "status": "FROZEN_VALIDATION_ONLY",
  "test_split_used": false
}
```

## Required datasets

The validator expects the following eight validation datasets, each with
`images/val`, `labels/val`, `metadata/val`, and `data.yaml`:

```text
datasets/pothole_yolo_sequence_disjoint_practical_sensor_calibrated_v2_{motion,defocus,lowlight,mixed}_val
datasets/road_damage_pcm_yolo_sequence_disjoint_practical_sensor_calibrated_v2_{motion,defocus,lowlight,mixed}_val
```

The IVCNZ split contains 146 validation identities and the PCM split contains
353. Create the partitions before any degradation. The train roots used for
matched adaptation are:

```text
data/pothole_restoration_practical_sensor_calibrated_v2_train
data/pcm_restoration_practical_sensor_calibrated_v2_train
```

Public sidecars contain the normalized 82-value practical packet. They must not
contain a scenario name, dataset identity, clean target, hidden kernel, blur
length, renderer noise sigma, or test label.

## Selected expert checkpoints

```text
experiments/matched_final_candidate_index_v28_epoch70_20260821/demoe/demoe_epoch_070.pth
  SHA-256 743f106d380bd8db5d9020917dbc53e668785c23e3b065fe74defda9f90c4cc8

experiments/matched_final_candidate_index_v28_epoch70_20260821/dfpir/dfpir_epoch_070.pth
  SHA-256 cca8f4b00a2e6468cb8bcd267393082b7c781500a7f10de1da626f4c0ee4caaf

experiments/matched_final_candidate_index_v28_epoch70_20260821/instructir/instructir_epoch_070.pth
  SHA-256 4d97860f360bf21cffa90011db3c5a14501ed27df9bbd5b743a8ebe8e0bc7ecc
```

InstructIR also requires `weights/instructir/lm_instructir-7d.pt`, SHA-256
`b239e5d5dbc811813a90e709f9647dead0e35a96a294a7d6c5263da549016fe6`.

## Detector checkpoints

```text
runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pothole_clean_80ep/weights/best.pt
  SHA-256 7d7e24e4e13e85456578b505dcf7ba327ab923d1fbd68fc2127e04766c96b4b9

runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pcm_clean_80ep/weights/best.pt
  SHA-256 7b6db99cd29da5ed4488d99a0afce2606491222251ce2669592290133562d290
```

## Frozen policy

```text
motion threshold        0.180
defocus threshold       0.200
low-light threshold     0.385
sensor support          0.500
low-light DFPIR weight  0.400
mixed DFPIR weight      0.075
gyro full scale         4.000
```

Motion routes to DFPIR, defocus to InstructIR, low light to a 0.4/0.6
DFPIR--DeMoE blend, mixed corruption to a 0.075/0.925 DFPIR--DeMoE blend, and
unsupported records to DeMoE automatic routing.

## Validation and paper rebuild

```powershell
.\.venv\Scripts\Activate.ps1

python tools\validate_rmrp_expert_fusion.py `
  --control correct `
  --control unavailable `
  --control cross_condition_shuffled `
  --out E:\RMRP_experiments\rmrp_expert_fusion_v50_validation_20260822

python tools\evaluate_rmrp_v50_validation_fidelity.py
python tools\freeze_rmrp_v50_validation_ledger.py
python tools\build_rmrp_v50_paper_assets.py
python tools\build_rmrp_v50_validation_qualitative.py

Push-Location paper_automation_in_construction_rmrnet
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
Pop-Location
```

The final ledger is the sole source for controlled paper numbers. KITTI and
CRID have separate, domain-specific manifests and checkpoints; do not merge
their selection records with the controlled expert-bank ledger.
