# RMR-Net Road Image Restoration

This repository contains the code used for the RMR-Net paper:

**RMR-Net: Metadata-Conditioned Road Image Restoration for Downstream Pavement Defect Detection**

RMR-Net is a lightweight PyTorch restorer for road monitoring images. The paper-facing method combines:

- metadata/image degradation-code fusion;
- evidence-conditioned restoration blocks;
- task-evidence attention for road edges, contrast, dark regions, and saturation;
- a bounded detail-preserving skip for crack, pothole-rim, patch, and lane-marking cues;
- low-weight detector-feature, Jacobian, detector-anchor, evidence non-regression, and detail-copy regularizers during training;
- a post-detection active-contour measurement stage.

The implementation class is still named `RCADNet` in several files for backwards compatibility with old checkpoints. The paper method name is **RMR-Net**.

## Environment

The final local runs used Windows, CUDA, and the project virtual environment:

```powershell
.\.venv\Scripts\activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Observed local hardware:

- NVIDIA GeForce RTX 3050, 6 GB VRAM
- PyTorch `2.11.0+cu128`
- CUDA execution enabled

Install extra packages as needed:

```powershell
python -m pip install -r requirements-windows-gpu.txt
python -m pip install -r requirements-detection-extra.txt
python -m pip install -r requirements-demoe-extra.txt
python -m pip install -r requirements-dfpir-extra.txt
```

## Main Files

- `rcadnet/model.py`: RMR-Net/RCADNet architecture, code fusion, FiLM, task-evidence attention, and bounded detail skip.
- `rcadnet/losses.py`: base restoration objective from the manuscript.
- `rcadnet/task_losses.py`: detector-feature TDP/CQMix, Hutchinson Jacobian, detector-anchor, evidence non-regression, detail-copy guard, and legacy optional active-contour loss.
- `rcadnet/scenario_codes.py`: scenario and metadata to normalized degradation code.
- `train_rcadnet.py`: training and task-loss integration.
- `benchmark_adapter_rcadnet.py`: benchmark adapter that loads the deployed RMR-Net checkpoint architecture.
- `tools/restore_yolo_split.py`: restores YOLO-format controlled-degradation splits.
- `tools/restore_native_yolo_split.py`: restores native-resolution geotagged field-test splits without resizing the source images.
- `tools/run_final_native_field_test.py`: all-49 Sony native-image field test, eta sweep, tau sweep, and sharpness audit.
- `tools/eval_native_tiled_detector.py`: native-resolution tiled detector evaluation.
- `tools/tune_residual_policy.py`: validation-selected residual/pass-through policy.
- `tools/tune_perception_gate.py`: validation-selected no-reference gate policy.
- `tools/snake_boundary_metrics.py`: post-detection active-contour boundary measurements.

## Final Configuration

The paper-facing configuration is in:

```text
configs/rmrnet_headline.yaml
```

Important final settings:

- patch size `128`
- batch size `1`
- width `40`
- detail skip enabled with gain `0.20`
- code supervision weight `0.05`
- TDP weight `0.001`
- Jacobian weight `0.00002`
- detector-anchor weight `0.0005`
- evidence non-regression weight `0.02`
- detail-copy weight `0.002`
- active-contour training loss disabled (`0.0`)

Active contours are used after detection for measurement only. They are not part of the final training objective.

## Training Command Pattern

RMR-Net is trained with paired degraded/clean road images. The exact data folders depend on the prepared local datasets.

```powershell
python train_rcadnet.py `
  --data-root data\pothole_restoration `
  --scenario motion_horizontal_medium --scenario defocus_medium --scenario lowlight_medium `
  --epochs 4 --batch-size 1 --patch-size 128 --width 40 `
  --device cuda --out runs\rmrnet_headline_pothole `
  --num-workers 0 --code-source metadata_fused `
  --block-type evidence --attention-type task --conditioning gated_basis `
  --detail-preserve --detail-gain 0.20 `
  --aux-code-weight 0.05 --metadata-dropout 0.10 --metadata-noise 0.01 `
  --edge-weight 0.15 --freq-weight 0.05 --defect-weight 0.10 --visibility-weight 0.08 `
  --use-task-losses --task-loss-warmup-epochs 2 `
  --lambda-tdp 0.001 --lambda-jacobian 0.00002 `
  --lambda-active-contour 0.0 `
  --lambda-detector-input-anchor 0.0005 `
  --lambda-evidence-nonregression 0.02 `
  --lambda-detail-copy 0.002 `
  --tdp-yolo-weights runs\detect\runs\yolo11s_v26\pothole_clean_80ep\weights\best.pt `
  --tdp-layers 2,4 --tdp-layer-weights 0.5,1 `
  --detector-input-size 256 --jacobian-probes 1 `
  --save-every-epoch
```

The PCM command is the same pattern with the PCM restoration root and PCM detector weights.

## Controlled Detection Evaluation

Restore a controlled YOLO split:

```powershell
python tools\restore_yolo_split.py `
  --data datasets\pcm_yolo_defocus_test\data.yaml `
  --split test --model rcadnet `
  --weights runs\rmrnet_headline_pcm\rcadnet_epoch_002.pth `
  --scenario defocus_medium `
  --rcadnet-code-source metadata `
  --out datasets\pcm_yolo_defocus_test_rmrnet `
  --device cuda
```

Evaluate with the frozen detector:

```powershell
python tools\yolo_eval.py `
  --weights runs\detect\runs\yolo11s_v26\pcm_clean_80ep\weights\best.pt `
  --data datasets\pcm_yolo_defocus_test_rmrnet\data.yaml `
  --split test --imgsz 640 --device 0
```

## Sony Native-Image Field Test

The high-resolution Sony field test uses all 49 matched annotated frames at native `4752 x 3168` resolution. Roboflow labels are mapped back to the original images, and real EXIF/pose metadata is used where available.

```powershell
python tools\run_final_native_field_test.py --no-overlays
```

Outputs are written under:

```text
experiments\roboflow_geotagged_v5_native_real\final_all49
```

The script writes:

- `paper_metric_summary_all49.csv`
- `geotagged_eta_sweep.csv`
- `geotagged_tau_sweep.csv`
- sharpness audit CSVs
- paper-ready figures and LaTeX tables

The eta/tau sweeps are descriptive held-out policy audits on the 49-image field set. Validation-selected deployment policies are implemented separately in `tools/tune_residual_policy.py` and `tools/tune_perception_gate.py`.

## Baselines

The release includes adapters/scripts for:

- DFPIR
- DeMoE-auto
- DeMoE-scenario
- NAFNet-road
- InstructIR adapters where weights/prompts are available

Large external weights and datasets are not bundled. Place weights in the folders expected by each adapter or pass explicit paths on the command line.

## Paper Release Packaging

Generate clean paper/source/evidence/code folders:

```powershell
python tools\package_final_release.py
```

The packaging script copies only manuscript-referenced tables/figures, the all-49 field-test evidence CSVs, selected provenance files, and code needed for reproduction.

## Important Claim Boundaries

- Controlled IVCNZ/PCM road-damage experiments use synthetic proxy metadata derived from known degradation settings.
- KITTI uses real OXTS telemetry under controlled blur, not naturally paired real blur.
- The Sony geotagged field test uses real native images and real metadata, but it has only 49 annotated frames and no clean target.
- Active contours are post-detection measurement tools, not deployed segmentation masks.
- RMR-Net is not claimed to be a universal blind enhancer; pass-through or weak residual output can be safer for already high-quality native images.
