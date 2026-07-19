# RMR-Net Road Image Restoration

Code release for the paper:

**RMR-Net: Road Metadata-Guided Restoration Network for Downstream Pavement-Defect Detection**

Code author: **Amir Ghorbani** `<amir.ghorbani@rmit.edu.au>`

Repository: https://github.com/AmirNetwork/rmrnet-restoration

## Audited submission release

The paper-aligned release is archived under [`audited_release/`](audited_release/README.md). Start there when reproducing the manuscript. It separates the exact implementation compatible with the reported IVCNZ/PCM epoch-28 checkpoints from later field-policy variants, includes validation-selection and held-out result ledgers, and provides file hashes plus a final integrity audit. The manuscript tables were manually assembled before 15 July 2026 and subsequently checked row by row against these frozen machine-readable artifacts.

The older root-level files are retained as development history. They must not be substituted silently for `audited_release/reported_controlled_main/` when reproducing the controlled headline tables.

RMR-Net is a PyTorch restoration model for road-monitoring imagery. It is designed to improve downstream pavement-defect detection under blur, defocus, low light, noise, and compression while keeping a pass-through path for already high-quality native frames.

The implementation class is still named `RCADNet` in some files for checkpoint compatibility. The paper method name is **RMR-Net**.

## What Is Included

- `rcadnet/model.py`: deployed RMR-Net backbone, sparse metadata/image code fusion, FiLM conditioning, task-evidence attention, and bounded detail skip.
- `models/rmrnet.py`: paper-facing wrapper that keeps legacy RCADNet checkpoint compatibility and exposes named outputs.
- `rcadnet/losses.py`: base restoration loss.
- `rcadnet/task_losses.py`: task-driven detector-feature loss, CQMix, Hutchinson Jacobian regularization, detector anchor, evidence non-regression, and disabled legacy active-contour ablation code.
- `train_rcadnet.py`: training entry point used by the paper experiments.
- `benchmark_adapter_rcadnet.py`: restoration benchmark adapter.
- `tools/restore_yolo_split.py`: restore controlled YOLO-format splits.
- `tools/restore_native_yolo_split.py`: restore native-resolution field images without resizing the source image.
- `tools/eval_yolo_suite.py` and `tools/yolo_eval.py`: detector evaluation helpers.
- `tools/eval_yolo26_coordinate_gt46.py` and `tools/run_gt46_yolo26_coordinate_all_methods.py`: G46 native-resolution field-test evaluation with the coordinate-aware YOLO26 wrapper.
- `Yolo26_coordinate/`: frozen field-detector wrapper and checkpoint used for the G46 field audit.
- `configs/rmrnet_headline.yaml`: paper-facing method and evaluation configuration.
- `results/`: compact CSV/JSON evidence copied from the final local run.

Large training datasets and full experiment outputs are not bundled. See `DATA_AND_WEIGHTS.md`.

## Environment

The final local runs used Windows, CUDA, and the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-windows-gpu.txt
python -m pip install -r requirements-detection-extra.txt
python -m pip install -r requirements-demoe-extra.txt
python -m pip install -r requirements-dfpir-extra.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Observed local hardware for the paper runs:

- Windows workstation
- NVIDIA GeForce RTX 3050, 6 GB VRAM
- PyTorch `2.11.0+cu128`
- CUDA execution enabled

## Training

RMR-Net is trained on paired degraded/clean road images organized as:

```text
data_root/
  scenarios/
    defocus_medium/
      input/
      gt/
    motion_horizontal_medium/
      input/
      gt/
```

Paper-facing command pattern:

```powershell
python train_rcadnet.py `
  --data-root data\pothole_restoration `
  --scenario motion_horizontal_medium --scenario defocus_medium --scenario lowlight_medium `
  --epochs 30 --batch-size 1 --patch-size 128 --width 40 `
  --device cuda --out runs\rmrnet_headline_pothole `
  --num-workers 0 --code-source metadata_fused `
  --block-type evidence --attention-type task --conditioning gated_basis `
  --detail-preserve --detail-gain 0.12 `
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
  --tdp-defect-mask-weight 4.0 `
  --detector-input-size 256 --jacobian-probes 1 `
  --save-every-epoch
```

Use the same pattern for PCM with the PCM restoration root and PCM detector weights.

Important final-training choices:

- active-contour training loss is disabled: `--lambda-active-contour 0.0`;
- active contours are used only after detection as a measurement stage;
- checkpoint choice is made using validation detector recovery before test evaluation;
- G46 field labels are not used for detector training, restoration training, checkpoint selection, or policy tuning.

## Controlled Detection Evaluation

Restore a controlled YOLO split:

```powershell
python tools\restore_yolo_split.py `
  --data datasets\pcm_yolo_defocus_test\data.yaml `
  --split test --model rcadnet `
  --weights runs\rmrnet_headline_pcm\rcadnet_epoch_028.pth `
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

## G46 Native-Resolution Field Test

The G46 field audit uses 46 annotated Sony cam1 frames at native `4752 x 3168` resolution. Roboflow annotations are mapped back to the original pixels. EXIF and pose/geotag metadata are joined by filename where available. No synthetic degradation is added.

The field detector is a frozen YOLO26-coordinate wrapper around `YOLO26s_RDD_FRDC_Distilled_v2.pt`. It is not fine-tuned on G46. The wrapper runs directly on the full native image at a 1280-pixel detector input, writes pixel and geo-coordinate outputs, and uses the four shared road-defect classes:

- D00 longitudinal crack
- D10 transverse crack
- D20 alligator crack
- D40 pothole

Run one image folder:

```powershell
python Yolo26_coordinate\Yolo26_coordinate.py `
  --images path\to\native_or_restored_images `
  --out experiments\g46_coordinate\raw `
  --csv Yolo26_coordinate\precise_cam2_coords0.csv `
  --model Yolo26_coordinate\YOLO26s_RDD_FRDC_Distilled_v2.pt
```

Run the paper-style all-method G46 audit:

```powershell
python tools\run_gt46_yolo26_coordinate_all_methods.py `
  --dataset-root road-defect-seg-9junedata.coco-segmentation `
  --native-root geotagged\cam1 `
  --metadata-csv geotagged\precise_cam1_coords.csv `
  --model Yolo26_coordinate\YOLO26s_RDD_FRDC_Distilled_v2.pt `
  --out experiments\gt46_yolo26_coordinate_revised
```

The G46 metric is intentionally detector-recovery oriented because exact high-resolution box boundaries are ambiguous. A ground-truth defect is counted as recovered when a same-primary-class prediction satisfies at least one condition:

- IoU is at least 0.10;
- prediction covers at least 25 percent of the GT area;
- prediction contains the GT-box center.

Conventional precision, recall, F1 at IoU 0.10 and 0.50 are reported beside this GT-success metric.

## Code-to-Paper Map

See `PAPER_CODE_MAP.md`. The most important links are:

- paper sparse basis and fusion: `rcadnet/model.py::CodeBasisFusion`;
- FiLM equation: `rcadnet/model.py::FiLM`;
- bounded detail skip: `rcadnet/model.py::EvidencePreservingDetailSkip`;
- TDP/CQMix/Jacobian objective: `rcadnet/task_losses.py`;
- composite objective: `rcadnet/task_losses.py::CompositeTaskLoss` plus `train_rcadnet.py`;
- G46 coordinate detector: `Yolo26_coordinate/Yolo26_coordinate.py` and `tools/run_gt46_yolo26_coordinate_all_methods.py`.

## Claim Boundaries

- Controlled IVCNZ/PCM road-damage experiments use synthetic proxy metadata derived from known degradation settings.
- KITTI uses real OXTS telemetry under controlled blur, not naturally paired real road-damage blur.
- The G46 Sony field test uses real native images, real metadata, and revised annotations, but it has no clean target.
- Active contours are post-detection measurement tools, not deployed segmentation masks.
- RMR-Net is not claimed to be a universal blind enhancer; pass-through or weak residual output can be safer for already high-quality native images.
