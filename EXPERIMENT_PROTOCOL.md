# RMR-Net Experiment Protocol

This file summarizes the code paths behind the RMR-Net manuscript and is intended as a reproducibility guide, not as a second paper.

## Core Claim

RMR-Net is a metadata-aware road-image restorer for downstream pavement-defect detection. The deployed model is intentionally small:

1. image-estimated and metadata-derived degradation codes;
2. reliability-gated monotone basis fusion;
3. FiLM and code-aware channel gates inside an efficient encoder-decoder;
4. task-evidence attention over edge, contrast, dark-region, and saturation cues;
5. bounded detail skip for local road evidence;
6. optional pass-through or residual-strength policy before the detector.

The final training objective excludes active-contour loss. Active contours are evaluated after detection as a measurement stage for area, perimeter, compactness, and boundary-quality metrics.

## Data Blocks

- IVCNZ pothole restoration/detection: controlled motion, defocus, low-light, and mixed degradations.
- PCM pothole/crack/manhole restoration/detection: controlled motion, defocus, low-light, and mixed degradations.
- KITTI raw telemetry audit: real OXTS metadata under controlled blur.
- Sony geotagged field test: 49 native-resolution annotated cam1 images, real EXIF/pose/geotag metadata, no synthetic degradation, no clean target.

## Training Objective

The code mirrors the manuscript equations:

- `rcadnet/losses.py`: base loss
  `L_base = L1 + edge + Fourier + defect-weighted + visibility`.
- `rcadnet/model.py`: monotone basis fusion
  `b(u) = [u, u^2, sqrt(u), s u, u(1-u)]`.
- `rcadnet/model.py`: detail skip
  `I_r = clip(I_b + eta_d G_d D(I_d), 0, 1)`.
- `rcadnet/task_losses.py`: task-driven perceptual loss with CQMix.
- `rcadnet/task_losses.py`: Hutchinson Jacobian penalty.
- `rcadnet/task_losses.py`: detector-input anchor.
- `rcadnet/task_losses.py`: road-evidence non-regression.
- `rcadnet/task_losses.py`: detail-copy guard.

The same backward pass optimizes the base restoration loss, code supervision, sparse basis loss, and enabled task regularizers after warmup.

## Headline Configuration

Use:

```text
configs/rmrnet_headline.yaml
```

The paper-facing settings are:

- width `40`;
- patch size `128`;
- batch size `1`;
- detail skip enabled with gain `0.12`;
- TDP `0.001`;
- Jacobian `0.00002`;
- detector anchor `0.0005`;
- evidence non-regression `0.02`;
- detail-copy `0.002`;
- active-contour training loss `0.0`.

## Controlled Road-Damage Evaluation

1. Train RMR-Net from paired restoration folders.
2. Save every epoch.
3. Restore validation YOLO splits for each epoch.
4. Select checkpoints by validation detector mAP50.
5. Restore the held-out test split once using the selected checkpoint.
6. Evaluate frozen YOLO11s controlled-dataset detectors; evaluate the native GT49 field set separately with the frozen YOLO26s RDD-FRDC high-resolution detector.
7. Run bootstrap and Holm-corrected uncertainty audits.

The test split is not used for checkpoint selection.

## Sony Native-Resolution Field Test

The Sony field test is not a severe real-blur benchmark. It is a high-quality native-image safety test with real metadata and known defect labels. The paper-facing GT49 protocol uses a frozen YOLO26s RDD-FRDC detector and does not update detector weights on the 49 GT49 labels.

Run:

```powershell
.\.venv\Scripts\python.exe tools\prepare_gt49_rdd4_eval_sets.py --input-root experiments\roboflow_geotagged_v5_native_real\v32_final_gt49_allmethods --out experiments\roboflow_geotagged_v5_native_real\v36_yolo26_rdd4_eval_sets
.\.venv\Scripts\python.exe tools\build_gt49_defect_first_assets.py
.\.venv\Scripts\python.exe tools\build_gt49_yolo26_segmentation_overlays.py --conf 0.10 --panel-width 420 --max-atlas-images 12
```

This evaluates raw native input, RMR-Net image-only, RMR-Net metadata, RMR-Net residual policies, NAFNet-road, DFPIR, DeMoE, and InstructIR variants on all 49 matched annotated frames. It also writes residual-strength eta and no-reference tau gate sweeps.

The eta/tau curves are held-out policy audits. Validation-selected policies for deployment are implemented by:

```text
tools\tune_residual_policy.py
tools\tune_perception_gate.py
```

## Boundary Measurement

Post-detection active contours are run with:

```text
tools\snake_boundary_metrics.py
```

The output CSV records accepted/rejected contours, area, perimeter, compactness, edge alignment, contrast, and failure reason. PCM polygons are preserved for IoU, Dice, boundary-F1, Chamfer, and Hausdorff audits where available.

## Baseline Reporting

Baselines are reported with their routing assumptions:

- DFPIR: official restoration baseline.
- DeMoE-auto: official automatic router.
- DeMoE-scenario: known-scenario routing, an oracle-like degradation-routing upper bound.
- NAFNet-road: road-trained efficient baseline.
- Raw/degraded input: pass-through baseline.

Runtime tables must report backend status and should not mix CPU-forced and GPU-confirmed timings in the same speed ranking.

## Release Packaging

Run:

```powershell
.\.venv\Scripts\python.exe tools\package_trc_final_release.py
```

The TRC package command creates clean paper, code, and evidence zips with the current paper-facing tables, figures, provenance files, and GT49 YOLO26 evidence.
