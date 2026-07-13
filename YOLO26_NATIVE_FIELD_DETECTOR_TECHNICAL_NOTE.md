# YOLO26 Coordinate Field Detector Note

This note documents the frozen detector used for the G46 native-resolution field audit in the paper.

## Checkpoint

Local file:

```text
Yolo26_coordinate/YOLO26s_RDD_FRDC_Distilled_v2.pt
```

Public model card:

```text
https://huggingface.co/TamAko783/YOLO26s_RDD_FRDC_Distilled_v2
```

The paper cites this model card. The checkpoint is treated as an external frozen detector, not as a contribution of RMR-Net.

## Why This Detector Was Used

Earlier local YOLOv8 and locally fine-tuned YOLO26 variants were inspected visually on native high-resolution road images. They either produced many false positives, missed longitudinal cracks and small potholes, or fragmented defects into many boxes. The public YOLO26s RDD-FRDC distilled checkpoint transferred better to the native field images, so the paper uses it as the frozen field detector.

The G46 labels are used only for evaluation. They are not used for detector fine-tuning, detector threshold selection, restoration training, RMR-Net checkpoint selection, or metadata-policy tuning.

## Coordinate Wrapper

The executed wrapper is:

```text
Yolo26_coordinate/Yolo26_coordinate.py
```

Example:

```powershell
python Yolo26_coordinate\Yolo26_coordinate.py `
  --images path\to\native_or_restored_images `
  --out experiments\g46_coordinate\raw `
  --csv Yolo26_coordinate\precise_cam2_coords0.csv `
  --model Yolo26_coordinate\YOLO26s_RDD_FRDC_Distilled_v2.pt
```

The wrapper:

- runs the frozen detector directly on native images;
- keeps detections in original pixel coordinates;
- joins pose/geotag rows by camera filename where available;
- writes CSV/GeoJSON style outputs for inspection;
- does not alter detector weights.

## Classes

The shared evaluation taxonomy is:

| ID | Name |
| -: | ---- |
| 0 | D00 longitudinal crack |
| 1 | D10 transverse crack |
| 2 | D20 alligator crack |
| 3 | D40 pothole |

Only common classes shared by the detector and G46 annotations are evaluated.

## Metrics

Because G46 is a small, manually annotated, high-resolution field audit, exact box boundaries can be ambiguous. The paper reports conventional precision, recall, and F1 at IoU 0.10 and IoU 0.50, but it also reports a recall-oriented known-GT success score.

A ground-truth defect is counted as recovered when a same-primary-class prediction satisfies at least one rule:

1. IoU is at least 0.10.
2. The prediction covers at least 25 percent of the GT area.
3. The prediction contains the GT-box center.

This metric is not a replacement for AP on large benchmarks. It is a field-safety audit asking whether known annotated defects remain visible to the frozen detector after restoration.

## Paper Interpretation

The G46 set is a high-quality Sony native-image field test, not a severe natural-blur benchmark. The important deployment question is whether restoration preserves detector evidence on real native images. The paper therefore reports raw-native, full restoration, metadata-conditioned restoration, residual policy, and dual-evidence variants with clear claim boundaries.
