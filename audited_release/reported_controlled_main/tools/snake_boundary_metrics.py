# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage import exposure
from skimage.filters import frangi
from skimage.measure import perimeter
from skimage.morphology import dilation, disk, remove_small_objects, remove_small_holes
from skimage.segmentation import morphological_chan_vese


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
warnings.filterwarnings("ignore", message="Parameter .* is deprecated.*", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optional detector-box active-contour refinement for road defects. "
            "The script reads YOLO-style boxes, runs a guarded MorphACWE snake "
            "inside each box crop, and writes overlays plus area/perimeter metrics."
        )
    )
    parser.add_argument("--data", required=True, help="YOLO data.yaml for the image set to analyze.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Image split inside data.yaml.")
    parser.add_argument("--out", required=True, help="Output folder for CSV metrics and overlays.")
    parser.add_argument("--box-dir", help="Optional YOLO txt box directory. Defaults to labels/<split> next to images/<split>.")
    parser.add_argument(
        "--gt-polygon-dir",
        help=(
            "Optional directory containing original YOLO-seg polygon labels. "
            "When provided and --box-dir uses matching GT order, the CSV also "
            "reports IoU/Dice/boundary-F/Chamfer/Hausdorff against polygons."
        ),
    )
    parser.add_argument("--classes", default="all", help="Comma list of class names or ids to process, or 'all'.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images for a quick audit. 0 means all images.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Shuffle images with this seed before --max-images. 0 preserves sorted order.")
    parser.add_argument("--pad-ratio", type=float, default=0.22, help="Context padding around each detector box.")
    parser.add_argument("--iterations", type=int, default=70, help="Morphological Chan-Vese iterations.")
    parser.add_argument("--smoothing", type=int, default=2, help="Morphological smoothing per iteration.")
    parser.add_argument("--min-area", type=int, default=18, help="Minimum accepted snake mask area in pixels.")
    parser.add_argument("--max-overlays", type=int, default=18, help="Maximum overlay panels included in the montage.")
    parser.add_argument("--thumb-width", type=int, default=360, help="Montage thumbnail width.")
    return parser.parse_args()


def load_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse as a YOLO data.yaml dictionary")
    return data


def normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError("data.yaml names must be a list or id:name dictionary")


def resolve_yolo_path(data_yaml: Path, data: dict, key: str) -> Path:
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = data.get(key, f"images/{key}")
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path
    return split_path.resolve()


def default_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def selected_class_ids(selection: str, names: dict[int, str]) -> set[int]:
    if selection.strip().lower() == "all":
        return set(names)
    wanted: set[int] = set()
    by_name = {name.lower(): idx for idx, name in names.items()}
    for token in selection.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            wanted.add(int(token))
        else:
            key = token.lower()
            if key not in by_name:
                raise ValueError(f"Unknown class '{token}'. Available: {sorted(by_name)}")
            wanted.add(by_name[key])
    return wanted


def read_yolo_boxes(label_path: Path, image_shape: tuple[int, int], keep_ids: set[int]) -> list[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    height, width = image_shape
    with label_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            if class_id not in keep_ids:
                continue
            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
            confidence = float(parts[5]) if len(parts) > 5 else None
            x1 = int(round((cx - bw / 2.0) * width))
            y1 = int(round((cy - bh / 2.0) * height))
            x2 = int(round((cx + bw / 2.0) * width))
            y2 = int(round((cy + bh / 2.0) * height))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "line_index": line_index,
                    "box": (x1, y1, x2, y2),
                }
            )
    return boxes


def read_polygon_line(label_path: Path, line_index: int) -> tuple[int, list[float]] | None:
    if not label_path.exists():
        return None
    lines = label_path.read_text(encoding="utf-8").splitlines()
    if line_index < 0 or line_index >= len(lines):
        return None
    parts = lines[line_index].strip().split()
    if len(parts) < 5:
        return None
    return int(float(parts[0])), [float(v) for v in parts[1:]]


def polygon_mask_from_line(
    label_path: Path,
    line_index: int,
    image_shape: tuple[int, int],
    crop_bounds: tuple[int, int, int, int],
) -> np.ndarray | None:
    item = read_polygon_line(label_path, line_index)
    if item is None:
        return None
    _class_id, values = item
    height, width = image_shape
    cx1, cy1, cx2, cy2 = crop_bounds
    crop_h = cy2 - cy1 + 1
    crop_w = cx2 - cx1 + 1
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    if len(values) == 4:
        x, y, w, h = values
        x1 = (x - w / 2.0) * width
        y1 = (y - h / 2.0) * height
        x2 = (x + w / 2.0) * width
        y2 = (y + h / 2.0) * height
        pts = np.array(
            [
                [x1 - cx1, y1 - cy1],
                [x2 - cx1, y1 - cy1],
                [x2 - cx1, y2 - cy1],
                [x1 - cx1, y2 - cy1],
            ],
            dtype=np.float32,
        )
    else:
        pts = np.array(
            [[values[i] * width - cx1, values[i + 1] * height - cy1] for i in range(0, len(values) - 1, 2)],
            dtype=np.float32,
        )
    if len(pts) < 3:
        return None
    pts[:, 0] = np.clip(pts[:, 0], 0, crop_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, crop_h - 1)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask_u8)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    return boundary > 0


def distance_metrics(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> dict[str, float | str]:
    if gt is None:
        return {
            "gt_iou": "",
            "gt_dice": "",
            "gt_boundary_f1": "",
            "gt_chamfer_px": "",
            "gt_hausdorff_px": "",
        }
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    iou = inter / union if union else 0.0
    dice = (2.0 * inter) / (pred_area + gt_area) if pred_area + gt_area else 0.0

    pred_b = mask_boundary(pred)
    gt_b = mask_boundary(gt)
    if not pred_b.any() or not gt_b.any():
        boundary_f1 = 0.0
        chamfer = 0.0
        hausdorff = 0.0
    else:
        gt_dist = cv2.distanceTransform((~gt_b).astype(np.uint8), cv2.DIST_L2, 3)
        pred_dist = cv2.distanceTransform((~pred_b).astype(np.uint8), cv2.DIST_L2, 3)
        pred_to_gt = gt_dist[pred_b]
        gt_to_pred = pred_dist[gt_b]
        precision = float((pred_to_gt <= tolerance).mean()) if pred_to_gt.size else 0.0
        recall = float((gt_to_pred <= tolerance).mean()) if gt_to_pred.size else 0.0
        boundary_f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        chamfer = 0.5 * (float(pred_to_gt.mean()) + float(gt_to_pred.mean()))
        hausdorff = max(float(pred_to_gt.max()), float(gt_to_pred.max()))
    return {
        "gt_iou": iou,
        "gt_dice": dice,
        "gt_boundary_f1": boundary_f1,
        "gt_chamfer_px": chamfer,
        "gt_hausdorff_px": hausdorff,
    }


def padded_crop_bounds(box: tuple[int, int, int, int], width: int, height: int, pad_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width - 1, x2 + pad_x),
        min(height - 1, y2 + pad_y),
    )


def initial_level_set(crop_shape: tuple[int, int], box_in_crop: tuple[int, int, int, int]) -> np.ndarray:
    height, width = crop_shape
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = box_in_crop
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    rx = max(3.0, 0.48 * (x2 - x1 + 1))
    ry = max(3.0, 0.48 * (y2 - y1 + 1))
    yy, xx = np.ogrid[:height, :width]
    mask[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0] = True
    return mask


def box_mask(crop_shape: tuple[int, int], box_in_crop: tuple[int, int, int, int], pad: int = 2) -> np.ndarray:
    height, width = crop_shape
    x1, y1, x2, y2 = box_in_crop
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(width - 1, x2 + pad), min(height - 1, y2 + pad)
    mask = np.zeros((height, width), dtype=bool)
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if min(gray.shape) >= 24:
        gray = exposure.equalize_adapthist(gray, clip_limit=0.015).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return np.clip(gray, 0.0, 1.0)


def crack_level_set(gray: np.ndarray, box_in_crop: tuple[int, int, int, int], min_area: int) -> np.ndarray:
    """Crack-aware active contour on ridge saliency.

    Region-based snakes tend to fill a whole crack bounding box because cracks
    are long, thin, and low contrast. For crack classes we therefore evolve the
    contour on a dark-ridge saliency map, initialized from the strongest ridges
    inside the detector box. It is still a guarded active-contour step, but with
    an image force that matches the line-like defect geometry.
    """

    inside_box = box_mask(gray.shape, box_in_crop, pad=3)
    ridge = frangi(gray, sigmas=range(1, 5), black_ridges=True)
    ridge = np.nan_to_num(ridge.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if ridge.max() > ridge.min():
        ridge = (ridge - ridge.min()) / (ridge.max() - ridge.min())
    values = ridge[inside_box]
    if values.size == 0 or float(values.max()) <= 1e-6:
        return initial_level_set(gray.shape, box_in_crop)
    threshold = max(float(np.quantile(values, 0.82)), float(values.mean() + 0.35 * values.std()))
    init = (ridge >= threshold) & inside_box
    init = dilation(init, footprint=disk(1))
    init = remove_small_objects(init, min_size=max(4, min_area // 3))
    if int(init.sum()) < min_area:
        init = initial_level_set(gray.shape, box_in_crop)
    try:
        mask = morphological_chan_vese(ridge, 28, init_level_set=init, smoothing=1, lambda1=1.6, lambda2=1.0)
    except TypeError:
        mask = morphological_chan_vese(ridge, num_iter=28, init_level_set=init, smoothing=1, lambda1=1.6, lambda2=1.0)
    mask = np.asarray(mask, dtype=bool) & inside_box
    mask = remove_small_objects(mask, min_size=max(4, min_area // 2))
    return dilation(mask, footprint=disk(1))


def boundary_quality(gray: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not mask.any():
        return 0.0, 0.0
    grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary = np.zeros_like(mask_u8)
    cv2.drawContours(boundary, contours, -1, 255, 1)
    boundary_pixels = boundary > 0
    if boundary_pixels.any():
        edge_alignment = float(grad[boundary_pixels].mean() / (grad.mean() + 1e-6))
    else:
        edge_alignment = 0.0
    inside = gray[mask]
    outside = gray[~mask]
    if inside.size == 0 or outside.size == 0:
        contrast = 0.0
    else:
        contrast = float(abs(float(inside.mean()) - float(outside.mean())) / (float(gray.std()) + 1e-6))
    return edge_alignment, contrast


def refine_box_with_snake(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    class_name: str,
    *,
    pad_ratio: float,
    iterations: int,
    smoothing: int,
    min_area: int,
) -> tuple[np.ndarray, dict]:
    height, width = image_bgr.shape[:2]
    cx1, cy1, cx2, cy2 = padded_crop_bounds(box, width, height, pad_ratio)
    crop = image_bgr[cy1 : cy2 + 1, cx1 : cx2 + 1]
    gray = preprocess_crop(crop)
    x1, y1, x2, y2 = box
    box_in_crop = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
    crack_like = "crack" in class_name.lower()
    if crack_like:
        mask = crack_level_set(gray, box_in_crop, min_area)
    else:
        init = initial_level_set(gray.shape, box_in_crop)
        try:
            mask = morphological_chan_vese(gray, iterations, init_level_set=init, smoothing=smoothing)
        except TypeError:
            mask = morphological_chan_vese(gray, num_iter=iterations, init_level_set=init, smoothing=smoothing)
        mask = np.asarray(mask, dtype=bool)
        mask = remove_small_objects(mask, min_size=max(4, min_area // 2))
        mask = remove_small_holes(mask, area_threshold=max(8, min_area))

    area = int(mask.sum())
    perim = float(perimeter(mask, neighborhood=8)) if area > 0 else 0.0
    crop_area = int(mask.shape[0] * mask.shape[1])
    bbox_area = int((x2 - x1 + 1) * (y2 - y1 + 1))
    area_bbox_ratio = area / max(bbox_area, 1)
    area_crop_ratio = area / max(crop_area, 1)
    compactness = 4.0 * math.pi * area / (perim * perim) if perim > 0 else 0.0
    edge_alignment, contrast = boundary_quality(gray, mask)
    touches_border = bool(
        mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    )

    failure_reasons = []
    if area < min_area:
        failure_reasons.append("too_small")
    max_crop_fill = 0.52 if crack_like else 0.88
    if area_crop_ratio > max_crop_fill:
        failure_reasons.append("leaked_to_crop")
    if touches_border:
        failure_reasons.append("touches_padding_border")
    if perim <= 0:
        failure_reasons.append("no_perimeter")
    success = not failure_reasons

    stats = {
        "crop_bounds": (cx1, cy1, cx2, cy2),
        "bbox_area_px": bbox_area,
        "crop_area_px": crop_area,
        "mask_area_px": area,
        "perimeter_px": perim,
        "compactness": compactness,
        "edge_alignment": edge_alignment,
        "foreground_background_contrast": contrast,
        "area_bbox_ratio": area_bbox_ratio,
        "area_crop_ratio": area_crop_ratio,
        "touches_padding_border": touches_border,
        "success": success,
        "failure_reason": ";".join(failure_reasons) if failure_reasons else "",
    }
    return mask, stats


def draw_overlay(
    image_bgr: np.ndarray,
    rows_for_image: list[dict],
    masks: list[np.ndarray],
    out_path: Path,
) -> None:
    overlay = image_bgr.copy()
    canvas = image_bgr.copy()
    for row, mask in zip(rows_for_image, masks):
        x1, y1, x2, y2 = [int(row[k]) for k in ("x1", "y1", "x2", "y2")]
        cx1, cy1, cx2, cy2 = [int(row[k]) for k in ("crop_x1", "crop_y1", "crop_x2", "crop_y2")]
        color = (64, 220, 255) if row["success"] else (60, 70, 255)
        mask_u8 = (mask.astype(np.uint8) * 255)
        colored = np.zeros_like(overlay[cy1 : cy2 + 1, cx1 : cx2 + 1])
        colored[:, :] = color
        roi = overlay[cy1 : cy2 + 1, cx1 : cx2 + 1]
        if mask.any():
            blended = cv2.addWeighted(roi[mask], 0.45, colored[mask], 0.55, 0)
            if blended is not None:
                roi[mask] = blended
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shifted = [cnt + np.array([[[cx1, cy1]]]) for cnt in contours]
        cv2.drawContours(canvas, shifted, -1, color, 2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
        text = f"{row['class_name']} {int(row['mask_area_px'])}px"
        cv2.putText(canvas, text, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    combined = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


def build_montage(overlay_paths: list[Path], out_path: Path, thumb_width: int) -> None:
    images = []
    for path in overlay_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = thumb_width / max(image.shape[1], 1)
        thumb = cv2.resize(image, (thumb_width, max(1, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        title = path.stem[:42]
        cv2.putText(thumb, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(thumb, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        images.append(thumb)
    if not images:
        return
    cols = 2 if len(images) <= 4 else 3
    rows = math.ceil(len(images) / cols)
    max_h = max(img.shape[0] for img in images)
    tile_w = thumb_width
    tile_h = max_h
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y0 = r * tile_h
        x0 = c * tile_w
        canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["class_name"]].append(row)
    summary = {
        "objects": len(rows),
        "successes": sum(1 for row in rows if row["success"]),
        "classes": {},
    }
    summary["success_rate"] = summary["successes"] / max(summary["objects"], 1)
    for class_name, class_rows in groups.items():
        ok_rows = [row for row in class_rows if row["success"]]
        summary["classes"][class_name] = {
            "objects": len(class_rows),
            "successes": len(ok_rows),
            "success_rate": len(ok_rows) / max(len(class_rows), 1),
            "mean_area_px": float(np.mean([row["mask_area_px"] for row in ok_rows])) if ok_rows else 0.0,
            "mean_perimeter_px": float(np.mean([row["perimeter_px"] for row in ok_rows])) if ok_rows else 0.0,
            "mean_compactness": float(np.mean([row["compactness"] for row in ok_rows])) if ok_rows else 0.0,
            "mean_edge_alignment": float(np.mean([row["edge_alignment"] for row in ok_rows])) if ok_rows else 0.0,
            "mean_contrast": float(np.mean([row["foreground_background_contrast"] for row in ok_rows])) if ok_rows else 0.0,
            "mean_area_bbox_ratio": float(np.mean([row["area_bbox_ratio"] for row in ok_rows])) if ok_rows else 0.0,
        }
        gt_rows = [row for row in ok_rows if row.get("gt_iou", "") != ""]
        if gt_rows:
            summary["classes"][class_name].update(
                {
                    "mean_gt_iou": float(np.mean([float(row["gt_iou"]) for row in gt_rows])),
                    "mean_gt_dice": float(np.mean([float(row["gt_dice"]) for row in gt_rows])),
                    "mean_gt_boundary_f1": float(np.mean([float(row["gt_boundary_f1"]) for row in gt_rows])),
                    "mean_gt_chamfer_px": float(np.mean([float(row["gt_chamfer_px"]) for row in gt_rows])),
                    "mean_gt_hausdorff_px": float(np.mean([float(row["gt_hausdorff_px"]) for row in gt_rows])),
                }
            )
    return summary


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    data = load_data_yaml(data_yaml)
    names = normalize_names(data["names"])
    keep_ids = selected_class_ids(args.classes, names)
    image_dir = resolve_yolo_path(data_yaml, data, args.split)
    label_dir = Path(args.box_dir).resolve() if args.box_dir else default_label_dir(image_dir)
    gt_polygon_dir = Path(args.gt_polygon_dir).resolve() if args.gt_polygon_dir else None
    out_dir = Path(args.out).resolve()
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if args.sample_seed:
        rng = np.random.default_rng(args.sample_seed)
        image_paths = [image_paths[i] for i in rng.permutation(len(image_paths))]
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    rows: list[dict] = []
    overlay_paths: list[Path] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = read_yolo_boxes(label_path, image.shape[:2], keep_ids)
        if not boxes:
            continue
        rows_for_image: list[dict] = []
        masks: list[np.ndarray] = []
        for object_index, item in enumerate(boxes):
            mask, stats = refine_box_with_snake(
                image,
                item["box"],
                names[item["class_id"]],
                pad_ratio=args.pad_ratio,
                iterations=args.iterations,
                smoothing=args.smoothing,
                min_area=args.min_area,
            )
            x1, y1, x2, y2 = item["box"]
            cx1, cy1, cx2, cy2 = stats["crop_bounds"]
            gt_metrics = {
                "gt_iou": "",
                "gt_dice": "",
                "gt_boundary_f1": "",
                "gt_chamfer_px": "",
                "gt_hausdorff_px": "",
            }
            if gt_polygon_dir is not None:
                gt_label_path = gt_polygon_dir / f"{image_path.stem}.txt"
                gt_mask = polygon_mask_from_line(gt_label_path, item["line_index"], image.shape[:2], stats["crop_bounds"])
                gt_metrics = distance_metrics(mask, gt_mask)
            row = {
                "image": image_path.name,
                "label_file": str(label_path),
                "class_id": item["class_id"],
                "class_name": names[item["class_id"]],
                "confidence": "" if item["confidence"] is None else item["confidence"],
                "object_index": object_index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_x1": cx1,
                "crop_y1": cy1,
                "crop_x2": cx2,
                "crop_y2": cy2,
                **{key: value for key, value in stats.items() if key != "crop_bounds"},
                **gt_metrics,
            }
            rows_for_image.append(row)
            rows.append(row)
            masks.append(mask)
        overlay_path = overlay_dir / f"{image_path.stem}_snake.png"
        draw_overlay(image, rows_for_image, masks, overlay_path)
        if len(overlay_paths) < args.max_overlays:
            overlay_paths.append(overlay_path)

    csv_path = out_dir / "snake_boundary_metrics.csv"
    fieldnames = [
        "image",
        "label_file",
        "class_id",
        "class_name",
        "confidence",
        "object_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "bbox_area_px",
        "crop_area_px",
        "mask_area_px",
        "perimeter_px",
        "compactness",
        "edge_alignment",
        "foreground_background_contrast",
        "area_bbox_ratio",
        "area_crop_ratio",
        "touches_padding_border",
        "success",
        "failure_reason",
        "gt_iou",
        "gt_dice",
        "gt_boundary_f1",
        "gt_chamfer_px",
        "gt_hausdorff_px",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary.update(
        {
            "data_yaml": str(data_yaml),
            "image_dir": str(image_dir),
            "box_dir": str(label_dir),
            "gt_polygon_dir": "" if gt_polygon_dir is None else str(gt_polygon_dir),
            "split": args.split,
            "max_images": args.max_images,
            "sample_seed": args.sample_seed,
            "selected_classes": [names[idx] for idx in sorted(keep_ids)],
            "iterations": args.iterations,
            "smoothing": args.smoothing,
            "pad_ratio": args.pad_ratio,
            "csv": str(csv_path),
        }
    )
    with (out_dir / "snake_boundary_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    build_montage(overlay_paths, out_dir / "snake_overlay_montage.png", args.thumb_width)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
