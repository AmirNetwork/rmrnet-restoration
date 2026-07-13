"""Native-resolution high-resolution detector evaluation for road-defect pilots.

Standard YOLO validation resizes a whole high-resolution image to one network
input size.  For the geotagged Cam1 pilot this can squash small cracks and
patches into a few pixels. Naive tiling, however, loses global road context and
fragments long cracks near tile borders. The released GT49 protocol therefore
supports a hybrid high-resolution strategy:

    full-image YOLO for global context
    + smart center-safe tiling for local detail
    + class-aware fusion NMS

* class-aware and class-agnostic greedy precision/recall at IoU 0.10/0.25/0.50
* crack-group-aware matching for detectors with crack subtypes
* ground-truth coverage by predicted regions (intersection / GT area)
* center-hit recall
* FROC-style recall at fixed false positives per image

It does not replace COCO mAP; it diagnoses whether a detector preserves useful
defect evidence under native-resolution deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Box:
    image: str
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0
    source: str = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--weights", required=True, help="YOLO detector weights")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--strategy",
        choices=["tiled", "smart_tile", "hybrid"],
        default="hybrid",
        help="High-resolution inference strategy. Hybrid is the GT49 release protocol.",
    )
    parser.add_argument("--tile", type=int, default=3072)
    parser.add_argument("--overlap", type=int, default=768)
    parser.add_argument("--center-margin", type=int, default=384)
    parser.add_argument("--infer-imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.03, help="Tile-branch prediction confidence.")
    parser.add_argument("--branch-iou", type=float, default=0.75, help="YOLO branch-level NMS IoU.")
    parser.add_argument("--full-imgsz", type=int, default=1536)
    parser.add_argument("--full-conf", type=float, default=0.03)
    parser.add_argument("--full-iou", type=float, default=0.75)
    parser.add_argument("--tile-fusion-conf", type=float, default=0.75)
    parser.add_argument("--nms-iou", type=float, default=0.45, help="Final class-aware fusion NMS IoU.")
    parser.add_argument(
        "--crop-bottom-half",
        action="store_true",
        help=(
            "Run YOLO26 only on the lower half of each native frame and map boxes "
            "back to full-resolution coordinates before GT49 evaluation."
        ),
    )
    parser.add_argument(
        "--road-filter",
        action="store_true",
        help=(
            "Apply the fixed pavement-likelihood gate used by the YOLO26 native-review protocol. "
            "Low-confidence boxes are kept only when their center lies on likely road surface; "
            "high-confidence boxes are retained regardless of the mask."
        ),
    )
    parser.add_argument(
        "--road-filter-high-conf",
        type=float,
        default=0.22,
        help="Confidence above which a prediction bypasses the road-surface gate.",
    )
    parser.add_argument(
        "--crack-classes",
        default="0,1,5",
        help="Comma-separated class IDs to treat as one crack group for crack_group metrics.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--save-overlays", action="store_true")
    return parser.parse_args()


def load_yolo_yaml(path: Path) -> tuple[Path, Path, dict[int, str]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(config.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    names_raw = config.get("names", {})
    if isinstance(names_raw, list):
        names = {idx: str(name) for idx, name in enumerate(names_raw)}
    else:
        names = {int(idx): str(name) for idx, name in dict(names_raw).items()}
    return root, config, names


def image_paths_from_yaml(data: Path, split: str) -> tuple[list[Path], Path, dict[int, str]]:
    root, config, names = load_yolo_yaml(data)
    image_dir = root / config[split]
    label_dir = root / str(config[split]).replace("images", "labels")
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return image_paths, label_dir, names


def read_labels(label_path: Path, image_name: str, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        boxes.append(Box(image=image_name, cls=cls, xyxy=(x1, y1, x2, y2)))
    return boxes


def grid_positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    positions = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def intersection_over_gt(pred: tuple[float, float, float, float], gt: tuple[float, float, float, float]) -> float:
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    return inter / gt_area if gt_area > 0 else 0.0


def parse_class_set(text: str) -> set[int]:
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def crack_group(cls: int, crack_classes: set[int]) -> int:
    # NewRoad six-class taxonomy uses {0,1,5}; YOLO26/RDD four-class taxonomy uses {0,1,2}.
    if cls in crack_classes:
        return 100
    return cls


def classes_match(pred_cls: int, gt_cls: int, mode: str, crack_classes: set[int]) -> bool:
    if mode == "agnostic":
        return True
    if mode == "crack_group":
        return crack_group(pred_cls, crack_classes) == crack_group(gt_cls, crack_classes)
    return pred_cls == gt_cls


def nms(boxes: list[Box], iou_thr: float) -> list[Box]:
    kept: list[Box] = []
    for cls in sorted({b.cls for b in boxes}):
        cls_boxes = sorted((b for b in boxes if b.cls == cls), key=lambda b: b.conf, reverse=True)
        while cls_boxes:
            best = cls_boxes.pop(0)
            kept.append(best)
            cls_boxes = [b for b in cls_boxes if box_iou(best.xyxy, b.xyxy) < iou_thr]
    return sorted(kept, key=lambda b: b.conf, reverse=True)


def center_is_safe(
    local_xyxy: tuple[float, float, float, float],
    *,
    tile_x: int,
    tile_y: int,
    crop_w: int,
    crop_h: int,
    image_w: int,
    image_h: int,
    margin: int,
) -> bool:
    """Reject detections near internal tile borders.

    External image borders are allowed. Only internal tile seams are guarded,
    which keeps the full image intact while avoiding duplicate/fragmented
    detections from partial border evidence.
    """

    if margin <= 0:
        return True
    x1, y1, x2, y2 = local_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    if tile_x > 0 and cx < margin:
        return False
    if tile_y > 0 and cy < margin:
        return False
    if tile_x + crop_w < image_w and cx > crop_w - margin:
        return False
    if tile_y + crop_h < image_h and cy > crop_h - margin:
        return False
    return True


def estimate_road_roi(image_path: Path) -> np.ndarray:
    """Estimate a conservative road-surface mask without using labels.

    This deterministic deployment prior is not fitted on GT49 labels. It only
    suppresses low-confidence detections whose centers fall outside likely
    pavement; high-confidence detector evidence bypasses the mask.
    """

    pil = Image.open(image_path).convert("RGB")
    width_orig, height_orig = pil.size
    pil = pil.crop((0, height_orig // 2, width_orig, height_orig))
    rgb = np.asarray(pil)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    chroma = maxc - minc
    exg = 2 * g - r - b
    height, width = rgb.shape[:2]
    yy = np.arange(height, dtype=np.float32)[:, None] / max(height - 1, 1)
    xx = np.arange(width, dtype=np.float32)[None, :] / max(width - 1, 1)
    green = (h > 26) & (h < 105) & (s > 24) & (exg > 8) & (g > r + 3)
    sky = (yy < 0.58) & (v > 130) & (s < 88) & (b > r + 3)
    grayish = (s < 88) | (chroma < 50)
    corridor = np.abs(xx - 0.50) < (0.10 + 0.72 * yy)
    seed = (grayish & corridor & (yy > 0.16) & ~green & ~sky).astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((seed > 0).astype(np.uint8), 8)
    keep = np.zeros_like(seed)
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(0.002 * height * width):
            continue
        comp = labels == idx
        ys, xs = np.where(comp)
        on_lower_road = np.any(ys > int(0.84 * height))
        on_center_road = np.any((ys > int(0.33 * height)) & (xs > int(0.20 * width)) & (xs < int(0.80 * width)))
        if on_lower_road or on_center_road:
            keep[comp] = 255
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0


def box_center_in_mask(box: tuple[float, float, float, float], mask: np.ndarray) -> bool:
    height, width = mask.shape
    x1, y1, x2, y2 = box
    cx = int(np.clip(0.5 * (x1 + x2), 0, width - 1))
    cy = int(np.clip(0.5 * (y1 + y2), 0, height - 1))
    return bool(mask[cy, cx])


def apply_road_filter(image_path: Path, preds: list[Box], high_conf: float) -> list[Box]:
    if not preds:
        return preds
    mask = estimate_road_roi(image_path)
    return [box for box in preds if box.conf >= high_conf or box_center_in_mask(box.xyxy, mask)]


def predict_full_image(
    model: YOLO,
    image_path: Path,
    infer_imgsz: int,
    conf: float,
    iou: float,
    device: str,
    *,
    crop_bottom_half: bool = False,
) -> list[Box]:
    y_offset = 0
    image_input: str | Image.Image = str(image_path)
    source = "full"
    if crop_bottom_half:
        pil = Image.open(image_path).convert("RGB")
        width_orig, height_orig = pil.size
        y_offset = height_orig // 2
        image_input = pil.crop((0, y_offset, width_orig, height_orig))
        source = "full_crop"
    result = model.predict(image_input, imgsz=infer_imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
    if result.boxes is None:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    return [
        Box(
            image=image_path.name,
            cls=int(c),
            xyxy=(float(coords[0]), float(coords[1]) + y_offset, float(coords[2]), float(coords[3]) + y_offset),
            conf=float(score),
            source=source,
        )
        for coords, c, score in zip(xyxy, cls, scores)
    ]


def predict_native_tiled(
    model: YOLO,
    image_path: Path,
    tile: int,
    overlap: int,
    infer_imgsz: int,
    conf: float,
    branch_iou: float,
    device: str,
    nms_iou: float,
    *,
    center_margin: int = 0,
    smart_center: bool = False,
    crop_bottom_half: bool = False,
) -> list[Box]:
    image = Image.open(image_path).convert("RGB")
    full_width, full_height = image.size
    y_offset = 0
    if crop_bottom_half:
        y_offset = full_height // 2
        image = image.crop((0, y_offset, full_width, full_height))
    width, height = image.size
    preds: list[Box] = []
    for y in grid_positions(height, tile, overlap):
        for x in grid_positions(width, tile, overlap):
            crop = image.crop((x, y, min(x + tile, width), min(y + tile, height)))
            result = model.predict(crop, imgsz=infer_imgsz, conf=conf, iou=branch_iou, device=device, verbose=False)[0]
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            cls = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            for coords, c, score in zip(xyxy, cls, scores):
                x1, y1, x2, y2 = coords.tolist()
                local = (float(x1), float(y1), float(x2), float(y2))
                if smart_center and not center_is_safe(
                    local,
                    tile_x=x,
                    tile_y=y,
                    crop_w=crop.width,
                    crop_h=crop.height,
                    image_w=width,
                    image_h=height,
                    margin=center_margin,
                ):
                    continue
                mapped = (
                    max(0.0, x1 + x),
                    max(0.0, y1 + y + y_offset),
                    min(float(width), x2 + x),
                    min(float(full_height), y2 + y + y_offset),
                )
                preds.append(Box(image=image_path.name, cls=int(c), xyxy=mapped, conf=float(score), source="tile"))
    return nms(preds, nms_iou)


def predict_hybrid(
    model: YOLO,
    image_path: Path,
    *,
    full_imgsz: int,
    full_conf: float,
    full_iou: float,
    tile: int,
    overlap: int,
    center_margin: int,
    infer_imgsz: int,
    tile_conf: float,
    branch_iou: float,
    tile_fusion_conf: float,
    fusion_nms_iou: float,
    device: str,
    crop_bottom_half: bool = False,
) -> list[Box]:
    full = predict_full_image(
        model,
        image_path,
        infer_imgsz=full_imgsz,
        conf=full_conf,
        iou=full_iou,
        device=device,
        crop_bottom_half=crop_bottom_half,
    )
    tiled = predict_native_tiled(
        model,
        image_path,
        tile=tile,
        overlap=overlap,
        infer_imgsz=infer_imgsz,
        conf=tile_conf,
        branch_iou=branch_iou,
        device=device,
        nms_iou=branch_iou,
        center_margin=center_margin,
        smart_center=True,
        crop_bottom_half=crop_bottom_half,
    )
    tiled = [box for box in tiled if box.conf >= tile_fusion_conf]
    return nms(full + tiled, fusion_nms_iou)


def greedy_match(gts: list[Box], preds: list[Box], iou_thr: float, mode: str, crack_classes: set[int]) -> dict:
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for pred in sorted(preds, key=lambda b: b.conf, reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gts):
            if idx in matched_gt or not classes_match(pred.cls, gt.cls, mode, crack_classes):
                continue
            iou = box_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = len(gts) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / len(gts) if gts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def coverage_recall(gts: list[Box], preds: list[Box], threshold: float, mode: str, crack_classes: set[int]) -> float:
    if not gts:
        return 0.0
    hit = 0
    for gt in gts:
        best = 0.0
        for pred in preds:
            if classes_match(pred.cls, gt.cls, mode, crack_classes):
                best = max(best, intersection_over_gt(pred.xyxy, gt.xyxy))
        hit += int(best >= threshold)
    return hit / len(gts)


def center_recall(gts: list[Box], preds: list[Box], mode: str, crack_classes: set[int]) -> float:
    if not gts:
        return 0.0
    hits = 0
    for gt in gts:
        gx1, gy1, gx2, gy2 = gt.xyxy
        cx, cy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5
        found = False
        for pred in preds:
            if not classes_match(pred.cls, gt.cls, mode, crack_classes):
                continue
            px1, py1, px2, py2 = pred.xyxy
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                found = True
                break
        hits += int(found)
    return hits / len(gts)


def froc_recalls(
    all_gts: dict[str, list[Box]],
    all_preds: dict[str, list[Box]],
    mode: str,
    crack_classes: set[int],
) -> dict[str, float]:
    thresholds = sorted({round(p.conf, 4) for preds in all_preds.values() for p in preds}, reverse=True)
    thresholds.append(1.01)
    budgets = [1, 2, 5, 10]
    best = {f"froc_recall_iou10_fp{b}_per_image": 0.0 for b in budgets}
    n_images = max(1, len(all_gts))
    for thr in thresholds:
        tps = fps = total_gt = 0
        for image_name, gts in all_gts.items():
            preds = [p for p in all_preds.get(image_name, []) if p.conf >= thr]
            counts = greedy_match(gts, preds, 0.10, mode, crack_classes)
            tps += counts["tp"]
            fps += counts["fp"]
            total_gt += len(gts)
        recall = tps / total_gt if total_gt else 0.0
        fppi = fps / n_images
        for budget in budgets:
            if fppi <= budget:
                key = f"froc_recall_iou10_fp{budget}_per_image"
                best[key] = max(best[key], recall)
    return best


def operating_curve_rows(
    all_gts: dict[str, list[Box]],
    all_preds: dict[str, list[Box]],
    thresholds: Iterable[float],
    crack_classes: set[int],
) -> list[dict]:
    """Return threshold-vs-recall/false-positive rows for operator calibration."""

    rows: list[dict] = []
    n_images = max(1, len(all_gts))
    flat_gts = [box for boxes in all_gts.values() for box in boxes]
    for threshold in thresholds:
        flat_preds = [
            box
            for boxes in all_preds.values()
            for box in boxes
            if box.conf >= threshold
        ]
        pred_count = len(flat_preds)
        for mode in ["aware", "crack_group", "agnostic"]:
            counts = greedy_match(flat_gts, flat_preds, 0.10, mode, crack_classes)
            rows.append(
                {
                    "confidence": threshold,
                    "mode": mode,
                    "predictions": pred_count,
                    "false_positives_per_image": counts["fp"] / n_images,
                    "iou10_precision": counts["precision"],
                    "iou10_recall": counts["recall"],
                    "coverage_ioa25_recall": coverage_recall(flat_gts, flat_preds, 0.25, mode, crack_classes),
                    "center_recall": center_recall(flat_gts, flat_preds, mode, crack_classes),
                }
            )
    return rows


def draw_overlay(image_path: Path, gts: list[Box], preds: list[Box], names: dict[int, str], out_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    width_orig, height_orig = image.size
    image = image.crop((0, height_orig // 2, width_orig, height_orig))
    width = 1400
    scale = width / image.width
    height = int(image.height * scale)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    def scale_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        return x1 * scale, y1 * scale, x2 * scale, y2 * scale

    for gt in gts:
        draw.rectangle(scale_box(gt.xyxy), outline=(255, 214, 10), width=3)
        x1, y1, _, _ = scale_box(gt.xyxy)
        draw.text((x1, max(0, y1 - 13)), f"GT {names.get(gt.cls, gt.cls)}", fill=(255, 214, 10), font=font)
    for pred in preds:
        draw.rectangle(scale_box(pred.xyxy), outline=(0, 220, 120), width=2)
        x1, y1, _, _ = scale_box(pred.xyxy)
        draw.text((x1, y1), f"{names.get(pred.cls, pred.cls)} {pred.conf:.2f}", fill=(0, 220, 120), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def write_atlas(overlay_paths: list[Path], out_path: Path) -> None:
    panels = [Image.open(p).convert("RGB") for p in overlay_paths]
    if not panels:
        return
    gap = 10
    width = max(p.width for p in panels)
    height = sum(p.height for p in panels) + gap * (len(panels) - 1)
    atlas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for panel in panels:
        atlas.paste(panel, (0, y))
        y += panel.height + gap
    atlas.save(out_path)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_paths, label_dir, names = image_paths_from_yaml(data_path, args.split)
    crack_classes = parse_class_set(args.crack_classes)
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    model = YOLO(args.weights)
    all_gts: dict[str, list[Box]] = {}
    all_preds: dict[str, list[Box]] = {}
    overlay_paths: list[Path] = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        gts_raw = read_labels(label_dir / f"{image_path.stem}.txt", image_path.name, width, height)
        if args.crop_bottom_half:
            # The modified YOLO26 native detector scans the road-focused lower
            # half only. Keep evaluation in original native coordinates, but
            # restrict GT to objects whose centers lie inside the scanned ROI.
            gts = [gt for gt in gts_raw if 0.5 * (gt.xyxy[1] + gt.xyxy[3]) >= height // 2]
        else:
            gts = gts_raw
        if args.strategy == "hybrid":
            preds = predict_hybrid(
                model,
                image_path,
                full_imgsz=args.full_imgsz,
                full_conf=args.full_conf,
                full_iou=args.full_iou,
                tile=args.tile,
                overlap=args.overlap,
                center_margin=args.center_margin,
                infer_imgsz=args.infer_imgsz,
                tile_conf=args.conf,
                branch_iou=args.branch_iou,
                tile_fusion_conf=args.tile_fusion_conf,
                fusion_nms_iou=args.nms_iou,
                device=args.device,
                crop_bottom_half=args.crop_bottom_half,
            )
        else:
            preds = predict_native_tiled(
                model,
                image_path,
                tile=args.tile,
                overlap=args.overlap,
                infer_imgsz=args.infer_imgsz,
                conf=args.conf,
                branch_iou=args.branch_iou,
                device=args.device,
                nms_iou=args.nms_iou,
                center_margin=args.center_margin,
                smart_center=args.strategy == "smart_tile",
                crop_bottom_half=args.crop_bottom_half,
            )
        if args.road_filter:
            preds = apply_road_filter(image_path, preds, args.road_filter_high_conf)
        all_gts[image_path.name] = gts
        all_preds[image_path.name] = preds
        if args.save_overlays:
            overlay_path = out / "overlays" / f"{image_path.stem}_native_{args.strategy}.png"
            draw_overlay(image_path, gts, preds, names, overlay_path)
            overlay_paths.append(overlay_path)

    metric_rows = []
    flat_gts = [box for boxes in all_gts.values() for box in boxes]
    flat_preds = [box for boxes in all_preds.values() for box in boxes]
    for mode in ["aware", "crack_group", "agnostic"]:
        for threshold in [0.10, 0.25, 0.50]:
            counts = greedy_match(flat_gts, flat_preds, threshold, mode, crack_classes)
            metric_rows.append({"metric": "iou_greedy", "mode": mode, "threshold": threshold, **counts})
        for threshold in [0.10, 0.25, 0.50]:
            metric_rows.append(
                {
                    "metric": "gt_coverage_ioa",
                    "mode": mode,
                    "threshold": threshold,
                    "recall": coverage_recall(flat_gts, flat_preds, threshold, mode, crack_classes),
                }
            )
        metric_rows.append(
            {
                "metric": "gt_center_inside_prediction",
                "mode": mode,
                "threshold": 0.0,
                "recall": center_recall(flat_gts, flat_preds, mode, crack_classes),
            }
        )
        for key, value in froc_recalls(all_gts, all_preds, mode, crack_classes).items():
            metric_rows.append({"metric": key, "mode": mode, "threshold": 0.10, "recall": value})

    metrics_path = out / "native_tiled_metrics.csv"
    fieldnames = ["metric", "mode", "threshold", "tp", "fp", "fn", "precision", "recall", "f1"]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metric_rows)

    pred_rows = []
    for image_name, preds in all_preds.items():
        for pred in preds:
            pred_rows.append(
                {
                    "image": image_name,
                    "class_id": pred.cls,
                    "class_name": names.get(pred.cls, str(pred.cls)),
                    "conf": pred.conf,
                    "source": pred.source,
                    "x1": pred.xyxy[0],
                    "y1": pred.xyxy[1],
                    "x2": pred.xyxy[2],
                    "y2": pred.xyxy[3],
                }
            )
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "class_id", "class_name", "conf", "source", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(pred_rows)

    op_rows = operating_curve_rows(
        all_gts,
        all_preds,
        thresholds=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60],
        crack_classes=crack_classes,
    )
    with (out / "operating_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "confidence",
                "mode",
                "predictions",
                "false_positives_per_image",
                "iou10_precision",
                "iou10_recall",
                "coverage_ioa25_recall",
                "center_recall",
            ],
        )
        writer.writeheader()
        writer.writerows(op_rows)

    summary = {
        "weights": args.weights,
        "data": args.data,
        "strategy": args.strategy,
        "images": len(image_paths),
        "gt_boxes": len(flat_gts),
        "pred_boxes": len(flat_preds),
        "full_imgsz": args.full_imgsz,
        "full_conf": args.full_conf,
        "full_iou": args.full_iou,
        "tile": args.tile,
        "overlap": args.overlap,
        "center_margin": args.center_margin,
        "infer_imgsz": args.infer_imgsz,
        "conf": args.conf,
        "branch_iou": args.branch_iou,
        "tile_fusion_conf": args.tile_fusion_conf,
        "nms_iou": args.nms_iou,
        "crop_bottom_half": bool(args.crop_bottom_half),
        "road_filter": bool(args.road_filter),
        "road_filter_high_conf": args.road_filter_high_conf,
        "crack_classes": sorted(crack_classes),
        "metrics_csv": str(metrics_path),
        "operating_points_csv": str(out / "operating_points.csv"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_overlays:
        write_atlas(overlay_paths, out / f"native_{args.strategy}_atlas.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
