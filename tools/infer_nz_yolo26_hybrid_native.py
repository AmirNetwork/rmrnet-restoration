"""YOLO26 hybrid native-resolution inference for NZ road images.

This is a clean detector-review script for the 4752x3168 NZ sample images.
It uses the stronger YOLO26 road-damage checkpoint with the GT49-style hybrid
strategy:

    full-frame inference for global road context
    + center-safe large tiles for high-resolution evidence
    + class-aware NMS
    + defect-first crack-group fusion for review overlays

The script writes native-resolution overlays, CSV predictions, and a quick atlas.
It intentionally separates two views:

* typed: original YOLO26 classes are shown.
* defect_first: longitudinal/transverse/alligator cracks are merged into one
  crack region; potholes remain separate. This is usually the more useful view
  when subtype predictions are unstable.
* defect_all_fused: all YOLO26 defect classes are merged into operational
  defect regions. This is best when the detector localizes damage but subtype
  labels are unreliable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CRACK_CLASSES = {0, 1, 2}
POTHOLE_CLASS = 3

NAMES = {
    0: "longitudinal",
    1: "transverse",
    2: "alligator",
    3: "pothole",
    100: "crack",
    200: "defect",
}

COLORS = {
    0: (0, 170, 255),
    1: (0, 210, 120),
    2: (255, 90, 70),
    3: (255, 205, 0),
    100: (255, 115, 0),
    200: (255, 40, 170),
}


@dataclass
class Box:
    image: str
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float
    source: str
    label_extra: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument(
        "--input-dir",
        default=Path("experiments/nz_cracks_yolo26_gt49_detector/input_images/NZ-cracks"),
        type=Path,
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--full-imgsz", type=int, default=1536)
    parser.add_argument("--tile-imgsz", type=int, default=1536)
    parser.add_argument("--tile", type=int, default=3072)
    parser.add_argument("--overlap", type=int, default=768)
    parser.add_argument("--center-margin", type=int, default=384)
    parser.add_argument("--full-conf", type=float, default=0.5)
    parser.add_argument("--tile-conf", type=float, default=0.5)
    parser.add_argument("--branch-iou", type=float, default=0.75)
    parser.add_argument("--typed-nms", type=float, default=0.45)
    parser.add_argument("--crack-merge-iou", type=float, default=0.05)
    parser.add_argument("--crack-merge-ioa", type=float, default=0.22)
    parser.add_argument("--crack-merge-distance", type=float, default=180.0)
    parser.add_argument("--min-conf-typed", type=float, default=0.5)
    parser.add_argument("--min-conf-review", type=float, default=0.5)
    parser.add_argument("--all-defect-merge-distance", type=float, default=240.0)
    parser.add_argument("--refine-review-boxes", action="store_true")
    parser.add_argument("--road-filter", action="store_true")
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def image_paths(input_dir: Path) -> list[Path]:
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images in {input_dir}")
    return paths


def tile_positions(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    values = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if values[-1] != last:
        values.append(last)
    return values


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = box_area(a) + box_area(b) - inter
    return inter / denom if denom > 0 else 0.0


def ioa_min(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = max(min(box_area(a), box_area(b)), 1e-6)
    return inter / small


def center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return math.hypot((ax1 + ax2 - bx1 - bx2) * 0.5, (ay1 + ay2 - by1 - by2) * 0.5)


def union_box(boxes: Iterable[Box], image: str, cls: int, source: str, label_extra: str = "") -> Box:
    items = list(boxes)
    x1 = min(b.xyxy[0] for b in items)
    y1 = min(b.xyxy[1] for b in items)
    x2 = max(b.xyxy[2] for b in items)
    y2 = max(b.xyxy[3] for b in items)
    conf = max(b.conf for b in items)
    return Box(image=image, cls=cls, xyxy=(x1, y1, x2, y2), conf=conf, source=source, label_extra=label_extra)


def subtype_summary(boxes: Iterable[Box]) -> str:
    names = []
    for box in boxes:
        name = NAMES.get(box.cls, str(box.cls))
        if name not in names:
            names.append(name)
    return " + ".join(names)


def class_aware_nms(boxes: list[Box], threshold: float) -> list[Box]:
    kept: list[Box] = []
    for cls in sorted({b.cls for b in boxes}):
        todo = sorted([b for b in boxes if b.cls == cls], key=lambda b: b.conf, reverse=True)
        while todo:
            best = todo.pop(0)
            kept.append(best)
            todo = [b for b in todo if iou(best.xyxy, b.xyxy) < threshold]
    return sorted(kept, key=lambda b: b.conf, reverse=True)


def center_safe(
    box: tuple[float, float, float, float],
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    image_w: int,
    image_h: int,
    margin: int,
) -> bool:
    x1, y1, x2, y2 = box
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    if x0 > 0 and cx < margin:
        return False
    if y0 > 0 and cy < margin:
        return False
    if x0 + crop_w < image_w and cx > crop_w - margin:
        return False
    if y0 + crop_h < image_h and cy > crop_h - margin:
        return False
    return True


def estimate_road_roi(rgb: np.ndarray) -> np.ndarray:
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
        if np.any(ys > int(0.84 * height)) or np.any((ys > int(0.33 * height)) & (xs > int(0.20 * width)) & (xs < int(0.80 * width))):
            keep[comp] = 255
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0


def box_center_in_mask(box: tuple[float, float, float, float], mask: np.ndarray) -> bool:
    h, w = mask.shape
    x1, y1, x2, y2 = box
    cx = int(np.clip(0.5 * (x1 + x2), 0, w - 1))
    cy = int(np.clip(0.5 * (y1 + y2), 0, h - 1))
    return bool(mask[cy, cx])


def predict_array(
    model: YOLO,
    image: np.ndarray | str,
    image_name: str,
    imgsz: int,
    conf: float,
    iou_thr: float,
    device: str,
    source: str,
    offset: tuple[int, int] = (0, 0),
    max_det: int = 500,
) -> list[Box]:
    result = model.predict(
        image,
        imgsz=imgsz,
        conf=conf,
        iou=iou_thr,
        device=device,
        max_det=max_det,
        verbose=False,
    )[0]
    if result.boxes is None:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    ox, oy = offset
    boxes: list[Box] = []
    for coords, c, score in zip(xyxy, cls, scores):
        x1, y1, x2, y2 = [float(v) for v in coords.tolist()]
        boxes.append(
            Box(
                image=image_name,
                cls=int(c),
                xyxy=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                conf=float(score),
                source=source,
                label_extra=NAMES.get(int(c), str(int(c))),
            )
        )
    return boxes


def hybrid_predict(model: YOLO, image_path: Path, args: argparse.Namespace) -> tuple[list[Box], np.ndarray | None]:
    pil = Image.open(image_path).convert("RGB")
    width_orig, height_orig = pil.size
    pil = pil.crop((0, height_orig // 2, width_orig, height_orig))
    
    rgb = np.array(pil)
    height, width = rgb.shape[:2]
    roi = estimate_road_roi(rgb) if args.road_filter else None

    boxes = predict_array(
        model,
        rgb,
        image_path.name,
        args.full_imgsz,
        args.full_conf,
        args.branch_iou,
        args.device,
        "full",
        max_det=args.max_det,
    )

    for y0 in tile_positions(height, args.tile, args.overlap):
        for x0 in tile_positions(width, args.tile, args.overlap):
            crop = rgb[y0 : min(y0 + args.tile, height), x0 : min(x0 + args.tile, width)]
            ch, cw = crop.shape[:2]
            local = predict_array(
                model,
                crop,
                image_path.name,
                args.tile_imgsz,
                args.tile_conf,
                args.branch_iou,
                args.device,
                "tile",
                offset=(x0, y0),
                max_det=args.max_det,
            )
            for b in local:
                local_box = (b.xyxy[0] - x0, b.xyxy[1] - y0, b.xyxy[2] - x0, b.xyxy[3] - y0)
                if center_safe(local_box, x0, y0, cw, ch, width, height, args.center_margin):
                    boxes.append(b)

    boxes = [b for b in boxes if b.conf >= args.min_conf_typed and b.cls in {0, 1, 2, 3}]
    if roi is not None:
        # Keep low-confidence detections only if centered on likely pavement.
        filtered = []
        for b in boxes:
            if b.conf >= 0.22 or box_center_in_mask(b.xyxy, roi):
                filtered.append(b)
        boxes = filtered
    return class_aware_nms(boxes, args.typed_nms), roi


def merge_crack_group(boxes: list[Box], args: argparse.Namespace) -> list[Box]:
    review: list[Box] = []
    cracks = [b for b in boxes if b.cls in CRACK_CLASSES and b.conf >= args.min_conf_review]
    potholes = [b for b in boxes if b.cls == POTHOLE_CLASS and b.conf >= args.min_conf_review]

    used = [False] * len(cracks)
    for i, box in enumerate(cracks):
        if used[i]:
            continue
        used[i] = True
        group = [box]
        changed = True
        while changed:
            changed = False
            union = union_box(group, box.image, 100, "crack_fused", subtype_summary(group))
            for j, other in enumerate(cracks):
                if used[j]:
                    continue
                same_region = (
                    iou(union.xyxy, other.xyxy) >= args.crack_merge_iou
                    or ioa_min(union.xyxy, other.xyxy) >= args.crack_merge_ioa
                    or center_distance(union.xyxy, other.xyxy) <= args.crack_merge_distance
                )
                if same_region:
                    used[j] = True
                    group.append(other)
                    changed = True
        review.append(union_box(group, box.image, 100, "crack_fused", subtype_summary(group)))

    review.extend(class_aware_nms(potholes, 0.35))
    return sorted(review, key=lambda b: b.conf, reverse=True)


def merge_all_defects(boxes: list[Box], args: argparse.Namespace) -> list[Box]:
    defects = [b for b in boxes if b.cls in {0, 1, 2, 3} and b.conf >= args.min_conf_review]
    used = [False] * len(defects)
    merged: list[Box] = []
    for i, box in enumerate(defects):
        if used[i]:
            continue
        used[i] = True
        group = [box]
        changed = True
        while changed:
            changed = False
            union = union_box(group, box.image, 200, "defect_fused", subtype_summary(group))
            for j, other in enumerate(defects):
                if used[j]:
                    continue
                same_region = (
                    iou(union.xyxy, other.xyxy) >= args.crack_merge_iou
                    or ioa_min(union.xyxy, other.xyxy) >= args.crack_merge_ioa
                    or center_distance(union.xyxy, other.xyxy) <= args.all_defect_merge_distance
                )
                if same_region:
                    used[j] = True
                    group.append(other)
                    changed = True
        merged.append(union_box(group, box.image, 200, "defect_fused", subtype_summary(group)))
    return sorted(merged, key=lambda b: b.conf, reverse=True)


def refine_boxes_with_local_evidence(image_path: Path, boxes: list[Box], min_size: int = 18) -> list[Box]:
    """Tighten review boxes around dark local contrast inside each YOLO box.

    This does not create new detections. It only adjusts review boxes so very
    loose YOLO regions better hug visible dark/edge evidence in the original
    native-resolution image.
    """

    if not boxes:
        return boxes
    pil = Image.open(image_path).convert("RGB")
    width_orig, height_orig = pil.size
    pil = pil.crop((0, height_orig // 2, width_orig, height_orig))
    rgb = np.asarray(pil)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(gray)
    dark = cv2.GaussianBlur(eq, (0, 0), 9) - eq
    dark = np.clip(dark, 0, 255).astype(np.uint8)
    grad = cv2.morphologyEx(eq, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    h, w = gray.shape
    refined: list[Box] = []
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy
        pad = 10
        ix1 = int(np.clip(math.floor(x1) - pad, 0, w - 1))
        iy1 = int(np.clip(math.floor(y1) - pad, 0, h - 1))
        ix2 = int(np.clip(math.ceil(x2) + pad, ix1 + 1, w))
        iy2 = int(np.clip(math.ceil(y2) + pad, iy1 + 1, h))
        patch_score = 0.65 * dark[iy1:iy2, ix1:ix2].astype(np.float32) + 0.35 * grad[iy1:iy2, ix1:ix2].astype(np.float32)
        if patch_score.size == 0:
            refined.append(b)
            continue
        threshold = max(float(np.percentile(patch_score, 82)), float(patch_score.mean() + 0.45 * patch_score.std()))
        mask = patch_score >= threshold
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            rx = int(stats[idx, cv2.CC_STAT_LEFT])
            ry = int(stats[idx, cv2.CC_STAT_TOP])
            rw = int(stats[idx, cv2.CC_STAT_WIDTH])
            rh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            candidates.append((area, rx, ry, rw, rh))
        if not candidates:
            refined.append(b)
            continue
        # Use all strong components so cracks split by texture still stay inside one box.
        xs1 = [c[1] for c in candidates]
        ys1 = [c[2] for c in candidates]
        xs2 = [c[1] + c[3] for c in candidates]
        ys2 = [c[2] + c[4] for c in candidates]
        nx1 = max(ix1, ix1 + min(xs1) - 8)
        ny1 = max(iy1, iy1 + min(ys1) - 8)
        nx2 = min(w, ix1 + max(xs2) + 8)
        ny2 = min(h, iy1 + max(ys2) + 8)
        if nx2 - nx1 < min_size or ny2 - ny1 < min_size:
            refined.append(b)
        else:
            refined.append(
                Box(
                    image=b.image,
                    cls=b.cls,
                    xyxy=(nx1, ny1, nx2, ny2),
                    conf=b.conf,
                    source=b.source + "_refined",
                    label_extra=b.label_extra,
                )
            )
    return refined


def draw_boxes(image_path: Path, boxes: list[Box], out_path: Path, title: str) -> None:
    img = Image.open(image_path).convert("RGB")
    width_orig, height_orig = img.size
    img = img.crop((0, height_orig // 2, width_orig, height_orig))
    
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
        small = ImageFont.truetype("arial.ttf", 21)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.rectangle((0, 0, img.width, 44), fill=(245, 245, 245))
    draw.text((12, 10), title, fill=(10, 10, 10), font=font)
    for b in boxes:
        color = COLORS.get(b.cls, (255, 255, 255))
        x1, y1, x2, y2 = b.xyxy
        # Rounded coordinates for drawing only; CSV keeps exact floats.
        coords = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        width = 5 if b.cls == 100 else 3
        draw.rectangle(coords, outline=color, width=width)
        name = NAMES.get(b.cls, b.cls)
        if b.cls in {100, 200} and b.label_extra:
            name = f"{name}: {b.label_extra}"
        label = f"{name} {b.conf:.2f}"
        tb = draw.textbbox((coords[0], max(45, coords[1] - 25)), label, font=small)
        draw.rectangle(tb, fill=(0, 0, 0))
        draw.text((tb[0], tb[1]), label, fill=color, font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def save_roi_overlay(image_path: Path, roi: np.ndarray | None, out_path: Path) -> None:
    if roi is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGBA")
    width_orig, height_orig = img.size
    img = img.crop((0, height_orig // 2, width_orig, height_orig))
    arr = np.zeros((img.height, img.width, 4), dtype=np.uint8)
    arr[..., 1] = 180
    arr[..., 3] = roi.astype(np.uint8) * 70
    Image.alpha_composite(img, Image.fromarray(arr, "RGBA")).convert("RGB").save(out_path, quality=92)


def make_atlas(paths: list[Path], out_path: Path, columns: int = 2) -> None:
    if not paths:
        return
    target_w = 960
    thumbs = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        scale = target_w / im.width
        thumb = im.resize((target_w, int(im.height * scale)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, thumb.height + 38), "white")
        canvas.paste(thumb, (0, 38))
        ImageDraw.Draw(canvas).text((8, 8), path.name, fill=(0, 0, 0))
        thumbs.append(canvas)
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    atlas = Image.new("RGB", (columns * cell_w, rows * cell_h), (245, 245, 245))
    for i, thumb in enumerate(thumbs):
        atlas.paste(thumb, ((i % columns) * cell_w, (i // columns) * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(out_path, quality=92)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    typed_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for image_path in image_paths(args.input_dir):
        print(f"[yolo26-hybrid] {image_path.name}")
        typed, roi = hybrid_predict(model, image_path, args)
        review = merge_crack_group(typed, args)
        all_fused = merge_all_defects(typed, args)
        if args.refine_review_boxes:
            review = refine_boxes_with_local_evidence(image_path, review)
            all_fused = refine_boxes_with_local_evidence(image_path, all_fused)

        draw_boxes(image_path, typed, args.out / "01_typed_full_resolution" / image_path.name, "YOLO26 hybrid typed")
        draw_boxes(image_path, review, args.out / "02_defect_first_fused_full_resolution" / image_path.name, "YOLO26 hybrid defect-first fused")
        draw_boxes(image_path, all_fused, args.out / "03_all_defects_fused_full_resolution" / image_path.name, "YOLO26 hybrid all defects fused")
        save_roi_overlay(image_path, roi, args.out / "03_road_roi_debug" / image_path.name)

        for b in typed:
            typed_rows.append(
                {
                    "image": image_path.name,
                    "view": "typed",
                    "class_id": b.cls,
                    "class_name": NAMES.get(b.cls, str(b.cls)),
                    "confidence": round(b.conf, 5),
                    "source": b.source,
                    "label_extra": b.label_extra,
                    "x1": round(b.xyxy[0], 2),
                    "y1": round(b.xyxy[1], 2),
                    "x2": round(b.xyxy[2], 2),
                    "y2": round(b.xyxy[3], 2),
                }
            )
        for b in review:
            review_rows.append(
                {
                    "image": image_path.name,
                    "view": "defect_first",
                    "class_id": b.cls,
                    "class_name": NAMES.get(b.cls, str(b.cls)),
                    "confidence": round(b.conf, 5),
                    "source": b.source,
                    "label_extra": b.label_extra,
                    "x1": round(b.xyxy[0], 2),
                    "y1": round(b.xyxy[1], 2),
                    "x2": round(b.xyxy[2], 2),
                    "y2": round(b.xyxy[3], 2),
                }
            )
        for b in all_fused:
            review_rows.append(
                {
                    "image": image_path.name,
                    "view": "all_defects_fused",
                    "class_id": b.cls,
                    "class_name": NAMES.get(b.cls, str(b.cls)),
                    "confidence": round(b.conf, 5),
                    "source": b.source,
                    "label_extra": b.label_extra,
                    "x1": round(b.xyxy[0], 2),
                    "y1": round(b.xyxy[1], 2),
                    "x2": round(b.xyxy[2], 2),
                    "y2": round(b.xyxy[3], 2),
                }
            )
        summary_rows.append(
            {
                "image": image_path.name,
                "typed_boxes": len(typed),
                "typed_crack_boxes": sum(1 for b in typed if b.cls in CRACK_CLASSES),
                "typed_pothole_boxes": sum(1 for b in typed if b.cls == POTHOLE_CLASS),
                "defect_first_boxes": len(review),
                "defect_first_crack_boxes": sum(1 for b in review if b.cls == 100),
                "defect_first_pothole_boxes": sum(1 for b in review if b.cls == POTHOLE_CLASS),
                "all_defects_fused_boxes": len(all_fused),
            }
        )

    write_csv(args.out / "typed_predictions.csv", typed_rows)
    write_csv(args.out / "defect_first_predictions.csv", review_rows)
    write_csv(args.out / "summary.csv", summary_rows)
    (args.out / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    make_atlas(sorted((args.out / "01_typed_full_resolution").glob("*.jpg")), args.out / "00_atlas" / "typed_atlas.jpg")
    make_atlas(sorted((args.out / "02_defect_first_fused_full_resolution").glob("*.jpg")), args.out / "00_atlas" / "defect_first_atlas.jpg")
    make_atlas(sorted((args.out / "03_all_defects_fused_full_resolution").glob("*.jpg")), args.out / "00_atlas" / "all_defects_fused_atlas.jpg")

    if args.zip:
        archive = shutil.make_archive(str(args.out.with_suffix("")), "zip", root_dir=args.out)
        print(f"[yolo26-hybrid] wrote {archive}")


if __name__ == "__main__":
    main()
