"""Run native-resolution crack/road-defect segmentation on NZ field images.

This tool is intentionally separate from the paper detector scripts.  It is for
manual inspection of high-resolution NZ road images where polygon masks are more
useful than loose detector boxes.

The script preserves the original image resolution.  YOLO masks predicted on the
full frame and on overlapping tiles are mapped back to native pixel coordinates,
merged class-wise, lightly cleaned, and drawn as transparent overlays.
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

CLASS_NAMES = {
    0: "alligator",
    1: "longitudinal",
    2: "others",
    3: "pothole",
    4: "road_intersection",
    5: "transverse",
}

COLORS = {
    0: (232, 81, 64),     # alligator crack
    1: (0, 166, 214),     # longitudinal crack
    2: (155, 155, 155),   # other
    3: (255, 191, 0),     # pothole
    4: (145, 92, 182),    # road intersection / non-defect context
    5: (0, 173, 95),      # transverse crack
}


@dataclass
class Component:
    cls: int
    area: int
    bbox: tuple[int, int, int, int]
    mean_conf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="YOLO segmentation checkpoint.")
    parser.add_argument(
        "--input-dir",
        default=Path("experiments/nz_cracks_yolo26_gt49_detector/input_images/NZ-cracks"),
        type=Path,
        help="Directory containing native-resolution NZ images.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory.")
    parser.add_argument("--device", default="0", help="Ultralytics device string, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", default=1536, type=int, help="YOLO inference image size.")
    parser.add_argument("--tile", default=2048, type=int, help="Native tile size in pixels.")
    parser.add_argument("--overlap", default=640, type=int, help="Native tile overlap in pixels.")
    parser.add_argument("--center-margin", default=192, type=int, help="Reject internal tile-edge detections.")
    parser.add_argument("--full-conf", default=0.20, type=float, help="Confidence threshold for full-frame branch.")
    parser.add_argument("--tile-conf", default=0.12, type=float, help="Confidence threshold for tile branch.")
    parser.add_argument("--iou", default=0.70, type=float, help="YOLO branch NMS IoU.")
    parser.add_argument("--min-area", default=160, type=int, help="Minimum native-pixel component area.")
    parser.add_argument("--close", default=5, type=int, help="Morphological close kernel; 0 disables.")
    parser.add_argument("--merge-distance", default=42, type=int, help="Merge nearby same-class boxes for display.")
    parser.add_argument(
        "--road-filter",
        action="store_true",
        help="Suppress predicted masks outside a classical low-saturation pavement region.",
    )
    parser.add_argument(
        "--defect-classes",
        default="0,1,3,5",
        help="Comma-separated classes to draw as defects. Defaults exclude 'others' and road_intersection.",
    )
    parser.add_argument("--draw-nondefect", action="store_true", help="Also draw non-defect/context classes.")
    parser.add_argument("--zip", action="store_true", help="Create a flat zip next to the output folder.")
    return parser.parse_args()


def image_paths(input_dir: Path) -> list[Path]:
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return paths


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def center_safe(
    bbox: tuple[float, float, float, float],
    tile_origin: tuple[int, int],
    tile_shape: tuple[int, int],
    image_shape: tuple[int, int],
    margin: int,
) -> bool:
    """Keep tile detections that are not centered on internal tile borders."""

    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    x0, y0 = tile_origin
    tw, th = tile_shape
    width, height = image_shape

    if x0 > 0 and cx < margin:
        return False
    if y0 > 0 and cy < margin:
        return False
    if x0 + tw < width and cx > tw - margin:
        return False
    if y0 + th < height and cy > th - margin:
        return False
    return True


def fill_prediction_masks(
    model: YOLO,
    image: np.ndarray,
    class_masks: dict[int, np.ndarray],
    confidence_masks: dict[int, np.ndarray],
    *,
    offset: tuple[int, int],
    conf: float,
    imgsz: int,
    iou: float,
    device: str,
    keep_classes: set[int],
    tile_filter: tuple[tuple[int, int], tuple[int, int], tuple[int, int], int] | None = None,
) -> int:
    """Run YOLO on an RGB array and paint predicted polygons into global masks."""

    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        retina_masks=True,
        device=device,
        verbose=False,
    )
    result = results[0]
    if result.masks is None or result.boxes is None:
        return 0

    count = 0
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()
    polygons = result.masks.xy
    ox, oy = offset

    for cls, score, box, poly in zip(classes, confs, boxes, polygons):
        if cls not in keep_classes or poly is None or len(poly) < 3:
            continue
        if tile_filter is not None:
            tile_origin, tile_shape, image_shape, margin = tile_filter
            if not center_safe(tuple(box.tolist()), tile_origin, tile_shape, image_shape, margin):
                continue
        pts = np.asarray(poly, dtype=np.float32)
        pts[:, 0] += ox
        pts[:, 1] += oy
        pts = np.rint(pts).astype(np.int32)
        h, w = class_masks[cls].shape
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(class_masks[cls], [pts], 255)
        cv2.fillPoly(confidence_masks[cls], [pts], int(max(1, min(255, round(float(score) * 255)))))
        count += 1
    return count


def clean_mask(mask: np.ndarray, close_kernel: int) -> np.ndarray:
    if close_kernel and close_kernel > 1:
        k = int(close_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def estimate_pavement_mask(rgb: np.ndarray) -> np.ndarray:
    """Estimate a broad pavement ROI from color and image geometry.

    This is deliberately conservative and classical: it removes obvious sky,
    grass, shrubs, roofs, and poles before accepting defect masks.  It does not
    create detections; it only gates neural segmentation masks to road-like
    surfaces in native field imagery.
    """

    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    chroma = maxc - minc

    yy = np.arange(height, dtype=np.float32)[:, None] / max(float(height - 1), 1.0)
    xx = np.arange(width, dtype=np.float32)[None, :] / max(float(width - 1), 1.0)

    exg = 2 * g - r - b
    gray_like = (s_ch < 82) | (chroma < 38)
    vegetation_like = (
        (h_ch > 25)
        & (h_ch < 105)
        & (s_ch > 22)
        & (exg > 10)
        & (g > r + 4)
    )
    not_green = ~vegetation_like
    not_sky = ~((yy < 0.58) & (v_ch > 138) & (s_ch < 82) & (b > r + 4))
    below_horizon = yy > 0.16

    # A soft road-corridor prior keeps far-side vegetation/roof regions from
    # being selected while leaving the near-lane and intersection area broad.
    corridor_half_width = 0.08 + 0.68 * yy
    in_corridor = np.abs(xx - 0.5) < corridor_half_width

    mask = (gray_like & not_green & not_sky & below_horizon & in_corridor).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n, labels, stats, _cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    min_area = int(0.004 * height * width)
    bottom_band = int(0.82 * height)
    center_l = int(0.22 * width)
    center_r = int(0.78 * width)
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == idx
        ys, xs = np.where(comp)
        touches_bottom = bool(np.any(ys >= bottom_band))
        crosses_center = bool(np.any((ys > int(0.35 * height)) & (xs > center_l) & (xs < center_r)))
        if touches_bottom or crosses_center:
            keep[comp] = 255

    if keep.sum() == 0:
        return mask
    dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    return cv2.dilate(keep, dilate, iterations=1)


def boxes_overlap_or_near(a: tuple[int, int, int, int], b: tuple[int, int, int, int], distance: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (
        ax2 + distance < bx1
        or bx2 + distance < ax1
        or ay2 + distance < by1
        or by2 + distance < ay1
    )


def merge_components(components: list[Component], distance: int) -> list[Component]:
    """Merge nearby same-class components for cleaner display boxes."""

    merged: list[Component] = []
    used = [False] * len(components)
    for i, comp in enumerate(components):
        if used[i]:
            continue
        used[i] = True
        group = [comp]
        changed = True
        while changed:
            changed = False
            group_box = union_box([g.bbox for g in group])
            for j, other in enumerate(components):
                if used[j] or other.cls != comp.cls:
                    continue
                if boxes_overlap_or_near(group_box, other.bbox, distance):
                    used[j] = True
                    group.append(other)
                    changed = True
        box = union_box([g.bbox for g in group])
        area = int(sum(g.area for g in group))
        mean_conf = float(np.mean([g.mean_conf for g in group])) if group else 0.0
        merged.append(Component(comp.cls, area, box, mean_conf))
    return merged


def union_box(boxes: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    xs1, ys1, xs2, ys2 = zip(*boxes)
    return min(xs1), min(ys1), max(xs2), max(ys2)


def components_from_masks(
    masks: dict[int, np.ndarray],
    confidence_masks: dict[int, np.ndarray],
    min_area: int,
    close_kernel: int,
    merge_distance: int,
) -> list[Component]:
    comps: list[Component] = []
    for cls, raw_mask in masks.items():
        mask = clean_mask(raw_mask, close_kernel)
        n, labels, stats, _cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            w = int(stats[idx, cv2.CC_STAT_WIDTH])
            h = int(stats[idx, cv2.CC_STAT_HEIGHT])
            component_mask = labels == idx
            conf_values = confidence_masks[cls][component_mask]
            mean_conf = float(conf_values.mean() / 255.0) if conf_values.size else 0.0
            comps.append(Component(cls, area, (x, y, x + w, y + h), mean_conf))
        masks[cls] = mask
    return merge_components(comps, merge_distance)


def suppress_context_components(
    masks: dict[int, np.ndarray],
    confidence_masks: dict[int, np.ndarray],
    rgb: np.ndarray,
    min_area: int,
    close_kernel: int,
) -> None:
    """Remove mask components whose own pixels do not look like road surface."""

    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, _v_ch = cv2.split(hsv)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    chroma = maxc - minc
    exg = 2 * g - r - b
    vegetation_like = (
        (h_ch > 25)
        & (h_ch < 105)
        & (s_ch > 22)
        & (exg > 10)
        & (g > r + 4)
    )
    pavement_like = ((s_ch < 86) | (chroma < 42)) & ~vegetation_like

    for cls, raw_mask in list(masks.items()):
        mask = clean_mask(raw_mask, close_kernel)
        n, labels, stats, cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        kept = np.zeros_like(mask)
        kept_conf = np.zeros_like(confidence_masks[cls])
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx, cy = cent[idx]
            x_norm = float(cx) / max(float(width - 1), 1.0)
            y_norm = float(cy) / max(float(height - 1), 1.0)
            if y_norm < 0.20:
                continue
            corridor_half = 0.10 + 0.60 * y_norm
            if abs(x_norm - 0.5) > corridor_half:
                continue
            comp = labels == idx
            veg_ratio = float(vegetation_like[comp].mean()) if comp.any() else 1.0
            pav_ratio = float(pavement_like[comp].mean()) if comp.any() else 0.0
            if veg_ratio > 0.22:
                continue
            if pav_ratio < 0.35 and y_norm < 0.78:
                continue
            kept[comp] = 255
            kept_conf[comp] = confidence_masks[cls][comp]
        masks[cls] = kept
        confidence_masks[cls] = kept_conf


def overlay_masks(image: Image.Image, masks: dict[int, np.ndarray], alpha: int = 105) -> Image.Image:
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    for cls, mask in masks.items():
        color = COLORS.get(cls, (255, 255, 255))
        arr = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        arr[..., 0] = color[0]
        arr[..., 1] = color[1]
        arr[..., 2] = color[2]
        arr[..., 3] = (mask > 0).astype(np.uint8) * alpha
        overlay = Image.alpha_composite(overlay, Image.fromarray(arr, "RGBA"))
    return Image.alpha_composite(rgba, overlay).convert("RGB")


def class_name(cls: int, names: dict[int, str] | None = None) -> str:
    if names and cls in names:
        return str(names[cls])
    return CLASS_NAMES.get(cls, str(cls))


def draw_components(image: Image.Image, components: list[Component], names: dict[int, str] | None = None) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    for comp in sorted(components, key=lambda c: c.area, reverse=True):
        color = COLORS.get(comp.cls, (255, 255, 255))
        x1, y1, x2, y2 = comp.bbox
        width = max(3, int(round(math.sqrt(max(comp.area, 1)) / 26)))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        label = f"{class_name(comp.cls, names)} {comp.mean_conf:.2f}"
        text_box = draw.textbbox((x1, max(0, y1 - 28)), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x1, max(0, y1 - 28)), label, fill=color, font=font)
    return image


def make_atlas(mask_paths: list[Path], out_path: Path, columns: int = 2) -> None:
    thumbs = []
    target_w = 960
    for path in mask_paths:
        img = Image.open(path).convert("RGB")
        scale = target_w / img.width
        thumb = img.resize((target_w, int(img.height * scale)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, thumb.height + 38), "white")
        canvas.paste(thumb, (0, 38))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), path.name, fill=(20, 20, 20))
        thumbs.append(canvas)
    if not thumbs:
        return
    rows = math.ceil(len(thumbs) / columns)
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    atlas = Image.new("RGB", (columns * cell_w, rows * cell_h), (245, 245, 245))
    for i, thumb in enumerate(thumbs):
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h
        atlas.paste(thumb, (x, y))
    atlas.save(out_path, quality=92)


def process_image(
    model: YOLO,
    path: Path,
    args: argparse.Namespace,
    keep_classes: set[int],
    names: dict[int, str] | None,
) -> dict[str, object]:
    pil = Image.open(path).convert("RGB")
    width, height = pil.size
    rgb = np.asarray(pil)

    masks = {cls: np.zeros((height, width), dtype=np.uint8) for cls in keep_classes}
    conf_masks = {cls: np.zeros((height, width), dtype=np.uint8) for cls in keep_classes}

    full_count = fill_prediction_masks(
        model,
        rgb,
        masks,
        conf_masks,
        offset=(0, 0),
        conf=args.full_conf,
        imgsz=args.imgsz,
        iou=args.iou,
        device=args.device,
        keep_classes=keep_classes,
    )

    tile_count = 0
    for y0 in tile_starts(height, args.tile, args.overlap):
        for x0 in tile_starts(width, args.tile, args.overlap):
            tile = rgb[y0 : min(y0 + args.tile, height), x0 : min(x0 + args.tile, width)]
            th, tw = tile.shape[:2]
            tile_count += fill_prediction_masks(
                model,
                tile,
                masks,
                conf_masks,
                offset=(x0, y0),
                conf=args.tile_conf,
                imgsz=args.imgsz,
                iou=args.iou,
                device=args.device,
                keep_classes=keep_classes,
                tile_filter=((x0, y0), (tw, th), (width, height), args.center_margin),
            )

    components = components_from_masks(masks, conf_masks, args.min_area, args.close, args.merge_distance)

    road_mask_path = None
    if args.road_filter:
        road_mask = estimate_pavement_mask(rgb)
        roi_dir = args.out / "03_pavement_roi_debug"
        roi_dir.mkdir(parents=True, exist_ok=True)
        road_mask_path = roi_dir / path.name
        roi_overlay = pil.copy().convert("RGBA")
        arr = np.zeros((height, width, 4), dtype=np.uint8)
        arr[..., 1] = 180
        arr[..., 3] = (road_mask > 0).astype(np.uint8) * 80
        Image.alpha_composite(roi_overlay, Image.fromarray(arr, "RGBA")).convert("RGB").save(road_mask_path, quality=92)
        for cls in masks:
            masks[cls] = cv2.bitwise_and(masks[cls], road_mask)
            conf_masks[cls] = cv2.bitwise_and(conf_masks[cls], road_mask)
        suppress_context_components(masks, conf_masks, rgb, args.min_area, args.close)
        components = components_from_masks(masks, conf_masks, args.min_area, args.close, args.merge_distance)

    mask_only = overlay_masks(pil, masks)
    mask_box = draw_components(mask_only.copy(), components, names)

    mask_dir = args.out / "00_mask_only_full_resolution"
    box_dir = args.out / "01_mask_and_tight_regions_full_resolution"
    mask_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / path.name
    box_path = box_dir / path.name
    mask_only.save(mask_path, quality=95)
    mask_box.save(box_path, quality=95)

    return {
        "image": path.name,
        "width": width,
        "height": height,
        "full_raw_masks": full_count,
        "tile_raw_masks": tile_count,
        "components": len(components),
        "mask_pixels": int(sum((m > 0).sum() for m in masks.values())),
        "classes": ";".join(sorted({class_name(c.cls, names) for c in components})),
        "road_filter": bool(args.road_filter),
        "road_roi_overlay": str(road_mask_path) if road_mask_path else "",
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    defect_classes = {int(x.strip()) for x in args.defect_classes.split(",") if x.strip()}
    keep_classes = set(CLASS_NAMES) if args.draw_nondefect else defect_classes

    model = YOLO(str(args.weights))
    model_names = {int(k): str(v) for k, v in getattr(model, "names", {}).items()}
    rows = []
    for path in image_paths(args.input_dir):
        print(f"[nz-crackseg] {path.name}")
        rows.append(process_image(model, path, args, keep_classes, model_names))

    with (args.out / "predictions_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    config = vars(args).copy()
    config["weights"] = str(args.weights)
    config["input_dir"] = str(args.input_dir)
    config["out"] = str(args.out)
    config["keep_classes"] = sorted(keep_classes)
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    atlas_dir = args.out / "02_quick_atlas_resized"
    atlas_dir.mkdir(exist_ok=True)
    mask_paths = sorted((args.out / "01_mask_and_tight_regions_full_resolution").glob("*.jpg"))
    make_atlas(mask_paths, atlas_dir / "nz_crackseg_atlas.jpg")

    if args.zip:
        zip_base = args.out.with_suffix("")
        archive = shutil.make_archive(str(zip_base), "zip", root_dir=args.out)
        print(f"[nz-crackseg] wrote {archive}")


if __name__ == "__main__":
    main()
