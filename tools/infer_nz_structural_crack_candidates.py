"""Native-resolution structural crack candidates for NZ field images.

This is a detector-repair tool for manual review, not a paper metric script.
It avoids YOLO box bias by using only image structure:

1. Estimate a road/pavement region from perspective and color.
2. Suppress sky, grass, lane markings, and bright road paint.
3. Enhance dark, thin, locally contrasted structures at several scales.
4. Keep connected components that look like plausible road-surface defects.
5. Draw native-resolution mask and skeleton overlays.

The output is intentionally conservative about class names: everything is a
``defect_candidate`` because the sample has no ground-truth labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Candidate:
    area: int
    bbox: tuple[int, int, int, int]
    score: float
    kind: str
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=Path("experiments/nz_cracks_yolo26_gt49_detector/input_images/NZ-cracks"),
        type=Path,
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-area", type=int, default=36)
    parser.add_argument("--max-area-frac", type=float, default=0.025)
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def normalize01(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32)
    if mask is not None and np.any(mask):
        vals = x[mask]
    else:
        vals = x.reshape(-1)
    lo, hi = np.percentile(vals, [1, 99.5]) if vals.size else (0, 1)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def image_paths(path: Path) -> list[Path]:
    paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(path)
    return paths


def color_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
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

    # These are deliberately strict; they remove obvious context before crack
    # enhancement so roadside texture does not become a false defect.
    green = (h > 26) & (h < 104) & (s > 24) & (exg > 8) & (g > r + 3)
    sky = (yy < 0.58) & (v > 130) & (s < 88) & (b > r + 3)
    white_paint = (v > 145) & (s < 52) & (yy > 0.18)
    yellow_paint = (h > 14) & (h < 44) & (s > 45) & (v > 80)
    paint = white_paint | yellow_paint

    grayish = (s < 84) | (chroma < 46)
    below_horizon = yy > 0.18
    corridor_half = 0.08 + 0.70 * yy
    in_corridor = np.abs(xx - 0.50) < corridor_half
    pavement_seed = grayish & below_horizon & in_corridor & ~green & ~sky

    seed = pavement_seed.astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)

    # Keep large road-like components connected to the lower image or central road.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((seed > 0).astype(np.uint8), 8)
    road = np.zeros_like(seed)
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < int(0.0025 * height * width):
            continue
        comp = labels == idx
        ys, xs = np.where(comp)
        touches_bottom = np.any(ys > int(0.84 * height))
        in_middle = np.any((ys > int(0.32 * height)) & (xs > int(0.22 * width)) & (xs < int(0.78 * width)))
        if touches_bottom or in_middle:
            road[comp] = 255
    road = cv2.dilate(road, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)

    return {
        "road": road > 0,
        "green": green,
        "sky": sky,
        "paint": paint,
        "pavement_like": grayish & ~green & ~sky,
    }


def crack_response(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(16, 16))
    eq = clahe.apply(gray)

    # Dark local contrast: cracks are darker than nearby pavement.
    dark21 = cv2.GaussianBlur(eq, (0, 0), 9) - eq
    dark41 = cv2.GaussianBlur(eq, (0, 0), 18) - eq
    dark = np.maximum(dark21, dark41)
    dark = np.clip(dark, 0, 255).astype(np.uint8)

    # Directional black-hat kernels catch horizontal, vertical, and slanted cracks.
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 35)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (23, 9)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 23)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    ]
    blackhat = np.zeros_like(eq)
    for kernel in kernels:
        blackhat = np.maximum(blackhat, cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, kernel))

    # Line-sensitive Gabor bank on the inverted equalized image.
    inv = 255 - eq
    gabor = np.zeros_like(eq, dtype=np.float32)
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        kernel = cv2.getGaborKernel((31, 31), sigma=4.0, theta=theta, lambd=13.0, gamma=0.18, psi=0)
        kernel -= kernel.mean()
        resp = cv2.filter2D(inv.astype(np.float32), cv2.CV_32F, kernel)
        gabor = np.maximum(gabor, resp)

    n_dark = normalize01(dark, valid)
    n_blackhat = normalize01(blackhat, valid)
    n_gabor = normalize01(gabor, valid)
    response = 0.48 * n_dark + 0.34 * n_blackhat + 0.18 * n_gabor
    response[~valid] = 0
    return response


def build_candidate_mask(response: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
    vals = response[valid]
    if vals.size == 0:
        return np.zeros_like(response, dtype=np.uint8)
    if mode == "review_sensitive":
        pct, min_resp = 94.4, 0.25
    elif mode == "sensitive":
        pct, min_resp = 96.8, 0.34
    elif mode == "balanced":
        pct, min_resp = 97.8, 0.42
    else:
        pct, min_resp = 98.6, 0.50
    thr = max(float(np.percentile(vals, pct)), min_resp)
    mask = ((response >= thr) & valid).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return mask


def filter_components(
    mask: np.ndarray,
    response: np.ndarray,
    masks: dict[str, np.ndarray],
    min_area: int,
    max_area: int,
    mode: str,
) -> tuple[np.ndarray, list[Candidate]]:
    n, labels, stats, cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    kept = np.zeros_like(mask)
    candidates: list[Candidate] = []
    height, width = mask.shape

    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = cent[idx]
        x_norm = float(cx) / max(width - 1, 1)
        y_norm = float(cy) / max(height - 1, 1)
        if y_norm < 0.20:
            continue
        corridor_half = 0.11 + 0.62 * y_norm
        if abs(x_norm - 0.5) > corridor_half:
            continue
        comp = labels == idx
        green_limit = 0.16 if mode == "review_sensitive" else 0.08
        paint_limit = 0.20 if mode == "review_sensitive" else 0.12
        pavement_floor = 0.28 if mode == "review_sensitive" else 0.42
        if float(masks["green"][comp].mean()) > green_limit:
            continue
        if float(masks["paint"][comp].mean()) > paint_limit:
            continue
        if float(masks["pavement_like"][comp].mean()) < pavement_floor and y_norm < 0.78:
            continue
        elongation = max(w, h) / max(min(w, h), 1)
        fill = area / max(w * h, 1)
        # Keep thin/elongated cracks and compact pothole-like dark regions.
        crack_elongation = 1.55 if mode == "review_sensitive" else 2.2
        crack_fill = 0.55 if mode == "review_sensitive" else 0.42
        crack_like = elongation >= crack_elongation and fill <= crack_fill
        pothole_area = 120 if mode == "review_sensitive" else 280
        pothole_like = area >= pothole_area and 0.04 <= fill <= 0.76 and y_norm > 0.35
        if not (crack_like or pothole_like):
            continue
        score = float(response[comp].mean())
        score_floor = 0.21 if mode == "review_sensitive" else 0.28
        if score < score_floor:
            continue
        kept[comp] = 255
        skel = skeletonize(comp).astype(np.uint8)
        length = int(skel.sum())
        kind = "linear" if crack_like else "patch"
        candidates.append(Candidate(area=area, bbox=(x, y, x + w, y + h), score=score, kind=kind, length=length))

    # Dilation only affects review visibility, not candidate measurement.
    k = 5 if mode == "review_sensitive" else 3
    visible = cv2.dilate(kept, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    return visible, candidates


def draw_overlay(rgb: np.ndarray, mask: np.ndarray, candidates: list[Candidate], mode: str) -> Image.Image:
    image = Image.fromarray(rgb).convert("RGBA")
    overlay = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    if mode == "review_sensitive":
        color = (255, 90, 230, 145)
    elif mode == "sensitive":
        color = (255, 195, 0, 110)
    elif mode == "balanced":
        color = (255, 82, 82, 120)
    else:
        color = (0, 200, 140, 135)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    overlay[..., 3] = (mask > 0).astype(np.uint8) * color[3]
    out = Image.alpha_composite(image, Image.fromarray(overlay, "RGBA")).convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    for cand in sorted(candidates, key=lambda c: c.score * math.log1p(c.length + c.area), reverse=True):
        x1, y1, x2, y2 = cand.bbox
        box_color = (255, 230, 70) if cand.kind == "linear" else (0, 220, 180)
        width = 2 if cand.kind == "linear" else 3
        draw.rectangle((x1, y1, x2, y2), outline=box_color, width=width)
        label = f"{cand.kind} {cand.score:.2f}"
        tx, ty = x1, max(0, y1 - 25)
        tb = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(tb, fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=box_color, font=font)
    return out


def save_debug_response(response: np.ndarray, out_path: Path) -> None:
    heat = (normalize01(response) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).save(out_path, quality=92)


def make_atlas(paths: list[Path], out_path: Path, columns: int = 2) -> None:
    target_w = 960
    thumbs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        scale = target_w / img.width
        thumb = img.resize((target_w, int(img.height * scale)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, thumb.height + 36), "white")
        canvas.paste(thumb, (0, 36))
        ImageDraw.Draw(canvas).text((8, 8), p.name, fill=(0, 0, 0))
        thumbs.append(canvas)
    if not thumbs:
        return
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    atlas = Image.new("RGB", (columns * cell_w, rows * cell_h), (245, 245, 245))
    for i, thumb in enumerate(thumbs):
        atlas.paste(thumb, ((i % columns) * cell_w, (i // columns) * cell_h))
    atlas.save(out_path, quality=92)


def process(path: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    pil = Image.open(path).convert("RGB")
    rgb = np.asarray(pil)
    masks = color_masks(rgb)
    valid = masks["road"] & ~masks["green"] & ~masks["sky"] & ~masks["paint"]
    response = crack_response(rgb, valid)
    max_area = int(args.max_area_frac * rgb.shape[0] * rgb.shape[1])

    rows: list[dict[str, object]] = []
    for mode in ["review_sensitive", "sensitive", "balanced", "conservative"]:
        raw_mask = build_candidate_mask(response, valid, mode)
        final_mask, candidates = filter_components(raw_mask, response, masks, args.min_area, max_area, mode)
        out_dir = args.out / f"{mode}_full_resolution"
        mask_dir = args.out / f"{mode}_binary_masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay = draw_overlay(rgb, final_mask, candidates, mode)
        overlay.save(out_dir / path.name, quality=95)
        Image.fromarray(final_mask).save(mask_dir / f"{path.stem}.png")
        rows.append(
            {
                "image": path.name,
                "mode": mode,
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "candidates": len(candidates),
                "mask_pixels": int((final_mask > 0).sum()),
                "linear_candidates": sum(1 for c in candidates if c.kind == "linear"),
                "patch_candidates": sum(1 for c in candidates if c.kind == "patch"),
                "mean_score": round(float(np.mean([c.score for c in candidates])) if candidates else 0.0, 4),
            }
        )

    debug_dir = args.out / "debug_response_and_roi"
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_debug_response(response, debug_dir / path.name)
    roi = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    roi[..., 1] = 180
    roi[..., 3] = valid.astype(np.uint8) * 82
    Image.alpha_composite(pil.convert("RGBA"), Image.fromarray(roi, "RGBA")).convert("RGB").save(
        debug_dir / f"{path.stem}_road_roi.jpg", quality=92
    )
    return rows


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for path in image_paths(args.input_dir):
        print(f"[structural] {path.name}")
        rows.extend(process(path, args))

    csv_path = args.out / "candidate_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    atlas_dir = args.out / "atlas"
    atlas_dir.mkdir(exist_ok=True)
    for mode in ["review_sensitive", "sensitive", "balanced", "conservative"]:
        make_atlas(sorted((args.out / f"{mode}_full_resolution").glob("*.jpg")), atlas_dir / f"{mode}_atlas.jpg")

    (args.out / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    if args.zip:
        archive = shutil.make_archive(str(args.out.with_suffix("")), "zip", root_dir=args.out)
        print(f"[structural] wrote {archive}")


if __name__ == "__main__":
    main()
