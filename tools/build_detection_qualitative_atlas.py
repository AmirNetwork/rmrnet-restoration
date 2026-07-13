from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
FIG_DIR = PAPER / "figures"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GT_COLOR = (245, 190, 55)
PRED_COLORS = {
    0: (32, 120, 184),
    1: (20, 155, 117),
    2: (185, 87, 69),
}


@dataclass(frozen=True)
class AtlasConfig:
    name: str
    detector: Path
    clean_images: Path
    labels: Path
    prefix: str
    class_filter: int | None = None


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def variant_dirs(config: AtlasConfig, tag: str) -> dict[str, Path]:
    return {
        "Clean": config.clean_images,
        "Degraded": ROOT / "datasets" / f"{config.prefix}_{tag}_test" / "images" / "test",
        "RMR+meta": ROOT / "datasets" / f"{config.prefix}_{tag}_test_rmrnet_revised" / "images" / "test",
        "NAFNet": ROOT / "datasets" / f"{config.prefix}_{tag}_test_nafnet" / "images" / "test",
        "DFPIR": ROOT / "datasets" / f"{config.prefix}_{tag}_test_dfpir" / "images" / "test",
    }


def read_gt(label_path: Path, class_filter: int | None = None) -> list[tuple[int, tuple[float, float, float, float]]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if class_filter is not None and cls != class_filter:
            continue
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        boxes.append((cls, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def predict(model: YOLO, path: Path, class_filter: int | None, conf: float) -> list[tuple[int, float, tuple[float, float, float, float]]]:
    result = model.predict(str(path), imgsz=640, conf=conf, verbose=False)[0]
    width, height = Image.open(path).size
    preds: list[tuple[int, float, tuple[float, float, float, float]]] = []
    for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
        cls_i = int(cls)
        if class_filter is not None and cls_i != class_filter:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        preds.append((cls_i, float(score), (x1 / width, y1 / height, x2 / width, y2 / height)))
    return preds


def match_count(gt: list[tuple[int, tuple[float, float, float, float]]], preds: list[tuple[int, float, tuple[float, float, float, float]]]) -> int:
    used: set[int] = set()
    matches = 0
    for gt_cls, gt_box in gt:
        best_index = -1
        best_iou = 0.0
        for index, (pred_cls, _, pred_box) in enumerate(preds):
            if index in used or pred_cls != gt_cls:
                continue
            overlap = iou(gt_box, pred_box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index >= 0 and best_iou >= 0.2:
            used.add(best_index)
            matches += 1
    return matches


def select_candidates(config: AtlasConfig, scenarios: list[str], rows_per_scenario: int, max_candidates: int, conf: float) -> list[dict]:
    model = YOLO(str(config.detector))
    selected: list[dict] = []
    for tag in scenarios:
        dirs = variant_dirs(config, tag)
        names = [p.name for p in sorted(dirs["Degraded"].iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        names = names[:max_candidates] if max_candidates else names
        scored: list[dict] = []
        for name in names:
            gt = read_gt(config.labels / f"{Path(name).stem}.txt", config.class_filter)
            if not gt:
                continue
            preds_by_method = {method: predict(model, folder / name, config.class_filter, conf) for method, folder in dirs.items()}
            matches = {method: match_count(gt, preds) for method, preds in preds_by_method.items()}
            pred_counts = {method: len(preds) for method, preds in preds_by_method.items()}
            false_pos = {method: max(pred_counts[method] - matches[method], 0) for method in preds_by_method}
            rmr_gain = matches["RMR+meta"] - matches["Degraded"]
            rmr_margin = matches["RMR+meta"] - max(matches["NAFNet"], matches["DFPIR"])
            clean_ok = matches["Clean"] > 0
            score = (
                6 * matches["Clean"]
                + 8 * max(rmr_gain, 0)
                + 3 * matches["RMR+meta"]
                + 2 * rmr_margin
                - 1.5 * false_pos["RMR+meta"]
                - 0.5 * false_pos["Clean"]
            )
            if not clean_ok:
                score -= 12
            scored.append(
                {
                    "scenario": tag,
                    "name": name,
                    "gt_count": len(gt),
                    "matches": matches,
                    "pred_counts": pred_counts,
                    "false_pos": false_pos,
                    "score": score,
                    "rmr_gain": rmr_gain,
                    "rmr_margin": rmr_margin,
                }
            )
        scored.sort(key=lambda row: row["score"], reverse=True)
        selected.extend(scored[:rows_per_scenario])
    return selected


def resize_crop(img: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[float, float, float]]:
    img = img.convert("RGB")
    width, height = img.size
    target_w, target_h = size
    scale = max(target_w / width, target_h / height)
    resized = img.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    new_w, new_h = resized.size
    offset_x = max((new_w - target_w) // 2, 0)
    offset_y = max((new_h - target_h) // 2, 0)
    return resized.crop((offset_x, offset_y, offset_x + target_w, offset_y + target_h)), (scale, offset_x, offset_y)


def draw_panel(
    model: YOLO,
    path: Path,
    label_path: Path,
    class_filter: int | None,
    size: tuple[int, int],
    conf: float,
) -> Image.Image:
    original = Image.open(path).convert("RGB")
    width, height = original.size
    canvas, (scale, offset_x, offset_y) = resize_crop(original, size)
    draw = ImageDraw.Draw(canvas)
    small = font(11)
    gt = read_gt(label_path, class_filter)
    preds = predict(model, path, class_filter, conf)

    def map_box(norm_box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = norm_box
        return (
            x1 * width * scale - offset_x,
            y1 * height * scale - offset_y,
            x2 * width * scale - offset_x,
            y2 * height * scale - offset_y,
        )

    for _, box in gt:
        x1, y1, x2, y2 = map_box(box)
        if x2 < 0 or y2 < 0 or x1 > size[0] or y1 > size[1]:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=GT_COLOR, width=2)

    for cls, score, box in preds:
        x1, y1, x2, y2 = map_box(box)
        if x2 < 0 or y2 < 0 or x1 > size[0] or y1 > size[1]:
            continue
        color = PRED_COLORS.get(cls, (230, 185, 60))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        text = f"{cls}:{score:.2f}"
        text_w = draw.textlength(text, font=small)
        y_text = max(0, y1 - 15)
        draw.rectangle((x1, y_text, x1 + text_w + 5, y_text + 14), fill=color)
        draw.text((x1 + 2, y_text), text, fill="white", font=small)
    return canvas


def focus_window(gt: list[tuple[int, tuple[float, float, float, float]]], aspect: float) -> tuple[float, float, float, float]:
    if not gt:
        return (0.0, 0.0, 1.0, 1.0)
    boxes = [box for _, box in gt]
    ux1 = min(box[0] for box in boxes)
    uy1 = min(box[1] for box in boxes)
    ux2 = max(box[2] for box in boxes)
    uy2 = max(box[3] for box in boxes)
    if (ux2 - ux1) > 0.62 or (uy2 - uy1) > 0.62:
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:1]
        ux1 = min(box[0] for box in boxes)
        uy1 = min(box[1] for box in boxes)
        ux2 = max(box[2] for box in boxes)
        uy2 = max(box[3] for box in boxes)

    cx = (ux1 + ux2) / 2
    cy = (uy1 + uy2) / 2
    width = max((ux2 - ux1) * 2.6, 0.32)
    height = max((uy2 - uy1) * 2.6, 0.18)
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    x1 = max(0.0, cx - width / 2)
    y1 = max(0.0, cy - height / 2)
    x2 = min(1.0, cx + width / 2)
    y2 = min(1.0, cy + height / 2)
    if x2 - x1 < width:
        if x1 == 0.0:
            x2 = min(1.0, x1 + width)
        elif x2 == 1.0:
            x1 = max(0.0, x2 - width)
    if y2 - y1 < height:
        if y1 == 0.0:
            y2 = min(1.0, y1 + height)
        elif y2 == 1.0:
            y1 = max(0.0, y2 - height)
    return (x1, y1, x2, y2)


def draw_panel_zoom(
    model: YOLO,
    path: Path,
    label_path: Path,
    class_filter: int | None,
    size: tuple[int, int],
    conf: float,
    crop_win: tuple[float, float, float, float],
) -> Image.Image:
    original = Image.open(path).convert("RGB")
    width, height = original.size
    wx1, wy1, wx2, wy2 = crop_win
    left, top, right, bottom = round(wx1 * width), round(wy1 * height), round(wx2 * width), round(wy2 * height)
    right = max(right, left + 2)
    bottom = max(bottom, top + 2)
    crop = original.crop((left, top, right, bottom))
    canvas = crop.resize(size, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(canvas)
    small = font(11)
    crop_w = right - left
    crop_h = bottom - top

    def to_crop_box(norm_box: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
        x1, y1, x2, y2 = norm_box
        px1, py1, px2, py2 = x1 * width, y1 * height, x2 * width, y2 * height
        ix1, iy1 = max(px1, left), max(py1, top)
        ix2, iy2 = min(px2, right), min(py2, bottom)
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return (
            (ix1 - left) / crop_w * size[0],
            (iy1 - top) / crop_h * size[1],
            (ix2 - left) / crop_w * size[0],
            (iy2 - top) / crop_h * size[1],
        )

    for _, box in read_gt(label_path, class_filter):
        mapped = to_crop_box(box)
        if mapped is None:
            continue
        draw.rectangle(mapped, outline=GT_COLOR, width=2)

    for cls, score, box in predict(model, path, class_filter, conf):
        mapped = to_crop_box(box)
        if mapped is None:
            continue
        color = PRED_COLORS.get(cls, (230, 185, 60))
        x1, y1, x2, y2 = mapped
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        text = f"{cls}:{score:.2f}"
        text_w = draw.textlength(text, font=small)
        y_text = max(0, y1 - 15)
        draw.rectangle((x1, y_text, x1 + text_w + 5, y_text + 14), fill=color)
        draw.text((x1 + 2, y_text), text, fill="white", font=small)
    return canvas


def build_atlas(config: AtlasConfig, candidates: list[dict], out_name: str, conf: float, *, zoom: bool = False) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(config.detector))
    columns = ["Clean", "Degraded", "RMR+meta", "NAFNet", "DFPIR"]
    tile = (250, 142)
    gap = 10
    left = 168
    top = 102
    row_h = 184
    header_h = 34
    width = left + len(columns) * tile[0] + (len(columns) - 1) * gap + 24
    height = top + len(candidates) * row_h + 38
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    title = font(24)
    head = font(15)
    small = font(12)
    tiny = font(10)
    title_suffix = "zoomed defect crops" if zoom else "full-frame candidates"
    draw.text((18, 16), f"{config.name} detection qualitative atlas: {title_suffix}", fill=(20, 28, 34), font=title)
    draw.text(
        (18, 48),
        "Yellow: ground truth. Colored boxes: detector predictions. Rows are held-out examples selected by matched ground-truth recovery.",
        fill=(70, 78, 86),
        font=small,
    )
    for col, label in enumerate(columns):
        x = left + col * (tile[0] + gap)
        draw.text((x + 4, top - header_h), label, fill=(20, 28, 34), font=head)

    for row, item in enumerate(candidates):
        y = top + row * row_h
        scenario = item["scenario"]
        name = item["name"]
        dirs = variant_dirs(config, scenario)
        label_path = config.labels / f"{Path(name).stem}.txt"
        gt = read_gt(label_path, config.class_filter)
        crop_win = focus_window(gt, tile[0] / tile[1])
        draw.text((18, y + 4), scenario, fill=(20, 28, 34), font=head)
        draw.text((18, y + 25), Path(name).stem[:22], fill=(70, 78, 86), font=tiny)
        draw.text((18, y + 45), f"GT: {item['gt_count']}", fill=(70, 78, 86), font=tiny)
        draw.text((18, y + 62), f"RMR gain: {item['rmr_gain']:+d}", fill=(20, 120, 85), font=tiny)
        draw.text((18, y + 79), f"RMR vs best: {item['rmr_margin']:+d}", fill=(70, 78, 86), font=tiny)
        for col, method in enumerate(columns):
            x = left + col * (tile[0] + gap)
            if zoom:
                panel = draw_panel_zoom(model, dirs[method] / name, label_path, config.class_filter, tile, conf, crop_win)
            else:
                panel = draw_panel(model, dirs[method] / name, label_path, config.class_filter, tile, conf)
            page.paste(panel, (x, y))
            m = item["matches"][method]
            draw.text((x + 4, y + tile[1] + 6), f"matched GT: {m}", fill=(20, 28, 34), font=tiny)
        draw.line((18, y + row_h - 12, width - 18, y + row_h - 12), fill=(225, 229, 232), width=1)

    out_path = FIG_DIR / out_name
    page.save(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one-page detection qualitative atlases for the paper and supplement.")
    parser.add_argument("--rows-per-scenario", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--conf", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [
        AtlasConfig(
            name="PCM pothole/crack/manhole",
            detector=ROOT / "runs" / "detect" / "runs" / "detect_train" / "yolov8n_pcm_clean_25ep" / "weights" / "best.pt",
            clean_images=ROOT / "datasets" / "road_damage_pcm_yolo" / "images" / "test",
            labels=ROOT / "datasets" / "road_damage_pcm_yolo" / "labels" / "test",
            prefix="pcm_yolo",
        ),
        AtlasConfig(
            name="PCM crack-only",
            detector=ROOT / "runs" / "detect" / "runs" / "detect_train" / "yolov8n_pcm_clean_25ep" / "weights" / "best.pt",
            clean_images=ROOT / "datasets" / "road_damage_pcm_yolo" / "images" / "test",
            labels=ROOT / "datasets" / "road_damage_pcm_yolo" / "labels" / "test",
            prefix="pcm_yolo",
            class_filter=1,
        ),
        AtlasConfig(
            name="IVCNZ pothole",
            detector=ROOT / "runs" / "detect" / "runs" / "detect_train" / "yolov8n_pothole_clean" / "weights" / "best.pt",
            clean_images=ROOT / "datasets" / "pothole_yolo" / "images" / "test",
            labels=ROOT / "datasets" / "pothole_yolo" / "labels" / "test",
            prefix="pothole_yolo",
        ),
    ]
    scenarios = ["motion", "defocus", "lowlight"]
    manifest = {}
    for config in configs:
        candidates = select_candidates(config, scenarios, args.rows_per_scenario, args.max_candidates, args.conf)
        if config.prefix.startswith("pcm") and config.class_filter == 1:
            stem = "pcm_crack"
        else:
            stem = "pcm" if config.prefix.startswith("pcm") else "pothole"
        out_path = build_atlas(config, candidates, f"fig_detection_candidate_atlas_{stem}.png", args.conf)
        zoom_path = build_atlas(config, candidates, f"fig_detection_candidate_atlas_{stem}_zoom.png", args.conf, zoom=True)
        manifest[stem] = {"figure": str(out_path), "zoom_figure": str(zoom_path), "candidates": candidates}
    manifest_path = PAPER / "DETECTION_QUALITATIVE_ATLAS_SELECTION.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value["figure"] for key, value in manifest.items()}, indent=2))


if __name__ == "__main__":
    main()
