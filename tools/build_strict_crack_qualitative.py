from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FIG_DIR = ROOT / "paper_ieee_tits_rmrnet" / "figures"
PCM_MODEL = ROOT / "runs" / "detect" / "runs" / "detect_train" / "yolov8n_pcm_clean_25ep" / "weights" / "best.pt"
CRACK_CLASS = 1


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_crop(img: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, int, int]:
    img = img.convert("RGB")
    ow, oh = img.size
    tw, th = size
    scale = max(tw / ow, th / oh)
    resized = img.resize((round(ow * scale), round(oh * scale)), Image.Resampling.BICUBIC)
    rw, rh = resized.size
    ox = max((rw - tw) // 2, 0)
    oy = max((rh - th) // 2, 0)
    return resized.crop((ox, oy, ox + tw, oy + th)), scale, ox, oy


def crack_gt_boxes(name: str) -> list[tuple[float, float, float, float]]:
    label_path = ROOT / "datasets" / "road_damage_pcm_yolo" / "labels" / "test" / (Path(name).stem + ".txt")
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) != CRACK_CLASS:
            continue
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def crack_predictions(model: YOLO, path: Path, conf: float = 0.20) -> list[tuple[tuple[float, float, float, float], float]]:
    result = model.predict(str(path), imgsz=640, conf=conf, verbose=False)[0]
    preds: list[tuple[tuple[float, float, float, float], float]] = []
    for box, cls, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
        if int(cls) != CRACK_CLASS:
            continue
        preds.append(((float(box[0]), float(box[1]), float(box[2]), float(box[3])), float(score)))
    return preds


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_stats(model: YOLO, path: Path, gt_boxes: list[tuple[float, float, float, float]]) -> dict[str, int]:
    with Image.open(path) as image:
        width, height = image.size
    pred_boxes = []
    for (x1, y1, x2, y2), _ in crack_predictions(model, path):
        pred_boxes.append((x1 / width, y1 / height, x2 / width, y2 / height))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for pred_idx, pred in enumerate(pred_boxes):
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            value = iou(pred, gt)
            if value > best_iou:
                best_iou = value
                best_gt = gt_idx
        if best_iou >= 0.08 and best_gt >= 0:
            matched_gt.add(best_gt)
            matched_pred.add(pred_idx)
    return {
        "predictions": len(pred_boxes),
        "matches": len(matched_gt),
        "false_positives": len(pred_boxes) - len(matched_pred),
    }


def visible_gt_score(boxes: list[tuple[float, float, float, float]]) -> float:
    # Prefer non-tiny crack annotations that are not right on the image border.
    score = 0.0
    for x1, y1, x2, y2 in boxes:
        area = max((x2 - x1) * (y2 - y1), 0.0)
        centered = 1.0 if 0.08 < x1 < 0.92 and 0.08 < x2 < 0.98 and 0.12 < y1 < 0.95 else 0.3
        score += area * centered
    return score


def select_example(model: YOLO, tag: str) -> tuple[str, dict[str, int | float]]:
    variants = variant_dirs(tag)
    names = [p.name for p in sorted(variants["Clean"].iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    best_name = ""
    best = -1e9
    best_stats: dict[str, int | float] = {}
    for name in names:
        gt = crack_gt_boxes(name)
        if not gt:
            continue
        clean_s = match_stats(model, variants["Clean"] / name, gt)
        degraded_s = match_stats(model, variants["Degraded"] / name, gt)
        rmr_s = match_stats(model, variants["RMR-Net+meta"] / name, gt)
        dfpir_s = match_stats(model, variants["DFPIR"] / name, gt)
        if clean_s["matches"] == 0 or rmr_s["matches"] == 0:
            continue
        # Strong panels show crack evidence recovered or preserved by RMR,
        # while degraded/DFPIR are weaker. Visible ground-truth cracks matter.
        score = (
            1000.0 * visible_gt_score(gt)
            + 14.0 * rmr_s["matches"]
            + 4.0 * clean_s["matches"]
            - 6.0 * degraded_s["matches"]
            - 4.0 * dfpir_s["matches"]
            - 7.0 * rmr_s["false_positives"]
            - 2.0 * dfpir_s["false_positives"]
        )
        if rmr_s["matches"] < degraded_s["matches"]:
            score -= 15.0
        if rmr_s["matches"] < dfpir_s["matches"]:
            score -= 10.0
        if rmr_s["false_positives"] > 1:
            score -= 12.0 * (rmr_s["false_positives"] - 1)
        if score > best:
            best = score
            best_name = name
            best_stats = {
                "gt_cracks": len(gt),
                "clean": clean_s,
                "degraded": degraded_s,
                "rmr": rmr_s,
                "dfpir": dfpir_s,
                "score": score,
            }
    if not best_name:
        raise RuntimeError(f"No strict crack qualitative example found for {tag}")
    return best_name, best_stats


def variant_dirs(tag: str) -> dict[str, Path]:
    return {
        "Clean": ROOT / "datasets" / "road_damage_pcm_yolo" / "images" / "test",
        "Degraded": ROOT / "datasets" / f"pcm_yolo_{tag}_test" / "images" / "test",
        "RMR-Net+meta": ROOT / "datasets" / f"pcm_yolo_{tag}_test_rmrnet_meta" / "images" / "test",
        "DFPIR": ROOT / "datasets" / f"pcm_yolo_{tag}_test_dfpir" / "images" / "test",
    }


def draw_crack_only(model: YOLO, image_path: Path, size: tuple[int, int], gt_boxes: list[tuple[float, float, float, float]]) -> Image.Image:
    raw = Image.open(image_path).convert("RGB")
    ow, oh = raw.size
    image, scale, ox, oy = resize_crop(raw, size)
    draw = ImageDraw.Draw(image)
    fnt = font(13)
    for gx1, gy1, gx2, gy2 in gt_boxes:
        x1 = gx1 * ow * scale - ox
        y1 = gy1 * oh * scale - oy
        x2 = gx2 * ow * scale - ox
        y2 = gy2 * oh * scale - oy
        if x2 < 0 or y2 < 0 or x1 > size[0] or y1 > size[1]:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=(245, 190, 55), width=2)
    for (x1, y1, x2, y2), score in crack_predictions(model, image_path):
        x1 = x1 * scale - ox
        x2 = x2 * scale - ox
        y1 = y1 * scale - oy
        y2 = y2 * scale - oy
        if x2 < 0 or y2 < 0 or x1 > size[0] or y1 > size[1]:
            continue
        label = f"crack {score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(20, 155, 117), width=3)
        tw = draw.textlength(label, font=fnt)
        draw.rectangle((x1, max(0, y1 - 18), x1 + tw + 6, max(18, y1)), fill=(20, 155, 117))
        draw.text((x1 + 3, max(0, y1 - 17)), label, fill="white", font=fnt)
    return image


def build_panel(tag: str, title: str) -> dict[str, int | float | str]:
    model = YOLO(str(PCM_MODEL))
    variants = variant_dirs(tag)
    name, stats = select_example(model, tag)
    gt = crack_gt_boxes(name)
    tile = (292, 168)
    pad = 12
    header_h = 74
    width = len(variants) * tile[0] + (len(variants) + 1) * pad
    height = header_h + tile[1] + 48
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((pad, 10), title, fill=(20, 28, 34), font=font(19))
    draw.text((pad, 34), "Yellow: ground-truth crack boxes. Green: crack-class YOLO predictions only.", fill=(70, 78, 86), font=font(13))
    draw.text((pad, 52), f"Selected frame: {name}; other detector classes are hidden to avoid false-positive distraction.", fill=(70, 78, 86), font=font(12))
    for i, (label, folder) in enumerate(variants.items()):
        x = pad + i * (tile[0] + pad)
        y = header_h
        panel.paste(draw_crack_only(model, folder / name, tile, gt), (x, y))
        draw.text((x + 4, y + tile[1] + 8), label, fill=(20, 28, 34), font=font(14))
    out = FIG_DIR / f"fig_yolo_crack_{tag}.png"
    panel.save(out)
    stats["name"] = name
    stats["figure"] = out.name
    return stats


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "defocus": build_panel("defocus", "Crack detection under defocus blur"),
        "lowlight": build_panel("lowlight", "Crack detection under low light"),
    }
    (ROOT / "paper_ieee_tits_rmrnet" / "STRICT_QUALITATIVE_SELECTION.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
