# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector-transfer diagnostics for small geotagged pilots.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_yolo(path: Path, width: int, height: int) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = [float(v) for v in parts[1:5]]
        x1 = (xc - w / 2.0) * width
        y1 = (yc - h / 2.0) * height
        x2 = (xc + w / 2.0) * width
        y2 = (yc + h / 2.0) * height
        rows.append({"cls": cls, "box": np.array([x1, y1, x2, y2], dtype=np.float32)})
    return rows


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(aa + bb - inter, 1e-9)


def box_ioa_gt(gt: np.ndarray, pred: np.ndarray) -> float:
    ix1 = max(float(gt[0]), float(pred[0]))
    iy1 = max(float(gt[1]), float(pred[1]))
    ix2 = min(float(gt[2]), float(pred[2]))
    iy2 = min(float(gt[3]), float(pred[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ga = max(0.0, float(gt[2] - gt[0])) * max(0.0, float(gt[3] - gt[1]))
    return inter / max(ga, 1e-9)


def center_inside(gt: np.ndarray, pred: np.ndarray) -> bool:
    cx = (float(gt[0]) + float(gt[2])) / 2.0
    cy = (float(gt[1]) + float(gt[3])) / 2.0
    return float(pred[0]) <= cx <= float(pred[2]) and float(pred[1]) <= cy <= float(pred[3])


def greedy_match(gt: list[dict], preds: list[dict], threshold: float, class_aware: bool) -> tuple[int, int, int]:
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            if class_aware and int(g["cls"]) != int(p["cls"]):
                continue
            score = box_iou(g["box"], p["box"])
            if score >= threshold:
                candidates.append((score, gi, pi))
    candidates.sort(reverse=True)
    used_gt = set()
    used_pred = set()
    tp = 0
    for _score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        tp += 1
    fp = len(preds) - len(used_pred)
    fn = len(gt) - len(used_gt)
    return tp, fp, fn


def metric_row(name: str, gt: list[dict], preds: list[dict], threshold: float, class_aware: bool) -> dict:
    tp, fp, fn = greedy_match(gt, preds, threshold, class_aware)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "metric": name,
        "threshold": threshold,
        "class_aware": class_aware,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
    }


def draw_overlay(
    image_path: Path,
    gt: list[dict],
    preds: list[dict],
    names: dict[int, str],
    native_ann_dir: Path | None,
    out_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(max(22, image.width // 170))
    native_ann = None
    if native_ann_dir:
        ann_path = native_ann_dir / f"{image_path.stem}.json"
        if ann_path.exists():
            native_ann = json.loads(ann_path.read_text(encoding="utf-8"))
    if native_ann:
        for ann in native_ann:
            for seg in ann.get("segmentation_native", []):
                pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                if len(pts) >= 3:
                    draw.polygon(pts, outline=(255, 215, 0, 230), fill=(255, 215, 0, 35))
                    draw.line(pts + [pts[0]], fill=(255, 215, 0, 255), width=max(4, image.width // 1000))
    for g in gt:
        x1, y1, x2, y2 = [float(v) for v in g["box"]]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 215, 0, 255), width=max(5, image.width // 900))
        draw.text((x1, max(0, y1 - 34)), f"GT {names.get(int(g['cls']), g['cls'])}", fill=(0, 0, 0, 255), font=font)
    for p in preds:
        x1, y1, x2, y2 = [float(v) for v in p["box"]]
        draw.rectangle([x1, y1, x2, y2], outline=(30, 220, 120, 255), width=max(4, image.width // 1000))
        draw.text((x1, max(0, y1 - 68)), f"P {names.get(int(p['cls']), p['cls'])} {p['conf']:.2f}", fill=(0, 120, 50, 255), font=font)
    scale = 1500 / image.width
    out = image.resize((1500, int(round(image.height * scale))), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root = Path(config["path"])
    image_dir = root / config[args.split]
    label_dir = root / config[args.split].replace("images", "labels")
    native_ann_dir = root / "native_annotations"
    if not native_ann_dir.exists():
        native_ann_dir = None
    names = {int(k): str(v) for k, v in dict(config["names"]).items()}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.max_images:
        images = images[: args.max_images]
    model = YOLO(args.weights)

    all_gt: list[dict] = []
    all_preds: list[dict] = []
    image_rows = []
    out = Path(args.out)
    overlay_dir = out / "overlays"
    for image_path in images:
        with Image.open(image_path) as im:
            width, height = im.size
        gt = read_yolo(label_dir / f"{image_path.stem}.txt", width, height)
        result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        preds = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confs = result.boxes.conf.detach().cpu().numpy()
            for box, cls, conf in zip(boxes, classes, confs):
                preds.append({"cls": int(cls), "box": box.astype(np.float32), "conf": float(conf)})
        all_gt.extend(gt)
        all_preds.extend(preds)
        center_hits = sum(any(center_inside(g["box"], p["box"]) for p in preds) for g in gt)
        ioa25_hits = sum(any(box_ioa_gt(g["box"], p["box"]) >= 0.25 for p in preds) for g in gt)
        image_rows.append(
            {
                "image": image_path.name,
                "gt": len(gt),
                "pred": len(preds),
                "gt_center_hit": center_hits,
                "gt_ioa25_hit": ioa25_hits,
            }
        )
        draw_overlay(image_path, gt, preds, names, native_ann_dir, overlay_dir / f"{image_path.stem}_diagnostic.png")

    rows = []
    for class_aware in (True, False):
        for thr in (0.10, 0.25, 0.50):
            rows.append(metric_row("iou_match", all_gt, all_preds, thr, class_aware))

    center_hits = sum(row["gt_center_hit"] for row in image_rows)
    ioa25_hits = sum(row["gt_ioa25_hit"] for row in image_rows)
    total_gt = max(len(all_gt), 1)
    rows.append(
        {
            "metric": "gt_center_inside_any_prediction",
            "threshold": 0.0,
            "class_aware": False,
            "tp": center_hits,
            "fp": "",
            "fn": len(all_gt) - center_hits,
            "precision": "",
            "recall": center_hits / total_gt,
            "f1": "",
        }
    )
    rows.append(
        {
            "metric": "gt_covered_by_prediction_ioa",
            "threshold": 0.25,
            "class_aware": False,
            "tp": ioa25_hits,
            "fp": "",
            "fn": len(all_gt) - ioa25_hits,
            "precision": "",
            "recall": ioa25_hits / total_gt,
            "f1": "",
        }
    )

    out.mkdir(parents=True, exist_ok=True)
    with (out / "relaxed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "per_image_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "weights": args.weights,
                "data": args.data,
                "images": len(images),
                "gt_boxes": len(all_gt),
                "pred_boxes": len(all_preds),
                "confidence": args.conf,
                "imgsz": args.imgsz,
                "interpretation": "Relaxed diagnostics for transfer/annotation QA; not a replacement for standard COCO mAP.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out.resolve())


if __name__ == "__main__":
    main()
