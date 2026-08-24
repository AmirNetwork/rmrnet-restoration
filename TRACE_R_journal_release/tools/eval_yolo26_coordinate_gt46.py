# TRACE-R release integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

"""Evaluate Yolo26_coordinate.py outputs on revised GT46 COCO annotations.

This evaluator is intentionally separate from the older GT49 crop/mask tools.
It consumes the per-class CSV files written by the user-provided
Yolo26_coordinate.py script and scores them against the revised Roboflow COCO
export at native 4752 x 3168 image coordinates.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COMMON_GT = {
    "Alligator Crack",
    "Longitudinal Crack",
    "Transverse Crack",
    "Potholes",
}

PRED_TO_GT = {
    "Alligator Crack (D20)": "Alligator Crack",
    "Longitudinal Crack (D00)": "Longitudinal Crack",
    "Transverse Crack (D10)": "Transverse Crack",
    "Pothole (D40)": "Potholes",
    "Potholes": "Potholes",
}

COLORS = {
    "Alligator Crack": (255, 120, 80),
    "Longitudinal Crack": (32, 180, 255),
    "Transverse Crack": (60, 220, 120),
    "Potholes": (220, 80, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, nargs="+", required=True)
    parser.add_argument("--pred-name", nargs="+", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--atlas-max", type=int, default=18)
    return parser.parse_args()


def native_name(file_name: str, extra: dict | None = None) -> str:
    if extra and extra.get("name"):
        return str(extra["name"])
    m = re.match(r"^(.*)_jpg\.rf\.[^.]+\.jpg$", file_name, re.I)
    if m:
        return m.group(1) + ".jpg"
    return file_name


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


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
    return inter / denom if denom else 0.0


def cover_gt(pred: tuple[float, float, float, float], gt: tuple[float, float, float, float]) -> float:
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    return inter / gt_area if gt_area else 0.0


def center_inside(pred: tuple[float, float, float, float], gt: tuple[float, float, float, float]) -> bool:
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt
    cx, cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
    return px1 <= cx <= px2 and py1 <= cy <= py2


def primary(label: str) -> str:
    return "pothole" if label == "Potholes" else "crack"


def load_gt(path: Path) -> tuple[dict[str, list[dict]], Counter]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(c["id"]): c["name"] for c in data["categories"]}
    image_by_id = {int(im["id"]): im for im in data["images"]}
    gt: dict[str, list[dict]] = defaultdict(list)
    counts = Counter()
    for ann in data["annotations"]:
        label = categories.get(int(ann["category_id"]), "")
        if label not in COMMON_GT:
            continue
        im = image_by_id[int(ann["image_id"])]
        name = native_name(im["file_name"], im.get("extra"))
        item = {
            "image": name,
            "label": label,
            "primary": primary(label),
            "box": xywh_to_xyxy([float(v) for v in ann["bbox"]]),
        }
        gt[name].append(item)
        counts[label] += 1
    return dict(gt), counts


def load_predictions(root: Path) -> dict[str, list[dict]]:
    preds: dict[str, list[dict]] = defaultdict(list)
    for csv_path in sorted(root.glob("*.csv")):
        if csv_path.name.startswith("_"):
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = PRED_TO_GT.get(row.get("class_name", ""))
                if label is None:
                    continue
                box = (
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                )
                # Restored views are lossless PNGs while the native COCO
                # records are JPEGs. They share the same capture stem, so the
                # extension must not determine sample identity.
                image = Path(row["image"]).stem + ".jpg"
                preds[image].append(
                    {
                        "image": image,
                        "label": label,
                        "primary": primary(label),
                        "conf": float(row["confidence"]),
                        "box": box,
                    }
                )
    return dict(preds)


def greedy_match(gt_items: list[dict], pred_items: list[dict], threshold: float, mode: str) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_items):
        for pi, p in enumerate(pred_items):
            if mode == "class" and g["label"] != p["label"]:
                continue
            if mode == "primary" and g["primary"] != p["primary"]:
                continue
            iou = box_iou(g["box"], p["box"])
            if iou >= threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_g, used_p = set(), set()
    for _iou, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
    tp = len(used_g)
    fp = len(pred_items) - len(used_p)
    fn = len(gt_items) - len(used_g)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def average_precision_101(recalls: list[float], precisions: list[float]) -> float:
    """COCO-style 101-point interpolated AP for one category and IoU."""

    if not recalls:
        return 0.0
    values = []
    for threshold in (index / 100.0 for index in range(101)):
        candidates = [precision for recall, precision in zip(recalls, precisions) if recall >= threshold]
        values.append(max(candidates, default=0.0))
    return sum(values) / len(values)


def corpus_ap(
    gt: dict[str, list[dict]],
    preds: dict[str, list[dict]],
    threshold: float,
    mode: str = "primary",
) -> float:
    """Macro AP over crack/pothole families or the exact four classes.

    Predictions are ranked globally by confidence. Each prediction can match at
    most one ground-truth box in the same image and category. The returned value
    is the unweighted mean over categories with at least one ground-truth box.
    """

    if mode == "primary":
        categories = ("crack", "pothole")
        key = "primary"
    elif mode == "class":
        categories = tuple(sorted(COMMON_GT))
        key = "label"
    else:
        raise ValueError(f"Unknown AP mode: {mode}")

    per_category: list[float] = []
    for category in categories:
        gt_category = {
            image: [item for item in items if item[key] == category]
            for image, items in gt.items()
        }
        n_gt = sum(len(items) for items in gt_category.values())
        if n_gt == 0:
            continue
        ranked = sorted(
            (
                item
                for image_items in preds.values()
                for item in image_items
                if item[key] == category
            ),
            key=lambda item: float(item["conf"]),
            reverse=True,
        )
        matched: dict[str, set[int]] = defaultdict(set)
        tp = fp = 0
        recalls: list[float] = []
        precisions: list[float] = []
        for prediction in ranked:
            image = str(prediction["image"])
            candidates = [
                (box_iou(prediction["box"], target["box"]), index)
                for index, target in enumerate(gt_category.get(image, []))
                if index not in matched[image]
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= threshold:
                matched[image].add(best_index)
                tp += 1
            else:
                fp += 1
            recalls.append(tp / n_gt)
            precisions.append(tp / (tp + fp))
        per_category.append(average_precision_101(recalls, precisions))
    return sum(per_category) / len(per_category) if per_category else 0.0


def success_count(gt_items: list[dict], pred_items: list[dict]) -> int:
    count = 0
    for g in gt_items:
        ok = False
        for p in pred_items:
            if g["primary"] != p["primary"]:
                continue
            if box_iou(g["box"], p["box"]) >= 0.10 or cover_gt(p["box"], g["box"]) >= 0.25 or center_inside(p["box"], g["box"]):
                ok = True
                break
        count += int(ok)
    return count


def evaluate_one(name: str, gt: dict[str, list[dict]], preds: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    images = sorted(gt)
    total_gt = sum(len(gt[i]) for i in images)
    total_pred = sum(len(preds.get(i, [])) for i in images)
    rows: list[dict] = []
    for mode in ["primary", "class"]:
        for thr in [0.10, 0.25, 0.50]:
            tp = fp = fn = 0
            for image in images:
                a, b, c = greedy_match(gt[image], preds.get(image, []), thr, mode)
                tp += a
                fp += b
                fn += c
            p, r, f1 = prf(tp, fp, fn)
            rows.append(
                {
                    "run": name,
                    "mode": mode,
                    "iou": thr,
                    "images": len(images),
                    "gt": total_gt,
                    "pred": total_pred,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                }
            )
    succ = sum(success_count(gt[i], preds.get(i, [])) for i in images)
    rows.append(
        {
            "run": name,
            "mode": "primary_success",
            "iou": -1,
            "images": len(images),
            "gt": total_gt,
            "pred": total_pred,
            "tp": succ,
            "fp": "",
            "fn": total_gt - succ,
            "precision": "",
            "recall": succ / total_gt if total_gt else 0.0,
            "f1": "",
        }
    )

    class_rows = []
    for label in sorted(COMMON_GT):
        gt_label = {i: [g for g in gt[i] if g["label"] == label] for i in images}
        pred_label = {i: [p for p in preds.get(i, []) if p["label"] == label] for i in images}
        tp = fp = fn = 0
        for image in images:
            a, b, c = greedy_match(gt_label[image], pred_label.get(image, []), 0.10, "class")
            tp += a
            fp += b
            fn += c
        p, r, f1 = prf(tp, fp, fn)
        class_rows.append({"run": name, "label": label, "gt": sum(len(v) for v in gt_label.values()), "pred": sum(len(v) for v in pred_label.values()), "tp_iou10": tp, "precision_iou10": p, "recall_iou10": r, "f1_iou10": f1})
    return rows, class_rows


def fmt(x) -> str:
    if x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def render_overlays(gt: dict[str, list[dict]], preds_by_run: dict[str, dict[str, list[dict]]], images_dir: Path, out_dir: Path, atlas_max: int) -> None:
    overlay_dir = out_dir / "visuals"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    atlas_panels = []
    for idx, image in enumerate(sorted(gt)):
        base = Image.open(images_dir / image).convert("RGB")
        base.thumbnail((960, 640), Image.Resampling.LANCZOS)
        scale_x = base.width / 4752.0
        scale_y = base.height / 3168.0
        panels = []
        for run, preds in preds_by_run.items():
            panel = base.copy()
            d = ImageDraw.Draw(panel)
            d.rectangle((0, 0, panel.width, 24), fill=(255, 255, 255))
            d.text((5, 5), f"{run} | GT={len(gt[image])}, pred={len(preds.get(image, []))}", fill=(0, 0, 0), font=font)
            for g in gt[image]:
                x1, y1, x2, y2 = g["box"]
                d.rectangle((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y), outline=(255, 214, 10), width=2)
                d.text((x1 * scale_x, max(24, y1 * scale_y - 10)), "GT " + g["label"].split()[0], fill=(255, 214, 10), font=font)
            for p in preds.get(image, []):
                x1, y1, x2, y2 = p["box"]
                color = COLORS[p["label"]]
                d.rectangle((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y), outline=color, width=2)
                d.text((x1 * scale_x, max(36, y1 * scale_y)), f"{p['label'].split()[0]} {p['conf']:.2f}", fill=color, font=font)
            panels.append(panel)
        combined = Image.new("RGB", (sum(p.width for p in panels) + 8 * (len(panels) - 1), max(p.height for p in panels)), (255, 255, 255))
        x = 0
        for p in panels:
            combined.paste(p, (x, 0))
            x += p.width + 8
        out_path = overlay_dir / f"{Path(image).stem}_gt_vs_predictions.jpg"
        combined.save(out_path, quality=92)
        if idx < atlas_max:
            atlas_panels.append(combined)
    if atlas_panels:
        width = max(p.width for p in atlas_panels)
        height = sum(p.height for p in atlas_panels) + 12 * (len(atlas_panels) - 1)
        atlas = Image.new("RGB", (width, height), (255, 255, 255))
        y = 0
        for p in atlas_panels:
            atlas.paste(p, (0, y))
            y += p.height + 12
        atlas.save(out_dir / "gt46_yolo26_coordinate_atlas.jpg", quality=92)


def write_table(rows: list[dict], class_rows: list[dict], out: Path) -> None:
    summary_csv = out / "summary_metrics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    class_csv = out / "class_metrics_iou10.csv"
    with class_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    key_rows = [r for r in rows if r["mode"] == "primary" and math.isclose(float(r["iou"]), 0.10)]
    key_rows += [r for r in rows if r["mode"] == "primary" and math.isclose(float(r["iou"]), 0.50)]
    success = {r["run"]: r for r in rows if r["mode"] == "primary_success"}
    lines = [
        "# GT46 YOLO26 Coordinate Evaluation",
        "",
        "| Run | Images | GT | Pred | GT success | P@0.10 | R@0.10 | F1@0.10 | F1@0.50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    runs = []
    for r in rows:
        if r["run"] not in runs:
            runs.append(r["run"])
    for run in runs:
        r10 = next(r for r in rows if r["run"] == run and r["mode"] == "primary" and math.isclose(float(r["iou"]), 0.10))
        r50 = next(r for r in rows if r["run"] == run and r["mode"] == "primary" and math.isclose(float(r["iou"]), 0.50))
        succ = success[run]["recall"]
        lines.append(
            f"| {run} | {r10['images']} | {r10['gt']} | {r10['pred']} | {succ:.3f} | {r10['precision']:.3f} | {r10['recall']:.3f} | {r10['f1']:.3f} | {r50['f1']:.3f} |"
        )
    lines.extend(["", "## Per-Class Exact-Class Metrics at IoU 0.10", "", "| Run | Class | GT | Pred | TP | Precision | Recall | F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in class_rows:
        lines.append(f"| {r['run']} | {r['label']} | {r['gt']} | {r['pred']} | {r['tp_iou10']} | {r['precision_iou10']:.3f} | {r['recall_iou10']:.3f} | {r['f1_iou10']:.3f} |")
    (out / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(args.pred_root) != len(args.pred_name):
        raise ValueError("--pred-root and --pred-name must have the same length")
    args.out.mkdir(parents=True, exist_ok=True)
    gt, gt_counts = load_gt(args.annotations)
    preds_by_run = {name: load_predictions(root) for name, root in zip(args.pred_name, args.pred_root)}
    rows_all: list[dict] = []
    class_all: list[dict] = []
    ap_rows: list[dict] = []
    for name, preds in preds_by_run.items():
        rows, class_rows = evaluate_one(name, gt, preds)
        rows_all.extend(rows)
        class_all.extend(class_rows)
        for mode in ("primary", "class"):
            for iou in (0.10, 0.50):
                ap_rows.append(
                    {
                        "run": name,
                        "mode": mode,
                        "iou": iou,
                        "ap_101": corpus_ap(gt, preds, iou, mode),
                        "implementation": "101-point interpolated corpus AP",
                    }
                )
    write_table(rows_all, class_all, args.out)
    with (args.out / "ap_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ap_rows[0]))
        writer.writeheader()
        writer.writerows(ap_rows)
    render_overlays(gt, preds_by_run, args.images, args.out, args.atlas_max)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "annotations": str(args.annotations),
                "images": str(args.images),
                "gt_common_counts": dict(gt_counts),
                "prediction_roots": {name: str(root) for name, root in zip(args.pred_name, args.pred_root)},
                "notes": "No crop/mask/tile protocol. Predictions come from Yolo26_coordinate.py per-class CSVs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print((args.out / "README_RESULTS.md").resolve())


if __name__ == "__main__":
    main()
