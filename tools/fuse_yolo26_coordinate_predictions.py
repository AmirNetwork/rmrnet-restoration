from __future__ import annotations

"""Fuse YOLO26-coordinate detections from complementary native-field views.

The G46/Sony native-field detector is strongest when it sees both the original
frame and a conservative RMR-Net evidence-preserving view.  This script fuses
the two detector outputs without using ground-truth annotations:

    P = NMS_primary({p in P_raw union P_rmr : conf(p) >= tau})

where ``NMS_primary`` suppresses boxes only inside the same primary defect
family (crack vs pothole).  The fused CSV keeps the same column schema emitted
by ``Yolo26_coordinate_revised.py`` so it can be evaluated by
``eval_yolo26_coordinate_gt46.py``.
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


PRED_TO_PRIMARY = {
    "Alligator Crack (D20)": "crack",
    "Longitudinal Crack (D00)": "crack",
    "Transverse Crack (D10)": "crack",
    "Pothole (D40)": "pothole",
    "Potholes": "pothole",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, nargs="+", required=True, help="Detector output folders to fuse.")
    parser.add_argument("--out", type=Path, required=True, help="Output detector folder with fused CSV rows.")
    parser.add_argument("--conf-min", type=float, default=0.08, help="Minimum confidence retained before fusion.")
    parser.add_argument("--nms-iou", type=float, default=0.20, help="Primary-class NMS IoU threshold.")
    parser.add_argument("--copy-views-from", type=Path, default=None, help="Optional detector folder whose view images are copied for browsing.")
    return parser.parse_args()


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def row_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    )


def load_rows(root: Path, conf_min: float) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    fieldnames: list[str] = []
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for csv_path in sorted(root.glob("*.csv")):
        if csv_path.name.startswith("_"):
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and not fieldnames:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                class_name = row.get("class_name", "")
                if class_name not in PRED_TO_PRIMARY:
                    continue
                if float(row.get("confidence", "0") or 0.0) < conf_min:
                    continue
                row = dict(row)
                row["_primary"] = PRED_TO_PRIMARY[class_name]
                by_image[row["image"]].append(row)
    return fieldnames, by_image


def fuse_rows(all_rows: list[dict[str, str]], nms_iou: float) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    ordered = sorted(all_rows, key=lambda r: float(r.get("confidence", "0") or 0.0), reverse=True)
    for row in ordered:
        primary = row["_primary"]
        box = row_box(row)
        duplicate = False
        for old in kept:
            if old["_primary"] != primary:
                continue
            if box_iou(box, row_box(old)) >= nms_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return kept


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    merged: dict[str, list[dict[str, str]]] = defaultdict(list)
    fieldnames: list[str] = []
    for root in args.pred_root:
        names, rows = load_rows(root, args.conf_min)
        if names and not fieldnames:
            fieldnames = names
        for image, image_rows in rows.items():
            merged[image].extend(image_rows)
    if not fieldnames:
        raise RuntimeError("No prediction CSV rows found to fuse.")

    fused: list[dict[str, str]] = []
    for image in sorted(merged):
        for row in fuse_rows(merged[image], args.nms_iou):
            clean = {k: row.get(k, "") for k in fieldnames}
            clean["min_conf_used"] = str(args.conf_min)
            fused.append(clean)

    out_csv = args.out / "fused_predictions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fused)

    if args.copy_views_from and args.copy_views_from.exists():
        for image_path in args.copy_views_from.glob("*_view_*.jpg"):
            shutil.copy2(image_path, args.out / image_path.name)

    print({"out": str(out_csv), "rows": len(fused), "images": len(merged), "conf_min": args.conf_min, "nms_iou": args.nms_iou})


if __name__ == "__main__":
    main()
