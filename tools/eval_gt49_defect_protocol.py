from __future__ import annotations

"""GT49 defect-primary evaluation with crack-fragment fusion.

The GT49 native field set has fine crack subtypes, but the practical first
question is whether a defect was localized. This evaluator therefore reports:

1. defect-localization metrics: all crack subtypes and potholes are defects;
2. primary detection metrics: crack group versus pothole;
3. subtype metrics: alligator, longitudinal, transverse, pothole;
4. fragmentation diagnostics: how often one GT defect is covered by many boxes.

No GT49 labels are used for training or threshold selection here. The fusion
rules are fixed geometry rules intended to remove duplicate crack fragments and
non-defect classes from already-generated detector predictions.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


DEFECT_CLASSES = {0, 1, 3, 5}
CRACK_CLASSES = {0, 1, 5}
POTHOLE_CLASS = 3
LONGITUDINAL_CLASS = 1
TRANSVERSE_CLASS = 5
ALLIGATOR_CLASS = 0
CLASS_NAMES = {
    0: "alligator_crack",
    1: "longitudinal_crack",
    3: "pothole",
    5: "transverse_crack",
}


@dataclass
class Box:
    image: str
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0
    source: str = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="GT49 YOLO data.yaml with native labels")
    parser.add_argument(
        "--eval-root",
        required=True,
        type=Path,
        help="Directory containing method subfolders with predictions.csv",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--methods",
        default="raw,rmr_blind,rmr_metadata,rmr_metadata_gated,nafnet,dfpir,demoe_auto,demoe_scenario,instructir_generic,instructir_metadata",
    )
    parser.add_argument("--duplicate-iou", type=float, default=0.15)
    parser.add_argument("--duplicate-ioa", type=float, default=0.30)
    parser.add_argument("--merge-gap", type=float, default=160.0)
    parser.add_argument("--defect-classes", default="0,1,3,5")
    parser.add_argument("--crack-classes", default="0,1,5")
    parser.add_argument("--pothole-class", type=int, default=3)
    parser.add_argument("--longitudinal-class", type=int, default=1)
    parser.add_argument("--transverse-class", type=int, default=5)
    parser.add_argument("--alligator-class", type=int, default=0)
    parser.add_argument(
        "--crop-bottom-half",
        action="store_true",
        help=(
            "Evaluate only GT boxes whose centers lie in the lower half of the native frame. "
            "Use this with the modified YOLO26 crop/mask detector protocol."
        ),
    )
    return parser.parse_args()


def parse_class_set(text: str) -> set[int]:
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def read_yaml(path: Path) -> tuple[Path, dict, dict[int, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    raw_names = data.get("names", {})
    if isinstance(raw_names, list):
        names = {idx: str(name) for idx, name in enumerate(raw_names)}
    else:
        names = {int(k): str(v) for k, v in dict(raw_names).items()}
    return root, data, names


def split_dir(root: Path, cfg: dict, split: str, kind: str) -> Path:
    value = str(cfg.get(split, cfg.get("val", f"images/{split}")))
    path = Path(value.replace("images", kind, 1))
    return path if path.is_absolute() else root / path


def read_gt(data_yaml: Path, split: str, *, crop_bottom_half: bool = False) -> dict[str, list[Box]]:
    root, cfg, _names = read_yaml(data_yaml)
    images = split_dir(root, cfg, split, "images")
    labels = split_dir(root, cfg, split, "labels")
    result: dict[str, list[Box]] = {}
    for image_path in sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        with Image.open(image_path) as image:
            width, height = image.size
        boxes: list[Box] = []
        label_path = labels / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                if cls not in DEFECT_CLASSES:
                    continue
                xc, yc, bw, bh = (float(v) for v in parts[1:5])
                x1 = (xc - bw / 2.0) * width
                y1 = (yc - bh / 2.0) * height
                x2 = (xc + bw / 2.0) * width
                y2 = (yc + bh / 2.0) * height
                if crop_bottom_half and 0.5 * (y1 + y2) < height // 2:
                    continue
                boxes.append(Box(image=image_path.name, cls=cls, xyxy=(x1, y1, x2, y2)))
        result[image_path.name] = boxes
    return result


def read_preds(path: Path) -> dict[str, list[Box]]:
    rows: dict[str, list[Box]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cls = int(float(row["class_id"]))
            if cls not in DEFECT_CLASSES:
                continue
            box = Box(
                image=row["image"],
                cls=cls,
                xyxy=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                conf=float(row["conf"]),
                source=row.get("source", "unknown"),
            )
            rows.setdefault(box.image, []).append(box)
    return rows


def area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = intersection(a, b)
    denom = area(a) + area(b) - inter
    return inter / denom if denom > 0 else 0.0


def ioa_min(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = intersection(a, b)
    denom = min(area(a), area(b))
    return inter / denom if denom > 0 else 0.0


def ioa_gt(pred: tuple[float, float, float, float], gt: tuple[float, float, float, float]) -> float:
    denom = area(gt)
    return intersection(pred, gt) / denom if denom > 0 else 0.0


def primary_class(cls: int) -> int:
    return 100 if cls in CRACK_CLASSES else cls


def match_class(pred_cls: int, gt_cls: int, mode: str) -> bool:
    if mode == "defect":
        return True
    if mode == "primary":
        return primary_class(pred_cls) == primary_class(gt_cls)
    if mode == "subtype":
        return pred_cls == gt_cls
    raise ValueError(mode)


def centers(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def gap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0.0


def should_merge_cracks(a: Box, b: Box, duplicate_iou: float, duplicate_ioa: float, merge_gap: float) -> bool:
    if a.cls not in CRACK_CLASSES or b.cls not in CRACK_CLASSES:
        return False
    if box_iou(a.xyxy, b.xyxy) >= duplicate_iou or ioa_min(a.xyxy, b.xyxy) >= duplicate_ioa:
        return True
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    y_overlap = overlap_1d(ay1, ay2, by1, by2) / max(1.0, min(ah, bh))
    x_overlap = overlap_1d(ax1, ax2, bx1, bx2) / max(1.0, min(aw, bw))
    x_gap = gap_1d(ax1, ax2, bx1, bx2)
    y_gap = gap_1d(ay1, ay2, by1, by2)
    if y_overlap >= 0.30 and x_gap <= max(merge_gap, 0.20 * max(aw, bw)):
        return True
    if x_overlap >= 0.30 and y_gap <= max(merge_gap, 0.20 * max(ah, bh)):
        return True
    return False


def geometry_relabel_crack(xyxy: tuple[float, float, float, float]) -> int:
    x1, y1, x2, y2 = xyxy
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    aspect = width / height
    if aspect >= 2.2:
        return TRANSVERSE_CLASS  # transverse in image coordinates
    if aspect <= 0.75:
        return LONGITUDINAL_CLASS  # longitudinal in image coordinates
    return ALLIGATOR_CLASS  # alligator/area-like crack


def merge_cluster(boxes: list[Box]) -> Box:
    x1 = min(b.xyxy[0] for b in boxes)
    y1 = min(b.xyxy[1] for b in boxes)
    x2 = max(b.xyxy[2] for b in boxes)
    y2 = max(b.xyxy[3] for b in boxes)
    conf = max(b.conf for b in boxes)
    image = boxes[0].image
    source = "+".join(sorted({b.source for b in boxes}))
    cls = geometry_relabel_crack((x1, y1, x2, y2))
    return Box(image=image, cls=cls, xyxy=(x1, y1, x2, y2), conf=conf, source=source)


def fuse_predictions(preds: list[Box], duplicate_iou: float, duplicate_ioa: float, merge_gap: float) -> list[Box]:
    potholes = [p for p in preds if p.cls == POTHOLE_CLASS]
    cracks = [p for p in preds if p.cls in CRACK_CLASSES]

    # Group-aware NMS for potholes.
    kept_potholes: list[Box] = []
    for box in sorted(potholes, key=lambda b: b.conf, reverse=True):
        if all(box_iou(box.xyxy, kept.xyxy) < 0.45 for kept in kept_potholes):
            kept_potholes.append(box)

    clusters: list[list[Box]] = []
    for box in sorted(cracks, key=lambda b: b.conf, reverse=True):
        placed = False
        for cluster in clusters:
            if any(should_merge_cracks(box, other, duplicate_iou, duplicate_ioa, merge_gap) for other in cluster):
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])

    changed = True
    while changed:
        changed = False
        merged: list[list[Box]] = []
        while clusters:
            cluster = clusters.pop(0)
            idx = 0
            while idx < len(clusters):
                if any(
                    should_merge_cracks(a, b, duplicate_iou, duplicate_ioa, merge_gap)
                    for a in cluster
                    for b in clusters[idx]
                ):
                    cluster.extend(clusters.pop(idx))
                    changed = True
                else:
                    idx += 1
            merged.append(cluster)
        clusters = merged

    fused = kept_potholes + [merge_cluster(cluster) for cluster in clusters]
    return sorted(fused, key=lambda b: b.conf, reverse=True)


def group_dedup_predictions(preds: list[Box], duplicate_iou: float, duplicate_ioa: float) -> list[Box]:
    """Remove duplicate boxes within crack-group/pothole classes without unioning.

    This is deliberately milder than fragment fusion. It fixes the common
    alligator/longitudinal/transverse duplicate-NMS failure while preserving
    separate boxes for nearby physical defects.
    """

    kept: list[Box] = []
    for pred in sorted(preds, key=lambda b: b.conf, reverse=True):
        duplicate = False
        for other in kept:
            if primary_class(pred.cls) != primary_class(other.cls):
                continue
            if box_iou(pred.xyxy, other.xyxy) >= duplicate_iou or ioa_min(pred.xyxy, other.xyxy) >= duplicate_ioa:
                duplicate = True
                break
        if not duplicate:
            kept.append(pred)
    return kept


def greedy_counts(gts: list[Box], preds: list[Box], threshold: float, mode: str) -> dict[str, float]:
    matched: set[int] = set()
    tp = 0
    fp = 0
    for pred in sorted(preds, key=lambda b: b.conf, reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gts):
            if idx in matched or not match_class(pred.cls, gt.cls, mode):
                continue
            iou = box_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= threshold:
            matched.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = len(gts) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(gts) if gts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def coverage_recall(gts: list[Box], preds: list[Box], threshold: float, mode: str) -> float:
    if not gts:
        return 0.0
    hits = 0
    for gt in gts:
        best = 0.0
        for pred in preds:
            if match_class(pred.cls, gt.cls, mode):
                best = max(best, ioa_gt(pred.xyxy, gt.xyxy))
        hits += int(best >= threshold)
    return hits / len(gts)


def center_recall(gts: list[Box], preds: list[Box], mode: str) -> float:
    if not gts:
        return 0.0
    hits = 0
    for gt in gts:
        cx, cy = centers(gt.xyxy)
        found = False
        for pred in preds:
            if not match_class(pred.cls, gt.cls, mode):
                continue
            x1, y1, x2, y2 = pred.xyxy
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                found = True
                break
        hits += int(found)
    return hits / len(gts)


def tolerant_gt_success(gts: list[Box], preds: list[Box], mode: str) -> dict[str, float]:
    """Recover labeled GT without over-trusting exact box overlap.

    Formula used in the paper for GT49:

        success(g, P) = 1[ max_p same_class (
            IoU(p, g) >= 0.10 OR IoA_gt(p, g) >= 0.25 OR center(g) in p
        ) ]

    This is intentionally recall-oriented.  GT49 labels are manually drawn and
    may miss visible non-annotated road defects, so extra predictions should be
    audited visually rather than automatically treated as equally severe errors.
    """

    if not gts:
        return {"count": 0, "recall": 0.0}
    hits = 0
    for gt in gts:
        cx, cy = centers(gt.xyxy)
        found = False
        for pred in preds:
            if not match_class(pred.cls, gt.cls, mode):
                continue
            x1, y1, x2, y2 = pred.xyxy
            center_inside = x1 <= cx <= x2 and y1 <= cy <= y2
            if box_iou(pred.xyxy, gt.xyxy) >= 0.10 or ioa_gt(pred.xyxy, gt.xyxy) >= 0.25 or center_inside:
                found = True
                break
        hits += int(found)
    return {"count": hits, "recall": hits / len(gts)}


def type_accuracy(gts: list[Box], preds: list[Box], threshold: float = 0.10) -> tuple[float, list[dict[str, object]]]:
    matched: set[int] = set()
    correct = 0
    total = 0
    rows: list[dict[str, object]] = []
    for pred in sorted(preds, key=lambda b: b.conf, reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gts):
            if idx in matched or primary_class(pred.cls) != primary_class(gt.cls):
                continue
            iou = box_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= threshold:
            gt = gts[best_idx]
            matched.add(best_idx)
            total += 1
            correct += int(pred.cls == gt.cls)
            rows.append(
                {
                    "gt_class": gt.cls,
                    "gt_name": CLASS_NAMES.get(gt.cls, str(gt.cls)),
                    "pred_class": pred.cls,
                    "pred_name": CLASS_NAMES.get(pred.cls, str(pred.cls)),
                    "iou": best_iou,
                }
            )
    return (correct / total if total else 0.0), rows


def fragmentation_stats(gts: list[Box], preds: list[Box]) -> dict[str, float]:
    counts: list[int] = []
    for gt in gts:
        count = 0
        for pred in preds:
            if primary_class(pred.cls) == primary_class(gt.cls) and ioa_gt(pred.xyxy, gt.xyxy) >= 0.10:
                count += 1
        counts.append(count)
    matched = [c for c in counts if c > 0]
    return {
        "covered_gt": len(matched),
        "mean_fragments_per_covered_gt": sum(matched) / len(matched) if matched else 0.0,
        "overfragmented_gt_fraction": sum(1 for c in counts if c > 1) / len(counts) if counts else 0.0,
    }


def flatten(values: Iterable[list[Box]]) -> list[Box]:
    return [box for boxes in values for box in boxes]


def aggregate_greedy_counts(
    gts_by_image: dict[str, list[Box]],
    preds_by_image: dict[str, list[Box]],
    threshold: float,
    mode: str,
) -> dict[str, float]:
    """Sum greedy matching per image before computing precision/recall.

    This prevents a prediction from one frame from matching a GT box in another
    frame.  Earlier flattened matching is invalid for GT49 and any other
    multi-image detection evaluation.
    """

    tp = fp = fn = 0
    for image, gts in gts_by_image.items():
        counts = greedy_counts(gts, preds_by_image.get(image, []), threshold, mode)
        tp += int(counts["tp"])
        fp += int(counts["fp"])
        fn += int(counts["fn"])
    for image, preds in preds_by_image.items():
        if image not in gts_by_image:
            fp += len(preds)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def aggregate_recall(
    gts_by_image: dict[str, list[Box]],
    preds_by_image: dict[str, list[Box]],
    mode: str,
    fn,
) -> float:
    total = sum(len(gts) for gts in gts_by_image.values())
    if total == 0:
        return 0.0
    hits = 0.0
    for image, gts in gts_by_image.items():
        value = fn(gts, preds_by_image.get(image, []), mode)
        hits += value * len(gts)
    return hits / total


def aggregate_tolerant_gt_success(
    gts_by_image: dict[str, list[Box]],
    preds_by_image: dict[str, list[Box]],
    mode: str,
) -> dict[str, float]:
    total = sum(len(gts) for gts in gts_by_image.values())
    hits = 0
    for image, gts in gts_by_image.items():
        result = tolerant_gt_success(gts, preds_by_image.get(image, []), mode)
        hits += int(result["count"])
    return {"count": hits, "recall": hits / total if total else 0.0}


def aggregate_fragmentation_stats(
    gts_by_image: dict[str, list[Box]],
    preds_by_image: dict[str, list[Box]],
) -> dict[str, float]:
    counts: list[int] = []
    for image, gts in gts_by_image.items():
        preds = preds_by_image.get(image, [])
        for gt in gts:
            count = 0
            for pred in preds:
                if primary_class(pred.cls) == primary_class(gt.cls) and ioa_gt(pred.xyxy, gt.xyxy) >= 0.10:
                    count += 1
            counts.append(count)
    matched = [c for c in counts if c > 0]
    return {
        "covered_gt": len(matched),
        "mean_fragments_per_covered_gt": sum(matched) / len(matched) if matched else 0.0,
        "overfragmented_gt_fraction": sum(1 for c in counts if c > 1) / len(counts) if counts else 0.0,
    }


def aggregate_type_accuracy(
    gts_by_image: dict[str, list[Box]],
    preds_by_image: dict[str, list[Box]],
    threshold: float = 0.10,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for image, gts in gts_by_image.items():
        acc, image_rows = type_accuracy(gts, preds_by_image.get(image, []), threshold)
        for row in image_rows:
            rows.append({"image": image, **row})
    correct = sum(1 for row in rows if row["gt_class"] == row["pred_class"])
    return (correct / len(rows) if rows else 0.0), rows


def per_class_rows(method: str, stage: str, gts: list[Box], preds: list[Box]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cls in sorted(DEFECT_CLASSES):
        gt_cls = [g for g in gts if g.cls == cls]
        pred_cls = [p for p in preds if p.cls == cls]
        counts = greedy_counts(gt_cls, pred_cls, 0.10, "subtype")
        rows.append({"method": method, "stage": stage, "class_id": cls, "class_name": CLASS_NAMES.get(cls, str(cls)), **counts})
    return rows


def per_class_rows_by_image(
    method: str,
    stage: str,
    gts_by_image: dict[str, list[Box]],
    preds_by_image: dict[str, list[Box]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cls in sorted(DEFECT_CLASSES):
        gt_cls_by = {image: [g for g in gts if g.cls == cls] for image, gts in gts_by_image.items()}
        pred_cls_by = {image: [p for p in preds if p.cls == cls] for image, preds in preds_by_image.items()}
        counts = aggregate_greedy_counts(gt_cls_by, pred_cls_by, 0.10, "subtype")
        rows.append({"method": method, "stage": stage, "class_id": cls, "class_name": CLASS_NAMES.get(cls, str(cls)), **counts})
    return rows


def align_predictions_to_gt_images(
    gts_by_image: dict[str, list[Box]],
    preds_by_image: dict[str, list[Box]],
) -> dict[str, list[Box]]:
    aligned = {image: list(preds_by_image.get(image, [])) for image in gts_by_image}
    for image, preds in preds_by_image.items():
        if image not in aligned:
            aligned[image] = list(preds)
    return aligned


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global DEFECT_CLASSES, CRACK_CLASSES, POTHOLE_CLASS, LONGITUDINAL_CLASS, TRANSVERSE_CLASS, ALLIGATOR_CLASS, CLASS_NAMES
    args = parse_args()
    DEFECT_CLASSES = parse_class_set(args.defect_classes)
    CRACK_CLASSES = parse_class_set(args.crack_classes)
    POTHOLE_CLASS = args.pothole_class
    LONGITUDINAL_CLASS = args.longitudinal_class
    TRANSVERSE_CLASS = args.transverse_class
    ALLIGATOR_CLASS = args.alligator_class
    _root, _cfg, yaml_names = read_yaml(args.data)
    CLASS_NAMES = {idx: yaml_names.get(idx, str(idx)) for idx in sorted(DEFECT_CLASSES)}
    args.out.mkdir(parents=True, exist_ok=True)
    all_gts = read_gt(args.data, args.split, crop_bottom_half=args.crop_bottom_half)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    flat_gts = flatten(all_gts.values())
    for method in methods:
        pred_csv = args.eval_root / method / "predictions.csv"
        if not pred_csv.exists():
            continue
        original_by_image = read_preds(pred_csv)
        fused_by_image = {
            image: fuse_predictions(preds, args.duplicate_iou, args.duplicate_ioa, args.merge_gap)
            for image, preds in original_by_image.items()
        }
        dedup_by_image = {
            image: group_dedup_predictions(preds, args.duplicate_iou, args.duplicate_ioa)
            for image, preds in original_by_image.items()
        }
        for stage, by_image in [("filtered", original_by_image), ("dedup", dedup_by_image), ("fused", fused_by_image)]:
            aligned_by_image = align_predictions_to_gt_images(all_gts, by_image)
            flat_preds = flatten(aligned_by_image.values())
            type_acc, confusion = aggregate_type_accuracy(all_gts, aligned_by_image, 0.10)
            frag = aggregate_fragmentation_stats(all_gts, aligned_by_image)
            row: dict[str, object] = {
                "method": method,
                "stage": stage,
                "images": len(all_gts),
                "gt_defects": len(flat_gts),
                "pred_defects": len(flat_preds),
                "type_accuracy_iou10_primary_matches": type_acc,
                **frag,
            }
            for mode in ["defect", "primary", "subtype"]:
                for thr in [0.10, 0.25, 0.50]:
                    counts = aggregate_greedy_counts(all_gts, aligned_by_image, thr, mode)
                    for key in ["precision", "recall", "f1"]:
                        row[f"{mode}_{key}_iou{int(thr*100):02d}"] = counts[key]
                row[f"{mode}_coverage_ioa25"] = aggregate_recall(
                    all_gts,
                    aligned_by_image,
                    mode,
                    lambda gts, preds, match_mode: coverage_recall(gts, preds, 0.25, match_mode),
                )
                row[f"{mode}_center_recall"] = aggregate_recall(all_gts, aligned_by_image, mode, center_recall)
                success = aggregate_tolerant_gt_success(all_gts, aligned_by_image, mode)
                row[f"{mode}_gt_success_count"] = success["count"]
                row[f"{mode}_gt_success_recall"] = success["recall"]
            summary_rows.append(row)
            class_rows.extend(per_class_rows_by_image(method, stage, all_gts, aligned_by_image))
            for item in confusion:
                confusion_rows.append({"method": method, "stage": stage, **item})

    write_csv(args.out / "defect_protocol_summary.csv", summary_rows)
    write_csv(args.out / "defect_protocol_class_metrics.csv", class_rows)
    write_csv(args.out / "defect_protocol_type_confusion.csv", confusion_rows)
    manifest = {
        "data": str(args.data),
        "eval_root": str(args.eval_root),
        "out": str(args.out),
        "defect_classes": {str(k): CLASS_NAMES.get(k, str(k)) for k in sorted(DEFECT_CLASSES)},
        "primary_classes": {"crack": sorted(CRACK_CLASSES), "pothole": [POTHOLE_CLASS]},
        "fusion": {
            "duplicate_iou": args.duplicate_iou,
            "duplicate_ioa": args.duplicate_ioa,
            "merge_gap": args.merge_gap,
            "note": "Fixed geometry-only fusion; GT49 labels are used only for evaluation.",
        },
        "crop_bottom_half": bool(args.crop_bottom_half),
    }
    (args.out / "defect_protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
