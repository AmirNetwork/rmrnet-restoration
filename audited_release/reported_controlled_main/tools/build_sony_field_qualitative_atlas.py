#!/usr/bin/env python3
"""Build the audited Sony field detector-evidence atlas used in the paper.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

The script reads frozen CSV predictions and revised COCO annotations. It never
runs a detector or modifies an experiment. Candidate frames must belong to the
chronological evaluation split. Panels use one raw-image crop so that only the
recorded detector evidence changes across methods.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_yolo26_coordinate_gt46 import box_iou, load_gt, load_predictions


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_automation_in_construction_rmrnet"
EXPERIMENT = ROOT / "experiments" / "gt46_yolo26_coordinate_revised"
TWO_VIEW = EXPERIMENT / "method_detections_two_view_fairness_20260714"
MANIFEST = ROOT / "experiments" / "major_revision_evidence_20260715" / "ilx_temporal_holdout_manifest.json"
ANNOTATIONS = ROOT / "road-defect-seg-9junedata.coco-segmentation" / "train" / "_annotations.coco.json"
IMAGES = EXPERIMENT / "gt46_native_images"

INK = (25, 40, 50)
MUTED = (88, 103, 113)
GT = (244, 183, 46)
MATCHED = (0, 148, 115)
UNMATCHED = (199, 75, 61)
TEAL = (11, 122, 117)

METHODS = {
    "Raw": EXPERIMENT / "method_detections_yolo26rev_imgsz1280_conf010_clsD00-015_D10-025_D20-025" / "raw",
    "Raw + NAFNet": TWO_VIEW / "raw_plus_nafnet",
    "Raw + DFPIR": TWO_VIEW / "raw_plus_dfpir",
    "Raw + DeMoE": TWO_VIEW / "raw_plus_demoe_scenario",
    "Raw + InstructIR": TWO_VIEW / "raw_plus_instructir_metadata",
    "Raw + RMR-Net": TWO_VIEW / "raw_plus_rmr_native_gate_gamma085",
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def assignments(gt_items: list[dict], pred_items: list[dict], threshold: float = 0.10) -> tuple[dict[int, int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, target in enumerate(gt_items):
        for pi, prediction in enumerate(pred_items):
            if target["primary"] != prediction["primary"]:
                continue
            overlap = box_iou(target["box"], prediction["box"])
            if overlap >= threshold:
                candidates.append((overlap, gi, pi))
    candidates.sort(reverse=True)
    gt_to_pred: dict[int, int] = {}
    used_pred: set[int] = set()
    for _overlap, gi, pi in candidates:
        if gi in gt_to_pred or pi in used_pred:
            continue
        gt_to_pred[gi] = pi
        used_pred.add(pi)
    return gt_to_pred, used_pred


def focus_box(target: tuple[float, float, float, float], image_size: tuple[int, int], aspect: float) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = target
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    crop_w = max((x2 - x1) * 3.0, width * 0.30)
    crop_h = max((y2 - y1) * 3.0, height * 0.22)
    if crop_w / crop_h < aspect:
        crop_w = crop_h * aspect
    else:
        crop_h = crop_w / aspect
    crop_w, crop_h = min(crop_w, width), min(crop_h, height)
    left = min(max(cx - crop_w / 2, 0), width - crop_w)
    top = min(max(cy - crop_h / 2, 0), height - crop_h)
    return round(left), round(top), round(left + crop_w), round(top + crop_h)


def draw_panel(
    raw: Image.Image,
    crop: tuple[int, int, int, int],
    gt_items: list[dict],
    pred_items: list[dict] | None,
    size: tuple[int, int],
) -> tuple[Image.Image, int, int]:
    left, top, right, bottom = crop
    panel = raw.crop(crop).resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    sx, sy = size[0] / (right - left), size[1] / (bottom - top)

    def mapped(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        return ((x1 - left) * sx, (y1 - top) * sy, (x2 - left) * sx, (y2 - top) * sy)

    for target in gt_items:
        box = mapped(target["box"])
        if box[2] > 0 and box[3] > 0 and box[0] < size[0] and box[1] < size[1]:
            draw.rectangle(box, outline=GT, width=4)

    if pred_items is None:
        return panel, 0, 0

    _gt_to_pred, matched_pred = assignments(gt_items, pred_items)
    for index, prediction in enumerate(pred_items):
        box = mapped(prediction["box"])
        if box[2] <= 0 or box[3] <= 0 or box[0] >= size[0] or box[1] >= size[1]:
            continue
        color = MATCHED if index in matched_pred else UNMATCHED
        draw.rectangle(box, outline=color, width=4)
    return panel, len(matched_pred), len(pred_items) - len(matched_pred)


def main() -> None:
    gt, _ = load_gt(ANNOTATIONS)
    predictions = {name: load_predictions(path) for name, path in METHODS.items()}
    evaluation = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["evaluation_frames"])
    requested = [
        "Cam1_2026-06-09_14-58-47_capt1680_1681.jpg",
        "Cam1_2026-06-09_14-59-49_capt2175_2176.jpg",
    ]
    if any(name not in evaluation for name in requested):
        raise RuntimeError("A selected qualitative frame is outside the chronological evaluation split.")

    columns = ["Ground truth", *METHODS]
    cell_w, cell_h = 360, 235
    left, top, gap_x, gap_y = 150, 82, 12, 72
    width = left + len(columns) * cell_w + (len(columns) - 1) * gap_x + 18
    height = top + len(requested) * cell_h + (len(requested) - 1) * gap_y + 38
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    for col, label in enumerate(columns):
        x = left + col * (cell_w + gap_x)
        draw.text((x + 6, 20), label, fill=INK, font=font(23, True))
        if label == "Raw + RMR-Net":
            draw.rectangle((x, 56, x + cell_w, 62), fill=TEAL)

    ledger: list[dict] = []
    for row, name in enumerate(requested):
        targets = gt[name]
        rmr_assignments, _ = assignments(targets, predictions["Raw + RMR-Net"].get(name, []))
        competitor_matches = set()
        for method in METHODS:
            if method == "Raw + RMR-Net":
                continue
            matched, _ = assignments(targets, predictions[method].get(name, []))
            competitor_matches.update(matched)
        unique_targets = [index for index in rmr_assignments if index not in competitor_matches]
        focus_index = unique_targets[0] if unique_targets else next(iter(rmr_assignments), 0)

        raw = Image.open(IMAGES / name).convert("RGB")
        crop = focus_box(targets[focus_index]["box"], raw.size, cell_w / cell_h)
        y = top + row * (cell_h + gap_y)
        draw.text((12, y + 10), f"Case {chr(65 + row)}", fill=INK, font=font(25, True))
        draw.text((12, y + 48), f"GT defects: {len(targets)}", fill=MUTED, font=font(17))
        draw.text((12, y + 74), "IoU >= 0.10", fill=MUTED, font=font(17))

        for col, label in enumerate(columns):
            x = left + col * (cell_w + gap_x)
            method_predictions = None if label == "Ground truth" else predictions[label].get(name, [])
            panel, tp, fp = draw_panel(raw, crop, targets, method_predictions, (cell_w, cell_h))
            page.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(185, 197, 204), width=2)
            if method_predictions is not None:
                draw.text((x + 6, y + cell_h + 8), f"matched {tp} | unmatched {fp}", fill=MUTED, font=font(17))
        ledger.append({"image": name, "focus_gt_index": focus_index, "crop_xyxy": crop})

    out = PAPER / "figures" / "fig_sony_field_evidence_atlas.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out, dpi=(320, 320), quality=96)
    (PAPER / "SONY_FIELD_ATLAS_SELECTION.json").write_text(
        json.dumps({"split": "chronological evaluation", "iou": 0.10, "cases": ledger}, indent=2),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
