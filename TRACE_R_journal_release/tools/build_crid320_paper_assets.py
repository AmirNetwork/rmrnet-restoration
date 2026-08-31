#!/usr/bin/env python3
"""Build CRID-320 paper tables, plots, and qualitative candidates.

This program consumes only the completed one-time sealed-test ledger. It never
trains or selects a model. Qualitative candidates are ranked by a declared
class-aware IoU-0.50 rule and remain secondary to dataset-level AP.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = Path(r"E:\TRACE_R_experiments\crid320_sealed_test_20260831")
DEFAULT_PAPER = ROOT / "paper_ieee_tits_trace_r"
METHODS = ("native", "nafnet", "instructir", "dfpir", "demoe", "rmrp")
DISPLAY_NAMES = {
    "native": "Unprocessed image",
    "nafnet": "NAFNet",
    "instructir": "InstructIR",
    "dfpir": "DFPIR",
    "demoe": "DeMoE-auto",
    "rmrp": "TRACE-R",
}
CLASS_NAMES = {
    0: "longitudinal",
    1: "transverse",
    2: "alligator",
    3: "pothole",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--options", type=int, default=10)
    return parser.parse_args()


def yolo_boxes(path: Path, *, prediction: bool, confidence: float) -> list[dict[str, float]]:
    boxes: list[dict[str, float]] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        score = float(values[5]) if prediction and len(values) > 5 else 1.0
        if score < confidence:
            continue
        cls = int(float(values[0]))
        cx, cy, width, height = map(float, values[1:5])
        boxes.append(
            {
                "class": cls,
                "x1": cx - width / 2.0,
                "y1": cy - height / 2.0,
                "x2": cx + width / 2.0,
                "y2": cy + height / 2.0,
                "confidence": score,
            }
        )
    return boxes


def iou(first: dict[str, float], second: dict[str, float]) -> float:
    left = max(first["x1"], second["x1"])
    top = max(first["y1"], second["y1"])
    right = min(first["x2"], second["x2"])
    bottom = min(first["y2"], second["y2"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first["x2"] - first["x1"]) * max(
        0.0, first["y2"] - first["y1"]
    )
    second_area = max(0.0, second["x2"] - second["x1"]) * max(
        0.0, second["y2"] - second["y1"]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_counts(
    ground_truth: list[dict[str, float]], predictions: list[dict[str, float]]
) -> tuple[int, int, int]:
    matches: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, target in enumerate(ground_truth):
            if int(prediction["class"]) != int(target["class"]):
                continue
            overlap = iou(prediction, target)
            if overlap >= 0.5:
                matches.append((overlap, pred_index, gt_index))
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for _, pred_index, gt_index in sorted(matches, reverse=True):
        if pred_index in used_predictions or gt_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(gt_index)
    true_positive = len(used_targets)
    return true_positive, len(predictions) - true_positive, len(ground_truth) - true_positive


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    best_map50 = max(float(row["map50"]) for row in rows)
    best_map95 = max(float(row["map50_95"]) for row in rows)
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Native-resolution CRID-320 confirmatory test (80 frames, 122 defects).}",
        r"\label{tab:crid}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Precision & Recall & mAP50 & mAP50--95 \\",
        r"\midrule",
    ]
    for row in rows:
        name = DISPLAY_NAMES[str(row["method"])]
        map50 = f"{float(row['map50']):.3f}"
        map95 = f"{float(row['map50_95']):.3f}"
        if math.isclose(float(row["map50"]), best_map50):
            map50 = rf"\textbf{{{map50}}}"
        if math.isclose(float(row["map50_95"]), best_map95):
            map95 = rf"\textbf{{{map95}}}"
        lines.append(
            f"{name} & {float(row['precision']):.3f} & "
            f"{float(row['recall']):.3f} & {map50} & {map95} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ap_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [DISPLAY_NAMES[str(row["method"])] for row in rows]
    values = [float(row["map50"]) for row in rows]
    colors = ["#8b949e", "#4c78a8", "#72b7b2", "#f58518", "#b279a2", "#1a7f5a"]
    figure, axis = plt.subplots(figsize=(6.9, 2.65))
    bars = axis.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="#202124", linewidth=0.55)
    axis.set_xlabel("mAP50")
    axis.set_xlim(0.18, max(values) + 0.012)
    axis.grid(axis="x", color="#d9dde3", linewidth=0.55)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values[::-1], strict=True):
        axis.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#202124")
    figure.tight_layout(pad=0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def candidate_rows(run: Path, confidence: float) -> list[dict[str, Any]]:
    test_manifest = run / "native" / "selected_view" / "test.txt"
    test_paths = [Path(row) for row in test_manifest.read_text(encoding="utf-8").splitlines() if row]
    rows: list[dict[str, Any]] = []
    for image_path in test_paths:
        label_path = Path(str(image_path).replace("\\images\\test\\", "\\labels\\test\\")).with_suffix(".txt")
        ground_truth = yolo_boxes(label_path, prediction=False, confidence=0.0)
        row: dict[str, Any] = {"stem": image_path.stem, "gt": len(ground_truth)}
        for method in METHODS:
            prediction_path = run / method / "detector_run" / "test" / "labels" / f"{image_path.stem}.txt"
            prediction = yolo_boxes(prediction_path, prediction=True, confidence=confidence)
            tp, fp, fn = match_counts(ground_truth, prediction)
            row[f"{method}_tp"] = tp
            row[f"{method}_fp"] = fp
            row[f"{method}_fn"] = fn
        comparator_tp = max(int(row[f"{method}_tp"]) for method in METHODS if method != "rmrp")
        comparator_fp = min(int(row[f"{method}_fp"]) for method in METHODS if method != "rmrp")
        row["trace_tp_advantage"] = int(row["rmrp_tp"]) - comparator_tp
        row["trace_fp_excess"] = int(row["rmrp_fp"]) - comparator_fp
        row["ranking_score"] = (
            20 * int(row["trace_tp_advantage"])
            - 2 * max(0, int(row["trace_fp_excess"]))
            + int(row["rmrp_tp"])
            - int(row["rmrp_fp"])
        )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            int(row["trace_tp_advantage"]),
            -int(row["trace_fp_excess"]),
            int(row["rmrp_tp"]),
            int(row["ranking_score"]),
        ),
        reverse=True,
    )


def draw_overlay(
    image_path: Path,
    ground_truth: list[dict[str, float]],
    predictions: list[dict[str, float]],
) -> Image.Image:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    line_width = max(4, int(round(min(width, height) / 750)))
    font = ImageFont.load_default(size=max(22, int(round(min(width, height) / 115))))

    def pixels(box: dict[str, float]) -> tuple[int, int, int, int]:
        return (
            int(round(box["x1"] * width)),
            int(round(box["y1"] * height)),
            int(round(box["x2"] * width)),
            int(round(box["y2"] * height)),
        )

    for box in ground_truth:
        coords = pixels(box)
        draw.rectangle(coords, outline=(255, 205, 0), width=line_width)
        draw.text(
            (coords[0], max(0, coords[1] - font.size - 5)),
            f"GT {CLASS_NAMES[int(box['class'])]}",
            fill=(255, 205, 0),
            font=font,
            stroke_width=2,
            stroke_fill=(25, 25, 25),
        )
    for box in predictions:
        coords = pixels(box)
        draw.rectangle(coords, outline=(0, 220, 110), width=line_width)
        draw.text(
            (coords[0], coords[1] + 4),
            f"{CLASS_NAMES[int(box['class'])]} {box['confidence']:.2f}",
            fill=(0, 220, 110),
            font=font,
            stroke_width=2,
            stroke_fill=(25, 25, 25),
        )
    return image


def panel_for_stem(
    run: Path, stem: str, output: Path, *, confidence: float
) -> None:
    panels: list[Image.Image] = []
    for method in METHODS:
        image_path = run / method / "selected_view" / "images" / "test" / f"{stem}.jpg"
        ground_truth = yolo_boxes(
            run / method / "selected_view" / "labels" / "test" / f"{stem}.txt",
            prediction=False,
            confidence=0.0,
        )
        predictions = yolo_boxes(
            run / method / "detector_run" / "test" / "labels" / f"{stem}.txt",
            prediction=True,
            confidence=confidence,
        )
        image = draw_overlay(image_path, ground_truth, predictions)
        image.thumbnail((900, 600), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (900, 650), "white")
        left = (900 - image.width) // 2
        canvas.paste(image, (left, 48))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=30)
        draw.text((18, 8), DISPLAY_NAMES[method], fill="black", font=font)
        panels.append(canvas)
    sheet = Image.new("RGB", (2700, 1300), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 3) * 900, (index // 3) * 650))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95, subsampling=0)


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    paper = args.paper.resolve()
    ledger_path = run / "SEALED_TEST_COMPLETE.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("status") != "ONE_TIME_CRID320_SEALED_TEST_COMPLETE":
        raise RuntimeError("CRID-320 sealed test is not complete")
    rows = list(ledger["results"])
    write_table(paper / "tables" / "table_crid.tex", rows)
    write_ap_plot(paper / "figures" / "fig_trace_crid_ap.pdf", rows)

    candidates = candidate_rows(run, args.confidence)
    audit_root = run / "paper_assets"
    audit_root.mkdir(parents=True, exist_ok=True)
    with (audit_root / "qualitative_candidate_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    option_root = audit_root / "qualitative_options"
    if option_root.exists():
        shutil.rmtree(option_root)
    for index, row in enumerate(candidates[: args.options], 1):
        panel_for_stem(
            run,
            str(row["stem"]),
            option_root / f"option_{index:02d}_{row['stem']}.jpg",
            confidence=args.confidence,
        )
    selected = candidates[0]
    selected_jpg = option_root / f"option_01_{selected['stem']}.jpg"
    with Image.open(selected_jpg) as image:
        image.save(
            paper / "figures" / "fig_trace_crid_qualitative.pdf",
            "PDF",
            resolution=300.0,
        )
    summary = {
        "status": "PAPER_ASSETS_FROM_COMPLETED_SEALED_TEST",
        "sealed_ledger": str(ledger_path),
        "confidence_for_display_and_candidate_audit": args.confidence,
        "selected_visual": selected,
        "table": str(paper / "tables" / "table_crid.tex"),
        "ap_figure": str(paper / "figures" / "fig_trace_crid_ap.pdf"),
        "qualitative_figure": str(paper / "figures" / "fig_trace_crid_qualitative.pdf"),
    }
    (audit_root / "paper_asset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
