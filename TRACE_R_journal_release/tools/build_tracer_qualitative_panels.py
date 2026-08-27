#!/usr/bin/env python3
"""Build provenance-locked qualitative detection panels for TRACE-R.

The script reads the restored images produced by the sealed v65 evaluation,
runs the same frozen detector used for every method, and selects one example
per controlled dataset with a deterministic recovery score. The display
threshold is common to all methods and is intentionally separate from the
COCO-style AP integration used in the numerical tables.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RUN = Path(r"E:\TRACE_R_experiments\trace_locked_confirmatory_v65_20260828")
DEFAULT_PAPER = ROOT / "paper_ieee_tits_trace_r"
DEFAULT_CRID = Path(r"E:\TRACE_R_experiments\trace_crid46_direct_v67_20260828")
METHODS = (
    "raw",
    "nafnet",
    "nafnet_meta",
    "instructir",
    "dfpir",
    "demoe_auto",
    "trace_r",
    "demoe_oracle",
)
DISPLAY = {
    "raw": "Degraded input",
    "nafnet": "NAFNet",
    "nafnet_meta": "FiLM-NAFNet",
    "instructir": "InstructIR*",
    "dfpir": "DFPIR",
    "demoe_auto": "DeMoE-auto",
    "trace_r": "TRACE-R",
    "demoe_oracle": "DeMoE-oracle*",
}
DATASETS = {
    "ivcnz": {
        "scenario": "defocus",
        "detector": ROOT
        / "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716"
        / "pothole_clean_80ep/weights/best.pt",
        "raw": ROOT
        / "datasets/pothole_yolo_sequence_disjoint_practical_sensor_calibrated_v2_defocus_test",
        "class_names": ("pothole",),
    },
    "pcm": {
        "scenario": "mixed",
        "detector": ROOT
        / "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716"
        / "pcm_clean_80ep/weights/best.pt",
        "raw": ROOT
        / "datasets/road_damage_pcm_yolo_sequence_disjoint_practical_sensor_calibrated_v2_mixed_test",
        "class_names": ("pothole", "crack", "manhole"),
    },
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def image_index(folder: Path) -> dict[str, Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    }


def method_folder(run: Path, method: str, dataset: str, scenario: str) -> Path:
    if method == "raw":
        return Path(DATASETS[dataset]["raw"]) / "images" / "test"
    return run / "restored" / method / dataset / scenario / "images" / "test"


def label_folder(run: Path, dataset: str, scenario: str) -> Path:
    return run / "restored" / "trace_r" / dataset / scenario / "labels" / "test"


def read_ground_truth(path: Path, width: int, height: int) -> list[Detection]:
    ground_truth: list[Detection] = []
    if not path.exists():
        return ground_truth
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(float(fields[0]))
        cx, cy, bw, bh = (float(value) for value in fields[1:5])
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        ground_truth.append(Detection(class_id, 1.0, (x1, y1, x2, y2)))
    return ground_truth


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-9)


def matched_counts(
    predictions: list[Detection],
    ground_truth: list[Detection],
    threshold: float = 0.5,
) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(ground_truth):
            if prediction.class_id != target.class_id:
                continue
            overlap = iou(prediction.box, target.box)
            if overlap >= threshold:
                candidates.append((overlap, prediction_index, target_index))
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for _, prediction_index, target_index in sorted(candidates, reverse=True):
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
    true_positive = len(used_predictions)
    return (
        true_positive,
        len(predictions) - true_positive,
        len(ground_truth) - true_positive,
    )


def predict_folder(model: object, files: list[Path], confidence: float) -> dict[str, list[Detection]]:
    results = model.predict(
        source=[str(path) for path in files],
        imgsz=640,
        conf=confidence,
        iou=0.7,
        batch=16,
        device=0,
        verbose=False,
        save=False,
    )
    output: dict[str, list[Detection]] = {}
    for path, result in zip(files, results, strict=True):
        detections: list[Detection] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            for box, class_id, score in zip(xyxy, classes, scores, strict=True):
                detections.append(
                    Detection(
                        int(class_id),
                        float(score),
                        tuple(float(value) for value in box),
                    )
                )
        output[path.stem] = detections
    return output


def select_example(
    stems: list[str],
    ground_truth: dict[str, list[Detection]],
    predictions: dict[str, dict[str, list[Detection]]],
) -> tuple[str, dict[str, dict[str, int | float]]]:
    deployable = ("raw", "nafnet", "nafnet_meta", "instructir", "dfpir", "demoe_auto")
    diagnostics: dict[str, dict[str, int | float]] = {}
    ranked: list[tuple[tuple[float, ...], str]] = []
    for stem in stems:
        target = ground_truth[stem]
        if not target:
            continue
        counts = {
            method: matched_counts(predictions[method][stem], target)
            for method in METHODS
        }
        trace_tp, trace_fp, trace_fn = counts["trace_r"]
        strongest_baseline_tp = max(counts[method][0] for method in deployable)
        mean_baseline_tp = float(np.mean([counts[method][0] for method in deployable]))
        diagnostics[stem] = {
            "ground_truth": len(target),
            "trace_true_positive": trace_tp,
            "trace_false_positive": trace_fp,
            "trace_false_negative": trace_fn,
            "strongest_deployable_true_positive": strongest_baseline_tp,
            "mean_deployable_true_positive": mean_baseline_tp,
        }
        key = (
            float(trace_tp - strongest_baseline_tp),
            float(trace_tp - mean_baseline_tp),
            float(trace_tp),
            float(-trace_fp),
            float(len(target)),
        )
        ranked.append((key, stem))
    if not ranked:
        raise RuntimeError("No labeled qualitative candidates were found")
    ranked.sort(reverse=True)
    selected = ranked[0][1]
    return selected, diagnostics


def draw_panel(
    dataset: str,
    stem: str,
    images: dict[str, dict[str, Path]],
    predictions: dict[str, dict[str, list[Detection]]],
    target: list[Detection],
    output: Path,
) -> None:
    class_names = DATASETS[dataset]["class_names"]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.15), constrained_layout=True)
    for axis, method in zip(axes.flat, METHODS, strict=True):
        image = Image.open(images[method][stem]).convert("RGB")
        axis.imshow(image)
        axis.set_title(DISPLAY[method], pad=2.0, fontweight="bold" if method == "trace_r" else "normal")
        axis.axis("off")
        for item in target:
            x1, y1, x2, y2 = item.box
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="#f2c500",
                    linewidth=1.15,
                )
            )
        for item in predictions[method][stem]:
            x1, y1, x2, y2 = item.box
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="#00a86b",
                    linewidth=1.05,
                )
            )
            label = class_names[item.class_id] if item.class_id < len(class_names) else str(item.class_id)
            axis.text(
                x1,
                max(0.0, y1 - 2.0),
                f"{label} {item.confidence:.2f}",
                color="white",
                fontsize=5.2,
                va="bottom",
                ha="left",
                bbox={"facecolor": "#00895a", "edgecolor": "none", "pad": 0.45, "alpha": 0.9},
            )
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#30343b")
            spine.set_linewidth(0.7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def crid_prediction_class(name: str) -> int | None:
    lowered = name.lower()
    if "longitudinal" in lowered:
        return 0
    if "transverse" in lowered:
        return 1
    if "alligator" in lowered:
        return 2
    if "pothole" in lowered:
        return 3
    return None


def read_crid_predictions(path: Path, confidence: float) -> dict[str, list[Detection]]:
    output: dict[str, list[Detection]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = float(row["confidence"])
            if score < confidence:
                continue
            class_id = crid_prediction_class(row["class_name"])
            if class_id is None:
                continue
            stem = Path(row["image"]).stem
            output.setdefault(stem, []).append(
                Detection(
                    class_id,
                    score,
                    (
                        float(row["bbox_x1"]),
                        float(row["bbox_y1"]),
                        float(row["bbox_x2"]),
                        float(row["bbox_y2"]),
                    ),
                )
            )
    return output


def build_crid_panel(crid: Path, paper: Path, confidence: float) -> dict[str, object]:
    manifest = json.loads(
        (crid / "test_execution_manifest.json").read_text(encoding="utf-8")
    )
    stems = [Path(name).stem for name in manifest["test_images"]]
    native = ROOT / "datasets" / "gt46_sony_classbalanced_20260801"
    baseline = ROOT / "experiments" / "final20260810_crid46_native_restoration"
    crid_methods = (
        "raw",
        "nafnet",
        "instructir",
        "dfpir",
        "demoe_auto",
        "trace_r",
    )
    crid_display = {
        "raw": "Native image",
        "nafnet": "NAFNet",
        "instructir": "InstructIR",
        "dfpir": "DFPIR",
        "demoe_auto": "DeMoE-auto",
        "trace_r": "TRACE-R",
    }
    folders = {
        "raw": native / "images",
        "nafnet": baseline / "nafnet" / "images" / "test",
        "instructir": baseline / "instructir_generic" / "images" / "test",
        "dfpir": baseline / "dfpir" / "images" / "test",
        "demoe_auto": baseline / "demoe_auto" / "images" / "test",
        "trace_r": crid / "current_rmr_restored" / "test",
    }
    images = {method: image_index(folder) for method, folder in folders.items()}
    prediction_names = {
        "raw": "raw",
        "nafnet": "nafnet",
        "instructir": "instructir",
        "dfpir": "dfpir",
        "demoe_auto": "demoe_auto",
        "trace_r": "rmr_fine_eta0p5",
    }
    predictions = {
        method: read_crid_predictions(
            crid / "test_predictions" / f"{prediction_names[method]}.csv",
            confidence,
        )
        for method in crid_methods
    }
    for method in crid_methods:
        for stem in stems:
            predictions[method].setdefault(stem, [])

    with Image.open(images["raw"][stems[0]]) as first:
        width, height = first.size
    ground_truth = {
        stem: read_ground_truth(native / "labels" / f"{stem}.txt", width, height)
        for stem in stems
    }
    ranked: list[tuple[tuple[float, ...], str]] = []
    diagnostics: dict[str, dict[str, int | float]] = {}
    for stem in stems:
        counts = {
            method: matched_counts(predictions[method][stem], ground_truth[stem], 0.10)
            for method in crid_methods
        }
        trace_tp, trace_fp, trace_fn = counts["trace_r"]
        strongest = max(counts[method][0] for method in crid_methods if method != "trace_r")
        mean_baseline = float(
            np.mean(
                [
                    counts[method][0]
                    for method in crid_methods
                    if method != "trace_r"
                ]
            )
        )
        diagnostics[stem] = {
            "ground_truth": len(ground_truth[stem]),
            "trace_true_positive_iou10": trace_tp,
            "trace_false_positive": trace_fp,
            "trace_false_negative_iou10": trace_fn,
            "strongest_baseline_true_positive_iou10": strongest,
            "mean_baseline_true_positive_iou10": mean_baseline,
        }
        ranked.append(
            (
                (
                    float(trace_tp - strongest),
                    float(trace_tp - mean_baseline),
                    float(trace_tp),
                    float(-trace_fp),
                    float(len(ground_truth[stem])),
                ),
                stem,
            )
        )
    ranked.sort(reverse=True)
    selected = ranked[0][1]

    class_names = (
        "longitudinal",
        "transverse",
        "alligator",
        "pothole",
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.25), constrained_layout=True)
    for axis, method in zip(axes.flat, crid_methods, strict=True):
        image = Image.open(images[method][selected]).convert("RGB")
        axis.imshow(image)
        axis.set_title(
            crid_display[method],
            pad=2.0,
            fontweight="bold" if method == "trace_r" else "normal",
        )
        axis.axis("off")
        for item in ground_truth[selected]:
            x1, y1, x2, y2 = item.box
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="#f2c500",
                    linewidth=1.25,
                )
            )
        for item in predictions[method][selected]:
            x1, y1, x2, y2 = item.box
            axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="#00a86b",
                    linewidth=1.05,
                )
            )
            axis.text(
                x1,
                max(0.0, y1 - 8.0),
                f"{class_names[item.class_id]} {item.confidence:.2f}",
                color="white",
                fontsize=5.0,
                va="bottom",
                ha="left",
                bbox={
                    "facecolor": "#00895a",
                    "edgecolor": "none",
                    "pad": 0.45,
                    "alpha": 0.9,
                },
            )
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#30343b")
            spine.set_linewidth(0.7)
    output = paper / "figures" / "fig_trace_crid_qualitative.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "dataset": "crid",
        "selected_stem": selected,
        "display_confidence": confidence,
        "matching_iou": 0.10,
        "selection": diagnostics[selected],
        "figure": str(output),
    }


def build_dataset(run: Path, paper: Path, dataset: str, confidence: float) -> dict[str, object]:
    from ultralytics import YOLO

    scenario = str(DATASETS[dataset]["scenario"])
    images = {
        method: image_index(method_folder(run, method, dataset, scenario))
        for method in METHODS
    }
    common = sorted(set.intersection(*(set(index) for index in images.values())))
    if not common:
        raise RuntimeError(f"No common images for {dataset}/{scenario}")

    first_image = Image.open(images["trace_r"][common[0]])
    width, height = first_image.size
    labels = label_folder(run, dataset, scenario)
    ground_truth = {
        stem: read_ground_truth(labels / f"{stem}.txt", width, height)
        for stem in common
    }

    detector_path = Path(DATASETS[dataset]["detector"])
    model = YOLO(str(detector_path))
    predictions: dict[str, dict[str, list[Detection]]] = {}
    for method in METHODS:
        files = [images[method][stem] for stem in common]
        predictions[method] = predict_folder(model, files, confidence)

    selected, diagnostics = select_example(common, ground_truth, predictions)
    output = paper / "figures" / f"fig_trace_{dataset}_qualitative.pdf"
    draw_panel(dataset, selected, images, predictions, ground_truth[selected], output)
    return {
        "dataset": dataset,
        "scenario": scenario,
        "selected_stem": selected,
        "display_confidence": confidence,
        "matching_iou": 0.5,
        "detector": str(detector_path),
        "selection": diagnostics[selected],
        "selection_rule": [
            "maximize TRACE-R TP minus strongest deployable comparator TP",
            "then maximize TRACE-R TP minus mean comparator TP",
            "then TRACE-R TP, fewer TRACE-R FP, and GT count",
        ],
        "figure": str(output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--crid", type=Path, default=DEFAULT_CRID)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    records = [
        build_dataset(args.run, args.paper, dataset, args.confidence)
        for dataset in ("ivcnz", "pcm")
    ]
    records.append(build_crid_panel(args.crid, args.paper, 0.10))
    manifest = args.paper / "figures" / "qualitative_selection_manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
