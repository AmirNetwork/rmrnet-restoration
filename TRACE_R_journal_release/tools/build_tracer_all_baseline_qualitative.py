#!/usr/bin/env python3
"""Build full-width controlled panels containing every reported baseline."""

from __future__ import annotations

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_trace_r"
FIGURES = PAPER / "figures"
MANIFEST = ROOT / "experiments/final_trace_r_qualitative_20260824/qualitative_manifest.json"
WORK = Path(r"E:\TRACE_R_experiments\qualitative_all_baselines_v1_20260824")
BASELINE_ROOT = Path(r"E:\RMRP_experiments\heterogeneous_expert_fusion_val_v1_20260822/full_validation")
TRACE_ROOT = Path(r"E:\RMRP_experiments\rmrp_expert_fusion_v50_validation_20260822/correct/restored")

DATASETS = {
    "ivcnz": {
        "prefix": "pothole_yolo_sequence_disjoint_practical_sensor_calibrated_v2",
        "detector": ROOT / "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pothole_clean_80ep/weights/best.pt",
        "figure": "fig_trace_ivcnz_qualitative.pdf",
    },
    "pcm": {
        "prefix": "road_damage_pcm_yolo_sequence_disjoint_practical_sensor_calibrated_v2",
        "detector": ROOT / "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pcm_clean_80ep/weights/best.pt",
        "figure": "fig_trace_pcm_qualitative.pdf",
    },
}
METHODS = ("raw", "nafnet", "nafnet_meta", "instructir", "dfpir", "demoe", "trace_r")
DISPLAY = {
    "raw": "Degraded",
    "nafnet": "NAFNet",
    "nafnet_meta": "FiLM-NAFNet",
    "instructir": "InstructIR",
    "dfpir": "DFPIR",
    "demoe": "DeMoE",
    "trace_r": "TRACE-R",
}
CAUSE_DISPLAY = {"motion": "Motion", "defocus": "Defocus", "lowlight": "Low light", "mixed": "Motion + low light"}


def source_root(dataset: str, cause: str) -> Path:
    data_yaml = ROOT / "datasets" / f"{DATASETS[dataset]['prefix']}_{cause}_val/data.yaml"
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    path = Path(payload["path"])
    return path if path.is_absolute() else (data_yaml.parent / path).resolve()


def locate(directory: Path, stem: str) -> Path:
    matches = [path for path in directory.iterdir() if path.is_file() and path.stem == stem]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one image for {stem} in {directory}, found {len(matches)}")
    return matches[0]


def prepare_miniset(dataset: str, cause: str, stems: list[str]) -> Path:
    root = source_root(dataset, cause)
    target = WORK / "source" / dataset / cause
    for kind in ("images", "labels", "metadata"):
        (target / kind / "val").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        image = locate(root / "images/val", stem)
        shutil.copy2(image, target / "images/val" / image.name)
        for kind, suffix in (("labels", ".txt"), ("metadata", ".json")):
            source = root / kind / "val" / f"{stem}{suffix}"
            shutil.copy2(source, target / kind / "val" / source.name)
    data_yaml = target / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(target),
                "train": "images/val",
                "val": "images/val",
                "test": "images/val",
                "names": {0: "pothole"} if dataset == "ivcnz" else {0: "pothole", 1: "crack", 2: "manhole"},
                "nc": 1 if dataset == "ivcnz" else 3,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def restore_naf(dataset: str, cause: str, data_yaml: Path, method: str) -> Path:
    checkpoint = ROOT / "experiments/matched_final_candidate_index_v28_epoch70_20260821" / method / f"{method}_epoch_070.pth"
    out = WORK / "restored" / dataset / cause / method
    output_yaml = out / "data.yaml"
    if not output_yaml.exists():
        command = [
            sys.executable,
            "tools/restore_yolo_split.py",
            "--data", str(data_yaml),
            "--split", "val",
            "--model", method,
            "--scenario", cause,
            "--out", str(out),
            "--device", "cuda",
            "--nafnet-weights", str(checkpoint),
        ]
        if method == "nafnet_meta":
            command += ["--require-metadata"]
        subprocess.run(command, cwd=ROOT, check=True)
    payload = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    restored_root = Path(payload["path"])
    if not restored_root.is_absolute():
        restored_root = (output_yaml.parent / restored_root).resolve()
    return restored_root / payload["val"]


def read_labels(path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        cls, cx, cy, bw, bh = map(float, fields[:5])
        rows.append((int(cls), np.asarray([
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ])))
    return rows


def iou(a: np.ndarray, b: np.ndarray) -> float:
    top_left = np.maximum(a[:2], b[:2])
    bottom_right = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0))
    area_a = np.prod(np.maximum(a[2:] - a[:2], 0.0))
    area_b = np.prod(np.maximum(b[2:] - b[:2], 0.0))
    return float(intersection / max(area_a + area_b - intersection, 1e-9))


def matched_count(labels: list[tuple[int, np.ndarray]], result: Any) -> int:
    pairs = []
    for prediction_index, (box, cls) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy())):
        for label_index, (gt_cls, gt_box) in enumerate(labels):
            overlap = iou(gt_box, box) if int(cls) == gt_cls else 0.0
            if overlap >= 0.5:
                pairs.append((overlap, label_index, prediction_index))
    used_labels, used_predictions = set(), set()
    for _, label_index, prediction_index in sorted(pairs, reverse=True):
        if label_index in used_labels or prediction_index in used_predictions:
            continue
        used_labels.add(label_index)
        used_predictions.add(prediction_index)
    return len(used_labels)


def crop_bounds(labels: list[tuple[int, np.ndarray]], width: int, height: int) -> tuple[int, int, int, int]:
    boxes = np.stack([box for _, box in labels])
    x1, y1 = boxes[:, :2].min(0)
    x2, y2 = boxes[:, 2:].max(0)
    px, py = max(40.0, 0.35 * (x2 - x1)), max(40.0, 0.45 * (y2 - y1))
    x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(width, x2 + px), min(height, y2 + py)
    # A near-landscape crop keeps road context and yields a readable panel
    # height after seven methods are arranged across an IEEE two-column page.
    target_ratio = 1.25
    ratio = (x2 - x1) / max(y2 - y1, 1.0)
    if ratio < target_ratio:
        extra = (target_ratio * (y2 - y1) - (x2 - x1)) / 2
        x1, x2 = max(0, x1 - extra), min(width, x2 + extra)
    elif ratio > target_ratio:
        extra = ((x2 - x1) / target_ratio - (y2 - y1)) / 2
        y1, y2 = max(0, y1 - extra), min(height, y2 + extra)
    return tuple(map(int, (x1, y1, x2, y2)))


def method_path(dataset: str, cause: str, stem: str, method: str, naf_dirs: dict[str, Path]) -> Path:
    if method == "raw":
        return locate(source_root(dataset, cause) / "images/val", stem)
    if method in {"nafnet", "nafnet_meta"}:
        return locate(naf_dirs[method], stem)
    if method == "trace_r":
        return locate(TRACE_ROOT / dataset / cause / "images/val", stem)
    return locate(BASELINE_ROOT / method / "epoch_070/restored" / dataset / cause / "images/val", stem)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.5,
    })


def build_dataset(dataset: str, selected: list[dict[str, Any]]) -> None:
    by_cause: dict[str, list[str]] = {}
    for item in selected:
        by_cause.setdefault(item["cause"], []).append(item["stem"])
    naf_roots: dict[str, dict[str, Path]] = {}
    for cause, stems in by_cause.items():
        mini = prepare_miniset(dataset, cause, stems)
        naf_roots[cause] = {
            method: restore_naf(dataset, cause, mini, method)
            for method in ("nafnet", "nafnet_meta")
        }
    detector = YOLO(str(DATASETS[dataset]["detector"]))
    rows = []
    for item in selected:
        cause, stem = item["cause"], item["stem"]
        paths = {method: method_path(dataset, cause, stem, method, naf_roots[cause]) for method in METHODS}
        predictions = {
            method: detector.predict(source=str(path), imgsz=640, conf=0.15, iou=0.70, device=0, verbose=False)[0]
            for method, path in paths.items()
        }
        with Image.open(paths["raw"]) as image:
            width, height = image.size
        labels = read_labels(source_root(dataset, cause) / "labels/val" / f"{stem}.txt", width, height)
        rows.append((item, paths, predictions, labels, crop_bounds(labels, width, height)))

    # Seven side-by-side road crops become too shallow in a two-column paper.
    # Wrap every example into a 4+3 grid so that all comparators remain legible.
    fig, axes = plt.subplots(
        2 * len(rows),
        4,
        figsize=(7.16, 2.45 * len(rows)),
        squeeze=False,
    )
    for example_index, (item, paths, predictions, labels, crop) in enumerate(rows):
        left, top, _, _ = crop
        for method_index, method in enumerate(METHODS):
            row_index = 2 * example_index + method_index // 4
            column = method_index % 4
            ax = axes[row_index, column]
            with Image.open(paths[method]) as image:
                ax.imshow(np.asarray(image.convert("RGB").crop(crop)))
            ax.set_title(DISPLAY[method], fontsize=7.2, fontweight="bold", pad=2)
            for _, box in labels:
                ax.add_patch(Rectangle((box[0]-left, box[1]-top), box[2]-box[0], box[3]-box[1], fill=False, edgecolor="#efc22e", linewidth=0.8))
            result = predictions[method]
            for box in result.boxes.xyxy.cpu().numpy():
                ax.add_patch(Rectangle((box[0]-left, box[1]-top), box[2]-box[0], box[3]-box[1], fill=False, edgecolor="#0aa674", linewidth=0.7))
            matches = matched_count(labels, result)
            ax.text(0.02, 0.03, f"GT recovered {matches}/{len(labels)}", transform=ax.transAxes, color="white", fontsize=5.7, bbox={"facecolor":"#16212b","edgecolor":"none","alpha":0.82,"pad":1.2})
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_linewidth(0.5); spine.set_color("black")
        axes[2 * example_index + 1, 3].axis("off")
        y = 1.0 - (example_index + 0.5) / len(rows)
        fig.text(0.008, y, CAUSE_DISPLAY[item["cause"]], rotation=90, va="center", fontsize=6.9)
    fig.subplots_adjust(
        left=0.035,
        right=0.995,
        top=0.955,
        bottom=0.02,
        wspace=0.025,
        hspace=0.23,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / DATASETS[dataset]["figure"],
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(fig)


def main() -> None:
    configure_style()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for dataset in DATASETS:
        build_dataset(dataset, manifest["datasets"][dataset]["selected"])
    print(json.dumps({"status":"COMPLETE","methods":METHODS,"work":str(WORK)}, indent=2))


if __name__ == "__main__":
    main()
