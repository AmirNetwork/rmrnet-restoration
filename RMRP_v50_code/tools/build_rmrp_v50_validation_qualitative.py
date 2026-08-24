#!/usr/bin/env python3
"""Build auditable RMR-P qualitative panels from retained validation outputs.

The script scans low-light and mixed validation outputs with the same frozen
detector used for the paper metrics. It selects examples where RMR-P ties the
best displayed restorer and recovers more class-correct ground-truth instances
than the degraded input. Exact expert routes can make RMR-P and one constituent
expert pixel-identical, so a unique visual win is neither required nor claimed.
This is an explicitly illustrative selection; aggregate claims remain based on
the complete validation partitions. The rule and image hashes are saved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_automation_in_construction_rmrnet"
FIGURES = PAPER / "figures"
OUT = ROOT / "experiments/final_rmrp_v50_qualitative_20260824"
LEDGER = ROOT / "experiments/final_rmrp_v50_validation_ledger_20260824/provenance_ledger.json"
RMRP_ROOT = Path(r"E:\RMRP_experiments\rmrp_expert_fusion_v50_validation_20260822\correct\restored")
BASELINE_ROOT = Path(r"E:\RMRP_experiments\heterogeneous_expert_fusion_val_v1_20260822\full_validation")

METHODS = ("raw", "instructir", "dfpir", "demoe", "rmrp")
DISPLAY = {
    "raw": "Degraded",
    "instructir": "InstructIR",
    "dfpir": "DFPIR",
    "demoe": "DeMoE",
    "rmrp": "RMR-P",
}
DATASETS = {
    "ivcnz": {
        "prefix": "pothole_yolo_sequence_disjoint_practical_sensor_calibrated_v2",
        "detector": ROOT / "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pothole_clean_80ep/weights/best.pt",
        "figure": "fig_ivcnz_v50_validation_comparison.png",
    },
    "pcm": {
        "prefix": "road_damage_pcm_yolo_sequence_disjoint_practical_sensor_calibrated_v2",
        "detector": ROOT / "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pcm_clean_80ep/weights/best.pt",
        "figure": "fig_pcm_v50_validation_comparison.png",
    },
}
CAUSES = ("lowlight", "mixed")
CAUSE_LABEL = {"lowlight": "Low light", "mixed": "Motion + low light"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_root(dataset: str, cause: str) -> Path:
    yaml_path = ROOT / "datasets" / f"{DATASETS[dataset]['prefix']}_{cause}_val/data.yaml"
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(document["path"])
    return root if root.is_absolute() else (yaml_path.parent / root).resolve()


def image_index(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }


def paths_for(dataset: str, cause: str) -> dict[str, dict[str, Path]]:
    root = source_root(dataset, cause)
    raw = image_index(root / "images" / "val")
    result: dict[str, dict[str, Path]] = {"raw": raw}
    for method in ("instructir", "dfpir", "demoe"):
        directory = BASELINE_ROOT / method / "epoch_070" / "restored" / dataset / cause / "images" / "val"
        result[method] = image_index(directory)
    result["rmrp"] = image_index(RMRP_ROOT / dataset / cause / "images" / "val")
    common = set.intersection(*(set(index) for index in result.values()))
    if common != set(raw):
        raise RuntimeError(
            f"Output/source identity mismatch for {dataset}/{cause}: "
            f"common={len(common)}, raw={len(raw)}"
        )
    return result


def read_labels(path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, xc, yc, bw, bh = map(float, values[:5])
        box = np.array(
            [
                (xc - bw / 2) * width,
                (yc - bh / 2) * height,
                (xc + bw / 2) * width,
                (yc + bh / 2) * height,
            ],
            dtype=np.float32,
        )
        result.append((int(cls), box))
    return result


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def match_stats(
    gt: list[tuple[int, np.ndarray]], prediction: Any, threshold: float = 0.50
) -> tuple[int, float]:
    candidates: list[tuple[float, float, int, int]] = []
    if prediction.boxes is not None:
        for pred_index, (box, cls, conf) in enumerate(
            zip(
                prediction.boxes.xyxy.cpu().numpy(),
                prediction.boxes.cls.cpu().numpy(),
                prediction.boxes.conf.cpu().numpy(),
            )
        ):
            for gt_index, (gt_cls, gt_box) in enumerate(gt):
                if int(cls) != gt_cls:
                    continue
                overlap = iou(gt_box, box)
                if overlap >= threshold:
                    candidates.append((overlap, float(conf), gt_index, pred_index))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    confidence_sum = 0.0
    for _, confidence, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        confidence_sum += confidence
    return len(used_gt), confidence_sum


def predict_paths(model: YOLO, paths: list[Path]) -> list[Any]:
    return model.predict(
        source=[str(path) for path in paths],
        imgsz=640,
        conf=0.15,
        iou=0.70,
        batch=16,
        device=0,
        verbose=False,
    )


def scan_dataset(dataset: str) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], Any], dict[str, dict[str, dict[str, Path]]]]:
    detector = YOLO(str(DATASETS[dataset]["detector"]))
    all_predictions: dict[tuple[str, str, str], Any] = {}
    all_paths: dict[str, dict[str, dict[str, Path]]] = {}
    records: list[dict[str, Any]] = []
    for cause in CAUSES:
        indexed = paths_for(dataset, cause)
        all_paths[cause] = indexed
        stems = sorted(indexed["raw"])
        root = source_root(dataset, cause)
        with Image.open(indexed["raw"][stems[0]]) as sample:
            width, height = sample.size
        labels = {
            stem: read_labels(root / "labels" / "val" / f"{stem}.txt", width, height)
            for stem in stems
        }
        for method in METHODS:
            print(f"[qualitative] {dataset}/{cause}/{method}: {len(stems)} images", flush=True)
            predictions = predict_paths(detector, [indexed[method][stem] for stem in stems])
            for stem, prediction in zip(stems, predictions):
                all_predictions[(cause, method, stem)] = prediction
        for stem in stems:
            stats = {
                method: match_stats(labels[stem], all_predictions[(cause, method, stem)])
                for method in METHODS
            }
            rmrp_tp, rmrp_conf = stats["rmrp"]
            peer_tp = max(stats[method][0] for method in METHODS if method != "rmrp")
            peer_conf = max(stats[method][1] for method in METHODS if method != "rmrp")
            records.append(
                {
                    "dataset": dataset,
                    "cause": cause,
                    "stem": stem,
                    "gt_count": len(labels[stem]),
                    "rmrp_tp": rmrp_tp,
                    "best_peer_tp": peer_tp,
                    "tp_margin": rmrp_tp - peer_tp,
                    "raw_tp_gain": rmrp_tp - stats["raw"][0],
                    "demoe_tp_gain": rmrp_tp - stats["demoe"][0],
                    "confidence_margin": rmrp_conf - peer_conf,
                    "stats": {method: {"tp": value[0], "matched_confidence": value[1]} for method, value in stats.items()},
                }
            )
    return records, all_predictions, all_paths


def crop_bounds(gt: list[tuple[int, np.ndarray]], width: int, height: int) -> tuple[int, int, int, int]:
    if not gt:
        return (0, 0, width, height)
    boxes = np.stack([box for _, box in gt])
    x1, y1 = boxes[:, :2].min(axis=0)
    x2, y2 = boxes[:, 2:].max(axis=0)
    pad_x = max(40.0, (x2 - x1) * 0.35)
    pad_y = max(40.0, (y2 - y1) * 0.45)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    # Keep a useful landscape context around small defects.
    crop_w, crop_h = x2 - x1, y2 - y1
    target_ratio = 1.55
    if crop_w / max(crop_h, 1.0) < target_ratio:
        extra = (target_ratio * crop_h - crop_w) / 2
        x1, x2 = max(0, x1 - extra), min(width, x2 + extra)
    return tuple(map(int, (x1, y1, x2, y2)))


def annotate(
    path: Path,
    labels: list[tuple[int, np.ndarray]],
    prediction: Any,
    crop: tuple[int, int, int, int],
) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, image.width // 320)
    names = prediction.names
    for cls, box in labels:
        draw.rectangle(tuple(box), outline="#f2c744", width=line_width)
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"GT {names[int(cls)]}", fill="#f2c744", font=font)
    if prediction.boxes is not None:
        for box, cls, conf in zip(
            prediction.boxes.xyxy.cpu().tolist(),
            prediction.boxes.cls.cpu().tolist(),
            prediction.boxes.conf.cpu().tolist(),
        ):
            draw.rectangle(tuple(box), outline="#17a36b", width=line_width)
            draw.text((box[0] + 2, box[1] + 2), f"{names[int(cls)]} {conf:.2f}", fill="#17a36b", font=font)
    return image.crop(crop)


def build_figure(
    dataset: str,
    selected: list[dict[str, Any]],
    predictions: dict[tuple[str, str, str], Any],
    paths: dict[str, dict[str, dict[str, Path]]],
) -> None:
    fig, axes = plt.subplots(len(selected), len(METHODS), figsize=(12.5, 4.6), dpi=220)
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row, record in enumerate(selected):
        cause, stem = record["cause"], record["stem"]
        root = source_root(dataset, cause)
        with Image.open(paths[cause]["raw"][stem]) as image:
            width, height = image.size
        labels = read_labels(root / "labels" / "val" / f"{stem}.txt", width, height)
        crop = crop_bounds(labels, width, height)
        record["crop_xyxy"] = crop
        for column, method in enumerate(METHODS):
            prediction = predictions[(cause, method, stem)]
            rendered = annotate(paths[cause][method][stem], labels, prediction, crop)
            ax = axes[row, column]
            ax.imshow(rendered)
            ax.axis("off")
            if row == 0:
                ax.set_title(DISPLAY[method], fontsize=8, fontweight="bold")
            if column == 0:
                ax.text(-0.025, 0.5, CAUSE_LABEL[cause], transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=8)
            stats = record["stats"][method]
            ax.text(
                0.02,
                0.03,
                f"matched GT: {stats['tp']}/{record['gt_count']}",
                transform=ax.transAxes,
                fontsize=6.5,
                color="white",
                bbox={"facecolor": "#16212b", "alpha": 0.78, "pad": 2, "edgecolor": "none"},
            )
    fig.subplots_adjust(left=0.035, right=0.995, top=0.91, bottom=0.02, wspace=0.02, hspace=0.06)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / DATASETS[dataset]["figure"], facecolor="white", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if ledger.get("status") != "FROZEN_VALIDATION_ONLY" or ledger.get("test_split_used") is not False:
        raise RuntimeError("Qualitative source is not the frozen validation-only release")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "running",
        "split": "validation",
        "test_split_used": False,
        "selection_rule": "RMR-P ties the best displayed restorer and recovers more class-correct IoU50 ground truth than degraded input; DeMoE gain, confidence, and GT-count tie-breaks",
        "display_confidence": 0.15,
        "detector_nms_iou": 0.70,
        "methods": list(METHODS),
        "datasets": {},
    }
    for dataset in DATASETS:
        records, predictions, paths = scan_dataset(dataset)
        records.sort(
            key=lambda row: (
                row["tp_margin"],
                row["raw_tp_gain"],
                row["demoe_tp_gain"],
                row["rmrp_tp"] / max(row["gt_count"], 1),
                -row["gt_count"],
                row["confidence_margin"],
                row["stem"],
            ),
            reverse=True,
        )
        strict = [
            row
            for row in records
            if row["tp_margin"] >= 0
            and row["raw_tp_gain"] > 0
            and row["demoe_tp_gain"] >= 0
            and row["rmrp_tp"] > 0
        ]
        if len(strict) < 2:
            raise RuntimeError(
                f"Only {len(strict)} honest RMR-P recovery examples found for {dataset}; "
                "refusing to manufacture a qualitative panel"
            )
        selected = strict[: (1 if dataset == "ivcnz" else 2)]
        build_figure(dataset, selected, predictions, paths)
        manifest["datasets"][dataset] = {
            "figure": f"paper_automation_in_construction_rmrnet/figures/{DATASETS[dataset]['figure']}",
            "detector": str(DATASETS[dataset]["detector"].relative_to(ROOT)),
            "detector_sha256": sha256(DATASETS[dataset]["detector"]),
            "selected": [
                {
                    **record,
                    "source_sha256": sha256(paths[record["cause"]]["raw"][record["stem"]]),
                }
                for record in selected
            ],
        }
    manifest["status"] = "COMPLETE"
    (OUT / "qualitative_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
