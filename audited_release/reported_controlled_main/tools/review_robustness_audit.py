# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

"""Run reviewer-facing detection robustness and uncertainty audits.

The manuscript's primary detector remains the frozen YOLO11s detector used for
checkpoint selection. This script adds an independent YOLOv8n detector audit and
paired bootstrap confidence intervals on AP50 deltas, using only held-out test
splits. The bootstrap AP implementation is intentionally compact and
transparent; it is not used to choose checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "v28_review_robustness"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class SerialPool:
    """Avoid Windows multiprocessing/cache edge cases inside Ultralytics eval."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> "SerialPool":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def imap(self, func: Any, iterable: Any) -> Any:
        return map(func, iterable)


def install_windows_safe_cache_pool() -> None:
    import ultralytics.data.dataset as dataset
    import ultralytics.data.utils as utils

    dataset.ThreadPool = SerialPool
    utils.ThreadPool = SerialPool


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    family: str
    pothole_weights: Path
    pcm_weights: Path


@dataclass(frozen=True)
class ScenarioPair:
    dataset: str
    scenario: str
    degraded_yaml: Path
    restored_yaml: Path


DETECTORS = [
    DetectorSpec(
        key="yolo11s",
        family="YOLO11s",
        pothole_weights=ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pothole_clean_80ep" / "weights" / "best.pt",
        pcm_weights=ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pcm_clean_80ep" / "weights" / "best.pt",
    ),
    DetectorSpec(
        key="yolov8n",
        family="YOLOv8n",
        pothole_weights=ROOT
        / "runs"
        / "detect"
        / "runs"
        / "detect"
        / "runs"
        / "yolov8n_review"
        / "pothole_clean_40ep_seed2026"
        / "weights"
        / "best.pt",
        pcm_weights=ROOT
        / "runs"
        / "detect"
        / "runs"
        / "detect"
        / "runs"
        / "yolov8n_review"
        / "pcm_clean_40ep_seed2026"
        / "weights"
        / "best.pt",
    ),
]

SCENARIO_PAIRS = [
    ScenarioPair(
        "IVCNZ pothole",
        "motion blur",
        ROOT / "datasets" / "pothole_yolo_motion_test" / "data.yaml",
        ROOT / "datasets" / "pothole_yolo_motion_test_rmrnet_noac_ep004" / "data.yaml",
    ),
    ScenarioPair(
        "IVCNZ pothole",
        "defocus",
        ROOT / "datasets" / "pothole_yolo_defocus_test" / "data.yaml",
        ROOT / "datasets" / "pothole_yolo_defocus_test_rmrnet_noac_ep004" / "data.yaml",
    ),
    ScenarioPair(
        "IVCNZ pothole",
        "low light",
        ROOT / "datasets" / "pothole_yolo_lowlight_test" / "data.yaml",
        ROOT / "datasets" / "pothole_yolo_lowlight_test_rmrnet_noac_ep004" / "data.yaml",
    ),
    ScenarioPair(
        "IVCNZ pothole",
        "mixed motion+low light",
        ROOT / "datasets" / "pothole_yolo_mixed_test" / "data.yaml",
        ROOT / "datasets" / "pothole_yolo_mixed_test_rmrnet_noac_ep004" / "data.yaml",
    ),
    ScenarioPair(
        "PCM",
        "motion blur",
        ROOT / "datasets" / "pcm_yolo_motion_test" / "data.yaml",
        ROOT / "datasets" / "pcm_yolo_motion_test_rmrnet_noac_ep001" / "data.yaml",
    ),
    ScenarioPair(
        "PCM",
        "defocus",
        ROOT / "datasets" / "pcm_yolo_defocus_test" / "data.yaml",
        ROOT / "datasets" / "pcm_yolo_defocus_test_rmrnet_noac_ep001" / "data.yaml",
    ),
    ScenarioPair(
        "PCM",
        "low light",
        ROOT / "datasets" / "pcm_yolo_lowlight_test" / "data.yaml",
        ROOT / "datasets" / "pcm_yolo_lowlight_test_rmrnet_noac_ep001" / "data.yaml",
    ),
    ScenarioPair(
        "PCM",
        "mixed motion+low light",
        ROOT / "datasets" / "pcm_yolo_mixed_test" / "data.yaml",
        ROOT / "datasets" / "pcm_yolo_mixed_test_rmrnet_noac_ep001" / "data.yaml",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--conf", type=float, default=0.001)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def names_from_yaml(data: dict[str, Any]) -> list[str]:
    names = data["names"]
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(x) for x in names]


def split_image_dir(data_yaml: Path, split: str = "test") -> Path:
    data = read_yaml(data_yaml)
    root = Path(data["path"])
    split_value = data.get(split) or data.get("val")
    return root / split_value


def image_paths(data_yaml: Path, split: str = "test") -> list[Path]:
    img_dir = split_image_dir(data_yaml, split)
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found under {img_dir}")
    return paths


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def xywhn_to_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = box
    x1 = (x - w / 2.0) * width
    y1 = (y - h / 2.0) * height
    x2 = (x + w / 2.0) * width
    y2 = (y + h / 2.0) * height
    return [x1, y1, x2, y2]


def load_gt(images: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for image_path in images:
        with Image.open(image_path) as im:
            width, height = im.size
        boxes: list[list[float]] = []
        classes: list[int] = []
        label_path = label_path_for_image(image_path)
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                box = [float(x) for x in parts[1:5]]
                classes.append(cls)
                boxes.append(xywhn_to_xyxy(box, width, height))
        records.append({"path": str(image_path), "boxes": boxes, "classes": classes})
    return records


def predict_records(model: YOLO, images: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    results = model.predict(
        source=[str(p) for p in images],
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=0.7,
        max_det=300,
        verbose=False,
    )
    pred_records: list[dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        pred_records.append(
            {
                "boxes": boxes.xyxy.detach().cpu().numpy().astype(float).tolist() if boxes is not None else [],
                "classes": boxes.cls.detach().cpu().numpy().astype(int).tolist() if boxes is not None else [],
                "scores": boxes.conf.detach().cpu().numpy().astype(float).tolist() if boxes is not None else [],
            }
        )
    return pred_records


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap50_for_class(
    gt_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    cls: int,
    sample_indices: list[int],
) -> float | None:
    gt_by_sample: list[list[list[float]]] = []
    detections: list[tuple[int, float, list[float]]] = []
    total_gt = 0
    for sample_pos, source_idx in enumerate(sample_indices):
        gt = gt_records[source_idx]
        pred = pred_records[source_idx]
        sample_gt = [box for box, c in zip(gt["boxes"], gt["classes"]) if c == cls]
        gt_by_sample.append(sample_gt)
        total_gt += len(sample_gt)
        for box, c, score in zip(pred["boxes"], pred["classes"], pred["scores"]):
            if c == cls:
                detections.append((sample_pos, float(score), box))
    if total_gt == 0:
        return None
    detections.sort(key=lambda x: x[1], reverse=True)
    matched = [set() for _ in sample_indices]
    tp = np.zeros(len(detections), dtype=float)
    fp = np.zeros(len(detections), dtype=float)
    for det_i, (sample_pos, _score, pred_box) in enumerate(detections):
        best_iou, best_gt = 0.0, -1
        for gt_i, gt_box in enumerate(gt_by_sample[sample_pos]):
            if gt_i in matched[sample_pos]:
                continue
            overlap = iou_xyxy(pred_box, gt_box)
            if overlap > best_iou:
                best_iou, best_gt = overlap, gt_i
        if best_iou >= 0.5 and best_gt >= 0:
            tp[det_i] = 1.0
            matched[sample_pos].add(best_gt)
        else:
            fp[det_i] = 1.0
    if len(detections) == 0:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(total_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return average_precision(recalls, precisions)


def map50(
    gt_records: list[dict[str, Any]],
    pred_records: list[dict[str, Any]],
    num_classes: int,
    sample_indices: list[int],
) -> float:
    aps = [ap50_for_class(gt_records, pred_records, cls, sample_indices) for cls in range(num_classes)]
    valid = [ap for ap in aps if ap is not None]
    return float(np.mean(valid)) if valid else 0.0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def bootstrap_pair(
    gt_records: list[dict[str, Any]],
    degraded_preds: list[dict[str, Any]],
    restored_preds: list[dict[str, Any]],
    num_classes: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    n = len(gt_records)
    full_indices = list(range(n))
    degraded_point = map50(gt_records, degraded_preds, num_classes, full_indices)
    restored_point = map50(gt_records, restored_preds, num_classes, full_indices)
    rng = random.Random(seed)
    degraded_samples: list[float] = []
    restored_samples: list[float] = []
    delta_samples: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.randrange(n) for _ in range(n)]
        d = map50(gt_records, degraded_preds, num_classes, sample)
        r = map50(gt_records, restored_preds, num_classes, sample)
        degraded_samples.append(d)
        restored_samples.append(r)
        delta_samples.append(r - d)
    le_zero = sum(1 for x in delta_samples if x <= 0) / len(delta_samples)
    ge_zero = sum(1 for x in delta_samples if x >= 0) / len(delta_samples)
    p_two_sided = min(1.0, 2.0 * min(le_zero, ge_zero))
    return {
        "n_images": n,
        "bootstrap": n_bootstrap,
        "ap50_degraded": degraded_point,
        "ap50_rmrnet": restored_point,
        "delta_ap50": restored_point - degraded_point,
        "delta_ci_low": percentile(delta_samples, 2.5),
        "delta_ci_high": percentile(delta_samples, 97.5),
        "degraded_ci_low": percentile(degraded_samples, 2.5),
        "degraded_ci_high": percentile(degraded_samples, 97.5),
        "rmrnet_ci_low": percentile(restored_samples, 2.5),
        "rmrnet_ci_high": percentile(restored_samples, 97.5),
        "paired_bootstrap_p": p_two_sided,
    }


def val_metric(model: YOLO, data_yaml: Path, detector_key: str, pair_key: str, args: argparse.Namespace) -> dict[str, float]:
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split="test",
        project=str(ROOT / "runs" / "detect" / "runs" / "detection_eval_v28_review"),
        name=f"{detector_key}_{pair_key}",
        verbose=False,
    )
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    install_windows_safe_cache_pool()
    EXP.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "purpose": "review-requested cross-detector and uncertainty audit",
        "checkpoint_selection": "not used for checkpoint selection; held-out test audit only",
        "bootstrap_seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "detectors": [],
    }

    for detector in DETECTORS:
        for dataset_key in ["pothole", "pcm"]:
            weights = detector.pothole_weights if dataset_key == "pothole" else detector.pcm_weights
            if not weights.exists():
                raise FileNotFoundError(weights)
            model = YOLO(str(weights))
            manifest["detectors"].append({"detector": detector.key, "dataset": dataset_key, "weights": str(weights)})
            for pair in SCENARIO_PAIRS:
                if dataset_key == "pothole" and pair.dataset != "IVCNZ pothole":
                    continue
                if dataset_key == "pcm" and pair.dataset != "PCM":
                    continue
                data = read_yaml(pair.degraded_yaml)
                names = names_from_yaml(data)
                num_classes = len(names)
                pair_key = f"{dataset_key}_{pair.scenario.replace(' ', '_')}"
                for state, yaml_path in [("degraded", pair.degraded_yaml), ("rmrnet", pair.restored_yaml)]:
                    exact = val_metric(model, yaml_path, detector.key, f"{pair_key}_{state}", args)
                    metric_rows.append(
                        {
                            "detector": detector.family,
                            "dataset": pair.dataset,
                            "scenario": pair.scenario,
                            "state": state,
                            "weights": str(weights),
                            "data_yaml": str(yaml_path),
                            **exact,
                        }
                    )
                degraded_images = image_paths(pair.degraded_yaml, "test")
                restored_images = image_paths(pair.restored_yaml, "test")
                if [p.name for p in degraded_images] != [p.name for p in restored_images]:
                    raise RuntimeError(f"Image order/name mismatch for {pair}")
                gt_records = load_gt(degraded_images)
                degraded_preds = predict_records(model, degraded_images, args)
                restored_preds = predict_records(model, restored_images, args)
                stats = bootstrap_pair(
                    gt_records,
                    degraded_preds,
                    restored_preds,
                    num_classes=num_classes,
                    n_bootstrap=args.bootstrap,
                    seed=args.seed,
                )
                bootstrap_rows.append(
                    {
                        "detector": detector.family,
                        "dataset": pair.dataset,
                        "scenario": pair.scenario,
                        "weights": str(weights),
                        "num_classes": num_classes,
                        "class_names": ";".join(names),
                        **stats,
                    }
                )
                print(json.dumps(bootstrap_rows[-1]), flush=True)

    write_csv(EXP / "cross_detector_metrics.csv", metric_rows)
    write_csv(EXP / "bootstrap_ap50_deltas.csv", bootstrap_rows)
    (EXP / "review_robustness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": str(EXP / "cross_detector_metrics.csv"), "bootstrap": str(EXP / "bootstrap_ap50_deltas.csv")}, indent=2))


if __name__ == "__main__":
    main()
