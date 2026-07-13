from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageFilter
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SerialPool:
    """Ultralytics cache helper for locked-down Windows environments."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> "SerialPool":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def imap(self, func: Any, iterable: Any) -> Any:
        return map(func, iterable)


def install_windows_safe_cache_pool() -> None:
    import ultralytics.data.dataset as dataset
    import ultralytics.data.utils as utils

    dataset.ThreadPool = SerialPool
    utils.ThreadPool = SerialPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and apply a no-reference input/restored perception gate.")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--val-input-data", required=True, help="Validation degraded/native YOLO data.yaml.")
    parser.add_argument("--val-restored-data", required=True, help="Validation restored YOLO data.yaml.")
    parser.add_argument("--test-input-data", required=True, help="Test degraded/native YOLO data.yaml.")
    parser.add_argument("--test-restored-data", required=True, help="Test restored YOLO data.yaml.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split-val", default="val")
    parser.add_argument("--split-test", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/detection_eval_perception_gate")
    parser.add_argument("--extra-threshold", action="append", type=float, default=[])
    parser.add_argument(
        "--min-gate-improvement",
        type=float,
        default=0.005,
        help="Minimum validation mAP50 gain over all-restored output required before using a mixed gate.",
    )
    return parser.parse_args()


def read_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    yaml_path = Path(path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(data.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root, data


def split_dir(root: Path, data: dict[str, Any], split: str, kind: str) -> Path:
    split_value = data.get(split, data.get("val", f"images/{split}"))
    split_path = Path(str(split_value).replace("images", kind, 1))
    return split_path if split_path.is_absolute() else root / split_path


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def road_evidence_score(path: Path) -> float:
    """No-reference road-evidence score used only for gating.

    The crop excludes the sky-heavy top quarter and scores edge strength, local
    contrast, and usable exposure. It deliberately avoids labels, detections,
    clean targets, or any test-set metric.
    """

    with Image.open(path) as image:
        gray_img = image.convert("L")
        width, height = gray_img.size
        gray_img = gray_img.crop((0, int(height * 0.25), width, height))
        blurred = gray_img.filter(ImageFilter.GaussianBlur(radius=2.0))
        gray = np.asarray(gray_img, dtype=np.float32) / 255.0
        low = np.asarray(blurred, dtype=np.float32) / 255.0

    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    edge = np.sqrt(gx * gx + gy * gy)
    contrast = np.abs(gray - low)
    mean = float(gray.mean())
    exposure_penalty = abs(mean - 0.45)
    clipped = float(((gray < 0.025) | (gray > 0.975)).mean())
    return (
        1.20 * float(np.percentile(edge, 90))
        + 0.70 * float(gray.std())
        + 0.50 * float(np.percentile(contrast, 90))
        - 0.30 * exposure_penalty
        - 0.25 * clipped
    )


def collect_delta_scores(input_data: str | Path, restored_data: str | Path, split: str) -> dict[str, float]:
    input_root, input_cfg = read_yaml(input_data)
    restored_root, restored_cfg = read_yaml(restored_data)
    input_dir = split_dir(input_root, input_cfg, split, "images")
    restored_dir = split_dir(restored_root, restored_cfg, split, "images")
    restored_by_name = {p.name: p for p in list_images(restored_dir)}
    deltas = {}
    for input_path in list_images(input_dir):
        restored_path = restored_by_name.get(input_path.name)
        if restored_path is None:
            continue
        deltas[input_path.name] = road_evidence_score(restored_path) - road_evidence_score(input_path)
    if not deltas:
        raise RuntimeError(f"No overlapping images between {input_dir} and {restored_dir}")
    return deltas


def candidate_thresholds(deltas: dict[str, float], extra: list[float]) -> list[float]:
    values = np.asarray(list(deltas.values()), dtype=np.float32)
    quantiles = [0, 5, 10, 25, 40, 50, 60, 75, 90, 95, 100]
    thresholds = {-1e9, 1e9}
    thresholds.update(float(np.percentile(values, q)) for q in quantiles)
    thresholds.update(float(item) for item in extra)
    return sorted(thresholds)


def make_gated_dataset(
    input_data: str | Path,
    restored_data: str | Path,
    split: str,
    threshold: float,
    out_dir: Path,
) -> tuple[Path, dict[str, int]]:
    input_root, input_cfg = read_yaml(input_data)
    restored_root, restored_cfg = read_yaml(restored_data)
    input_image_dir = split_dir(input_root, input_cfg, split, "images")
    restored_image_dir = split_dir(restored_root, restored_cfg, split, "images")
    input_label_dir = split_dir(input_root, input_cfg, split, "labels")
    restored_by_name = {p.name: p for p in list_images(restored_image_dir)}

    out_image_dir = out_dir / "images" / split
    out_label_dir = out_dir / "labels" / split
    for target_dir in (out_image_dir, out_label_dir):
        if target_dir.exists():
            shutil.rmtree(target_dir)
    cache_file = out_dir / "labels" / f"{split}.cache"
    if cache_file.exists():
        cache_file.unlink()
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"input": 0, "restored": 0}
    for input_path in list_images(input_image_dir):
        restored_path = restored_by_name.get(input_path.name)
        if restored_path is None:
            continue
        delta = road_evidence_score(restored_path) - road_evidence_score(input_path)
        use_restored = delta > threshold
        source_path = restored_path if use_restored else input_path
        shutil.copy2(source_path, out_image_dir / input_path.name)
        label_path = input_label_dir / input_path.with_suffix(".txt").name
        if label_path.exists():
            shutil.copy2(label_path, out_label_dir / label_path.name)
        choice = "restored" if use_restored else "input"
        counts[choice] += 1
        rows.append({"image": input_path.name, "delta_score": delta, "choice": choice})

    data_yaml = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": input_cfg.get("train", f"images/{split}"),
        "val": f"images/{split}",
        "test": f"images/{split}",
        "names": input_cfg["names"],
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    with (out_dir / "gate_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "delta_score", "choice"])
        writer.writeheader()
        writer.writerows(rows)
    return yaml_path, counts


def evaluate(model: YOLO, data_yaml: Path, split: str, args: argparse.Namespace, name: str) -> dict[str, float | str]:
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=split,
        project=args.project,
        name=name,
        workers=args.workers,
        verbose=False,
    )
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main() -> None:
    args = parse_args()
    install_windows_safe_cache_pool()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.yolo_weights)

    deltas = collect_delta_scores(args.val_input_data, args.val_restored_data, args.split_val)
    rows = []
    best: dict[str, Any] | None = None
    restored_reference: dict[str, Any] | None = None
    for index, threshold in enumerate(candidate_thresholds(deltas, args.extra_threshold)):
        candidate_dir = out_root / f"{args.name}_val_tau_{index:02d}"
        data_yaml, counts = make_gated_dataset(args.val_input_data, args.val_restored_data, args.split_val, threshold, candidate_dir)
        metrics = evaluate(model, data_yaml, args.split_val, args, f"{args.name}_val_tau_{index:02d}")
        row = {"threshold": threshold, **counts, **metrics}
        rows.append(row)
        if threshold == -1e9:
            restored_reference = row
        if best is None or (row["map50"], row["map50_95"], row["restored"]) > (best["map50"], best["map50_95"], best["restored"]):
            best = row
        print(json.dumps({"candidate": index, **row}), flush=True)

    if best is None:
        raise RuntimeError("No gate candidate evaluated")
    if restored_reference is not None and best is not restored_reference:
        improvement = float(best["map50"]) - float(restored_reference["map50"])
        if improvement < args.min_gate_improvement:
            best = restored_reference
            print(
                json.dumps(
                    {
                        "gate_policy": "fallback_all_restored",
                        "reason": "validation_gain_below_margin",
                        "min_gate_improvement": args.min_gate_improvement,
                        "observed_gain": improvement,
                    }
                ),
                flush=True,
            )
    test_dir = out_root / f"{args.name}_test_gated"
    test_yaml, test_counts = make_gated_dataset(
        args.test_input_data,
        args.test_restored_data,
        args.split_test,
        float(best["threshold"]),
        test_dir,
    )
    test_metrics = evaluate(model, test_yaml, args.split_test, args, f"{args.name}_test_gated")
    summary = {
        "name": args.name,
        "best_threshold": best["threshold"],
        "best_val": best,
        "test_data": str(test_yaml),
        "test_counts": test_counts,
        "test_metrics": test_metrics,
    }

    with (out_root / f"{args.name}_val_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_root / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
