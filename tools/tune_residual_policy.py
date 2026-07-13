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
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tune_perception_gate import (  # noqa: E402
    IMAGE_EXTS,
    candidate_thresholds,
    install_windows_safe_cache_pool,
    list_images,
    read_yaml,
    road_evidence_score,
    split_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune residual-strength and pass-through policy on validation mAP.")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--val-input-data", required=True)
    parser.add_argument("--val-restored-data", required=True)
    parser.add_argument("--test-input-data", required=True)
    parser.add_argument("--test-restored-data", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split-val", default="val")
    parser.add_argument("--split-test", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/detection_eval_residual_policy")
    parser.add_argument("--strength", action="append", type=float, default=[])
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--gain", action="append", type=float, default=[], help="Optional gain applied to residual-policy frames.")
    parser.add_argument("--contrast", action="append", type=float, default=[], help="Optional contrast factor applied to residual-policy frames.")
    parser.add_argument("--only-thresholds", action="store_true", help="Use only explicit --threshold values instead of quantile thresholds.")
    parser.add_argument("--min-policy-improvement", type=float, default=0.005)
    return parser.parse_args()


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


def blend_images(input_path: Path, restored_path: Path, strength: float, gain: float, contrast: float, out_path: Path) -> None:
    """Apply the manuscript residual policy.

    Equation:
        I_o = I_d + eta * (I_r - I_d)

    ``strength`` is eta. Optional gain/contrast are validation-searched
    calibration factors and must not be confused with eta.
    """
    with Image.open(input_path) as input_image, Image.open(restored_path) as restored_image:
        base = np.asarray(input_image.convert("RGB"), dtype=np.float32)
        restored = np.asarray(restored_image.convert("RGB").resize(input_image.size), dtype=np.float32)
    blended = np.clip(base + strength * (restored - base), 0.0, 255.0).astype(np.uint8)
    if gain != 1.0 or contrast != 1.0:
        calibrated = blended.astype(np.float32) / 255.0
        mean = calibrated.mean(axis=(0, 1), keepdims=True)
        calibrated = (calibrated - mean) * contrast + mean
        calibrated = calibrated * gain
        blended = np.clip(calibrated * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(blended).save(out_path)


def make_policy_dataset(
    input_data: str | Path,
    restored_data: str | Path,
    split: str,
    threshold: float,
    strength: float,
    gain: float,
    contrast: float,
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
    counts = {"input": 0, "residual": 0}
    for input_path in list_images(input_image_dir):
        restored_path = restored_by_name.get(input_path.name)
        if restored_path is None:
            continue
        delta = road_evidence_score(restored_path) - road_evidence_score(input_path)
        use_residual = delta > threshold and strength > 0.0
        if use_residual:
            blend_images(input_path, restored_path, strength, gain, contrast, out_image_dir / input_path.name)
            counts["residual"] += 1
            choice = f"residual_{strength:.2f}_gain_{gain:.2f}_contrast_{contrast:.2f}"
        else:
            shutil.copy2(input_path, out_image_dir / input_path.name)
            counts["input"] += 1
            choice = "input"

        label_path = input_label_dir / input_path.with_suffix(".txt").name
        if label_path.exists():
            shutil.copy2(label_path, out_label_dir / label_path.name)
        rows.append(
            {
                "image": input_path.name,
                "delta_score": delta,
                "choice": choice,
                "strength": strength,
                "gain": gain,
                "contrast": contrast,
            }
        )

    data_yaml = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": input_cfg.get("train", f"images/{split}"),
        "val": f"images/{split}",
        "test": f"images/{split}",
        "names": input_cfg["names"],
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    with (out_dir / "policy_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "delta_score", "choice", "strength", "gain", "contrast"])
        writer.writeheader()
        writer.writerows(rows)
    return yaml_path, counts


def evaluate(model: YOLO, data_yaml: Path, split: str, args: argparse.Namespace, name: str) -> dict[str, float]:
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
        plots=False,
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
    strengths = args.strength or [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]
    thresholds = sorted(args.threshold) if args.only_thresholds and args.threshold else candidate_thresholds(deltas, args.threshold)
    gains = args.gain or [1.0]
    contrasts = args.contrast or [1.0]

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    index = 0
    for strength in strengths:
        for threshold in thresholds:
            for gain in gains:
                for contrast in contrasts:
                    candidate_dir = out_root / f"{args.name}_val_policy_{index:03d}"
                    data_yaml, counts = make_policy_dataset(
                        args.val_input_data,
                        args.val_restored_data,
                        args.split_val,
                        threshold,
                        strength,
                        gain,
                        contrast,
                        candidate_dir,
                    )
                    metrics = evaluate(model, data_yaml, args.split_val, args, f"{args.name}_val_policy_{index:03d}")
                    row = {"strength": strength, "threshold": threshold, "gain": gain, "contrast": contrast, **counts, **metrics}
                    rows.append(row)
                    # Fallback is the better validation endpoint between input and all-restored.
                    if (
                        gain == 1.0
                        and contrast == 1.0
                        and ((strength == 0.0 and threshold == -1e9) or (strength == 1.0 and threshold == -1e9))
                    ):
                        if fallback is None or (row["map50"], row["map50_95"]) > (fallback["map50"], fallback["map50_95"]):
                            fallback = row
                    if best is None or (row["map50"], row["map50_95"], row["residual"]) > (
                        best["map50"],
                        best["map50_95"],
                        best["residual"],
                    ):
                        best = row
                    print(json.dumps({"candidate": index, **row}), flush=True)
                    index += 1

    if best is None:
        raise RuntimeError("No residual policy candidate evaluated")
    if fallback is not None and best is not fallback:
        improvement = float(best["map50"]) - float(fallback["map50"])
        if improvement < args.min_policy_improvement:
            print(
                json.dumps(
                    {
                        "policy": "fallback_endpoint",
                        "reason": "validation_gain_below_margin",
                        "observed_gain": improvement,
                        "min_policy_improvement": args.min_policy_improvement,
                    }
                ),
                flush=True,
            )
            best = fallback

    test_dir = out_root / f"{args.name}_test_policy"
    test_yaml, test_counts = make_policy_dataset(
        args.test_input_data,
        args.test_restored_data,
        args.split_test,
        float(best["threshold"]),
        # Argument order is threshold, eta, gain, contrast.
        float(best["strength"]),
        float(best["gain"]),
        float(best["contrast"]),
        test_dir,
    )
    test_metrics = evaluate(model, test_yaml, args.split_test, args, f"{args.name}_test_policy")
    summary = {
        "name": args.name,
        "best_policy": best,
        "test_data": str(test_yaml),
        "test_counts": test_counts,
        "test_metrics": test_metrics,
    }
    with (out_root / f"{args.name}_val_policies.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_root / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
