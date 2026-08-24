#!/usr/bin/env python3
"""Compute paired validation PSNR/SSIM for frozen RMR-P v50 outputs."""

from __future__ import annotations

import csv
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
EXECUTED = Path(r"E:\RMRP_experiments\rmrp_expert_fusion_v50_validation_20260822")
BASELINE = Path(r"E:\RMRP_experiments\heterogeneous_expert_fusion_val_v1_20260822\full_validation")
OUT = ROOT / "experiments/final_rmrp_v50_validation_ledger_20260824"
SCENARIOS = {
    "motion": "motion_horizontal_medium",
    "defocus": "defocus_medium",
    "lowlight": "lowlight_medium",
    "mixed": "mixed_motion_lowlight",
}
PAIRED = {
    "ivcnz": ROOT / "data/pothole_restoration_practical_sensor_calibrated_v2_val",
    "pcm": ROOT / "data/pcm_restoration_practical_sensor_calibrated_v2_val",
}
MODELS = ("raw", "rmrp", "demoe", "dfpir", "instructir")


def load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def prediction_directory(model: str, dataset: str, cause: str) -> Path:
    if model == "raw":
        return PAIRED[dataset] / "scenarios" / SCENARIOS[cause] / "input"
    if model == "rmrp":
        return EXECUTED / "correct/restored" / dataset / cause / "images/val"
    return BASELINE / model / "epoch_070/restored" / dataset / cause / "images/val"


def score(pair: tuple[Path, Path]) -> tuple[str, float, float]:
    target_path, prediction_path = pair
    target = load(target_path)
    prediction = load(prediction_path)
    if prediction.shape != target.shape:
        raise RuntimeError(
            f"Shape mismatch: {prediction_path} {prediction.shape} != "
            f"{target_path} {target.shape}"
        )
    mse = float(np.mean((prediction - target) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    ssim = float(
        structural_similarity(target, prediction, channel_axis=2, data_range=1.0)
    )
    return target_path.stem, psnr, ssim


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    detailed: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for model in MODELS:
        for dataset in PAIRED:
            for cause in SCENARIOS:
                target_root = PAIRED[dataset] / "scenarios" / SCENARIOS[cause] / "gt"
                prediction_root = prediction_directory(model, dataset, cause)
                predictions = {
                    path.stem: path for path in prediction_root.iterdir() if path.is_file()
                }
                targets = sorted(path for path in target_root.iterdir() if path.is_file())
                pairs = []
                for target in targets:
                    prediction = predictions.get(target.stem)
                    if prediction is None:
                        raise FileNotFoundError(
                            f"No {model} prediction for {target.stem!r} in {prediction_root}"
                        )
                    pairs.append((target, prediction))
                with ThreadPoolExecutor(max_workers=8) as pool:
                    scores = list(pool.map(score, pairs))
                for stem, psnr, ssim in scores:
                    detailed.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "condition": cause,
                            "image": stem,
                            "psnr": psnr,
                            "ssim": ssim,
                            "selection_split": "validation",
                            "test_split_used": False,
                        }
                    )
                summary.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "condition": cause,
                        "n": len(scores),
                        "psnr": float(np.mean([value[1] for value in scores])),
                        "ssim": float(np.mean([value[2] for value in scores])),
                        "selection_split": "validation",
                        "test_split_used": False,
                    }
                )
                print(
                    f"{model:10s} {dataset:5s} {cause:8s} "
                    f"PSNR={summary[-1]['psnr']:.4f} SSIM={summary[-1]['ssim']:.4f}",
                    flush=True,
                )
    write_csv(OUT / "fidelity_per_image.csv", detailed)
    write_csv(OUT / "fidelity_summary.csv", summary)


if __name__ == "__main__":
    main()
