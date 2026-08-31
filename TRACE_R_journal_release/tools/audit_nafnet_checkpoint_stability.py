#!/usr/bin/env python3
# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Audit NAFNet checkpoints for output instability on validation images.

Detector-aware restoration can improve validation AP while producing sparse,
large RGB residuals that are hidden by the final image clamp.  This audit runs
the *raw* model output on the existing validation-only subset and records both
range violations and conspicuous chromatic changes.  It never reads test data
or detector labels and therefore can be used as a checkpoint validity gate
before validation mAP selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.nafnet_road import build_nafnet_from_payload


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--severe-range-margin", type=float, default=0.25)
    parser.add_argument("--large-residual-threshold", type=float, default=0.35)
    parser.add_argument("--max-severe-image-fraction", type=float, default=0.01)
    parser.add_argument("--max-large-chroma-image-fraction", type=float, default=0.02)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(root: Path) -> list[Path]:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "images" in {part.lower() for part in path.parts}
    ]
    if not paths:
        raise FileNotFoundError(f"No validation images found under {root}")
    return sorted(paths)


def load_image(path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    height, width = tensor.shape[-2:]
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (height, width)


def image_metrics(
    raw: torch.Tensor,
    source: torch.Tensor,
    *,
    severe_range_margin: float,
    large_residual_threshold: float,
) -> dict[str, float]:
    raw = raw.float()
    source = source.float()
    clipped = raw.clamp(0.0, 1.0)
    residual = raw - source
    residual_rgb_range = residual.amax(dim=1) - residual.amin(dim=1)
    residual_magnitude = residual.abs().amax(dim=1)
    outside = ((raw < 0.0) | (raw > 1.0)).any(dim=1)
    severe = (
        (raw < -float(severe_range_margin))
        | (raw > 1.0 + float(severe_range_margin))
    ).any(dim=1)
    large_chroma = (
        residual_rgb_range > float(large_residual_threshold)
    ) & (residual_magnitude > float(large_residual_threshold))
    clamp_delta = (raw - clipped).abs().amax(dim=1)
    flat = residual.abs().flatten()
    return {
        "outside_fraction": float(outside.float().mean().cpu()),
        "severe_fraction": float(severe.float().mean().cpu()),
        "large_chroma_fraction": float(large_chroma.float().mean().cpu()),
        "clamp_delta_mean": float(clamp_delta.mean().cpu()),
        "residual_abs_mean": float(flat.mean().cpu()),
        "residual_abs_p99": float(torch.quantile(flat, 0.99).cpu()),
        "residual_abs_max": float(flat.max().cpu()),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in (
        "outside_fraction",
        "severe_fraction",
        "large_chroma_fraction",
        "clamp_delta_mean",
        "residual_abs_mean",
        "residual_abs_p99",
        "residual_abs_max",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"p95_image_{key}"] = float(np.quantile(values, 0.95))
        summary[f"max_image_{key}"] = float(values.max())
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoints = sorted(args.checkpoint_dir.glob("nafnet_epoch_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No retained NAFNet checkpoints in {args.checkpoint_dir}")
    paths = image_paths(args.validation_root)
    args.out.mkdir(parents=True, exist_ok=True)
    all_images: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []

    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model, _, official = build_nafnet_from_payload(payload)
        if not official:
            raise RuntimeError(f"Checkpoint is not the official width-32 NAFNet: {checkpoint}")
        model = model.to(device).eval()
        per_image: list[dict[str, float]] = []
        with torch.inference_mode():
            for path in paths:
                source, (height, width) = load_image(path, device)
                raw = model(source)[..., :height, :width]
                source_crop = source[..., :height, :width]
                metrics = image_metrics(
                    raw,
                    source_crop,
                    severe_range_margin=float(args.severe_range_margin),
                    large_residual_threshold=float(args.large_residual_threshold),
                )
                per_image.append(metrics)
                all_images.append(
                    {
                        "checkpoint": checkpoint.name,
                        "image": str(path.relative_to(args.validation_root)),
                        **metrics,
                    }
                )
        summary = summarize(per_image)
        valid = (
            summary["max_image_severe_fraction"]
            <= float(args.max_severe_image_fraction)
            and summary["max_image_large_chroma_fraction"]
            <= float(args.max_large_chroma_image_fraction)
        )
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint.name,
                "sha256": sha256(checkpoint),
                "official_width32": True,
                "validation_images": len(paths),
                "stability_valid": bool(valid),
                **summary,
            }
        )
        print(
            json.dumps(
                {
                    "checkpoint": checkpoint.name,
                    "stability_valid": bool(valid),
                    "max_image_severe_fraction": summary["max_image_severe_fraction"],
                    "max_image_large_chroma_fraction": summary[
                        "max_image_large_chroma_fraction"
                    ],
                }
            ),
            flush=True,
        )
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (args.out / "checkpoint_stability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checkpoint_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    with (args.out / "per_image_stability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_images[0]))
        writer.writeheader()
        writer.writerows(all_images)

    report = {
        "protocol": "validation-only NAFNet output-stability gate",
        "test_data_used": False,
        "validation_root": str(args.validation_root.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "images_per_checkpoint": len(paths),
        "thresholds": {
            "severe_range": [
                -float(args.severe_range_margin),
                1.0 + float(args.severe_range_margin),
            ],
            "large_residual": float(args.large_residual_threshold),
            "max_severe_image_fraction": float(args.max_severe_image_fraction),
            "max_large_chroma_image_fraction": float(
                args.max_large_chroma_image_fraction
            ),
        },
        "checkpoints": checkpoint_rows,
    }
    (args.out / "checkpoint_stability.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
