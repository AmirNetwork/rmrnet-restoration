#!/usr/bin/env python3
"""Validation-only low-dimensional detail calibration for sharp CRID frames.

This diagnostic never reads the sealed CRID test block.  It asks whether a
small, globally interpretable operation around the validation-selected DFPIR
image generalizes better than a learned spatial refiner.  Failed candidates
are deleted; the complete metric ledger and selected images are retained.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_crid320_validation import (
    CLASS_NAMES,
    hardlink_or_copy,
    label_for_image,
    metric_payload,
    sha256,
    validation_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=ROOT / "datasets" / "crid320_annotation_20260829" / "exports" / "latest",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="0")
    parser.add_argument("--only-sigma", type=float)
    parser.add_argument("--only-alpha", type=float)
    return parser.parse_args()


def candidates() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = [{"kind": "identity"}]
    for sigma in (0.75, 1.5, 3.0, 5.0):
        for alpha in (-0.35, -0.20, -0.10, 0.10, 0.20, 0.35):
            rows.append({"kind": "unsharp", "sigma": sigma, "alpha": alpha})
    for gamma in (0.90, 0.95, 1.05, 1.10):
        rows.append({"kind": "gamma", "gamma": gamma})
    for contrast in (0.94, 0.97, 1.03, 1.06):
        rows.append({"kind": "contrast", "contrast": contrast})
    return rows


def tag(spec: dict[str, float | str]) -> str:
    if spec["kind"] == "identity":
        return "identity"
    return "_".join(
        [str(spec["kind"])]
        + [f"{key}{str(value).replace('-', 'm').replace('.', 'p')}" for key, value in spec.items() if key != "kind"]
    )


def transform(image: np.ndarray, spec: dict[str, float | str]) -> np.ndarray:
    floating = image.astype(np.float32) / 255.0
    kind = spec["kind"]
    if kind == "unsharp":
        sigma = float(spec["sigma"])
        alpha = float(spec["alpha"])
        low = cv2.GaussianBlur(floating, (0, 0), sigmaX=sigma, sigmaY=sigma)
        floating = floating + alpha * (floating - low)
    elif kind == "gamma":
        floating = np.power(np.clip(floating, 0.0, 1.0), float(spec["gamma"]))
    elif kind == "contrast":
        # Preserve each channel's global mean while scaling local contrast.
        mean = floating.mean(axis=(0, 1), keepdims=True)
        floating = mean + float(spec["contrast"]) * (floating - mean)
    elif kind != "identity":
        raise ValueError(kind)
    return np.clip(np.rint(floating * 255.0), 0, 255).astype(np.uint8)


def build_dataset(
    spec: dict[str, float | str],
    paths: list[Path],
    base_dir: Path,
    root: Path,
) -> Path:
    if root.exists():
        shutil.rmtree(root)
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    outputs = []
    for source in paths:
        base = base_dir / f"{source.stem}.jpg"
        if not base.exists():
            raise FileNotFoundError(base)
        target = image_dir / base.name
        if spec["kind"] == "identity":
            hardlink_or_copy(base, target)
        else:
            image = cv2.imread(str(base), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not decode {base}")
            output = transform(image, spec)
            if not cv2.imwrite(
                str(target), output, [cv2.IMWRITE_JPEG_QUALITY, 98]
            ):
                raise RuntimeError(f"Could not write {target}")
        hardlink_or_copy(label_for_image(source), label_dir / f"{source.stem}.txt")
        outputs.append(target.resolve())
    manifest = root / "val.txt"
    manifest.write_text("\n".join(map(str, outputs)) + "\n", encoding="utf-8")
    data = {
        "path": str(root.resolve()),
        "train": str(manifest.resolve()),
        "val": str(manifest.resolve()),
        "names": CLASS_NAMES,
    }
    data_path = root / "data_val_only.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_path


def main() -> None:
    args = parse_args()
    args.base_dir = args.base_dir.resolve()
    args.detector = args.detector.resolve()
    args.export_root = args.export_root.resolve()
    args.out = args.out.resolve()
    for path in (args.base_dir, args.detector, args.export_root):
        if not path.exists():
            raise FileNotFoundError(path)
    paths = validation_paths(args.export_root)
    detector = YOLO(str(args.detector))
    workspace = args.out / "_candidate"
    best_root = args.out / "selected_images"
    records = []
    best = None
    specs = candidates()
    if args.only_sigma is not None or args.only_alpha is not None:
        if args.only_sigma is None or args.only_alpha is None:
            raise ValueError("--only-sigma and --only-alpha must be used together")
        specs = [
            {"kind": "unsharp", "sigma": args.only_sigma, "alpha": args.only_alpha}
        ]
    for spec in specs:
        name = tag(spec)
        data = build_dataset(spec, paths, args.base_dir, workspace)
        metrics = detector.val(
            data=str(data),
            imgsz=args.imgsz,
            batch=1,
            device=args.device,
            conf=0.001,
            iou=0.70,
            plots=False,
            save_json=False,
            verbose=False,
            project=str(args.out / "detector_runs"),
            name=name,
        )
        row = {"tag": name, "spec": spec, **metric_payload(metrics)}
        records.append(row)
        print(json.dumps({"tag": name, "map50": row["map50"], "map50_95": row["map50_95"]}), flush=True)
        rank = (float(row["map50"]), float(row["map50_95"]))
        if best is None or rank > best[0]:
            best = (rank, row)
            if best_root.exists():
                shutil.rmtree(best_root)
            shutil.copytree(workspace / "images" / "val", best_root)
    shutil.rmtree(workspace, ignore_errors=True)
    assert best is not None
    report = {
        "status": "VALIDATION_ONLY_LOW_DIMENSIONAL_SWEEP_TEST_UNOPENED",
        "base_dir": str(args.base_dir),
        "detector": str(args.detector),
        "detector_sha256": sha256(args.detector),
        "validation_manifest": str((args.export_root / "validation.txt").resolve()),
        "selection_metric": "validation mAP50; mAP50-95 breaks ties",
        "candidates": records,
        "selected": best[1],
        "test_images_or_labels_read": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detail_sweep_selection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
