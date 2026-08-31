#!/usr/bin/env python3
"""Fine-tune and freeze the CRID-320 field detector on train/validation only.

The script deliberately accepts a dataset YAML with no ``test`` key.  Model
selection is therefore made by Ultralytics' validation metric without opening
the sealed temporal test block.  A provenance record is written beside the
training run before and after optimization.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT
    / "datasets"
    / "crid320_annotation_20260829"
    / "exports"
    / "latest"
    / "data_train_val.yaml"
)
DEFAULT_FREEZE = (
    ROOT
    / "datasets"
    / "crid320_annotation_20260829"
    / "annotation_freeze.json"
)
DEFAULT_INIT = ROOT / "Yolo26_coordinate" / "YOLO26s_RDD_FRDC_Distilled_v2.pt"
DEFAULT_OUT = Path(r"E:\TRACE_R_experiments\crid320_detector_v1_20260829")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--annotation-freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--weights", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return payload


def resolve_list(data: dict[str, Any], key: str) -> Path:
    root = Path(str(data["path"]))
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    return (root / str(data[key])).resolve()


def count_list(path: Path) -> int:
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.annotation_freeze = args.annotation_freeze.resolve()
    args.weights = args.weights.resolve()
    args.out = args.out.resolve()
    for path in (args.data, args.annotation_freeze, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)

    data = read_yaml(args.data)
    if "test" in data:
        raise RuntimeError(
            "Detector selection YAML must not expose the sealed CRID-320 test split"
        )
    if set(("train", "val")) - set(data):
        raise RuntimeError("Detector selection YAML requires train and val entries")
    train_list = resolve_list(data, "train")
    val_list = resolve_list(data, "val")
    if not train_list.exists() or not val_list.exists():
        raise FileNotFoundError("CRID-320 train/validation manifests are incomplete")
    if count_list(train_list) != 180 or count_list(val_list) != 60:
        raise RuntimeError("Expected the locked 180/60 CRID-320 train/validation split")

    freeze = json.loads(args.annotation_freeze.read_text(encoding="utf-8"))
    frozen_annotations = Path(freeze["frozen_annotations_path"])
    if sha256(frozen_annotations) != freeze["frozen_annotations_sha256"]:
        raise RuntimeError("Frozen CRID-320 annotations changed after review")

    args.out.mkdir(parents=True, exist_ok=True)
    project = args.out.parent
    name = args.out.name
    audit = {
        "protocol": "CRID-320 sequence-disjoint detector adaptation",
        "selection": "Ultralytics best checkpoint by validation fitness only",
        "test_split_opened": False,
        "annotation_freeze": str(args.annotation_freeze),
        "annotation_freeze_sha256": sha256(args.annotation_freeze),
        "frozen_annotations_sha256": freeze["frozen_annotations_sha256"],
        "locked_split_sha256": freeze["locked_split_sha256"],
        "dataset_yaml": str(args.data),
        "dataset_yaml_sha256": sha256(args.data),
        "train_manifest_sha256": sha256(train_list),
        "validation_manifest_sha256": sha256(val_list),
        "train_images": count_list(train_list),
        "validation_images": count_list(val_list),
        "initial_checkpoint": str(args.weights),
        "initial_checkpoint_sha256": sha256(args.weights),
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": "AdamW",
        "lr0": 2.0e-4,
        "lrf": 0.05,
        "weight_decay": 5.0e-4,
        "close_mosaic": 12,
        "started_unix": time.time(),
    }
    audit_path = args.out / "detector_training_audit.json"
    if not audit_path.exists():
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    last = args.out / "weights" / "last.pt"
    source = last if args.resume and last.exists() else args.weights
    model = YOLO(str(source))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        optimizer="AdamW",
        lr0=2.0e-4,
        lrf=0.05,
        momentum=0.9,
        weight_decay=5.0e-4,
        warmup_epochs=3.0,
        cos_lr=True,
        close_mosaic=12,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        degrees=2.0,
        translate=0.08,
        scale=0.25,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.20,
        cache=False,
        amp=True,
        save=True,
        save_period=5,
        plots=True,
        val=True,
        project=str(project),
        name=name,
        exist_ok=True,
        resume=bool(args.resume and last.exists()),
        verbose=True,
    )

    best = args.out / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    selected = {
        **audit,
        "status": "FROZEN_BEFORE_CRID320_TEST",
        "completed_unix": time.time(),
        "selected_checkpoint": str(best),
        "selected_checkpoint_sha256": sha256(best),
        "test_split_opened": False,
    }
    (args.out / "detector_selection_freeze.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    shutil.copy2(args.data, args.out / "data_train_val.yaml")
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
