from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import ultralytics
from ultralytics import YOLO


class SerialPool:
    """Tiny drop-in replacement for Ultralytics cache ThreadPool on locked-down Windows."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __enter__(self) -> "SerialPool":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def imap(self, func, iterable):
        return map(func, iterable)


def install_windows_safe_cache_pool() -> None:
    import ultralytics.data.dataset as dataset
    import ultralytics.data.utils as utils

    dataset.ThreadPool = SerialPool
    utils.ThreadPool = SerialPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one YOLO detector on many data.yaml files.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--item", action="append", required=True, help="name=path/to/data.yaml")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence floor used to construct the PR curve.")
    parser.add_argument("--iou", type=float, default=0.70, help="Class-aware NMS IoU threshold.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", default="test")
    parser.add_argument("--project", default="runs/detection_eval_suite")
    parser.add_argument("--out", required=True, help="Output path without extension or with .csv/.json extension.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_windows_safe_cache_pool()
    model = YOLO(args.weights)
    out = Path(args.out)
    if out.suffix.lower() in {".csv", ".json"}:
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in args.item:
        if "=" not in item:
            raise ValueError(f"Expected --item name=path/to/data.yaml, got {item}")
        name, data = item.split("=", 1)
        metrics = model.val(
            data=data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split=args.split,
            project=args.project,
            name=name,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            seed=args.seed,
            deterministic=True,
            verbose=False,
        )
        row = {
            "name": name,
            "data": data,
            "split": args.split,
            "map50_95": float(metrics.box.map),
            "map50": float(metrics.box.map50),
            "map75": float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "class_names": json.dumps([metrics.names[int(i)] for i in metrics.box.ap_class_index]),
            "class_map50_95": json.dumps([float(v) for v in metrics.box.maps]),
            "class_map50": json.dumps([float(v) for v in metrics.box.ap50]),
            "class_precision": json.dumps([float(v) for v in metrics.box.p]),
            "class_recall": json.dumps([float(v) for v in metrics.box.r]),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "confidence_floor": args.conf,
            "nms_iou": args.iou,
            "seed": args.seed,
            "ultralytics_version": ultralytics.__version__,
            "resize_policy": "Ultralytics aspect-preserving letterbox",
            "ap_implementation": "Ultralytics COCO-style interpolated PR integration",
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
