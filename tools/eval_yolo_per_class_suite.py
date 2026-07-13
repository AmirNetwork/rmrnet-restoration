from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Evaluate YOLO per-class metrics on many data.yaml files.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--item", action="append", required=True, help="name=path/to/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--project", default="runs/detection_eval_per_class")
    parser.add_argument("--out", required=True)
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
        metrics = model.val(data=data, imgsz=args.imgsz, batch=args.batch, device=args.device, split=args.split, project=args.project, name=name, workers=args.workers, verbose=False)
        for class_id in metrics.box.ap_class_index.tolist():
            precision, recall, map50, map50_95 = metrics.box.class_result(int(class_id))
            row = {
                "eval_name": name,
                "class_id": int(class_id),
                "class_name": metrics.names[int(class_id)],
                "precision": float(precision),
                "recall": float(recall),
                "map50": float(map50),
                "map50_95": float(map50_95),
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
