from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO detector and save compact metrics JSON.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default="runs/detection_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device, split=args.split, project=args.out, name=args.name)
    row = {
        "name": args.name,
        "weights": args.weights,
        "data": args.data,
        "split": args.split,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    out_path = Path(args.out) / f"{args.name}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
