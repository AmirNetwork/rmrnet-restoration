"""Train an Ultralytics YOLO detector with a reproducible road-defect recipe.

This wrapper keeps detector training commands short and records a compact
summary next to the Ultralytics run. It is intentionally detector-only: RMR-Net
training and restoration evaluation remain in the existing project scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for road-defect detection.")
    parser.add_argument("--data", required=True, help="Ultralytics data.yaml file.")
    parser.add_argument("--base-weights", default="yolo11s.pt", help="Pretrained detector weights.")
    parser.add_argument("--project", default="runs/yolo11s_v26", help="Output project directory.")
    parser.add_argument("--name", required=True, help="Run name under the project directory.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics image caching.")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    args = parse_args()
    model = YOLO(args.base_weights)
    project = Path(args.project)
    if not project.is_absolute():
        project = ROOT / project

    result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=False,
        patience=args.patience,
        seed=args.seed,
        cos_lr=True,
        close_mosaic=args.close_mosaic,
        save_period=args.save_period,
        cache=args.cache,
    )

    save_dir = Path(getattr(result, "save_dir", Path(args.project) / args.name))
    summary = {
        "data": str(Path(args.data)),
        "base_weights": args.base_weights,
        "project": str(project),
        "name": args.name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.patience,
        "save_period": args.save_period,
        "close_mosaic": args.close_mosaic,
        "cos_lr": True,
        "cache": args.cache,
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "results_csv": str(save_dir / "results.csv"),
    }
    (save_dir / "training_recipe.json").write_text(json.dumps(_jsonable(summary), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
