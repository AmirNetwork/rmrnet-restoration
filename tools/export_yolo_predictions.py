from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


class SerialPool:
    """Avoid Windows cache thread-pool issues in restricted workspaces."""

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
    parser = argparse.ArgumentParser(description="Export YOLO prediction txt files for Snake boundary evaluation.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--project", default="runs/yolo_predictions")
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def image_dir_from_data(data_yaml: Path, split: str) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = data.get(split, f"images/{split}")
    split_path = Path(split_value)
    return split_path if split_path.is_absolute() else root / split_path


def main() -> None:
    args = parse_args()
    install_windows_safe_cache_pool()
    data_yaml = Path(args.data)
    source = image_dir_from_data(data_yaml, args.split)
    model = YOLO(args.weights)
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=False,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=False,
    )
    label_dir = Path(args.project) / args.name / "labels"
    print({"name": args.name, "images": len(results), "labels": str(label_dir)}, flush=True)


if __name__ == "__main__":
    main()
