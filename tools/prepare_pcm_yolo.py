from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the Zenodo pothole/crack/manhole dataset to YOLO boxes.")
    parser.add_argument("--raw-root", default="datasets/raw/road_damage_pcm_zenodo_17834373/data")
    parser.add_argument("--out", default="datasets/road_damage_pcm_yolo")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    return parser.parse_args()


def polygon_line_to_box(line: str) -> str | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        x, y, w, h = values
    else:
        xs = values[0::2]
        ys = values[1::2]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(1.0, max(xs))
        y2 = min(1.0, max(ys))
        x = (x1 + x2) / 2.0
        y = (y1 + y2) / 2.0
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return None
    return f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    image_dir = raw_root / "images"
    label_dir = raw_root / "labels"
    out = Path(args.out)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and (label_dir / f"{p.stem}.txt").exists())
    random.Random(args.seed).shuffle(images)

    n = len(images)
    n_train = round(n * args.train_frac)
    n_val = round(n * args.val_frac)
    split_map = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split, paths in split_map.items():
        out_img = out / "images" / split
        out_lab = out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lab.mkdir(parents=True, exist_ok=True)
        for image_path in paths:
            with Image.open(image_path) as image:
                image.convert("RGB").save(out_img / image_path.name)
            label_path = label_dir / f"{image_path.stem}.txt"
            converted = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                box = polygon_line_to_box(line)
                if box is not None:
                    converted.append(box)
            (out_lab / f"{image_path.stem}.txt").write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")

    data = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole", 1: "crack", 2: "manhole"},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print({split: len(paths) for split, paths in split_map.items()})


if __name__ == "__main__":
    main()
