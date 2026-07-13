from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the IVCNZ pothole release into Ultralytics YOLO layout.")
    parser.add_argument("--raw-dir", required=True, help="Folder containing img-N.jpg and img-N.txt files.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out = Path(args.out)
    pairs = []
    for image_path in sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label_path = image_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((image_path, label_path))
    if not pairs:
        raise RuntimeError(f"No image/label pairs found in {raw_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    train_end = int(len(pairs) * args.train)
    val_end = train_end + int(len(pairs) * args.val)
    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    for split, split_pairs in splits.items():
        image_dir = out / "images" / split
        label_dir = out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in split_pairs:
            shutil.copy2(image_path, image_dir / image_path.name)
            shutil.copy2(label_path, label_dir / label_path.name)

    data_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole"},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print({"total": len(pairs), **{split: len(items) for split, items in splits.items()}})


if __name__ == "__main__":
    main()

