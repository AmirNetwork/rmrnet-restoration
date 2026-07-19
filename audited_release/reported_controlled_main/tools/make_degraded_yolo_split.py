# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcadnet.synthetic_metadata import synthetic_metadata_from_scenario


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create degraded YOLO split while preserving normalized labels.")
    parser.add_argument("--data", required=True, help="Source YOLO data.yaml.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=640)
    return parser.parse_args()


def resize_max(image: Image.Image, max_side: int) -> Image.Image:
    if not max_side or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.BICUBIC)


def motion_blur(image: Image.Image, angle_deg: float, length: int) -> Image.Image:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    angle = math.radians(angle_deg)
    for i in range(length):
        x = int(round(center + (i - center) * math.cos(angle)))
        y = int(round(center + (i - center) * math.sin(angle)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    kernel /= kernel.sum()
    arr = np.asarray(image)
    out = cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def degrade(image: Image.Image, scenario: str, seed: int) -> Image.Image:
    name = scenario.lower()
    rng = np.random.default_rng(seed)
    out = image
    applied = False
    if "horizontal" in name or ("motion" in name and "vertical" not in name and "diagonal" not in name):
        out = motion_blur(out, 0, 17)
        applied = True
    if "vertical" in name:
        out = motion_blur(out, 90, 17)
        applied = True
    if "diagonal" in name:
        out = motion_blur(out, 35, 17)
        applied = True
    if "defocus" in name:
        out = out.filter(ImageFilter.GaussianBlur(radius=2.2))
        applied = True
    if "lowlight" in name:
        arr = np.asarray(out).astype(np.float32) * 0.45 + rng.normal(0, 5, (*out.size[::-1], 3))
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        applied = True
    if "gaussian" in name or "noise" in name:
        arr = np.asarray(out).astype(np.float32) + rng.normal(0, 8, (*out.size[::-1], 3))
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        applied = True
    if applied:
        return out
    raise ValueError(f"Unsupported scenario: {scenario}")


def main() -> None:
    args = parse_args()
    source_yaml = Path(args.data)
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source_root = Path(config["path"])
    image_dir = source_root / config[args.split]
    label_dir = source_root / config[args.split].replace("images", "labels")
    out = Path(args.out)
    out_image_dir = out / "images" / args.split
    out_label_dir = out / "labels" / args.split
    out_metadata_dir = out / "metadata" / args.split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_metadata_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        image_paths = image_paths[: args.limit]
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            degraded = degrade(resize_max(image.convert("RGB"), args.max_side), args.scenario, index)
        degraded.save(out_image_dir / image_path.name)
        label_path = label_dir / image_path.with_suffix(".txt").name
        if label_path.exists():
            shutil.copy2(label_path, out_label_dir / label_path.name)
        metadata = synthetic_metadata_from_scenario(args.scenario, seed=index)
        (out_metadata_dir / f"{image_path.stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    data_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": config.get("train", "images/train"),
        "val": f"images/{args.split}",
        "test": f"images/{args.split}",
        "names": config["names"],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print({"scenario": args.scenario, "split": args.split, "images": len(image_paths), "out": str(out)})


if __name__ == "__main__":
    main()
