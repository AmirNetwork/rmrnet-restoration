# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import json
import math
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
    parser = argparse.ArgumentParser(description="Create RCAD-Net restoration pairs from a YOLO image split.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenario", action="append", dest="scenarios", required=True)
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
    config = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root = Path(config["path"])
    image_dir = root / config[args.split]
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        paths = paths[: args.limit]

    out = Path(args.out)
    for scenario in args.scenarios:
        input_dir = out / "scenarios" / scenario / "input"
        gt_dir = out / "scenarios" / scenario / "gt"
        metadata_dir = out / "scenarios" / scenario / "metadata"
        input_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                clean = resize_max(image.convert("RGB"), args.max_side)
            clean.save(gt_dir / path.name)
            degrade(clean, scenario, seed=index).save(input_dir / path.name)
            metadata = synthetic_metadata_from_scenario(scenario, seed=index)
            (metadata_dir / f"{path.stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print({"images_per_scenario": len(paths), "scenarios": args.scenarios, "out": str(out)})


if __name__ == "__main__":
    main()
