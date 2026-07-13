from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


SCENARIOS = (
    "motion_horizontal_medium",
    "motion_vertical_medium",
    "defocus_medium",
    "lowlight_medium",
    "mixed_vibration_noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic paired road-restoration smoke-test dataset.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--size", type=int, default=192)
    return parser.parse_args()


def road_like_clean(size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    base = Image.new("RGB", (size, size), (76, 78, 74))
    draw = ImageDraw.Draw(base)

    for _ in range(350):
        x = rng.randrange(size)
        y = rng.randrange(size)
        gray = rng.randrange(45, 120)
        draw.point((x, y), fill=(gray, gray, gray))

    for _ in range(rng.randrange(3, 8)):
        x0 = rng.randrange(size)
        y0 = rng.randrange(size)
        length = rng.randrange(size // 5, size)
        angle = rng.uniform(-0.6, 0.6)
        points = []
        for step in range(18):
            t = step / 17
            x = x0 + math.cos(angle) * length * t + rng.uniform(-4, 4)
            y = y0 + math.sin(angle) * length * t + rng.uniform(-4, 4)
            points.append((x, y))
        draw.line(points, fill=(25, 25, 24), width=rng.choice([1, 1, 2]))

    for _ in range(rng.randrange(1, 4)):
        x = rng.randrange(size // 8, size - size // 8)
        y = rng.randrange(size // 8, size - size // 8)
        rx = rng.randrange(8, 22)
        ry = rng.randrange(5, 16)
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=(30, 29, 28), width=2)

    if rng.random() < 0.7:
        x = rng.randrange(size // 4, size * 3 // 4)
        draw.line((x, 0, x + rng.randrange(-20, 20), size), fill=(205, 195, 130), width=2)

    return base.filter(ImageFilter.GaussianBlur(radius=0.25))


def motion_blur(image: Image.Image, horizontal: bool = True, length: int = 9) -> Image.Image:
    kernel = np.zeros((length, length), dtype=np.float32)
    if horizontal:
        kernel[length // 2, :] = 1.0 / length
    else:
        kernel[:, length // 2] = 1.0 / length
    arr = np.asarray(image).astype(np.float32)
    pad = length // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            patch = padded[y : y + length, x : x + length]
            out[y, x] = (patch * kernel[..., None]).sum(axis=(0, 1))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def degrade(clean: Image.Image, scenario: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    if "horizontal" in scenario:
        degraded = motion_blur(clean, horizontal=True, length=11)
    elif "vertical" in scenario:
        degraded = motion_blur(clean, horizontal=False, length=11)
    elif "defocus" in scenario:
        degraded = clean.filter(ImageFilter.GaussianBlur(radius=2.0))
    elif "lowlight" in scenario:
        arr = np.asarray(clean).astype(np.float32) * 0.45
        arr += rng.normal(0, 4, arr.shape)
        degraded = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    else:
        arr = np.asarray(clean.filter(ImageFilter.GaussianBlur(radius=1.2))).astype(np.float32)
        arr += rng.normal(0, 9, arr.shape)
        degraded = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return degraded


def main() -> None:
    args = parse_args()
    root = Path(args.out)
    for scenario in SCENARIOS:
        input_dir = root / "scenarios" / scenario / "input"
        gt_dir = root / "scenarios" / scenario / "gt"
        input_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        for index in range(args.count):
            clean = road_like_clean(args.size, seed=index)
            degraded = degrade(clean, scenario, seed=index + 1000)
            name = f"{index:05d}.png"
            clean.save(gt_dir / name)
            degraded.save(input_dir / name)
    print(root)


if __name__ == "__main__":
    main()

