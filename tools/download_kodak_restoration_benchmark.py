from __future__ import annotations

import argparse
import io
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim{index:02d}.png"

SCENARIOS = (
    "motion_horizontal_medium",
    "motion_vertical_medium",
    "defocus_medium",
    "gaussian_sigma3",
    "lowlight_medium",
    "jpeg40_motion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kodak images and synthesize benchmark-style restoration pairs.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--max-size", type=int, default=768)
    return parser.parse_args()


def download_image(index: int) -> Image.Image:
    url = KODAK_URL.format(index=index)
    with urllib.request.urlopen(url, timeout=60) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def resize_max(image: Image.Image, max_size: int) -> Image.Image:
    scale = min(max_size / max(image.size), 1.0)
    if scale >= 1.0:
        return image
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.BICUBIC)


def motion_blur(image: Image.Image, angle_deg: float, length: int = 15) -> Image.Image:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    angle = math.radians(angle_deg)
    for i in range(length):
        x = center + (i - center) * math.cos(angle)
        y = center + (i - center) * math.sin(angle)
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < length and 0 <= yi < length:
            kernel[yi, xi] = 1.0
    kernel /= kernel.sum()
    arr = np.asarray(image).astype(np.float32)
    pad = length // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            out[y, x] = (padded[y : y + length, x : x + length] * kernel[..., None]).sum(axis=(0, 1))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def degrade(clean: Image.Image, scenario: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    if scenario == "motion_horizontal_medium":
        return motion_blur(clean, 0, 15)
    if scenario == "motion_vertical_medium":
        return motion_blur(clean, 90, 15)
    if scenario == "defocus_medium":
        return clean.filter(ImageFilter.GaussianBlur(radius=2.0))
    if scenario == "gaussian_sigma3":
        arr = np.asarray(clean).astype(np.float32) + rng.normal(0, 3, (*clean.size[::-1], 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if scenario == "lowlight_medium":
        arr = np.asarray(clean).astype(np.float32) * 0.50 + rng.normal(0, 5, (*clean.size[::-1], 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if scenario == "jpeg40_motion":
        blurred = motion_blur(clean, 0, 11)
        buffer = io.BytesIO()
        blurred.save(buffer, format="JPEG", quality=40)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    raise ValueError(scenario)


def main() -> None:
    args = parse_args()
    root = Path(args.out)
    images = []
    for index in range(1, min(args.limit, 24) + 1):
        image = resize_max(download_image(index), args.max_size)
        images.append((f"kodim{index:02d}.png", image))

    for scenario in SCENARIOS:
        input_dir = root / "scenarios" / scenario / "input"
        gt_dir = root / "scenarios" / scenario / "gt"
        input_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        for index, (name, clean) in enumerate(images):
            clean.save(gt_dir / name)
            degrade(clean, scenario, seed=index).save(input_dir / name)
    print(root)


if __name__ == "__main__":
    main()

