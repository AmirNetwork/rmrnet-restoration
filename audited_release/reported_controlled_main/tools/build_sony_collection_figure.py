#!/usr/bin/env python3
"""Compose the Sony collection-platform and trajectory figure.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

The source photographs are retained without generative editing. This script
only performs deterministic cropping, resizing, labeling, and layout.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_automation_in_construction_rmrnet"
SOURCE = PAPER / "source_assets"
OUTPUT = PAPER / "figures" / "fig_sony_collection_system.png"


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def cover(image: Image.Image, size: tuple[int, int], anchor_x: float = 0.5) -> Image.Image:
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = round((resized.width - target_w) * anchor_x)
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def main() -> None:
    platform = Image.open(SOURCE / "sony_sensor_suite.jpg").convert("RGB")
    trajectory = Image.open(SOURCE / "sony_cam1_trajectory.png").convert("RGB")
    panel_h = 630
    platform_w, trajectory_w = 470, 970
    gap, margin, header = 18, 24, 54
    page = Image.new("RGB", (2 * margin + platform_w + gap + trajectory_w, 2 * margin + header + panel_h), "white")
    draw = ImageDraw.Draw(page)
    panels = [
        ("(a) Collection platform", cover(platform, (platform_w, panel_h), anchor_x=0.48), margin),
        ("(b) Camera trajectory and positioning audit", cover(trajectory, (trajectory_w, panel_h)), margin + platform_w + gap),
    ]
    for title, panel, x in panels:
        draw.text((x, margin), title, fill=(25, 40, 50), font=font(24, True))
        y = margin + header
        page.paste(panel, (x, y))
        draw.rectangle((x, y, x + panel.width - 1, y + panel.height - 1), outline=(176, 188, 196), width=2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    page.save(OUTPUT, dpi=(320, 320), quality=96)
    print(OUTPUT)


if __name__ == "__main__":
    main()
