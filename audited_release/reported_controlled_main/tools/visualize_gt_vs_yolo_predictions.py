# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw GT boxes and YOLO predictions for visual QA.")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--panel-width", type=int, default=1400)
    parser.add_argument("--device", default="0")
    parser.add_argument("--title", default="")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_yolo_labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = [float(v) for v in parts[1:5]]
        x1 = (xc - w / 2.0) * width
        y1 = (yc - h / 2.0) * height
        x2 = (xc + w / 2.0) * width
        y2 = (yc + h / 2.0) * height
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def label_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    width: int = 5,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    text_bbox = draw.textbbox((x1, y1), text, font=font)
    pad = 4
    bg = [text_bbox[0], max(0, text_bbox[1] - pad), text_bbox[2] + pad * 2, text_bbox[3] + pad]
    draw.rectangle(bg, fill=color)
    draw.text((x1 + pad, max(0, y1 - pad)), text, fill=(0, 0, 0), font=font)


def fit_panel(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    height = int(round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def make_overlay(
    image_path: Path,
    label_path: Path,
    names: dict[int, str],
    model: YOLO,
    imgsz: int,
    conf: float,
    device: str,
    panel_width: int,
    title: str,
) -> tuple[Image.Image, dict[str, int]]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    gt = read_yolo_labels(label_path, width, height)
    result = model.predict(str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    font = load_font(max(20, width // 160))
    pred_count = 0
    if result.boxes is not None:
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confs = result.boxes.conf.detach().cpu().numpy()
        for box, cls, score in zip(boxes, classes, confs):
            pred_count += 1
            label_box(
                draw,
                tuple(float(v) for v in box),
                f"P {names.get(int(cls), str(cls))} {float(score):.2f}",
                (32, 220, 120),
                font,
                width=max(4, width // 900),
            )
    for cls, x1, y1, x2, y2 in gt:
        label_box(
            draw,
            (x1, y1, x2, y2),
            f"GT {names.get(int(cls), str(cls))}",
            (255, 215, 0),
            font,
            width=max(5, width // 800),
        )

    panel = fit_panel(overlay, panel_width)
    header_h = 66
    header = Image.new("RGB", (panel.width, header_h), (250, 250, 250))
    header_draw = ImageDraw.Draw(header)
    header_font = load_font(28)
    header_text = f"{title} | {image_path.name} | GT={len(gt)} Pred={pred_count}"
    header_draw.text((18, 16), header_text, fill=(0, 0, 0), font=header_font)
    framed = Image.new("RGB", (panel.width, header_h + panel.height), (255, 255, 255))
    framed.paste(header, (0, 0))
    framed.paste(panel, (0, header_h))
    return framed, {"gt": len(gt), "pred": pred_count}


def make_atlas(panels: list[Image.Image], columns: int = 1, gap: int = 22) -> Image.Image:
    if not panels:
        raise ValueError("No panels to combine")
    columns = max(1, columns)
    rows = (len(panels) + columns - 1) // columns
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    atlas = Image.new("RGB", (columns * cell_w + (columns - 1) * gap, rows * cell_h + (rows - 1) * gap), (255, 255, 255))
    for index, panel in enumerate(panels):
        row = index // columns
        col = index % columns
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        atlas.paste(panel, (x, y))
    return atlas


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root = Path(config["path"])
    image_dir = root / config[args.split]
    label_dir = root / config[args.split].replace("images", "labels")
    names = {int(k): str(v) for k, v in dict(config["names"]).items()}
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    image_paths = image_paths[: args.max_images]

    out = Path(args.out)
    per_image_dir = out / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    panels: list[Image.Image] = []
    rows = []
    for image_path in image_paths:
        panel, counts = make_overlay(
            image_path,
            label_dir / f"{image_path.stem}.txt",
            names,
            model,
            args.imgsz,
            args.conf,
            args.device,
            args.panel_width,
            args.title or root.name,
        )
        panel.save(per_image_dir / f"{image_path.stem}_gt_pred.png")
        panels.append(panel)
        rows.append({"image": image_path.name, **counts})

    atlas = make_atlas(panels, columns=1)
    atlas_path = out / "gt_vs_predictions_atlas.png"
    atlas.save(atlas_path)
    (out / "counts.csv").write_text(
        "image,gt,pred\n" + "\n".join(f"{row['image']},{row['gt']},{row['pred']}" for row in rows) + "\n",
        encoding="utf-8",
    )
    print(atlas_path.resolve())


if __name__ == "__main__":
    main()
