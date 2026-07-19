# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile a native-resolution YOLO dataset without changing pixel scale.")
    parser.add_argument("--data", required=True, help="Source YOLO data.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--overlap", type=int, default=160)
    parser.add_argument("--min-intersection", type=float, default=0.30)
    return parser.parse_args()


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def read_labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    out = []
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
        out.append((cls, x1, y1, x2, y2))
    return out


def clip_box(
    box: tuple[int, float, float, float, float],
    tx: int,
    ty: int,
    tw: int,
    th: int,
    min_intersection: float,
) -> str | None:
    cls, x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0:
        return None
    ix1 = max(x1, tx)
    iy1 = max(y1, ty)
    ix2 = min(x2, tx + tw)
    iy2 = min(y2, ty + th)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter / area < min_intersection:
        return None
    ix1 -= tx
    ix2 -= tx
    iy1 -= ty
    iy2 -= ty
    bw = max(1.0, ix2 - ix1)
    bh = max(1.0, iy2 - iy1)
    xc = (ix1 + bw / 2.0) / tw
    yc = (iy1 + bh / 2.0) / th
    return f"{cls} {xc:.8f} {yc:.8f} {bw / tw:.8f} {bh / th:.8f}"


def main() -> None:
    args = parse_args()
    source_yaml = Path(args.data)
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source_root = Path(config["path"])
    image_dir = source_root / config[args.split]
    label_dir = source_root / config[args.split].replace("images", "labels")
    out_root = Path(args.out)
    out_image_dir = out_root / "images" / args.split
    out_label_dir = out_root / "labels" / args.split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    tile_count = 0
    kept_box_count = 0
    source_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for image_path in source_images:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            labels = read_labels(label_dir / f"{image_path.stem}.txt", width, height)
            for ty in tile_starts(height, args.tile, args.overlap):
                for tx in tile_starts(width, args.tile, args.overlap):
                    tw = min(args.tile, width - tx)
                    th = min(args.tile, height - ty)
                    tile_labels = [
                        clipped
                        for box in labels
                        if (clipped := clip_box(box, tx, ty, tw, th, args.min_intersection)) is not None
                    ]
                    if not tile_labels:
                        continue
                    tile_name = f"{image_path.stem}__x{tx}_y{ty}_w{tw}_h{th}{image_path.suffix}"
                    crop = image.crop((tx, ty, tx + tw, ty + th))
                    crop.save(out_image_dir / tile_name, quality=95)
                    (out_label_dir / f"{Path(tile_name).stem}.txt").write_text(
                        "\n".join(tile_labels) + "\n", encoding="utf-8"
                    )
                    tile_count += 1
                    kept_box_count += len(tile_labels)
                    rows.append(
                        {
                            "tile": tile_name,
                            "source": image_path.name,
                            "x": tx,
                            "y": ty,
                            "width": tw,
                            "height": th,
                            "labels": len(tile_labels),
                        }
                    )

    data_yaml = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": f"images/{args.split}",
        "val": f"images/{args.split}",
        "test": f"images/{args.split}",
        "names": config["names"],
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (out_root / "tile_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = {
        "source_images": len(source_images),
        "tiles_with_labels": tile_count,
        "labels_in_tiles": kept_box_count,
        "tile": args.tile,
        "overlap": args.overlap,
        "min_intersection": args.min_intersection,
        "data": str((out_root / "data.yaml").resolve()),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
