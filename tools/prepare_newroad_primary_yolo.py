from __future__ import annotations

"""Prepare a crack-vs-pothole detector dataset from New Road Crack.

GT49 evaluation should prioritize defect localization before subtype naming.
This script remaps the external six-class New Road Crack dataset into a
two-class detector training set:

    alligator_crack, longitudinal_crack, transverse_crack -> crack
    pothole -> pothole
    others, road_intersection -> ignored

The GT49 test labels are never read by this script.
"""

import argparse
import shutil
from pathlib import Path

import yaml


CRACK_CLASSES = {0, 1, 5}
POTHOLE_CLASS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("datasets/new_road_crack_detection_v1i_yolov8"))
    parser.add_argument("--out", type=Path, default=Path("datasets/new_road_crack_detection_v1i_yolov8_primary"))
    return parser.parse_args()


def convert_label(src: Path, dst: Path) -> int:
    rows = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls in CRACK_CLASSES:
                parts[0] = "0"
            elif cls == POTHOLE_CLASS:
                parts[0] = "1"
            else:
                continue
            rows.append(" ".join(parts[:5]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(args.src)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split in ["train", "valid", "test"]:
        src_images = args.src / split / "images"
        src_labels = args.src / split / "labels"
        out_images = args.out / split / "images"
        out_labels = args.out / split / "labels"
        kept_images = 0
        kept_boxes = 0
        for image in sorted(src_images.glob("*")):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            label = src_labels / f"{image.stem}.txt"
            out_label = out_labels / f"{image.stem}.txt"
            n_boxes = convert_label(label, out_label)
            if n_boxes <= 0:
                out_label.unlink(missing_ok=True)
                continue
            copy_or_link(image, out_images / image.name)
            kept_images += 1
            kept_boxes += n_boxes
        manifest.append({"split": split, "images": kept_images, "boxes": kept_boxes})
    data = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": 2,
        "names": ["crack", "pothole"],
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (args.out / "primary_manifest.yaml").write_text(yaml.safe_dump({"splits": manifest}, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({"out": str(args.out), "splits": manifest}, sort_keys=False))


if __name__ == "__main__":
    main()
