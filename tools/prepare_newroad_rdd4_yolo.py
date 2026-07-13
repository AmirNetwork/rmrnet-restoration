from __future__ import annotations

"""Convert the external New Road Crack split into YOLO26/RDD four-class order.

Source Roboflow classes:
    0 alligator_crack
    1 longitudinal_crack
    2 others                  (ignored)
    3 pothole
    4 road_intersection       (ignored)
    5 transverse_crack

Target YOLO26/RDD order used by TamAko783/YOLO26s_RDD_FRDC_Distilled_v2:
    0 Longitudinal Crack (D00)
    1 Transverse Crack (D10)
    2 Alligator Crack (D20)
    3 Pothole (D40)

This keeps detector fine-tuning aligned with the pretrained head rather than
silently swapping alligator/longitudinal/transverse labels.
"""

import argparse
import shutil
from pathlib import Path

import yaml


CLASS_MAP = {
    0: 2,  # alligator -> D20
    1: 0,  # longitudinal -> D00
    3: 3,  # pothole -> D40
    5: 1,  # transverse -> D10
}

TARGET_NAMES = [
    "Longitudinal Crack (D00)",
    "Transverse Crack (D10)",
    "Alligator Crack (D20)",
    "Pothole (D40)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path("datasets/new_road_crack_detection_v1i_yolov8"))
    parser.add_argument("--out", type=Path, default=Path("datasets/new_road_crack_detection_v1i_yolov8_rdd4"))
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def convert_label(src: Path, dst: Path) -> int:
    rows: list[str] = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_cls = int(float(parts[0]))
            if old_cls not in CLASS_MAP:
                continue
            parts[0] = str(CLASS_MAP[old_cls])
            rows.append(" ".join(parts[:5]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def convert_split(src_root: Path, out_root: Path, split: str) -> dict[str, int]:
    src_images = src_root / split / "images"
    src_labels = src_root / split / "labels"
    out_images = out_root / split / "images"
    out_labels = out_root / split / "labels"
    n_images = 0
    n_boxes = 0
    for image in sorted(src_images.glob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        link_or_copy(image, out_images / image.name)
        n_boxes += convert_label(src_labels / f"{image.stem}.txt", out_labels / f"{image.stem}.txt")
        n_images += 1
    return {"images": n_images, "boxes": n_boxes}


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {split: convert_split(args.src, args.out, split) for split in ["train", "valid", "test"]}
    data = {
        "path": str(args.out.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 4,
        "names": TARGET_NAMES,
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({"out": str(args.out), "stats": stats, "names": TARGET_NAMES}, sort_keys=False))


if __name__ == "__main__":
    main()
