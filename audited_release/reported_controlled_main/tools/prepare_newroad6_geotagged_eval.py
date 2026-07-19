# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

"""Prepare geotagged pilot labels for the New Road Crack YOLOv8 taxonomy.

The native geotagged pilot was first exported with the paper's collapsed
classes (pothole/crack/manhole).  The newly trained YOLOv8 detector uses the
six-class Roboflow taxonomy from "NEW Road Crack detection.v1i.yolov8":

0 alligator_crack
1 longitudinal_crack
2 others
3 pothole
4 road_intersection
5 transverse_crack

This script reuses the already matched native-resolution images and scaled
annotation JSON files, and writes class-compatible YOLO labels without resizing
or modifying the source images.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


NAMES = [
    "alligator_crack",
    "longitudinal_crack",
    "others",
    "pothole",
    "road_intersection",
    "transverse_crack",
]

SOURCE_CATEGORY_TO_ID = {
    "alligator crack": 0,
    "longitudinal crack": 1,
    "others": 2,
    "potholes": 3,
    "pothole": 3,
    "road_intersection": 4,
    "road intersection": 4,
    "transverse crack": 5,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def convert_condition(src_condition: Path, dst_condition: Path) -> dict:
    src_images = src_condition / "images" / "test"
    src_ann = src_condition / "native_annotations"
    dst_images = dst_condition / "images" / "test"
    dst_labels = dst_condition / "labels" / "test"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    skipped_unknown = 0
    class_counts = {name: 0 for name in NAMES}

    for image_path in sorted(src_images.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image_count += 1
        out_image = dst_images / image_path.name
        if not out_image.exists():
            shutil.copy2(image_path, out_image)

        with Image.open(image_path) as im:
            width, height = im.size

        ann_path = src_ann / f"{image_path.stem}.json"
        annotations = json.loads(ann_path.read_text()) if ann_path.exists() else []
        lines: list[str] = []

        for ann in annotations:
            source_category = str(ann.get("source_category", "")).strip().lower()
            cls_id = SOURCE_CATEGORY_TO_ID.get(source_category)
            if cls_id is None:
                skipped_unknown += 1
                continue

            x, y, w, h = [float(v) for v in ann["bbox_xywh_native"]]
            x1 = clamp(x, 0.0, width)
            y1 = clamp(y, 0.0, height)
            x2 = clamp(x + w, 0.0, width)
            y2 = clamp(y + h, 0.0, height)
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 1.0 or bh <= 1.0:
                continue

            xc = (x1 + x2) * 0.5 / width
            yc = (y1 + y2) * 0.5 / height
            nw = bw / width
            nh = bh / height
            lines.append(f"{cls_id} {xc:.8f} {yc:.8f} {nw:.8f} {nh:.8f}")
            label_count += 1
            class_counts[NAMES[cls_id]] += 1

        (dst_labels / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    data_yaml = dst_condition / "data.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(NAMES))
    data_yaml.write_text(
        f"path: {dst_condition.resolve().as_posix()}\n"
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n"
        f"nc: {len(NAMES)}\n"
        "names:\n"
        f"{names_block}\n"
    )

    return {
        "condition": src_condition.name,
        "images": image_count,
        "labels": label_count,
        "skipped_unknown": skipped_unknown,
        "class_counts": class_counts,
        "data_yaml": str(data_yaml),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="experiments/roboflow_geotagged_native_annotation_pilot/native_yolo",
        help="Existing native geotagged pilot root.",
    )
    parser.add_argument(
        "--out-root",
        default="experiments/roboflow_geotagged_native_annotation_pilot/native_yolo_newroad6",
        help="Output root for six-class New Road Crack labels.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["sharp", "motion_horizontal_medium"],
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for condition in args.conditions:
        summaries.append(convert_condition(source_root / condition, out_root / condition))

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(summary_path)
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
