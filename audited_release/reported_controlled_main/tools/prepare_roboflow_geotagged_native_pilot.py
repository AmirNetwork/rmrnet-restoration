# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageFilter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

YOLO_NAMES = {0: "pothole", 1: "crack", 2: "manhole"}
COCO_TO_YOLO = {
    "Potholes": 0,
    "Alligator Crack": 1,
    "Longitudinal Crack": 1,
    "Transverse Crack": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map a Roboflow COCO-seg export back to original geotagged cam1 images, "
            "rescale annotations if Roboflow resized them, and create sharp/degraded YOLO datasets."
        )
    )
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--geotagged-root", default="geotagged")
    parser.add_argument("--out", default="experiments/roboflow_geotagged_native_annotation_pilot/native_yolo")
    parser.add_argument("--scenario", default="motion_horizontal_medium")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def original_name_from_coco(image_record: dict[str, Any]) -> str:
    extra = image_record.get("extra") or {}
    if extra.get("name"):
        return str(extra["name"])
    name = str(image_record["file_name"])
    marker = "_jpg.rf."
    if marker in name:
        return name.split(marker, 1)[0] + ".jpg"
    return name


def motion_blur(image: Image.Image, angle_deg: float, length: int) -> Image.Image:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    angle = math.radians(angle_deg)
    for i in range(length):
        x = int(round(center + (i - center) * math.cos(angle)))
        y = int(round(center + (i - center) * math.sin(angle)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    kernel /= max(float(kernel.sum()), 1e-6)
    out = cv2.filter2D(np.asarray(image), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def degrade(image: Image.Image, scenario: str) -> Image.Image:
    name = scenario.lower()
    if "motion" in name:
        if "vertical" in name:
            return motion_blur(image, 90, 21)
        if "diagonal" in name:
            return motion_blur(image, 35, 21)
        return motion_blur(image, 0, 21)
    if "defocus" in name:
        return image.filter(ImageFilter.GaussianBlur(radius=2.4))
    raise ValueError(f"Unsupported pilot scenario: {scenario}")


def read_pose_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("camera_name") or ""
        if key:
            out[key] = row
            out[Path(key).stem] = row
    return out


def polygon_bounds(segmentation: Any) -> list[float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for poly in segmentation or []:
        if not isinstance(poly, list):
            continue
        coords = [float(v) for v in poly]
        xs.extend(coords[0::2])
        ys.extend(coords[1::2])
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def normalize_box(x: float, y: float, w: float, h: float, width: int, height: int) -> tuple[float, float, float, float]:
    x = max(0.0, min(x, width - 1.0))
    y = max(0.0, min(y, height - 1.0))
    w = max(1.0, min(w, width - x))
    h = max(1.0, min(h, height - y))
    return ((x + w / 2.0) / width, (y + h / 2.0) / height, w / width, h / height)


def main() -> None:
    args = parse_args()
    coco_json = Path(args.coco_json)
    geotagged_root = Path(args.geotagged_root)
    image_root = geotagged_root / "cam1"
    pose_by_name = read_pose_csv(geotagged_root / "precise_cam1_coords.csv")
    out_root = Path(args.out)
    sharp_root = out_root / "sharp"
    degraded_root = out_root / args.scenario
    split = args.split

    data = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
    images = data.get("images", [])
    if args.limit:
        images = images[: args.limit]
    selected_ids = {int(img["id"]) for img in images}

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if int(ann["image_id"]) in selected_ids:
            annotations_by_image[int(ann["image_id"])].append(ann)

    for root in (sharp_root, degraded_root):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "metadata" / split).mkdir(parents=True, exist_ok=True)
        (root / "native_annotations").mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for image_record in images:
        image_id = int(image_record["id"])
        original_name = original_name_from_coco(image_record)
        original_path = image_root / original_name
        if not original_path.exists():
            raise FileNotFoundError(f"Cannot find original geotagged image for {original_name}: {original_path}")

        with Image.open(original_path) as pil:
            native = pil.convert("RGB")
            native_w, native_h = native.size
            degraded = degrade(native, args.scenario)

        coco_w = int(image_record["width"])
        coco_h = int(image_record["height"])
        sx = native_w / max(coco_w, 1)
        sy = native_h / max(coco_h, 1)

        shutil.copy2(original_path, sharp_root / "images" / split / original_name)
        degraded.save(degraded_root / "images" / split / original_name, quality=95)

        yolo_lines: list[str] = []
        native_anns: list[dict[str, Any]] = []
        for ann in annotations_by_image[image_id]:
            category_name = categories[int(ann["category_id"])]
            if category_name not in COCO_TO_YOLO:
                continue
            cls = COCO_TO_YOLO[category_name]
            bbox = ann.get("bbox") or polygon_bounds(ann.get("segmentation"))
            if not bbox:
                continue
            x, y, w, h = [float(v) for v in bbox]
            x *= sx
            y *= sy
            w *= sx
            h *= sy
            xc, yc, wn, hn = normalize_box(x, y, w, h, native_w, native_h)
            yolo_lines.append(f"{cls} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}")

            scaled_segmentation = []
            for poly in ann.get("segmentation") or []:
                coords = [float(v) for v in poly]
                scaled = []
                for px, py in zip(coords[0::2], coords[1::2]):
                    scaled.extend([px * sx, py * sy])
                scaled_segmentation.append(scaled)
            native_anns.append(
                {
                    "source_category": category_name,
                    "yolo_class": YOLO_NAMES[cls],
                    "bbox_xywh_native": [x, y, w, h],
                    "segmentation_native": scaled_segmentation,
                }
            )

        for root in (sharp_root, degraded_root):
            (root / "labels" / split / f"{Path(original_name).stem}.txt").write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
            )
            metadata = {
                "original_name": original_name,
                "coco_file_name": image_record["file_name"],
                "coco_size": [coco_w, coco_h],
                "native_size": [native_w, native_h],
                "scale_x": sx,
                "scale_y": sy,
                "pose_csv": pose_by_name.get(original_name) or pose_by_name.get(Path(original_name).stem) or {},
            }
            (root / "metadata" / split / f"{Path(original_name).stem}.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            (root / "native_annotations" / f"{Path(original_name).stem}.json").write_text(
                json.dumps(native_anns, indent=2), encoding="utf-8"
            )

        manifest.append(
            {
                "image": original_name,
                "coco_width": coco_w,
                "coco_height": coco_h,
                "native_width": native_w,
                "native_height": native_h,
                "scale_x": sx,
                "scale_y": sy,
                "annotations": len(yolo_lines),
                "pose_matched": bool(pose_by_name.get(original_name) or pose_by_name.get(Path(original_name).stem)),
            }
        )

    names = {i: name for i, name in YOLO_NAMES.items()}
    for root in (sharp_root, degraded_root):
        config = {
            "path": str(root.resolve()).replace("\\", "/"),
            "train": f"images/{split}",
            "val": f"images/{split}",
            "test": f"images/{split}",
            "names": names,
        }
        (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    summary = {
        "images": len(manifest),
        "annotations": sum(int(row["annotations"]) for row in manifest),
        "pose_matched": sum(1 for row in manifest if row["pose_matched"]),
        "scenario": args.scenario,
        "sharp_data": str((sharp_root / "data.yaml").resolve()),
        "degraded_data": str((degraded_root / "data.yaml").resolve()),
        "class_mapping": COCO_TO_YOLO,
        "resolution_policy": "Roboflow COCO coordinates are scaled to original geotagged image dimensions before YOLO labels are written.",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
