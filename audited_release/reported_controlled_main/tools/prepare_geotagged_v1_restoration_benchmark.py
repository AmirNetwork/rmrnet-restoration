# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

"""Prepare Roboflow v1 geotagged road-defect data for restoration evaluation.

This script treats the Roboflow export as annotation metadata and maps it back
to the original native-resolution geotagged Cam1 images.  It writes:

1. YOLO detection folders for sharp and degraded images.
2. Native annotation JSON files with scaled boxes/polygons.
3. Restoration folders with scenarios/<scenario>/input, gt, metadata.

The Roboflow export should not be used as training data for this experiment;
the output is a test-only benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcadnet.synthetic_metadata import synthetic_metadata_from_scenario


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

NEWROAD_NAMES = {
    0: "alligator_crack",
    1: "longitudinal_crack",
    2: "others",
    3: "pothole",
    4: "road_intersection",
    5: "transverse_crack",
}

COCO_TO_NEWROAD = {
    "Alligator Crack": 0,
    "Longitudinal Crack": 1,
    "Potholes": 3,
    "Transverse Crack": 5,
}

DEFAULT_SCENARIOS = [
    "motion_horizontal_medium",
    "defocus_medium",
    "lowlight_medium",
    "mixed_motion_lowlight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco-json",
        default=(
            "experiments/roboflow_geotagged_v1_restoration_benchmark/"
            "roboflow_export/road-defect-seg-9junedata-1/train/_annotations.coco.json"
        ),
    )
    parser.add_argument("--geotagged-root", default="geotagged")
    parser.add_argument("--out", default="experiments/roboflow_geotagged_v1_restoration_benchmark")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario", action="append", dest="scenarios", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-existing-images", action="store_true")
    return parser.parse_args()


def original_name_from_coco(image_record: dict[str, Any]) -> str:
    extra = image_record.get("extra") or {}
    if isinstance(extra, dict) and extra.get("name"):
        return str(extra["name"])
    name = str(image_record["file_name"])
    marker = "_jpg.rf."
    if marker in name:
        return name.split(marker, 1)[0] + ".jpg"
    return Path(name).name


def build_native_index(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.parent.name.lower() != "q":
            index[path.name.lower()] = path
            index[path.stem.lower()] = path
    return index


def read_pose_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("camera_name") or ""
        if key:
            out[key.lower()] = row
            out[Path(key).stem.lower()] = row
    return out


def float_or_zero(value: Any) -> float:
    try:
        if value in {"", None}:  # type: ignore[comparison-overlap]
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def telemetry_from_pose(row: dict[str, str]) -> dict[str, float]:
    std_e = float_or_zero(row.get("std_east"))
    std_n = float_or_zero(row.get("std_north"))
    std_h = float_or_zero(row.get("std_ht"))
    std_roll = float_or_zero(row.get("std_roll"))
    std_pitch = float_or_zero(row.get("std_pitch"))
    std_yaw = float_or_zero(row.get("std_yaw"))
    pose_uncertainty = math.sqrt(std_e * std_e + std_n * std_n + std_h * std_h)
    angular_uncertainty = math.sqrt(std_roll * std_roll + std_pitch * std_pitch + std_yaw * std_yaw)
    return {
        "gyro_x": min(std_pitch / 0.25, 1.0),
        "gyro_y": min(std_roll / 0.25, 1.0),
        "accel_norm": min((pose_uncertainty * 10.0 + angular_uncertainty) / 5.0, 1.0),
        "speed_mps": 0.0,
    }


def motion_blur(image: Image.Image, angle_deg: float, length: int) -> Image.Image:
    length = max(3, int(length) | 1)
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


def add_sensor_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(image).astype(np.float32)
    arr += rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def degrade(image: Image.Image, scenario: str, seed: int) -> Image.Image:
    name = scenario.lower()
    out = image
    if "motion" in name:
        if "vertical" in name:
            out = motion_blur(out, 90.0, 21)
        elif "diagonal" in name:
            out = motion_blur(out, 35.0, 21)
        else:
            out = motion_blur(out, 0.0, 21)
    if "defocus" in name:
        out = out.filter(ImageFilter.GaussianBlur(radius=2.4))
    if "lowlight" in name or "low_light" in name:
        out = ImageEnhance.Brightness(out).enhance(0.45)
        out = ImageEnhance.Contrast(out).enhance(0.82)
        out = add_sensor_noise(out, sigma=5.0, seed=seed)
    if "jpeg40" in name:
        # Handled by save quality, but keep branch for metadata clarity.
        pass
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
    x1 = max(0.0, min(x, width - 1.0))
    y1 = max(0.0, min(y, height - 1.0))
    x2 = max(0.0, min(x + w, width))
    y2 = max(0.0, min(y + h, height))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    return ((x1 + bw / 2.0) / width, (y1 + bh / 2.0) / height, bw / width, bh / height)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def save_image(image: Image.Image, path: Path, quality: int, skip_existing: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and path.exists():
        return
    image.save(path, quality=quality, subsampling=0)


def write_yolo_yaml(root: Path, split: str) -> None:
    config = {
        "path": str(root.resolve()).replace("\\", "/"),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "nc": len(NEWROAD_NAMES),
        "names": NEWROAD_NAMES,
    }
    (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    coco_json = Path(args.coco_json)
    geotagged_root = Path(args.geotagged_root)
    out_root = Path(args.out)
    yolo_root = out_root / "native_yolo_newroad6"
    restoration_root = out_root / "restoration"
    split = args.split
    scenarios = args.scenarios or DEFAULT_SCENARIOS

    data = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        annotations_by_image[int(ann["image_id"])].append(ann)

    native_index = build_native_index(geotagged_root / "cam1")
    pose_by_name = read_pose_csv(geotagged_root / "precise_cam1_coords.csv")

    selected_records = []
    skipped_missing = []
    for image_record in data.get("images", []):
        original_name = original_name_from_coco(image_record)
        native_path = native_index.get(original_name.lower()) or native_index.get(Path(original_name).stem.lower())
        if native_path is None:
            skipped_missing.append(original_name)
            continue
        selected_records.append((image_record, original_name, native_path))
        if args.limit and len(selected_records) >= args.limit:
            break

    for condition in ["sharp", *scenarios]:
        root = yolo_root / condition
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "metadata" / split).mkdir(parents=True, exist_ok=True)
        (root / "native_annotations").mkdir(parents=True, exist_ok=True)
        write_yolo_yaml(root, split)

    for scenario in scenarios:
        base = restoration_root / "scenarios" / scenario
        (base / "input").mkdir(parents=True, exist_ok=True)
        (base / "gt").mkdir(parents=True, exist_ok=True)
        (base / "metadata").mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()

    for index, (image_record, original_name, native_path) in enumerate(selected_records):
        with Image.open(native_path) as pil:
            native = pil.convert("RGB")
            native_w, native_h = native.size

        coco_w = int(image_record["width"])
        coco_h = int(image_record["height"])
        sx = native_w / max(coco_w, 1)
        sy = native_h / max(coco_h, 1)
        pose_row = pose_by_name.get(original_name.lower()) or pose_by_name.get(Path(original_name).stem.lower()) or {}
        telemetry = telemetry_from_pose(pose_row)

        yolo_lines: list[str] = []
        native_anns: list[dict[str, Any]] = []
        for ann in annotations_by_image[int(image_record["id"])]:
            category_name = categories[int(ann["category_id"])]
            if category_name not in COCO_TO_NEWROAD:
                continue
            cls = COCO_TO_NEWROAD[category_name]
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
            class_counts[NEWROAD_NAMES[cls]] += 1

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
                    "yolo_class_id": cls,
                    "yolo_class": NEWROAD_NAMES[cls],
                    "bbox_xywh_native": [x, y, w, h],
                    "segmentation_native": scaled_segmentation,
                }
            )

        # Sharp/native YOLO condition.
        sharp_root = yolo_root / "sharp"
        link_or_copy(native_path, sharp_root / "images" / split / original_name)
        for condition in ["sharp", *scenarios]:
            root = yolo_root / condition
            (root / "labels" / split / f"{Path(original_name).stem}.txt").write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
            )
            (root / "native_annotations" / f"{Path(original_name).stem}.json").write_text(
                json.dumps(native_anns, indent=2), encoding="utf-8"
            )

        for scenario in scenarios:
            scenario_meta = synthetic_metadata_from_scenario(scenario, seed=index)
            scenario_meta.update(telemetry)
            scenario_meta.update(
                {
                    "scenario": scenario,
                    "original_name": original_name,
                    "coco_file_name": image_record["file_name"],
                    "coco_size": [coco_w, coco_h],
                    "native_size": [native_w, native_h],
                    "scale_x": sx,
                    "scale_y": sy,
                    "pose_csv": pose_row,
                    "metadata_source": "synthetic_degradation_plus_geotagged_pose",
                }
            )
            degraded = degrade(native, scenario, seed=index)
            degraded_yolo_path = yolo_root / scenario / "images" / split / original_name
            save_image(degraded, degraded_yolo_path, quality=args.jpeg_quality, skip_existing=args.skip_existing_images)
            (yolo_root / scenario / "metadata" / split / f"{Path(original_name).stem}.json").write_text(
                json.dumps(scenario_meta, indent=2), encoding="utf-8"
            )

            rest_base = restoration_root / "scenarios" / scenario
            link_or_copy(degraded_yolo_path, rest_base / "input" / original_name)
            link_or_copy(native_path, rest_base / "gt" / original_name)
            (rest_base / "metadata" / f"{Path(original_name).stem}.json").write_text(
                json.dumps(scenario_meta, indent=2), encoding="utf-8"
            )

        # Sharp metadata is real/native only.
        sharp_meta = {
            "scenario": "sharp",
            "original_name": original_name,
            "coco_file_name": image_record["file_name"],
            "coco_size": [coco_w, coco_h],
            "native_size": [native_w, native_h],
            "scale_x": sx,
            "scale_y": sy,
            "pose_csv": pose_row,
            "metadata_source": "geotagged_pose_only",
            **telemetry,
        }
        (sharp_root / "metadata" / split / f"{Path(original_name).stem}.json").write_text(
            json.dumps(sharp_meta, indent=2), encoding="utf-8"
        )

        manifest.append(
            {
                "image": original_name,
                "native_path": str(native_path),
                "coco_file_name": image_record["file_name"],
                "coco_width": coco_w,
                "coco_height": coco_h,
                "native_width": native_w,
                "native_height": native_h,
                "scale_x": sx,
                "scale_y": sy,
                "annotations": len(yolo_lines),
                "pose_matched": bool(pose_row),
            }
        )
        if (index + 1) % 50 == 0:
            print(json.dumps({"prepared": index + 1, "total": len(selected_records)}), flush=True)

    summary = {
        "roboflow_images": len(data.get("images", [])),
        "matched_native_images": len(selected_records),
        "skipped_missing_native_images": len(skipped_missing),
        "annotations": sum(row["annotations"] for row in manifest),
        "class_counts": dict(class_counts),
        "scenarios": scenarios,
        "sharp_data": str((yolo_root / "sharp" / "data.yaml").resolve()),
        "restoration_root": str(restoration_root.resolve()),
        "resolution_policy": "All reported images are original native geotagged resolution; COCO coordinates are scaled only if needed.",
        "class_mapping": COCO_TO_NEWROAD,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "skipped_missing_native_images.json").write_text(json.dumps(skipped_missing, indent=2), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
