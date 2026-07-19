# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import ExifTags, Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

NEWROAD6_NAMES = {
    0: "alligator_crack",
    1: "longitudinal_crack",
    2: "others",
    3: "pothole",
    4: "road_intersection",
    5: "transverse_crack",
}

COCO_TO_NEWROAD6 = {
    "Alligator Crack": 0,
    "Longitudinal Crack": 1,
    "Potholes": 3,
    "Transverse Crack": 5,
    # The current detector has no dedicated heads for these surface-defect
    # subclasses. Map them to the trained "others" class and report relaxed
    # agnostic metrics alongside class-aware metrics.
    "dirt": 2,
    "manhole": 2,
    "patch": 2,
    "rutting": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Roboflow v5 COCO annotations as a native-resolution real "
            "geotagged YOLO test split. The pixel data is always read from "
            "geotagged/cam1, not from the Roboflow export."
        )
    )
    parser.add_argument("--coco-json", required=True, type=Path)
    parser.add_argument("--geotagged-root", type=Path, default=Path("geotagged"))
    parser.add_argument("--out", type=Path, default=Path("experiments/roboflow_geotagged_v5_native_real"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="0 keeps all annotated images.")
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


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_pose_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name: dict[str, dict[str, str]] = {}
    for idx, row in enumerate(rows):
        row["_row_index"] = str(idx)
        name = row.get("camera_name") or ""
        if name:
            by_name[name] = row
            by_name[Path(name).stem] = row
    return by_name


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:  # type: ignore[comparison-overlap]
            return default
        return float(value)
    except Exception:
        return default


def capture_index(name: str) -> int:
    match = re.search(r"_capt(\d+)_", name)
    return int(match.group(1)) if match else -1


def capture_time(name: str) -> datetime | None:
    match = re.search(r"Cam1_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", name)
    if not match:
        return None
    return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H-%M-%S")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def build_motion_lookup(pose_rows: dict[str, dict[str, str]]) -> dict[str, dict[str, float]]:
    unique = [row for key, row in pose_rows.items() if key.endswith(".jpg")]
    unique.sort(key=lambda r: capture_index(r.get("camera_name", "")))
    motion: dict[str, dict[str, float]] = {}
    for idx, row in enumerate(unique):
        name = row.get("camera_name", "")
        prev_row = unique[idx - 1] if idx > 0 else None
        next_row = unique[idx + 1] if idx + 1 < len(unique) else None
        ref = next_row or prev_row
        if not name or ref is None:
            continue
        t0 = capture_time(name)
        t1 = capture_time(ref.get("camera_name", ""))
        cap_dt = abs(capture_index(ref.get("camera_name", "")) - capture_index(name)) / 10.0
        clock_dt = abs((t1 - t0).total_seconds()) if t0 and t1 else 0.0
        dt = max(clock_dt, cap_dt, 0.1)
        dist = haversine_m(
            parse_float(row.get("lat")),
            parse_float(row.get("lon")),
            parse_float(ref.get("lat")),
            parse_float(ref.get("lon")),
        )
        speed = min(dist / dt, 35.0)
        yaw0 = math.radians(parse_float(row.get("yaw")))
        yaw1 = math.radians(parse_float(ref.get("yaw")))
        pitch0 = parse_float(row.get("pitch"))
        pitch1 = parse_float(ref.get("pitch"))
        roll0 = parse_float(row.get("roll"))
        roll1 = parse_float(ref.get("roll"))
        yaw_rate = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0)) / dt
        pitch_rate_norm = min(abs(pitch1 - pitch0) / max(dt, 1e-6) / 20.0, 1.0)
        roll_rate_norm = min(abs(roll1 - roll0) / max(dt, 1e-6) / 20.0, 1.0)
        motion[name] = {
            "speed_mps": speed,
            "raw_oxts_yaw_rate_radps": yaw_rate,
            "gyro_x": pitch_rate_norm,
            "gyro_y": roll_rate_norm,
        }
    return motion


def read_exif(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        with Image.open(path) as image:
            raw = image.getexif()
            out["width"], out["height"] = image.size
        tag_names = {value: key for key, value in ExifTags.TAGS.items()}
        for key, value in raw.items():
            name = tag_names.get(key, str(key))
            try:
                out[name] = str(value)
            except Exception:
                out[name] = repr(value)
    except Exception:
        pass
    return out


def exposure_ms(exif: dict[str, Any]) -> float:
    value = exif.get("ExposureTime", "")
    if "/" in str(value):
        a, b = str(value).split("/", 1)
        return 1000.0 * parse_float(a) / max(parse_float(b, 1.0), 1e-6)
    v = parse_float(value, 0.0)
    return 1000.0 * v if 0 < v < 1 else v


def brightness_low_light(exif: dict[str, Any]) -> float:
    # BrightnessValue is not guaranteed to exist. Keep this conservative so the
    # metadata path is real but does not pretend to know unavailable lighting.
    brightness = parse_float(exif.get("BrightnessValue", ""), 4.0)
    return max(0.0, min((1.0 - brightness) / 4.0, 1.0))


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
    return ((x + w * 0.5) / width, (y + h * 0.5) / height, w / width, h / height)


def scale_segmentation(segmentation: Any, sx: float, sy: float) -> list[list[float]]:
    scaled: list[list[float]] = []
    for poly in segmentation or []:
        if not isinstance(poly, list):
            continue
        coords = [float(v) for v in poly]
        out: list[float] = []
        for x, y in zip(coords[0::2], coords[1::2]):
            out.extend([x * sx, y * sy])
        scaled.append(out)
    return scaled


def main() -> None:
    args = parse_args()
    data = json.loads(args.coco_json.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
    pose_by_name = read_pose_csv(args.geotagged_root / "precise_cam1_coords.csv")
    motion_by_name = build_motion_lookup(pose_by_name)
    image_root = args.geotagged_root / "cam1"
    out_root = args.out / "native_real_yolo_newroad6"
    image_out = out_root / "images" / args.split
    label_out = out_root / "labels" / args.split
    metadata_out = out_root / "metadata" / args.split
    contour_out = out_root / "native_annotations"
    for directory in (image_out, label_out, metadata_out, contour_out):
        directory.mkdir(parents=True, exist_ok=True)

    images = list(data.get("images", []))
    if args.limit > 0:
        images = images[: args.limit]
    image_ids = {int(img["id"]) for img in images}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if int(ann["image_id"]) in image_ids:
            anns_by_image[int(ann["image_id"])].append(ann)

    manifest: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    skipped: list[dict[str, str]] = []
    for image_record in images:
        image_id = int(image_record["id"])
        original_name = original_name_from_coco(image_record)
        native_path = image_root / original_name
        if not native_path.exists():
            skipped.append({"image": original_name, "reason": "missing_native_image"})
            continue
        with Image.open(native_path) as image:
            native_w, native_h = image.size
        coco_w = int(image_record["width"])
        coco_h = int(image_record["height"])
        sx = native_w / max(coco_w, 1)
        sy = native_h / max(coco_h, 1)

        yolo_lines: list[str] = []
        native_annotations: list[dict[str, Any]] = []
        for ann in anns_by_image[image_id]:
            category_name = categories.get(int(ann["category_id"]), "")
            if category_name not in COCO_TO_NEWROAD6:
                continue
            cls = COCO_TO_NEWROAD6[category_name]
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
            class_counts[NEWROAD6_NAMES[cls]] += 1
            native_annotations.append(
                {
                    "source_category": category_name,
                    "mapped_class_id": cls,
                    "mapped_class_name": NEWROAD6_NAMES[cls],
                    "bbox_xywh_native": [x, y, w, h],
                    "segmentation_native": scale_segmentation(ann.get("segmentation"), sx, sy),
                }
            )

        link_or_copy(native_path, image_out / original_name)
        (label_out / f"{Path(original_name).stem}.txt").write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
        )
        exif = read_exif(native_path)
        pose = pose_by_name.get(original_name) or pose_by_name.get(Path(original_name).stem) or {}
        motion = motion_by_name.get(original_name, {})
        std_e = parse_float(pose.get("std_east"))
        std_n = parse_float(pose.get("std_north"))
        std_h = parse_float(pose.get("std_ht"))
        pose_uncertainty = math.sqrt(std_e * std_e + std_n * std_n + std_h * std_h)
        low_light = brightness_low_light(exif)
        metadata = {
            "metadata_source": "real_geotagged_pose_exif_native",
            "original_name": original_name,
            "coco_file_name": image_record["file_name"],
            "coco_size": [coco_w, coco_h],
            "native_size": [native_w, native_h],
            "scale_x": sx,
            "scale_y": sy,
            "pose_csv": pose,
            "exif": exif,
            "gyro_x": motion.get("gyro_x", 0.0),
            "gyro_y": motion.get("gyro_y", 0.0),
            "accel_norm": min(pose_uncertainty * 10.0, 1.0),
            "speed_mps": motion.get("speed_mps", 0.0),
            "exposure_ms": exposure_ms(exif),
            "defocus_score": 0.0,
            "noise_score": 0.0,
            "low_light_score": low_light,
            "jpeg_quality": None,
            "raw_oxts_yaw_rate_radps": motion.get("raw_oxts_yaw_rate_radps", 0.0),
            "raw_oxts_forward_accel_mps2": 0.0,
            "raw_oxts_lateral_accel_mps2": 0.0,
        }
        (metadata_out / f"{Path(original_name).stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (contour_out / f"{Path(original_name).stem}.json").write_text(
            json.dumps(native_annotations, indent=2), encoding="utf-8"
        )
        manifest.append(
            {
                "image": original_name,
                "annotations": len(yolo_lines),
                "native_width": native_w,
                "native_height": native_h,
                "coco_width": coco_w,
                "coco_height": coco_h,
                "scale_x": sx,
                "scale_y": sy,
                "pose_matched": bool(pose),
                "exif_fields": len(exif),
            }
        )

    data_yaml = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": f"images/{args.split}",
        "val": f"images/{args.split}",
        "test": f"images/{args.split}",
        "nc": len(NEWROAD6_NAMES),
        "names": NEWROAD6_NAMES,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    summary = {
        "images": len(manifest),
        "annotations": sum(int(row["annotations"]) for row in manifest),
        "class_counts": dict(class_counts),
        "pose_matched": sum(1 for row in manifest if row["pose_matched"]),
        "skipped": skipped,
        "data_yaml": str((out_root / "data.yaml").resolve()),
        "resolution_policy": "Native geotagged images are hardlinked/copied unchanged; COCO coordinates are scaled only when necessary.",
        "metadata_policy": "Uses real precise_cam1_coords.csv and image EXIF only; no synthetic degradation metadata is inserted.",
        "class_mapping": COCO_TO_NEWROAD6,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
