"""Prepare a small real road-damage + telemetry pilot split.

This helper is intentionally conservative. It does not invent telemetry and it
does not create benchmark numbers. It converts a user-collected frame-level CSV
into the metadata format consumed by RMR-Net so a synchronized road-damage pilot
can be trained/evaluated when real vehicle or camera telemetry is available.

Expected telemetry CSV columns can be changed with CLI flags, but the defaults
are:

image,speed_mps,gyro_z_dps,accel_x,accel_y,accel_z,exposure_ms,iso,jpeg_quality

The output contains:
  - metadata.jsonl: one record per image with raw telemetry and normalized code.
  - telemetry_metadata.csv: compact audit table.
  - data.yaml: YOLO-style detector split reference.
  - README_real_telemetry_pilot.md: claim-boundary notes.

Example:
python tools/prepare_real_road_telemetry_pilot.py ^
  --images datasets/raw/my_pilot/images ^
  --labels datasets/raw/my_pilot/labels ^
  --telemetry-csv datasets/raw/my_pilot/telemetry.csv ^
  --out datasets/real_telemetry_pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def get_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return default
    try:
        return float(value)
    except ValueError:
        return default


def telemetry_code(row: dict[str, str], args: argparse.Namespace) -> dict[str, float]:
    speed = get_float(row, args.speed_col)
    gyro_z = get_float(row, args.gyro_z_col)
    ax = get_float(row, args.accel_x_col)
    ay = get_float(row, args.accel_y_col)
    az = get_float(row, args.accel_z_col)
    exposure = get_float(row, args.exposure_col, args.default_exposure_ms)
    iso = get_float(row, args.iso_col, args.default_iso)
    jpeg_q = get_float(row, args.jpeg_col, args.default_jpeg_quality)

    vibration = math.sqrt(ax * ax + ay * ay + az * az)
    motion_strength = clamp01((abs(speed) * exposure) / args.speed_exposure_norm)
    yaw_strength = clamp01(abs(gyro_z) / args.gyro_norm)
    vibration_score = clamp01(vibration / args.accel_norm)
    low_light = clamp01((iso - args.iso_low) / max(1.0, args.iso_high - args.iso_low))
    compression = clamp01((100.0 - jpeg_q) / 100.0)
    severity = clamp01(0.45 * motion_strength + 0.25 * yaw_strength + 0.2 * vibration_score + 0.1 * low_light)

    horizontal = motion_strength * (1.0 - yaw_strength)
    vertical = motion_strength * yaw_strength
    random_vibration = vibration_score

    return {
        "horizontal_motion": clamp01(horizontal),
        "vertical_motion": clamp01(vertical),
        "random_vibration": clamp01(random_vibration),
        "defocus": 0.0,
        "noise": low_light,
        "low_light": low_light,
        "compression": compression,
        "severity": severity,
    }


def read_telemetry(path: Path, image_col: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or image_col not in reader.fieldnames:
            raise ValueError(f"Telemetry CSV must contain image column {image_col!r}.")
        rows = {}
        for row in reader:
            name = Path(row[image_col]).name
            rows[name] = row
        return rows


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def maybe_copy(src: Path, dst: Path, enabled: bool) -> Path:
    if not enabled:
        return src
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--telemetry-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--copy", action="store_true", help="Copy images/labels into the output folder instead of referencing original paths.")
    parser.add_argument("--image-col", default="image")
    parser.add_argument("--speed-col", default="speed_mps")
    parser.add_argument("--gyro-z-col", default="gyro_z_dps")
    parser.add_argument("--accel-x-col", default="accel_x")
    parser.add_argument("--accel-y-col", default="accel_y")
    parser.add_argument("--accel-z-col", default="accel_z")
    parser.add_argument("--exposure-col", default="exposure_ms")
    parser.add_argument("--iso-col", default="iso")
    parser.add_argument("--jpeg-col", default="jpeg_quality")
    parser.add_argument("--default-exposure-ms", type=float, default=8.0)
    parser.add_argument("--default-iso", type=float, default=100.0)
    parser.add_argument("--default-jpeg-quality", type=float, default=95.0)
    parser.add_argument("--speed-exposure-norm", type=float, default=0.25, help="Normalizes speed*exposure_ms/1000 to [0,1].")
    parser.add_argument("--gyro-norm", type=float, default=60.0)
    parser.add_argument("--accel-norm", type=float, default=8.0)
    parser.add_argument("--iso-low", type=float, default=200.0)
    parser.add_argument("--iso-high", type=float, default=1600.0)
    parser.add_argument("--names", default="pothole,crack,manhole")
    args = parser.parse_args()

    if not args.images.exists():
        raise FileNotFoundError(args.images)
    if not args.labels.exists():
        raise FileNotFoundError(args.labels)
    if not args.telemetry_csv.exists():
        raise FileNotFoundError(args.telemetry_csv)

    args.out.mkdir(parents=True, exist_ok=True)
    telemetry = read_telemetry(args.telemetry_csv, args.image_col)
    images = list_images(args.images)
    if not images:
        raise RuntimeError(f"No images found under {args.images}")

    metadata_rows = []
    csv_rows = []
    missing = []

    out_img_dir = args.out / "images" / "all"
    out_lab_dir = args.out / "labels" / "all"
    for img in images:
        row = telemetry.get(img.name)
        if row is None:
            missing.append(img.name)
            continue
        label = args.labels / f"{img.stem}.txt"
        if not label.exists():
            missing.append(f"{img.name} (missing label)")
            continue
        img_ref = maybe_copy(img, out_img_dir / img.name, args.copy)
        lab_ref = maybe_copy(label, out_lab_dir / label.name, args.copy)
        code = telemetry_code(row, args)
        record = {
            "image": str(img_ref),
            "label": str(lab_ref),
            "source": "real_road_telemetry_pilot",
            "metadata_type": "real_telemetry",
            "code": code,
            "raw": {k: row.get(k, "") for k in row.keys()},
        }
        metadata_rows.append(record)
        csv_rows.append({"image": img.name, **code})

    if not metadata_rows:
        raise RuntimeError("No synchronized image/label/telemetry records were matched.")

    with (args.out / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for record in metadata_rows:
            f.write(json.dumps(record) + "\n")

    with (args.out / "telemetry_metadata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    yaml_text = (
        f"path: {args.out.as_posix()}\n"
        "train: images/all\n"
        "val: images/all\n"
        "test: images/all\n"
        f"nc: {len(names)}\n"
        f"names: {names!r}\n"
    )
    (args.out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    readme = f"""# Real Road-Damage Telemetry Pilot

Matched records: {len(metadata_rows)}
Skipped records: {len(missing)}

This folder is a pilot-preparation artifact only. It should be split by route,
drive, camera, or day before any publishable train/validation/test experiment.
Do not mix adjacent video frames across splits.

Metadata is real only if the input CSV was collected from synchronized vehicle
or camera logs. The script does not infer or fabricate telemetry.
"""
    (args.out / "README_real_telemetry_pilot.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"matched": len(metadata_rows), "missing": len(missing), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
