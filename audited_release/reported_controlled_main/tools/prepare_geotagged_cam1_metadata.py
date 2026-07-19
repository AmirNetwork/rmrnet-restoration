# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXIF_KEEP = {
    "Model",
    "Make",
    "ExposureTime",
    "BrightnessValue",
    "DigitalZoomRatio",
    "FNumber",
    "ISOSpeedRatings",
    "PhotographicSensitivity",
    "FocalLength",
    "DateTimeOriginal",
    "GPSInfo",
}


def scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, tuple):
        return [scalar(v) for v in value]
    if isinstance(value, list):
        return [scalar(v) for v in value]
    try:
        if value.__class__.__name__ == "IFDRational":
            return float(value)
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def gps_to_float(value: Any, ref: str | None = None) -> float | None:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    deg, minute, sec = [float(v) for v in value[:3]]
    out = deg + minute / 60.0 + sec / 3600.0
    if ref in {"S", "W"}:
        out *= -1
    return out


def read_exif(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        exif = image.getexif()
        width, height = image.size
    out: dict[str, Any] = {"width": width, "height": height}
    gps_raw = None
    for key, value in exif.items():
        tag = ExifTags.TAGS.get(key, key)
        if tag not in EXIF_KEEP:
            continue
        if tag == "GPSInfo":
            try:
                gps_raw = exif.get_ifd(key)
            except Exception:
                gps_raw = value if isinstance(value, dict) else None
            continue
        out[str(tag)] = scalar(value)
    if gps_raw:
        gps = {}
        for gkey, value in gps_raw.items():
            name = ExifTags.GPSTAGS.get(gkey, gkey)
            gps[str(name)] = scalar(value)
        lat = gps_to_float(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
        lon = gps_to_float(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
        if lat is not None:
            out["GPSLatitude"] = lat
        if lon is not None:
            out["GPSLongitude"] = lon
        if "GPSAltitude" in gps:
            out["GPSAltitude"] = gps["GPSAltitude"]
    return out


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "camera_name" not in reader.fieldnames:
            raise ValueError("precise_cam1_coords.csv must contain a camera_name column.")
        return {Path(row["camera_name"]).name: row for row in reader}


def list_cam1_images(root: Path) -> list[Path]:
    cam1 = root / "cam1"
    if not cam1.exists():
        raise FileNotFoundError(cam1)
    images = []
    for path in cam1.rglob("*"):
        if path.parent.name.lower() == "q":
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract native-resolution geotagged cam1 metadata.")
    parser.add_argument("--root", type=Path, default=Path("geotagged"))
    parser.add_argument("--csv", type=Path, default=Path("geotagged/precise_cam1_coords.csv"))
    parser.add_argument("--out", type=Path, default=Path("experiments/geotagged_cam1_native"))
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. 0 means all usable cam1 images.")
    args = parser.parse_args()

    coords = read_csv(args.csv)
    images = list_cam1_images(args.root)
    if args.limit > 0:
        images = images[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    missing_csv = []
    for path in images:
        csv_row = coords.get(path.name)
        if csv_row is None:
            missing_csv.append(path.name)
        record = {
            "image": str(path.resolve()),
            "name": path.name,
            "stem": path.stem,
            "has_precise_csv": csv_row is not None,
            "csv": csv_row or {},
            "exif": read_exif(path),
        }
        records.append(record)

    jsonl_path = args.out / "geotagged_metadata.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    csv_path = args.out / "geotagged_metadata_summary.csv"
    fieldnames = [
        "name",
        "width",
        "height",
        "has_precise_csv",
        "lat",
        "lon",
        "ht",
        "yaw",
        "pitch",
        "roll",
        "std_east",
        "std_north",
        "std_ht",
        "model",
        "exposure_time",
        "brightness",
        "zoom",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = record["csv"]
            exif = record["exif"]
            writer.writerow(
                {
                    "name": record["name"],
                    "width": exif.get("width"),
                    "height": exif.get("height"),
                    "has_precise_csv": record["has_precise_csv"],
                    "lat": row.get("lat", exif.get("GPSLatitude", "")),
                    "lon": row.get("lon", exif.get("GPSLongitude", "")),
                    "ht": row.get("ht", exif.get("GPSAltitude", "")),
                    "yaw": row.get("yaw", ""),
                    "pitch": row.get("pitch", ""),
                    "roll": row.get("roll", ""),
                    "std_east": row.get("std_east", ""),
                    "std_north": row.get("std_north", ""),
                    "std_ht": row.get("std_ht", ""),
                    "model": exif.get("Model", ""),
                    "exposure_time": exif.get("ExposureTime", ""),
                    "brightness": exif.get("BrightnessValue", ""),
                    "zoom": exif.get("DigitalZoomRatio", ""),
                }
            )

    report = {
        "images": len(records),
        "csv_matched": sum(1 for r in records if r["has_precise_csv"]),
        "missing_csv": len(missing_csv),
        "ignored_q_folder": True,
        "resolution_policy": "native image dimensions are recorded and not resized by this preparation step",
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
    }
    (args.out / "metadata_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
