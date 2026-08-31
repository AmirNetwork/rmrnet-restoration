#!/usr/bin/env python3
# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Create the locked, native-resolution CRID-320 annotation pool.

The split is fixed before detector proposals are generated. Previously inspected
CRID-46 frames are development data only and are assigned to training. New
validation and test frames are selected uniformly from later, disjoint temporal
blocks without consulting labels, restorations, or detector outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_RE = re.compile(r"_capt(?P<capture>\d+)_")
CLASS_NAMES = [
    "Longitudinal Crack (D00)",
    "Transverse Crack (D10)",
    "Alligator Crack (D20)",
    "Pothole (D40)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "geotagged/cam1")
    parser.add_argument(
        "--coordinates", type=Path, default=ROOT / "geotagged/precise_cam1_coords.csv"
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=ROOT
        / "experiments/geotagged_cam1_complete_sbg_ins_metadata_v4_20260811"
        / "crid_ins_all_alignment.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT
        / "experiments/geotagged_cam1_complete_sbg_ins_metadata_v4_20260811"
        / "metadata",
    )
    parser.add_argument(
        "--existing-labels",
        type=Path,
        default=ROOT / "datasets/gt46_sony_classbalanced_20260801/labels",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "datasets/crid320_annotation_20260829",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_index(path_or_name: Path | str) -> int:
    match = CAPTURE_RE.search(Path(path_or_name).name)
    if not match:
        raise ValueError(f"Cannot parse capture index: {path_or_name}")
    return int(match.group("capture"))


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row[key]).strip(): row for row in rows}


def select_uniform(candidates: list[Path], count: int) -> list[Path]:
    """Select one model-blind frame near each equal-width temporal bin centre."""

    ordered = sorted(candidates, key=capture_index)
    if len(ordered) < count:
        raise ValueError(f"Need {count} candidates, found {len(ordered)}")
    chosen: list[Path] = []
    used: set[Path] = set()
    for index in range(count):
        target = (index + 0.5) * len(ordered) / count - 0.5
        center = int(round(target))
        for offset in range(len(ordered)):
            for position in (center - offset, center + offset):
                if 0 <= position < len(ordered) and ordered[position] not in used:
                    chosen.append(ordered[position])
                    used.add(ordered[position])
                    break
            else:
                continue
            break
    return sorted(chosen, key=capture_index)


def select_farthest(
    candidates: list[Path], anchors: Iterable[Path], count: int
) -> list[Path]:
    """Fill training coverage while avoiding near-duplicate capture times."""

    remaining = {path: capture_index(path) for path in candidates}
    selected = list(anchors)
    selected_indices = [capture_index(path) for path in selected]
    while len(selected) < count:
        if not remaining:
            raise RuntimeError("Insufficient training candidates")
        winner = max(
            remaining,
            key=lambda path: (
                min(abs(remaining[path] - anchor) for anchor in selected_indices),
                -remaining[path],
            ),
        )
        selected.append(winner)
        selected_indices.append(remaining.pop(winner))
    return sorted(selected, key=capture_index)


def haversine_m(a: dict[str, Any], b: dict[str, Any]) -> float:
    lat1 = math.radians(float(a["latitude_audit_only"]))
    lon1 = math.radians(float(a["longitude_audit_only"]))
    lat2 = math.radians(float(b["latitude_audit_only"]))
    lon2 = math.radians(float(b["longitude_audit_only"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(math.sqrt(value))


def import_yolo_boxes(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if not label_path.exists():
        return boxes
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row at {label_path}:{line_number}")
        class_id = int(parts[0])
        cx, cy, box_w, box_h = map(float, parts[1:])
        boxes.append(
            {
                "id": f"legacy-{line_number:03d}",
                "class_id": class_id,
                "x1": max(0.0, (cx - box_w / 2) * width),
                "y1": max(0.0, (cy - box_h / 2) * height),
                "x2": min(float(width), (cx + box_w / 2) * width),
                "y2": min(float(height), (cy + box_h / 2) * height),
                "source": "existing_crid46_manual",
            }
        )
    return boxes


def temporal_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indices = sorted(int(row["capture_index"]) for row in rows)
    gaps = [right - left for left, right in zip(indices, indices[1:])]
    return {
        "count": len(indices),
        "capture_min": min(indices),
        "capture_max": max(indices),
        "minimum_within_split_capture_gap": min(gaps) if gaps else None,
        "median_within_split_capture_gap": sorted(gaps)[len(gaps) // 2] if gaps else None,
    }


def minimum_path_distance_m(
    left: list[Path], right: list[Path], coordinates: dict[str, dict[str, str]]
) -> float:
    def audit_row(path: Path) -> dict[str, float]:
        row = coordinates[path.name]
        return {
            "latitude_audit_only": float(row["lat"]),
            "longitude_audit_only": float(row["lon"]),
        }

    left_rows = [audit_row(path) for path in left]
    right_rows = [audit_row(path) for path in right]
    return min(haversine_m(a, b) for a in left_rows for b in right_rows)


def main() -> None:
    args = parse_args()
    args.images = args.images.resolve()
    args.coordinates = args.coordinates.resolve()
    args.alignment = args.alignment.resolve()
    args.metadata = args.metadata.resolve()
    args.existing_labels = args.existing_labels.resolve()
    args.out = args.out.resolve()

    lock_path = args.out / "locked_split.json"
    if lock_path.exists() and not args.force:
        raise FileExistsError(
            f"Split is already locked: {lock_path}. Refusing to overwrite without --force."
        )

    images = sorted(args.images.glob("*.jpg"), key=capture_index)
    if len(images) != 4_134:
        raise RuntimeError(f"Expected 4,134 CRID frames, found {len(images)}")
    by_name = {path.name: path for path in images}

    coordinate_rows = read_csv_by_key(args.coordinates, "camera_name")
    alignment_rows = read_csv_by_key(args.alignment, "original_name")
    existing_names = sorted(
        f"{path.stem}.jpg"
        for path in args.existing_labels.glob("*.txt")
        if path.name != "classes.txt"
    )
    if len(existing_names) != 46:
        raise RuntimeError(f"Expected 46 existing labels, found {len(existing_names)}")
    missing_existing = sorted(set(existing_names) - set(by_name))
    if missing_existing:
        raise FileNotFoundError(f"Missing existing images: {missing_existing[:3]}")

    existing_train = [by_name[name] for name in existing_names]
    # The blocks are declared before any detector-assisted annotation:
    # training <= 2200, validation 2700..3299, test 3500..4192. These
    # intervals are also spatially disjoint by at least 20 m over the complete
    # candidate blocks, which prevents opposite-direction revisits of the same
    # pavement location from crossing splits.
    train_candidates = [path for path in images if 176 <= capture_index(path) <= 2_200]
    validation_candidates = [
        path for path in images if 2_700 <= capture_index(path) <= 3_299
    ]
    test_candidates = [path for path in images if 3_500 <= capture_index(path) <= 4_192]
    training = select_farthest(train_candidates, existing_train, 180)
    validation = select_uniform(validation_candidates, 60)
    test = select_uniform(test_candidates, 80)

    block_spatial_distances = {
        "train_to_validation_minimum_gps_m": minimum_path_distance_m(
            train_candidates, validation_candidates, coordinate_rows
        ),
        "validation_to_test_minimum_gps_m": minimum_path_distance_m(
            validation_candidates, test_candidates, coordinate_rows
        ),
        "train_to_test_minimum_gps_m": minimum_path_distance_m(
            train_candidates, test_candidates, coordinate_rows
        ),
    }
    if min(block_spatial_distances.values()) < 20.0:
        raise RuntimeError(
            f"Candidate blocks violate the 20 m spatial embargo: {block_spatial_distances}"
        )

    selected = {"train": training, "validation": validation, "test": test}
    flattened = [path for paths in selected.values() for path in paths]
    if len(flattened) != 320 or len(set(flattened)) != 320:
        raise RuntimeError("CRID-320 split is not disjoint or does not contain 320 frames")

    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    annotations: dict[str, Any] = {
        "schema": "crid_native_bbox_annotations_v1",
        "classes": CLASS_NAMES,
        "native_size": [4752, 3168],
        "images": {},
    }
    for split, paths in selected.items():
        for path in paths:
            coordinate = coordinate_rows.get(path.name)
            alignment = alignment_rows.get(path.name)
            metadata_path = args.metadata / f"{path.stem}.json"
            if coordinate is None or alignment is None or not metadata_path.exists():
                raise FileNotFoundError(f"Missing coordinate/alignment/metadata for {path.name}")
            with Image.open(path) as image:
                width, height = image.size
            if (width, height) != (4752, 3168):
                raise RuntimeError(f"Native size changed for {path.name}: {(width, height)}")
            label_path = args.existing_labels / f"{path.stem}.txt"
            imported_boxes = import_yolo_boxes(label_path, width, height)
            record = {
                "image": path.name,
                "split": split,
                "capture_index": capture_index(path),
                "width": width,
                "height": height,
                "timestamp_utc": alignment["capture_timestamp_utc"],
                "latitude_audit_only": float(coordinate["lat"]),
                "longitude_audit_only": float(coordinate["lon"]),
                "imu_reliability": float(alignment["imu_reliability"]),
                "vehicle_reliability": float(alignment["vehicle_reliability"]),
                "nearest_sample_offset_ms": float(alignment["nearest_sample_offset_ms"]),
                "complete_sbg_ins_available": alignment["complete_sbg_ins_available"].lower()
                == "true",
                "existing_label": label_path.exists(),
                "image_path": str(path),
                "metadata_path": str(metadata_path),
                "image_sha256": sha256(path),
                "metadata_sha256": sha256(metadata_path),
            }
            records.append(record)
            annotations["images"][path.name] = {
                "split": split,
                "width": width,
                "height": height,
                "review_status": "needs_review",
                "no_defect": False,
                "uncertain": False,
                "boxes": imported_boxes,
                "notes": "Imported CRID-46 boxes require confirmation."
                if imported_boxes or label_path.exists()
                else "",
            }

    records.sort(key=lambda row: (row["split"], int(row["capture_index"])))
    manifest_csv = args.out / "split_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest_json = args.out / "split_manifest.json"
    manifest_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    annotations_path = args.out / "annotations_working.json"
    annotations_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")

    split_rows = {
        split: [row for row in records if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    cross_split_distances: dict[str, float] = {}
    for left, right in (("train", "validation"), ("validation", "test"), ("train", "test")):
        cross_split_distances[f"{left}_to_{right}_minimum_gps_m"] = min(
            haversine_m(a, b) for a in split_rows[left] for b in split_rows[right]
        )

    lock = {
        "protocol": "CRID-320 native-resolution human annotation study",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_blinding": (
            "Split membership was fixed from capture index and existing-label provenance "
            "before detector proposals, restoration outputs, or test metrics were consulted."
        ),
        "split_counts": {key: len(value) for key, value in selected.items()},
        "selection_blocks": {
            "train": [176, 2200],
            "validation": [2700, 3299],
            "test": [3500, 4192],
        },
        "temporal_embargo_capture_frames": {
            "train_to_validation": 499,
            "validation_to_test": 200,
        },
        "spatial_block_audit": (
            "Every frame in a later candidate block is at least 20 m from every "
            "frame in each earlier candidate block, preventing route revisits from "
            "placing the same pavement location in multiple splits."
        ),
        "candidate_block_minimum_gps_m": block_spatial_distances,
        "existing_crid46_policy": (
            "All 46 previously inspected frames are assigned to training/development only."
        ),
        "native_geometry": [4752, 3168],
        "coordinate_policy": "Annotations are stored in native pixel coordinates.",
        "telemetry_policy": (
            "All selected frames require synchronized Sony EXIF and complete 200 Hz SBG data; "
            "latitude/longitude are retained only for split and synchronization audit."
        ),
        "splits": {key: temporal_stats(value) for key, value in split_rows.items()},
        "cross_split_spatial_audit": cross_split_distances,
        "complete_sbg_ins_coverage": sum(
            bool(row["complete_sbg_ins_available"]) for row in records
        ),
        "manifest_csv_sha256": sha256(manifest_csv),
        "manifest_json_sha256": sha256(manifest_json),
        "annotations_initial_sha256": sha256(annotations_path),
        "source_files": {
            "coordinates": {"path": str(args.coordinates), "sha256": sha256(args.coordinates)},
            "alignment": {"path": str(args.alignment), "sha256": sha256(args.alignment)},
        },
        "records": [
            {
                "image": row["image"],
                "split": row["split"],
                "capture_index": row["capture_index"],
                "image_sha256": row["image_sha256"],
                "metadata_sha256": row["metadata_sha256"],
            }
            for row in records
        ],
    }
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in lock.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
