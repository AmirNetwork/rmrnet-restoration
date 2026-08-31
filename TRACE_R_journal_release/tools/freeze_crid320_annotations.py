#!/usr/bin/env python3
# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Validate, freeze, and export the completed CRID-320 annotations."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from serve_crid320_annotation import Store, atomic_json


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", type=Path, default=ROOT / "datasets/crid320_annotation_20260829"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    intersection_w = max(0.0, min(left["x2"], right["x2"]) - max(left["x1"], right["x1"]))
    intersection_h = max(0.0, min(left["y2"], right["y2"]) - max(left["y1"], right["y1"]))
    intersection = intersection_w * intersection_h
    union = (
        (left["x2"] - left["x1"]) * (left["y2"] - left["y1"])
        + (right["x2"] - right["x1"]) * (right["y2"] - right["y1"])
        - intersection
    )
    return intersection / union if union else 0.0


def main() -> None:
    args = parse_args()
    pool = args.pool.resolve()
    working_path = pool / "annotations_working.json"
    frozen_path = pool / "annotations_frozen.json"
    freeze_path = pool / "annotation_freeze.json"
    if frozen_path.exists() or freeze_path.exists():
        raise FileExistsError("CRID-320 annotations are already frozen")

    store = Store(pool)
    document = store.annotations()
    frames = document["images"]
    if len(frames) != 320:
        raise RuntimeError(f"Expected 320 annotations, found {len(frames)}")
    incomplete = [name for name, value in frames.items() if value["review_status"] != "reviewed"]
    uncertain = [name for name, value in frames.items() if value["uncertain"]]
    if incomplete or uncertain:
        raise RuntimeError(
            f"Review gate failed: incomplete={len(incomplete)}, uncertain={len(uncertain)}"
        )

    duplicate_pairs: list[dict[str, Any]] = []
    class_counts: collections.Counter[int] = collections.Counter()
    split_counts: collections.Counter[str] = collections.Counter()
    source_counts: collections.Counter[str] = collections.Counter()
    zero_box_counts: collections.Counter[str] = collections.Counter()
    for name, value in frames.items():
        if not value["boxes"]:
            zero_box_counts[value["split"]] += 1
        for box in value["boxes"]:
            if int(box["class_id"]) not in range(4):
                raise ValueError(f"Invalid class for {name}: {box}")
            if not (
                0 <= box["x1"] < box["x2"] <= value["width"]
                and 0 <= box["y1"] < box["y2"] <= value["height"]
            ):
                raise ValueError(f"Invalid native geometry for {name}: {box}")
            if box["x2"] - box["x1"] < 16 or box["y2"] - box["y1"] < 16:
                raise ValueError(f"Sub-16-pixel box for {name}: {box}")
            class_counts[int(box["class_id"])] += 1
            split_counts[value["split"]] += 1
            source_counts[str(box.get("source", "unknown"))] += 1
        for index, left in enumerate(value["boxes"]):
            for right in value["boxes"][index + 1 :]:
                if left["class_id"] == right["class_id"] and box_iou(left, right) > 0.95:
                    duplicate_pairs.append(
                        {"image": name, "left": left["id"], "right": right["id"]}
                    )
    if duplicate_pairs:
        raise RuntimeError(f"Near-duplicate same-class boxes found: {duplicate_pairs[:3]}")

    source_bytes = working_path.read_bytes()
    temporary = frozen_path.with_suffix(".json.tmp")
    temporary.write_bytes(source_bytes)
    os.replace(temporary, frozen_path)
    if sha256(working_path) != sha256(frozen_path):
        raise RuntimeError("Frozen annotation bytes differ from the reviewed source")

    export = store.export()
    if not export["complete"]:
        raise RuntimeError("Export is incomplete despite passing the review gate")
    if sha256(working_path) != sha256(frozen_path):
        raise RuntimeError("Working annotations changed during export")

    export_audit_path = pool / "exports/latest/export_audit.json"
    export_audit = json.loads(export_audit_path.read_text(encoding="utf-8"))
    export_audit.update(
        {
            "frozen_annotations": str(frozen_path),
            "frozen_annotations_sha256": sha256(frozen_path),
        }
    )
    atomic_json(export_audit_path, export_audit)
    freeze = {
        "protocol": "CRID-320 human annotation freeze",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "frames": len(frames),
        "boxes": sum(class_counts.values()),
        "reviewed_frames": sum(value["review_status"] == "reviewed" for value in frames.values()),
        "uncertain_frames": sum(value["uncertain"] for value in frames.values()),
        "zero_box_frames_by_split": dict(zero_box_counts),
        "boxes_by_class": {str(key): value for key, value in sorted(class_counts.items())},
        "boxes_by_split": dict(split_counts),
        "boxes_by_source": dict(source_counts),
        "working_annotations_sha256": sha256(working_path),
        "frozen_annotations_path": str(frozen_path),
        "frozen_annotations_sha256": sha256(frozen_path),
        "locked_split_sha256": sha256(pool / "locked_split.json"),
        "manifest_sha256": sha256(pool / "split_manifest.json"),
        "annotation_audit_sha256": sha256(pool / "annotation_audit.jsonl"),
        "export_audit_path": str(export_audit_path),
        "export_audit_sha256": sha256(export_audit_path),
        "quality_checks": {
            "native_bounds_valid": True,
            "minimum_box_side_pixels": 16,
            "same_class_iou_duplicate_threshold": 0.95,
            "same_class_duplicates": 0,
            "all_frames_reviewed": True,
            "unresolved_uncertainty": 0,
        },
    }
    atomic_json(freeze_path, freeze)
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
