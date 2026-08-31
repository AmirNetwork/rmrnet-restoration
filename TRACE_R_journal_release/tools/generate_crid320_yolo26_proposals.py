#!/usr/bin/env python3
# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Generate resumable YOLO26 annotation proposals for a locked CRID pool.

Proposals are stored separately from human annotations and never become ground
truth until accepted or corrected in the review interface. Ultralytics maps the
network input back to each original image, and this tool verifies that all boxes
are expressed in native 4752 x 3168 pixel coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", type=Path, default=ROOT / "datasets/crid320_annotation_20260829"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "Yolo26_coordinate/YOLO26s_RDD_FRDC_Distilled_v2.pt",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=200)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def names_as_dict(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def main() -> None:
    args = parse_args()
    args.pool = args.pool.resolve()
    args.model = args.model.resolve()
    lock_path = args.pool / "locked_split.json"
    manifest_path = args.pool / "split_manifest.json"
    proposals_path = args.pool / "detector_proposals.json"
    audit_path = args.pool / "detector_proposals_audit.json"
    if not lock_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("The CRID split must be locked before proposal generation")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256(manifest_path) != lock["manifest_json_sha256"]:
        raise RuntimeError("Manifest hash no longer matches locked_split.json")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(records) != 320:
        raise RuntimeError(f"Expected 320 locked records, found {len(records)}")

    model_hash = sha256(args.model)
    configuration = {
        "model_path": str(args.model),
        "model_sha256": model_hash,
        "split_lock_sha256": sha256(lock_path),
        "manifest_sha256": sha256(manifest_path),
        "imgsz": args.imgsz,
        "confidence": args.conf,
        "nms_iou": args.iou,
        "max_det": args.max_det,
        "device": args.device,
        "native_coordinate_policy": True,
    }
    if proposals_path.exists() and not args.force:
        proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
        if proposals.get("configuration") != configuration:
            raise RuntimeError(
                "Existing proposal configuration differs. Use --force to start a new run."
            )
    else:
        proposals = {
            "schema": "crid_detector_proposals_v1",
            "configuration": configuration,
            "classes": {},
            "images": {},
        }
        atomic_json(proposals_path, proposals)

    model = YOLO(str(args.model))
    class_names = names_as_dict(model.names)
    if len(class_names) != 4:
        raise RuntimeError(f"Expected four road-defect classes, found {class_names}")
    proposals["classes"] = {str(key): value for key, value in class_names.items()}

    pending = [row for row in records if row["image"] not in proposals["images"]]
    print(
        json.dumps(
            {
                "locked_frames": len(records),
                "already_processed": len(records) - len(pending),
                "pending": len(pending),
                "configuration": configuration,
                "classes": class_names,
            },
            indent=2,
        )
    )
    for index, row in enumerate(pending, 1):
        image_path = Path(row["image_path"])
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            half=str(args.device).lower() not in {"cpu", "mps"},
            agnostic_nms=False,
            verbose=False,
        )[0]
        native_height, native_width = map(int, result.orig_shape)
        if (native_width, native_height) != (int(row["width"]), int(row["height"])):
            raise RuntimeError(
                f"Detector coordinate geometry mismatch for {image_path.name}: "
                f"{(native_width, native_height)}"
            )
        boxes: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box_index, box in enumerate(result.boxes):
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                boxes.append(
                    {
                        "id": f"proposal-{box_index + 1:03d}",
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "confidence": float(box.conf[0]),
                        "x1": max(0.0, min(float(native_width), x1)),
                        "y1": max(0.0, min(float(native_height), y1)),
                        "x2": max(0.0, min(float(native_width), x2)),
                        "y2": max(0.0, min(float(native_height), y2)),
                        "source": "frozen_yolo26_proposal",
                    }
                )
        proposals["images"][image_path.name] = {
            "split": row["split"],
            "width": native_width,
            "height": native_height,
            "boxes": boxes,
        }
        atomic_json(proposals_path, proposals)
        print(
            f"[{len(records) - len(pending) + index:03d}/320] "
            f"{row['split']:<10} proposals={len(boxes):3d} {image_path.name}",
            flush=True,
        )

    counts_by_split = {
        split: sum(
            len(value["boxes"])
            for value in proposals["images"].values()
            if value["split"] == split
        )
        for split in ("train", "validation", "test")
    }
    audit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_file": str(proposals_path),
        "proposal_file_sha256": sha256(proposals_path),
        "processed_images": len(proposals["images"]),
        "proposal_count": sum(
            len(value["boxes"]) for value in proposals["images"].values()
        ),
        "proposal_count_by_split": counts_by_split,
        "human_review_required": True,
        "ground_truth_status": (
            "Detector boxes are untrusted proposals and are excluded from evaluation "
            "until a human reviewer accepts or corrects them."
        ),
        "configuration": configuration,
        "classes": proposals["classes"],
    }
    atomic_json(audit_path, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
