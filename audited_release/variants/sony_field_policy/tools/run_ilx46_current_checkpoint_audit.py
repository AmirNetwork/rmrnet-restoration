# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Restore and detect the exact revised ILX-RD46 native-image corpus.

This script performs inference only.  Ground-truth annotations define the
46-frame corpus and are converted for later scoring, but no box coordinate is
used by restoration, detector inference, or output-strength selection.  It
creates predeclared residual views that a separate chronological calibration
and evaluation script may compare.  Exact filename agreement with the revised
COCO export prevents the earlier 49-image YAML from entering this experiment.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "road-defect-seg-9junedata.coco-segmentation" / "train" / "_annotations.coco.json"
NATIVE = ROOT / "experiments" / "gt46_yolo26_coordinate_revised" / "gt46_native_images"
SOURCE = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "native_real_yolo_newroad6"
DETECTOR = ROOT / "Yolo26_coordinate" / "Yolo26_coordinate_revised.py"
DETECTOR_WEIGHTS = ROOT / "Yolo26_coordinate" / "YOLO26s_RDD_FRDC_Distilled_v2.pt"
COORDS = ROOT / "geotagged" / "precise_cam1_coords.csv"
ETAS = (0.05, 0.10, 0.25, 0.50, 1.00)
COCO_TO_YOLO = {
    2: 0,  # Alligator Crack
    3: 1,  # Longitudinal Crack
    4: 3,  # Potholes
    5: 5,  # Transverse Crack
    6: 2,  # manhole -> other
    7: 2,  # patch -> other
    8: 2,  # rutting -> other
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tag", default="major_revision_ilx46_current_20260716")
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print(json.dumps({"cmd": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def native_name(file_name: str) -> str:
    match = re.match(r"^(.*)_jpg\.rf\.[^.]+\.jpg$", file_name, re.IGNORECASE)
    return match.group(1) + ".jpg" if match else file_name


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_exact_dataset(root: Path, force: bool) -> tuple[Path, list[str]]:
    if force and root.exists():
        shutil.rmtree(root)
    coco = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    expected = sorted(native_name(item["file_name"]) for item in coco["images"])
    if len(expected) != 46 or len(set(expected)) != 46:
        raise RuntimeError(f"Revised COCO export must contain 46 unique images, got {len(expected)}")
    present = sorted(path.name for path in NATIVE.glob("*.jpg"))
    if present != expected:
        raise RuntimeError(
            "Native/annotation filename mismatch: "
            + json.dumps({"annotation_only": sorted(set(expected) - set(present)), "native_only": sorted(set(present) - set(expected))})
        )
    image_records = {native_name(item["file_name"]): item for item in coco["images"]}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    generated_boxes = 0
    for name in expected:
        stem = Path(name).stem
        sources = {
            "images": NATIVE / name,
            "metadata": SOURCE / "metadata" / "test" / f"{stem}.json",
        }
        missing = [str(path) for path in sources.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Incomplete ILX-RD46 record {name}: {missing}")
        link_or_copy(sources["images"], root / "images" / "test" / name)
        link_or_copy(sources["metadata"], root / "metadata" / "test" / f"{stem}.json")
        record = image_records[name]
        width, height = int(record["width"]), int(record["height"])
        if (width, height) != (4752, 3168):
            raise RuntimeError(f"Revised COCO annotation is not in native coordinates for {name}: {(width, height)}")
        label_rows: list[str] = []
        for annotation in annotations_by_image.get(int(record["id"]), []):
            category = int(annotation["category_id"])
            if category not in COCO_TO_YOLO:
                continue
            x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
            cx = (x + box_width / 2.0) / width
            cy = (y + box_height / 2.0) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
            label_rows.append(
                f"{COCO_TO_YOLO[category]} {cx:.9f} {cy:.9f} {normalized_width:.9f} {normalized_height:.9f}"
            )
        label_path = root / "labels" / "test" / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
        generated_boxes += len(label_rows)
    source_yaml = yaml.safe_load((SOURCE / "data.yaml").read_text(encoding="utf-8"))
    data = {
        "path": str(root.resolve()).replace("\\", "/"),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "nc": source_yaml["nc"],
        "names": source_yaml["names"],
    }
    data_path = root / "data.yaml"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (root / "exact_manifest.json").write_text(
        json.dumps(
            {
                "images": expected,
                "count": 46,
                "native_resolution": [4752, 3168],
                "synthetic_degradation_added": False,
                "annotation_file": str(ANNOTATIONS.relative_to(ROOT)),
                "annotation_sha256": sha256(ANNOTATIONS),
                "labels_generated_directly_from_revised_coco": True,
                "category_mapping_coco_to_yolo": COCO_TO_YOLO,
                "generated_boxes": generated_boxes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return data_path, expected


def complete(folder: Path, expected: list[str]) -> bool:
    expected_stems = {Path(name).stem for name in expected}
    present = {
        path.stem
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    } if folder.exists() else set()
    return present == expected_stems


def images_by_stem(folder: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def restore(checkpoint: Path, experiment: Path, data: Path, expected: list[str], force: bool) -> Path:
    root = experiment / "restored"
    if force and root.exists():
        shutil.rmtree(root)
    blind = root / "rmr_blind" / "images" / "test"
    metadata = root / "rmr_metadata" / "images" / "test"
    if force or not (complete(blind, expected) and complete(metadata, expected)):
        run(
            [
                sys.executable,
                "tools/restore_native_yolo_split.py",
                "--data", str(data),
                "--split", "test",
                "--scenario", "native_real",
                "--out", str(root),
                "--models", "rmr_blind,rmr_metadata",
                "--device", "cuda",
                "--tile", "768",
                "--overlap", "96",
                "--jpeg-quality", "98",
                "--output-format", "png",
                "--rcadnet-weights", str(checkpoint),
            ]
        )
    if not complete(blind, expected) or not complete(metadata, expected):
        raise RuntimeError("RMR-Net restoration did not produce the exact 46-frame corpus")
    return root


def blend_views(restored: Path, experiment: Path, expected: list[str], force: bool) -> dict[str, Path]:
    views: dict[str, Path] = {}
    for source in ("blind", "metadata"):
        full = restored / f"rmr_{source}" / "images" / "test"
        restored_by_stem = images_by_stem(full)
        for eta in ETAS:
            tag = f"current_{source}_eta{eta:.2f}".replace(".", "p")
            destination = experiment / "views" / tag
            if force and destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
            views[tag] = destination
            for name in expected:
                stem = Path(name).stem
                target = destination / f"{stem}.png"
                if target.exists() and not force:
                    continue
                raw = cv2.imread(str(NATIVE / name), cv2.IMREAD_COLOR)
                result = cv2.imread(str(restored_by_stem[stem]), cv2.IMREAD_COLOR)
                if raw is None or result is None or raw.shape != result.shape:
                    raise RuntimeError(f"Native-resolution pair failed for {name}")
                blended = cv2.addWeighted(raw, 1.0 - eta, result, eta, 0.0)
                if not cv2.imwrite(str(target), blended, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError(f"Could not write {target}")
            if not complete(destination, expected):
                raise RuntimeError(f"Incomplete residual view {tag}")
    return views


def detect(views: dict[str, Path], experiment: Path, device: str, force: bool) -> Path:
    prediction_root = experiment / "detections"
    for tag, images in views.items():
        out = prediction_root / tag
        if force or not (out / "detections.geojson").exists():
            run(
                [
                    sys.executable,
                    str(DETECTOR),
                    "--images", str(images),
                    "--out", str(out),
                    "--csv", str(COORDS),
                    "--model", str(DETECTOR_WEIGHTS),
                    "--device", device,
                    "--workers", "1",
                    "--imgsz", "1280",
                    "--conf", "0.10",
                    "--class_conf", "D00=0.15,D10=0.25,D20=0.25",
                ]
            )
    return prediction_root


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    experiment = ROOT / "experiments" / args.tag
    data, expected = prepare_exact_dataset(experiment / "native_exact", args.force)
    restored = restore(checkpoint, experiment, data, expected, args.force)
    views = {"raw": NATIVE, **blend_views(restored, experiment, expected, args.force)}
    predictions = detect(views, experiment, args.device, args.force)
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_selection": "controlled PCM validation only; ILX labels not read by this script",
        "images": 46,
        "native_resolution_preserved": True,
        "restored_output_encoding": "lossless PNG; raw view retains source JPEG encoding",
        "synthetic_degradation_added": False,
        "annotation_use": "defines the 46-image corpus and later scoring labels only; boxes do not enter restoration, detector inference, or policy selection",
        "residual_strength_candidates": list(ETAS),
        "detector": str(DETECTOR_WEIGHTS.relative_to(ROOT)),
        "detector_sha256": sha256(DETECTOR_WEIGHTS),
        "detector_protocol": {
            "imgsz": 1280,
            "base_confidence": 0.10,
            "class_confidence": {"D00": 0.15, "D10": 0.25, "D20": 0.25, "D40": 0.10},
            "crop_or_mask": False,
            "detector_tiling": False,
            "direct_head": "end-to-end one-to-one, NMS-free",
        },
        "prediction_root": str(predictions.relative_to(ROOT)),
    }
    (experiment / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
