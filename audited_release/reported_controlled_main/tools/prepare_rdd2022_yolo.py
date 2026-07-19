# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

"""Convert an unpacked RDD2022-style VOC dataset into YOLO splits.

The script is intentionally conservative: it only prepares data and writes a
manifest.  It does not train detectors or add manuscript results.  Use it when
the external RDD2022 archive has been downloaded outside the repository.

Example:
    python tools/prepare_rdd2022_yolo.py --rdd-root D:/datasets/RDD2022 \
        --out datasets/rdd2022_yolo_leave_japan --test-source Japan
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CLASS_MAP = {
    "D00": "crack",
    "D01": "crack",
    "D10": "crack",
    "D11": "crack",
    "D20": "crack",
    "D40": "pothole",
    "D43": "pothole",
    "Repair": "patch",
    "repair": "patch",
}


@dataclass(frozen=True)
class Record:
    image: Path
    boxes: list[tuple[str, float, float, float, float]]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdd-root", required=True, type=Path, help="Unpacked RDD2022 root containing VOC XML files.")
    parser.add_argument("--out", required=True, type=Path, help="Output YOLO dataset directory.")
    parser.add_argument("--test-source", default="", help="Substring used for a leave-one-source test split, e.g. Japan.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Used only when --test-source is omitted.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of trying hardlinks first. Copy is safest on locked-down filesystems.")
    parser.add_argument("--keep-rdd-labels", action="store_true", help="Keep RDD labels such as D00/D10/D40 instead of coarse crack/pothole/patch labels.")
    return parser.parse_args()


def find_image(xml_path: Path, filename: str) -> Path | None:
    candidates = []
    if filename:
        candidates.extend(xml_path.parents[i] / filename for i in range(min(4, len(xml_path.parents))))
        candidates.extend(xml_path.parents[i] / "images" / filename for i in range(min(4, len(xml_path.parents))))
        candidates.extend(xml_path.parents[i] / "JPEGImages" / filename for i in range(min(4, len(xml_path.parents))))
    for ext in IMAGE_EXTS:
        candidates.extend(xml_path.parents[i] / f"{xml_path.stem}{ext}" for i in range(min(4, len(xml_path.parents))))
        candidates.extend(xml_path.parents[i] / "images" / f"{xml_path.stem}{ext}" for i in range(min(4, len(xml_path.parents))))
        candidates.extend(xml_path.parents[i] / "JPEGImages" / f"{xml_path.stem}{ext}" for i in range(min(4, len(xml_path.parents))))
    for path in candidates:
        if path.exists():
            return path
    root = xml_path.parents[-2] if len(xml_path.parents) >= 2 else xml_path.parent
    for path in root.rglob(filename or f"{xml_path.stem}.*"):
        if path.suffix.lower() in IMAGE_EXTS:
            return path
    return None


def source_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if rel.parts else "unknown"


def read_record(xml_path: Path, root: Path, keep_labels: bool) -> Record | None:
    tree = ET.parse(xml_path)
    doc = tree.getroot()
    filename = (doc.findtext("filename") or "").strip()
    image = find_image(xml_path, filename)
    if image is None:
        return None
    width = float(doc.findtext("size/width") or 0)
    height = float(doc.findtext("size/height") or 0)
    if width <= 0 or height <= 0:
        return None
    boxes: list[tuple[str, float, float, float, float]] = []
    for obj in doc.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        label = raw_name if keep_labels else DEFAULT_CLASS_MAP.get(raw_name)
        if not label:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin") or 0)
        ymin = float(box.findtext("ymin") or 0)
        xmax = float(box.findtext("xmax") or 0)
        ymax = float(box.findtext("ymax") or 0)
        xmin = max(0.0, min(width - 1.0, xmin))
        ymin = max(0.0, min(height - 1.0, ymin))
        xmax = max(0.0, min(width - 1.0, xmax))
        ymax = max(0.0, min(height - 1.0, ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        xc = ((xmin + xmax) / 2.0) / width
        yc = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        boxes.append((label, xc, yc, bw, bh))
    if not boxes:
        return None
    return Record(image=image, boxes=boxes, source=source_name(xml_path, root))


def load_records(root: Path, keep_labels: bool) -> list[Record]:
    records = []
    for xml_path in sorted(root.rglob("*.xml")):
        record = read_record(xml_path, root, keep_labels)
        if record is not None:
            records.append(record)
    return records


def split_records(records: list[Record], test_source: str, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[Record]]:
    rng = random.Random(seed)
    if test_source:
        test_key = test_source.lower()
        test = [r for r in records if test_key in str(r.image).lower() or test_key in r.source.lower()]
        rest = [r for r in records if r not in test]
        rng.shuffle(rest)
        val_n = max(1, round(len(rest) * val_ratio))
        return {"train": rest[val_n:], "val": rest[:val_n], "test": test}
    shuffled = records[:]
    rng.shuffle(shuffled)
    test_n = max(1, round(len(shuffled) * test_ratio))
    val_n = max(1, round(len(shuffled) * val_ratio))
    return {"train": shuffled[test_n + val_n :], "val": shuffled[test_n : test_n + val_n], "test": shuffled[:test_n]}


def class_names(records: Iterable[Record]) -> list[str]:
    labels = sorted({box[0] for record in records for box in record.boxes})
    return labels


def write_split(out: Path, split: str, records: list[Record], names: list[str], copy_images: bool) -> dict[str, int]:
    img_dir = out / "images" / split
    lab_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    label_to_id = {name: i for i, name in enumerate(names)}
    boxes = 0
    for idx, record in enumerate(records):
        target_name = f"{record.image.stem}_{idx:06d}{record.image.suffix.lower()}"
        target_image = img_dir / target_name
        if copy_images:
            shutil.copy2(record.image, target_image)
        else:
            if not target_image.exists():
                try:
                    os.link(record.image, target_image)
                except OSError:
                    shutil.copy2(record.image, target_image)
        label_lines = []
        for label, xc, yc, bw, bh in record.boxes:
            boxes += 1
            label_lines.append(f"{label_to_id[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (lab_dir / f"{Path(target_name).stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return {"images": len(records), "boxes": boxes}


def main() -> None:
    args = parse_args()
    root = args.rdd_root.resolve()
    out = args.out.resolve()
    if not root.exists():
        raise FileNotFoundError(f"RDD root does not exist: {root}")
    records = load_records(root, args.keep_rdd_labels)
    if not records:
        raise RuntimeError(f"No annotated RDD records found under {root}")
    splits = split_records(records, args.test_source, args.val_ratio, args.test_ratio, args.seed)
    names = class_names(records)
    stats = {split: write_split(out, split, split_records_, names, args.copy) for split, split_records_ in splits.items()}
    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(names)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    manifest = {
        "source_root": str(root),
        "output": str(out),
        "test_source": args.test_source,
        "seed": args.seed,
        "copy_images": args.copy,
        "keep_rdd_labels": args.keep_rdd_labels,
        "names": names,
        "stats": stats,
        "note": "Prepared dataset only. Train/evaluate before adding any RDD2022 results to the manuscript.",
    }
    (out / "prepare_rdd2022_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
