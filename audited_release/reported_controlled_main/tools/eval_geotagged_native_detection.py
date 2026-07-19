# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class SerialPool:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> "SerialPool":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def imap(self, func: Any, iterable: Any) -> Any:
        return map(func, iterable)


def install_windows_safe_cache_pool() -> None:
    import ultralytics.data.dataset as dataset
    import ultralytics.data.utils as utils

    dataset.ThreadPool = SerialPool
    utils.ThreadPool = SerialPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run native-resolution tiled YOLO prediction on geotagged restored images. "
            "No mAP is reported because the geotagged folder has no ground-truth labels."
        )
    )
    parser.add_argument("--restored-root", type=Path, default=Path("experiments/geotagged_cam1_native/restored"))
    parser.add_argument("--methods", default="raw,rmr_blind,rmr_metadata,nafnet,dfpir,demoe_auto,demoe_scenario,instructir_generic,instructir_metadata")
    parser.add_argument("--weights", type=Path, default=Path("runs/detect/runs/yolo11s_v26/pcm_clean_80ep/weights/best.pt"))
    parser.add_argument("--out", type=Path, default=Path("experiments/geotagged_cam1_native/detection"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--tile", type=int, default=1280)
    parser.add_argument("--overlap", type=int, default=160)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names", default="pothole,crack,manhole")
    return parser.parse_args()


def starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(tile - overlap, 1)
    values = list(range(0, max(length - tile, 0), step))
    values.append(length - tile)
    return sorted(set(values))


def list_images(path: Path, limit: int = 0) -> list[Path]:
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return images[:limit] if limit > 0 else images


def predict_image(
    model: YOLO,
    image_path: Path,
    *,
    tile: int,
    overlap: int,
    conf: float,
    device: str,
) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    preds: list[dict[str, Any]] = []
    for y in starts(height, tile, overlap):
        for x in starts(width, tile, overlap):
            crop = image.crop((x, y, min(x + tile, width), min(y + tile, height)))
            result = model.predict(
                source=np.asarray(crop),
                imgsz=max(crop.size),
                conf=conf,
                device=device,
                verbose=False,
            )[0]
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.tolist()
                preds.append(
                    {
                        "class_id": int(cls),
                        "confidence": float(score),
                        "x1": max(0.0, min(width - 1.0, x1 + x)),
                        "y1": max(0.0, min(height - 1.0, y1 + y)),
                        "x2": max(0.0, min(width - 1.0, x2 + x)),
                        "y2": max(0.0, min(height - 1.0, y2 + y)),
                    }
                )
    return (width, height), preds


def apply_nms(preds: list[dict[str, Any]], iou: float) -> list[dict[str, Any]]:
    if not preds:
        return []
    kept: list[dict[str, Any]] = []
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in preds:
        if pred["x2"] <= pred["x1"] or pred["y2"] <= pred["y1"]:
            continue
        by_class[int(pred["class_id"])].append(pred)
    for cls, cls_preds in by_class.items():
        boxes = torch.tensor([[p["x1"], p["y1"], p["x2"], p["y2"]] for p in cls_preds], dtype=torch.float32)
        scores = torch.tensor([p["confidence"] for p in cls_preds], dtype=torch.float32)
        for idx in nms(boxes, scores, iou).tolist():
            kept.append(cls_preds[idx])
    return sorted(kept, key=lambda p: p["confidence"], reverse=True)


def write_yolo_labels(path: Path, preds: list[dict[str, Any]], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pred in preds:
        x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{pred['class_id']} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f} {pred['confidence']:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    install_windows_safe_cache_pool()
    model = YOLO(str(args.weights))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary_rows = []
    for method in methods:
        method_root = args.restored_root / method
        image_dir = method_root / "images"
        if not image_dir.exists():
            print(json.dumps({"method": method, "status": "missing_images", "path": str(image_dir)}), file=sys.stderr)
            continue
        label_dir = method_root / "labels"
        images = list_images(image_dir, args.limit)
        method_rows = []
        for image_path in images:
            (width, height), preds = predict_image(
                model,
                image_path,
                tile=args.tile,
                overlap=args.overlap,
                conf=args.conf,
                device=args.device,
            )
            preds = apply_nms(preds, args.iou)
            write_yolo_labels(label_dir / f"{image_path.stem}.txt", preds, width, height)
            for pred in preds:
                row = {"method": method, "image": image_path.name, "width": width, "height": height, **pred}
                method_rows.append(row)
                all_rows.append(row)
            print(json.dumps({"method": method, "image": image_path.name, "detections": len(preds)}), flush=True)

        data_yaml = {
            "path": str(method_root.resolve()).replace("\\", "/"),
            "test": "images",
            "val": "images",
            "train": "images",
            "nc": len(names),
            "names": names,
        }
        (method_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
        detections = len(method_rows)
        mean_conf = sum(float(r["confidence"]) for r in method_rows) / detections if detections else 0.0
        class_counts = defaultdict(int)
        for row in method_rows:
            class_counts[int(row["class_id"])] += 1
        summary_rows.append(
            {
                "method": method,
                "images": len(images),
                "detections": detections,
                "detections_per_image": detections / max(len(images), 1),
                "mean_confidence": mean_conf,
                **{f"class_{i}_{names[i]}": class_counts[i] for i in range(len(names))},
            }
        )

    pred_csv = args.out / "geotagged_native_predictions.csv"
    if all_rows:
        with pred_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    summary_csv = args.out / "geotagged_native_detection_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.out / "geotagged_native_detection_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_csv), "predictions": str(pred_csv), "rows": len(all_rows)}, indent=2))


if __name__ == "__main__":
    main()
