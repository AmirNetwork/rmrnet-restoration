from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a real/native blurry held-out YOLO subset using no synthetic degradation."
    )
    parser.add_argument("--data", required=True, help="Source YOLO data.yaml.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, help="Output YOLO dataset folder.")
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=0,
        help="For multi-class sets, keep at least this many images containing each class if possible.",
    )
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument(
        "--resize-long-side",
        type=int,
        default=0,
        help="Resize copied images to this long side while keeping YOLO-normalized labels valid.",
    )
    return parser.parse_args()


def image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def class_ids(label_path: Path) -> set[int]:
    ids: set[int] = set()
    if not label_path.exists():
        return ids
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            ids.add(int(float(parts[0])))
    return ids


def read_gray(path: Path, max_side: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def sharpness_metrics(path: Path, max_side: int) -> dict[str, float]:
    gray = read_gray(path, max_side)
    # Focus on the lower road-heavy region, but keep a full-image score as a
    # fallback because pothole close-ups are not always dashcam-framed.
    h = gray.shape[0]
    roi = gray[int(0.30 * h) :, :]
    if roi.size < 256:
        roi = gray
    roi_f = roi.astype(np.float32) / 255.0
    lap_var = float(cv2.Laplacian(roi_f, cv2.CV_32F).var())
    gx = cv2.Sobel(roi_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_f, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    brightness = float(np.mean(roi_f))
    contrast = float(np.std(roi_f))
    return {
        "laplacian_variance": lap_var,
        "tenengrad": tenengrad,
        "brightness": brightness,
        "contrast": contrast,
    }


def ranked_records(config: dict, split: str, max_side: int) -> list[dict]:
    root = Path(config["path"])
    image_dir = root / config[split]
    label_dir = root / config[split].replace("images", "labels")
    records = []
    for path in image_paths(image_dir):
        label_path = label_dir / f"{path.stem}.txt"
        ids = class_ids(label_path)
        if not ids:
            continue
        metrics = sharpness_metrics(path, max_side)
        records.append(
            {
                "path": path,
                "label_path": label_path,
                "classes": ids,
                **metrics,
            }
        )
    # Lower Laplacian/Tenengrad means the native frame is blurrier or less
    # locally detailed. The average rank is stable across brightness changes.
    by_lap = {id(rec): rank for rank, rec in enumerate(sorted(records, key=lambda r: r["laplacian_variance"]))}
    by_ten = {id(rec): rank for rank, rec in enumerate(sorted(records, key=lambda r: r["tenengrad"]))}
    for rec in records:
        rec["blur_rank"] = 0.5 * by_lap[id(rec)] + 0.5 * by_ten[id(rec)]
    return sorted(records, key=lambda r: r["blur_rank"])


def select_records(records: list[dict], max_images: int, min_per_class: int) -> list[dict]:
    selected = list(records[:max_images])
    if min_per_class <= 0:
        return selected
    all_classes = sorted({cls for rec in records for cls in rec["classes"]})
    selected_paths = {rec["path"] for rec in selected}
    for cls in all_classes:
        have = sum(1 for rec in selected if cls in rec["classes"])
        for rec in records:
            if have >= min_per_class:
                break
            if cls not in rec["classes"] or rec["path"] in selected_paths:
                continue
            # Drop the currently sharpest selected image that does not contain
            # the under-represented class. This keeps the subset blurry while
            # avoiding a crack/manhole-free stress test.
            drop_index = None
            for idx in range(len(selected) - 1, -1, -1):
                if cls not in selected[idx]["classes"]:
                    drop_index = idx
                    break
            if drop_index is None:
                break
            selected_paths.remove(selected[drop_index]["path"])
            selected[drop_index] = rec
            selected_paths.add(rec["path"])
            have += 1
    return sorted(selected, key=lambda r: r["path"].name)


def copy_image(src: Path, dst: Path, resize_long_side: int) -> None:
    if resize_long_side <= 0:
        shutil.copy2(src, dst)
        return
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(src)
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side > resize_long_side:
        scale = resize_long_side / long_side
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), image)


def copy_subset(config: dict, split: str, selected: list[dict], out: Path, resize_long_side: int) -> None:
    names = config["names"]
    out_images = out / "images" / split
    out_labels = out / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for rec in selected:
        copy_image(rec["path"], out_images / rec["path"].name, resize_long_side)
        shutil.copy2(rec["label_path"], out_labels / rec["label_path"].name)
    data_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "names": names,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")


def write_manifest(out: Path, selected: list[dict]) -> None:
    fieldnames = [
        "image",
        "classes",
        "laplacian_variance",
        "tenengrad",
        "brightness",
        "contrast",
        "blur_rank",
    ]
    with (out / "native_blur_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in selected:
            writer.writerow(
                {
                    "image": rec["path"].name,
                    "classes": " ".join(str(cls) for cls in sorted(rec["classes"])),
                    "laplacian_variance": f"{rec['laplacian_variance']:.8f}",
                    "tenengrad": f"{rec['tenengrad']:.8f}",
                    "brightness": f"{rec['brightness']:.5f}",
                    "contrast": f"{rec['contrast']:.5f}",
                    "blur_rank": f"{rec['blur_rank']:.1f}",
                }
            )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    records = ranked_records(config, args.split, args.max_side)
    selected = select_records(records, args.max_images, args.min_per_class)
    copy_subset(config, args.split, selected, out, args.resize_long_side)
    write_manifest(out, selected)
    counts = Counter(cls for rec in selected for cls in rec["classes"])
    print(
        {
            "out": str(out),
            "images": len(selected),
            "class_image_counts": dict(sorted(counts.items())),
            "mean_laplacian_variance": float(np.mean([rec["laplacian_variance"] for rec in selected])),
            "mean_tenengrad": float(np.mean([rec["tenengrad"] for rec in selected])),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
