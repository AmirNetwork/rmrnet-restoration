from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep deterministic native evidence-preserving transforms on GT49."
    )
    parser.add_argument("--data", type=Path, default=Path("experiments/roboflow_geotagged_v5_native_real/native_real_yolo_newroad6/data.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("runs/detect/geotagged_yolo11s_newroad/yolo11s_newroad_ext_1280_120ep/weights/best.pt"))
    parser.add_argument("--out", type=Path, default=Path("experiments/roboflow_geotagged_v5_native_real/v31_native_pass_sweep"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--tags", default="", help="Optional comma-separated subset of transform tags.")
    return parser.parse_args()


def read_yaml(path: Path) -> tuple[Path, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return root, data


def split_dir(root: Path, data: dict, split: str, kind: str) -> Path:
    value = str(data.get(split, data.get("val", f"images/{split}")))
    path = Path(value.replace("images", kind, 1))
    return path if path.is_absolute() else root / path


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    for path in src.iterdir():
        if path.is_file():
            shutil.copy2(path, dst / path.name)


def gamma_image(image: Image.Image, gamma: float) -> Image.Image:
    if abs(gamma - 1.0) < 1e-6:
        return image
    table = [max(0, min(255, int(round((i / 255.0) ** gamma * 255.0)))) for i in range(256)]
    return image.point(table * len(image.getbands()))


def transform_factory(tag: str) -> Callable[[Image.Image], Image.Image]:
    def base(image: Image.Image) -> Image.Image:
        return image

    def gamma_sharp(gamma: float, sharp: float) -> Callable[[Image.Image], Image.Image]:
        def fn(image: Image.Image) -> Image.Image:
            out = gamma_image(image, gamma)
            return ImageEnhance.Sharpness(out).enhance(sharp)
        return fn

    def denoise_unsharp(radius: float, percent: int, threshold: int, gamma: float = 1.0) -> Callable[[Image.Image], Image.Image]:
        def fn(image: Image.Image) -> Image.Image:
            out = gamma_image(image, gamma)
            out = out.filter(ImageFilter.MedianFilter(size=3))
            return out.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        return fn

    def contrast_sharp(contrast: float, sharp: float, gamma: float = 1.0) -> Callable[[Image.Image], Image.Image]:
        def fn(image: Image.Image) -> Image.Image:
            out = gamma_image(image, gamma)
            out = ImageEnhance.Contrast(out).enhance(contrast)
            return ImageEnhance.Sharpness(out).enhance(sharp)
        return fn

    def autocontrast_sharp(cutoff: float, sharp: float) -> Callable[[Image.Image], Image.Image]:
        def fn(image: Image.Image) -> Image.Image:
            out = ImageOps.autocontrast(image, cutoff=cutoff)
            return ImageEnhance.Sharpness(out).enhance(sharp)
        return fn

    variants: dict[str, Callable[[Image.Image], Image.Image]] = {
        "identity_q100": base,
        "gamma090_sharp110": gamma_sharp(0.90, 1.10),
        "gamma095_sharp115": gamma_sharp(0.95, 1.15),
        "sharp120": gamma_sharp(1.00, 1.20),
        "contrast105_sharp110": contrast_sharp(1.05, 1.10),
        "contrast110_gamma095_sharp110": contrast_sharp(1.10, 1.10, 0.95),
        "denoise_unsharp_r10_p80": denoise_unsharp(1.0, 80, 3, 1.0),
        "denoise_gamma095_unsharp_r10_p90": denoise_unsharp(1.0, 90, 3, 0.95),
        "autocontrast02_sharp110": autocontrast_sharp(0.2, 1.10),
        "autocontrast02_sharp100": autocontrast_sharp(0.2, 1.00),
        "autocontrast02_sharp105": autocontrast_sharp(0.2, 1.05),
        "autocontrast02_sharp115": autocontrast_sharp(0.2, 1.15),
        "autocontrast02_sharp120": autocontrast_sharp(0.2, 1.20),
        "autocontrast025_sharp100": autocontrast_sharp(0.25, 1.00),
        "autocontrast025_sharp105": autocontrast_sharp(0.25, 1.05),
        "autocontrast025_sharp110": autocontrast_sharp(0.25, 1.10),
        "autocontrast03_sharp100": autocontrast_sharp(0.3, 1.00),
        "autocontrast03_sharp105": autocontrast_sharp(0.3, 1.05),
        "autocontrast03_sharp110": autocontrast_sharp(0.3, 1.10),
        "autocontrast04_sharp100": autocontrast_sharp(0.4, 1.00),
        "autocontrast04_sharp105": autocontrast_sharp(0.4, 1.05),
        "autocontrast04_sharp110": autocontrast_sharp(0.4, 1.10),
        "autocontrast05_sharp105": autocontrast_sharp(0.5, 1.05),
        "autocontrast05_sharp110": autocontrast_sharp(0.5, 1.10),
        "autocontrast10_sharp105": autocontrast_sharp(1.0, 1.05),
        "autocontrast10_sharp110": autocontrast_sharp(1.0, 1.10),
        "autocontrast02_contrast105_sharp110": lambda image: ImageEnhance.Sharpness(
            ImageEnhance.Contrast(ImageOps.autocontrast(image, cutoff=0.2)).enhance(1.05)
        ).enhance(1.10),
    }
    if tag not in variants:
        raise KeyError(tag)
    return variants[tag]


def prepare_variant(args: argparse.Namespace, tag: str) -> Path:
    root, cfg = read_yaml(args.data)
    image_dir = split_dir(root, cfg, "test", "images")
    label_dir = split_dir(root, cfg, "test", "labels")
    metadata_dir = split_dir(root, cfg, "test", "metadata")
    out_root = args.out / "datasets" / tag
    out_images = out_root / "images" / "test"
    if out_images.exists():
        shutil.rmtree(out_images)
    out_images.mkdir(parents=True, exist_ok=True)
    fn = transform_factory(tag)
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        with Image.open(image_path) as image:
            out = fn(image.convert("RGB"))
            out.save(out_images / image_path.name, quality=100, subsampling=0)
    copy_tree(label_dir, out_root / "labels" / "test")
    copy_tree(metadata_dir, out_root / "metadata" / "test")
    out_yaml = out_root / "data.yaml"
    out_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_root.resolve()).replace("\\", "/"),
                "train": "images/test",
                "val": "images/test",
                "test": "images/test",
                "names": cfg["names"],
                "nc": cfg.get("nc", len(cfg["names"]) if isinstance(cfg["names"], list) else len(dict(cfg["names"]))),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return out_yaml


def eval_variant(args: argparse.Namespace, tag: str, data_yaml: Path) -> dict[str, str]:
    out_dir = args.out / "eval" / tag
    cmd = [
        sys.executable,
        "tools/eval_native_tiled_detector.py",
        "--data",
        str(data_yaml),
        "--weights",
        str(args.weights),
        "--out",
        str(out_dir),
        "--split",
        "test",
        "--strategy",
        "hybrid",
        "--tile",
        "3072",
        "--overlap",
        "768",
        "--center-margin",
        "384",
        "--infer-imgsz",
        "1536",
        "--conf",
        "0.03",
        "--branch-iou",
        "0.75",
        "--full-imgsz",
        "1536",
        "--full-conf",
        "0.03",
        "--full-iou",
        "0.75",
        "--tile-fusion-conf",
        "0.75",
        "--nms-iou",
        "0.45",
        "--device",
        str(args.device),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    metrics_path = out_dir / "native_tiled_metrics.csv"
    selected: dict[str, str] = {"tag": tag}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "iou_greedy" and row["mode"] == "crack_group" and row["threshold"] in {"0.1", "0.5"}:
                suffix = row["threshold"].replace(".", "")
                selected[f"precision_iou{suffix}"] = row["precision"]
                selected[f"recall_iou{suffix}"] = row["recall"]
                selected[f"f1_iou{suffix}"] = row["f1"]
    return selected


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    default_tags = [
        "identity_q100",
        "gamma090_sharp110",
        "gamma095_sharp115",
        "sharp120",
        "contrast105_sharp110",
        "contrast110_gamma095_sharp110",
        "denoise_unsharp_r10_p80",
        "denoise_gamma095_unsharp_r10_p90",
        "autocontrast02_sharp110",
    ]
    tags = [item.strip() for item in args.tags.split(",") if item.strip()] or default_tags
    for tag in tags:
        print({"stage": "prepare", "tag": tag}, flush=True)
        data_yaml = prepare_variant(args, tag)
        print({"stage": "eval", "tag": tag}, flush=True)
        rows.append(eval_variant(args, tag, data_yaml))
    out_csv = args.out / "native_pass_sweep_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print({"summary": str(out_csv)}, flush=True)


if __name__ == "__main__":
    main()
