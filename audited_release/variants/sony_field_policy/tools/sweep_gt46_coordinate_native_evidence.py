# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Sweep simple native evidence policies with the GT46 YOLO26-coordinate evaluator.

This script does not train on GT46. It applies deterministic image-only field
policies to the 46 native images, runs the supplied coordinate-aware YOLO26
detector, and evaluates against the revised COCO annotations. The resulting
table is useful for deciding whether a lightweight deployment gate is safer
than full restoration on already high-quality Sony field frames.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments" / "gt46_yolo26_coordinate_revised"
NATIVE_IMAGES = EXP_ROOT / "gt46_native_images"
ANNOTATIONS = ROOT / "road-defect-seg-9junedata.coco-segmentation" / "train" / "_annotations.coco.json"
CAM1_METADATA = ROOT / "geotagged" / "precise_cam1_coords.csv"
DETECTOR_SCRIPT = ROOT / "Yolo26_coordinate" / "Yolo26_coordinate_revised.py"
DETECTOR_WEIGHTS = ROOT / "Yolo26_coordinate" / "YOLO26s_RDD_FRDC_Distilled_v2.pt"
EVAL_SCRIPT = ROOT / "tools" / "eval_yolo26_coordinate_gt46.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP_ROOT / "native_evidence_policy_sweep_yolo26_coordinate")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--class-conf", default="D00=0.15,D10=0.25,D20=0.25")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tags", default="", help="Optional comma-separated policy tags.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def gamma_image(image: Image.Image, gamma: float) -> Image.Image:
    if abs(gamma - 1.0) < 1e-6:
        return image
    table = [max(0, min(255, int(round((i / 255.0) ** gamma * 255.0)))) for i in range(256)]
    return image.point(table * len(image.getbands()))


def transform_factory(tag: str) -> Callable[[Image.Image], Image.Image]:
    def identity(image: Image.Image) -> Image.Image:
        return image

    def gamma_sharp(gamma: float, sharp: float) -> Callable[[Image.Image], Image.Image]:
        def fn(image: Image.Image) -> Image.Image:
            out = gamma_image(image, gamma)
            return ImageEnhance.Sharpness(out).enhance(sharp)

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

    def unsharp(gamma: float, radius: float, percent: int, threshold: int) -> Callable[[Image.Image], Image.Image]:
        def fn(image: Image.Image) -> Image.Image:
            out = gamma_image(image, gamma)
            return out.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

        return fn

    variants: dict[str, Callable[[Image.Image], Image.Image]] = {
        "identity_q100": identity,
        "gamma090_sharp110": gamma_sharp(0.90, 1.10),
        "gamma090_sharp120": gamma_sharp(0.90, 1.20),
        "gamma085_sharp110": gamma_sharp(0.85, 1.10),
        "gamma095_sharp115": gamma_sharp(0.95, 1.15),
        "sharp120": gamma_sharp(1.00, 1.20),
        "contrast105_sharp110": contrast_sharp(1.05, 1.10),
        "contrast110_gamma095_sharp110": contrast_sharp(1.10, 1.10, 0.95),
        "contrast115_gamma095_sharp115": contrast_sharp(1.15, 1.15, 0.95),
        "autocontrast02_sharp105": autocontrast_sharp(0.2, 1.05),
        "autocontrast02_sharp110": autocontrast_sharp(0.2, 1.10),
        "autocontrast03_sharp105": autocontrast_sharp(0.3, 1.05),
        "unsharp_g095_r08_p80_t3": unsharp(0.95, 0.8, 80, 3),
        "unsharp_g090_r10_p90_t3": unsharp(0.90, 1.0, 90, 3),
    }
    if tag not in variants:
        raise KeyError(f"Unknown policy tag '{tag}'. Available: {sorted(variants)}")
    return variants[tag]


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(str(item) for item in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def prepare_images(args: argparse.Namespace, tag: str) -> Path:
    out_dir = args.out / "images" / tag
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) == 46:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = transform_factory(tag)
    for path in sorted(NATIVE_IMAGES.glob("*.jpg")):
        with Image.open(path) as image:
            out = fn(image.convert("RGB"))
            out.save(out_dir / path.name, quality=100, subsampling=0)
    return out_dir


def run_detector(args: argparse.Namespace, tag: str, image_dir: Path) -> Path:
    out_dir = args.out / "detections" / tag
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    if (out_dir / "detections.geojson").exists() and any(out_dir.glob("*.csv")):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(DETECTOR_SCRIPT),
        "--images",
        str(image_dir),
        "--out",
        str(out_dir),
        "--csv",
        str(CAM1_METADATA),
        "--model",
        str(DETECTOR_WEIGHTS),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
    ]
    if args.class_conf.strip():
        cmd.extend(["--class_conf", args.class_conf])
    run(cmd)
    return out_dir


def run_evaluation(args: argparse.Namespace, pred_roots: list[tuple[str, Path]]) -> Path:
    eval_out = args.out / "evaluation"
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--annotations",
        str(ANNOTATIONS),
        "--images",
        str(NATIVE_IMAGES),
        "--out",
        str(eval_out),
        "--atlas-max",
        "8",
        "--pred-root",
    ]
    cmd.extend(str(path) for _tag, path in pred_roots)
    cmd.append("--pred-name")
    cmd.extend(tag for tag, _path in pred_roots)
    run(cmd)
    return eval_out


def summarize(eval_out: Path, out_csv: Path) -> None:
    rows = []
    with (eval_out / "summary_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == "primary" and row["iou"] == "0.1":
                rows.append(
                    {
                        "tag": row["run"],
                        "pred": row["pred"],
                        "precision_iou10": row["precision"],
                        "recall_iou10": row["recall"],
                        "f1_iou10": row["f1"],
                    }
                )
    success = {}
    with (eval_out / "summary_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] == "primary_success":
                success[row["run"]] = row["recall"]
    for row in rows:
        row["gt_success"] = success.get(row["tag"], "")
    rows.sort(key=lambda r: float(r["f1_iou10"]), reverse=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    default_tags = [
        "identity_q100",
        "gamma090_sharp110",
        "gamma090_sharp120",
        "gamma085_sharp110",
        "gamma095_sharp115",
        "sharp120",
        "contrast105_sharp110",
        "contrast110_gamma095_sharp110",
        "contrast115_gamma095_sharp115",
        "autocontrast02_sharp105",
        "autocontrast02_sharp110",
        "autocontrast03_sharp105",
        "unsharp_g095_r08_p80_t3",
        "unsharp_g090_r10_p90_t3",
    ]
    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()] or default_tags
    pred_roots = []
    for tag in tags:
        print({"stage": "prepare", "tag": tag}, flush=True)
        image_dir = prepare_images(args, tag)
        print({"stage": "detect", "tag": tag}, flush=True)
        pred_roots.append((tag, run_detector(args, tag, image_dir)))
    eval_out = run_evaluation(args, pred_roots)
    out_csv = args.out / "policy_sweep_summary.csv"
    summarize(eval_out, out_csv)
    print({"summary": str(out_csv), "evaluation": str(eval_out)}, flush=True)


if __name__ == "__main__":
    main()
