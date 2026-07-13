from __future__ import annotations

"""Create native-safe GT49 evidence-enhanced images for detector evaluation.

The normal RMR restoration branch can be harmful on already sharp Sony native
frames because full image-to-image restoration may change contrast globally or
introduce boundary artifacts.  This script creates a deployment-time native
evidence branch that keeps the original image as the anchor and applies only
bounded local evidence enhancement in the lower road ROI used by the GT49
detector.

No GT49 labels are read by this script.  Labels are copied only so that the
existing evaluator can score the fixed outputs afterward.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "roboflow_geotagged_v5_native_real"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "v36_yolo26_rdd4_eval_sets/raw")
    parser.add_argument("--out", type=Path, default=BASE / "v42_native_evidence_eval_sets")
    parser.add_argument(
        "--variants",
        default="nee_mild,nee_balanced,nee_strong,nee_darkline",
        help="Comma-separated native-evidence variants to create.",
    )
    return parser.parse_args()


def clahe_luma(rgb: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)


def unsharp(rgb: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(rgb, 1.0 + amount, blur, -amount, 0)


def dark_line_boost(rgb: np.ndarray, strength: float, kernel: int) -> np.ndarray:
    """Darken locally thin dark structures without moving pixels.

    For cracks, black-hat morphology highlights pixels that are darker than
    their immediate neighborhood.  We subtract a bounded fraction of that cue
    from the luminance channel, making weak cracks easier for a detector to see
    while leaving most pavement texture unchanged.
    """

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    blackhat = cv2.morphologyEx(l, cv2.MORPH_BLACKHAT, k)
    blackhat = cv2.GaussianBlur(blackhat, (0, 0), sigmaX=1.1, sigmaY=1.1)
    l2 = np.clip(l.astype(np.float32) - strength * blackhat.astype(np.float32), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)


def edge_microcontrast(rgb: np.ndarray, strength: float) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 1, cv2.NORM_MINMAX)
    detail = (mag[..., None] * (rgb.astype(np.float32) - cv2.GaussianBlur(rgb, (0, 0), 1.6))).astype(np.float32)
    return np.clip(rgb.astype(np.float32) + strength * detail, 0, 255).astype(np.uint8)


def enhance_roi(rgb: np.ndarray, variant: str) -> np.ndarray:
    out = rgb.copy()
    h, _w = out.shape[:2]
    y0 = h // 2
    roi = out[y0:].copy()

    if variant == "nee_mild":
        roi = clahe_luma(roi, clip_limit=1.4, grid=8)
        roi = unsharp(roi, amount=0.18, sigma=1.2)
        roi = dark_line_boost(roi, strength=0.35, kernel=9)
    elif variant == "nee_balanced":
        roi = clahe_luma(roi, clip_limit=1.8, grid=8)
        roi = unsharp(roi, amount=0.28, sigma=1.2)
        roi = dark_line_boost(roi, strength=0.55, kernel=11)
        roi = edge_microcontrast(roi, strength=0.22)
    elif variant == "nee_strong":
        roi = clahe_luma(roi, clip_limit=2.4, grid=8)
        roi = unsharp(roi, amount=0.40, sigma=1.1)
        roi = dark_line_boost(roi, strength=0.85, kernel=13)
        roi = edge_microcontrast(roi, strength=0.35)
    elif variant == "nee_darkline":
        roi = dark_line_boost(roi, strength=1.05, kernel=15)
        roi = clahe_luma(roi, clip_limit=1.35, grid=12)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Blend back with the native image to guarantee bounded deviation.
    if variant == "nee_mild":
        alpha = 0.42
    elif variant == "nee_balanced":
        alpha = 0.55
    elif variant == "nee_strong":
        alpha = 0.68
    else:
        alpha = 0.55
    out[y0:] = cv2.addWeighted(rgb[y0:], 1.0 - alpha, roi, alpha, 0)
    return out


def copy_labels_and_write_yaml(source: Path, target: Path) -> None:
    labels_src = source / "labels" / "test"
    labels_dst = target / "labels" / "test"
    labels_dst.mkdir(parents=True, exist_ok=True)
    for label in labels_src.glob("*.txt"):
        shutil.copy2(label, labels_dst / label.name)

    yaml_data = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    yaml_data["path"] = str(target.resolve()).replace("\\", "/")
    yaml_data["train"] = "images/test"
    yaml_data["val"] = "images/test"
    yaml_data["test"] = "images/test"
    (target / "data.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    src_images = args.source / "images" / "test"
    image_paths = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    args.out.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        target = args.out / variant
        if target.exists():
            shutil.rmtree(target)
        image_dst = target / "images" / "test"
        image_dst.mkdir(parents=True, exist_ok=True)
        copy_labels_and_write_yaml(args.source, target)
        for image_path in image_paths:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            enhanced = enhance_roi(rgb, variant)
            out_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_dst / image_path.name), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        print(f"{variant}: wrote {len(image_paths)} images to {target}")


if __name__ == "__main__":
    main()
