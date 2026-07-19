# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Generate label-free ILX field controls requested by the system-level audit.

Two views are produced without changing the native 4752x3168 dimensions:

1. a weak gamma correction (gamma=0.90), representing inexpensive
   photometric enhancement rather than learned restoration;
2. an ordinary raw-image detector pass with Ultralytics test-time
   augmentation enabled.

Ground-truth annotations are deliberately not accepted by this script. The
resulting prediction folders are evaluated later with exactly the same fusion,
NMS, prediction budget, and chronological holdout as the learned restorers.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP = ROOT / "experiments" / "major_revision_ilx46_sequence_disjoint_20260716"
DEFAULT_DETECTOR = ROOT / "Yolo26_coordinate" / "Yolo26_coordinate_revised.py"
DEFAULT_WEIGHTS = ROOT / "Yolo26_coordinate" / "YOLO26s_RDD_FRDC_Distilled_v2.pt"
DEFAULT_COORDS = ROOT / "geotagged" / "precise_cam1_coords.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXP)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--class-conf", default="D00=0.15,D10=0.25,D20=0.25")
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def make_gamma_images(images: Path, control_images: Path, force: bool) -> None:
    control_images.mkdir(parents=True, exist_ok=True)
    lut = np.clip(255.0 * (np.arange(256, dtype=np.float32) / 255.0) ** 0.90, 0, 255).astype(np.uint8)
    sources = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not sources:
        raise RuntimeError(f"No native images found in {images}")
    for source in sources:
        # PNG avoids introducing a second JPEG encode into this control arm.
        target = control_images / f"{source.stem}.png"
        if target.exists() and not force:
            continue
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {source}")
        corrected = cv2.LUT(image, lut)
        if corrected.shape != image.shape:
            raise AssertionError("Photometric control changed image dimensions")
        if not cv2.imwrite(str(target), corrected, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise RuntimeError(f"Could not write {target}")


def complete(path: Path) -> bool:
    return (path / "detections.geojson").exists() and any(path.glob("*.csv"))


def run_detector(
    images: Path,
    output: Path,
    *,
    detector: Path,
    weights: Path,
    coords: Path,
    imgsz: int,
    conf: float,
    class_conf: str,
    augment: bool,
    device: str,
    force: bool,
) -> None:
    if complete(output) and not force:
        print(f"[SKIP] {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(detector),
        "--images",
        str(images),
        "--out",
        str(output),
        "--csv",
        str(coords),
        "--model",
        str(weights),
        "--device",
        device,
        "--workers",
        "1",
        "--imgsz",
        str(imgsz),
        "--conf",
        str(conf),
        "--class_conf",
        class_conf,
    ]
    if augment:
        command.append("--augment")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    experiment = args.experiment.resolve()
    images = (args.images or experiment / "gt46_native_images").resolve()
    prediction_root = (args.prediction_root or experiment / "field_fairness_predictions").resolve()
    control_images = experiment / "field_fairness_controls" / "weak_gamma090"
    for required in (images, args.detector, args.weights, args.coords):
        if not required.exists():
            raise FileNotFoundError(required)
    make_gamma_images(images, control_images, args.force)
    common = dict(
        detector=args.detector.resolve(),
        weights=args.weights.resolve(),
        coords=args.coords.resolve(),
        imgsz=args.imgsz,
        conf=args.conf,
        class_conf=args.class_conf,
        device=args.device,
        force=args.force,
    )
    run_detector(control_images, prediction_root / "weak_gamma090", augment=False, **common)
    run_detector(images, prediction_root / "raw_tta", augment=True, **common)
    print(f"[OK] controls written under {prediction_root}")


if __name__ == "__main__":
    main()
