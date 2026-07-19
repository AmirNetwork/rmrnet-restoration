# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Run the fixed 30-epoch PCM RMR-Net checkpoint on ILX-RD46.

This audit replaces historical field outputs from detector-dominant pilot
checkpoints. It restores all 46 native 4752x3168 frames with the global-epoch
28 checkpoint selected on PCM validation mAP50, forms predeclared residual
strength views, and runs the unchanged YOLO26 coordinate detector. ILX labels
are neither read nor accepted by this script.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "native_real_yolo_newroad6" / "data.yaml"
NATIVE = ROOT / "experiments" / "gt46_yolo26_coordinate_revised" / "gt46_native_images"
RESTORED = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v47_rmr30_pcm_ep028_eval_sets"
PRED_ROOT = (
    ROOT
    / "experiments"
    / "gt46_yolo26_coordinate_revised"
    / "method_detections_yolo26rev_imgsz1280_conf010_clsD00-015_D10-025_D20-025_detdom2ep"
)
CHECKPOINT = ROOT / "runs" / "trc_final_rmrnet_pcm_30ep" / "rcadnet_epoch_028.pth"
DETECTOR = ROOT / "Yolo26_coordinate" / "Yolo26_coordinate_revised.py"
DETECTOR_WEIGHTS = ROOT / "Yolo26_coordinate" / "YOLO26s_RDD_FRDC_Distilled_v2.pt"
COORDS = ROOT / "geotagged" / "precise_cam1_coords.csv"
ETAS = (0.05, 0.10, 0.25, 0.50, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="CUDA index for YOLO; RMR-Net uses cuda when available.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def complete_images(folder: Path) -> bool:
    return folder.exists() and len(list(folder.glob("*.jpg"))) == 46


def restore_full(force: bool) -> None:
    blind = RESTORED / "rmr_blind" / "images" / "test"
    metadata = RESTORED / "rmr_metadata" / "images" / "test"
    if complete_images(blind) and complete_images(metadata) and not force:
        return
    run(
        [
            sys.executable,
            "tools/restore_native_yolo_split.py",
            "--data",
            str(DATA),
            "--split",
            "test",
            "--scenario",
            "native_real",
            "--out",
            str(RESTORED),
            "--models",
            "rmr_blind,rmr_metadata",
            "--device",
            "cuda",
            "--tile",
            "768",
            "--overlap",
            "96",
            "--jpeg-quality",
            "98",
            "--rcadnet-weights",
            str(CHECKPOINT),
        ]
    )


def eta_tag(source: str, eta: float) -> str:
    return f"rmr30_pcm_ep028_{source}_eta{eta:.2f}".replace(".", "p")


def blend_views(force: bool) -> dict[str, Path]:
    """Create I_eta = I_d + eta (I_r - I_d) at native resolution."""

    outputs: dict[str, Path] = {}
    for source in ("blind", "metadata"):
        full = RESTORED / f"rmr_{source}" / "images" / "test"
        for eta in ETAS:
            tag = eta_tag(source, eta)
            target_dir = RESTORED / tag / "images" / "test"
            outputs[tag] = target_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            for raw_path in sorted(NATIVE.glob("*.jpg")):
                target = target_dir / raw_path.name
                if target.exists() and not force:
                    continue
                restored_path = full / raw_path.name
                raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
                restored = cv2.imread(str(restored_path), cv2.IMREAD_COLOR)
                if raw is None or restored is None:
                    raise RuntimeError(f"Missing raw/restored pair for {raw_path.name}")
                if raw.shape != restored.shape:
                    raise AssertionError(f"Native dimensions changed for {raw_path.name}: {raw.shape} vs {restored.shape}")
                blended = cv2.addWeighted(raw, 1.0 - eta, restored, eta, 0.0)
                if not cv2.imwrite(str(target), blended, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                    raise RuntimeError(f"Could not write {target}")
    return outputs


def detector_complete(folder: Path) -> bool:
    return (folder / "detections.geojson").exists() and any(folder.glob("*.csv"))


def detect(tag: str, images: Path, *, device: str, force: bool) -> None:
    output = PRED_ROOT / tag
    if detector_complete(output) and not force:
        return
    output.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(DETECTOR),
            "--images",
            str(images),
            "--out",
            str(output),
            "--csv",
            str(COORDS),
            "--model",
            str(DETECTOR_WEIGHTS),
            "--device",
            device,
            "--workers",
            "1",
            "--imgsz",
            "1280",
            "--conf",
            "0.10",
            "--class_conf",
            "D00=0.15,D10=0.25,D20=0.25",
        ]
    )


def main() -> None:
    args = parse_args()
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    restore_full(args.force)
    views = blend_views(args.force)
    for tag, images in views.items():
        detect(tag, images, device=args.device, force=args.force)
    manifest = {
        "purpose": "ILX-RD46 field audit using the same fixed checkpoint as the 30-epoch PCM headline result",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_selection": "mean validation mAP50 across PCM degradations; global epoch 28; ILX labels not used",
        "etas": list(ETAS),
        "views": {tag: str(path.relative_to(ROOT)) for tag, path in views.items()},
        "prediction_root": str(PRED_ROOT.relative_to(ROOT)),
        "detector": str(DETECTOR_WEIGHTS.relative_to(ROOT)),
        "detector_protocol": {
            "imgsz": 1280,
            "conf": 0.10,
            "class_conf": "D00=0.15,D10=0.25,D20=0.25",
            "crop_or_mask": False,
            "detector_tiling": False,
            "detector_head": "end-to-end one-to-one; NMS-free direct inference",
        },
        "native_resolution_preserved": True,
        "labels_read": False,
    }
    RESTORED.mkdir(parents=True, exist_ok=True)
    (RESTORED / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
