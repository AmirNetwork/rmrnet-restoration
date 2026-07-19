# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Run direct single-view ILX-RD46 controls under one detector protocol.

The 46 native Sony frames are never synthetically degraded or resized on disk.
Restoration is tiled internally, written as lossless PNG at 4752x3168, and then
passed to the same full-image YOLO26-coordinate detector.  This script does not
select an RMR residual strength or fuse detector outputs; those decisions belong
to the separate chronological policy analysis.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "road-defect-seg-9junedata.coco-segmentation" / "train" / "_annotations.coco.json"
NATIVE = ROOT / "experiments" / "gt46_yolo26_coordinate_revised" / "gt46_native_images"
DETECTOR = ROOT / "Yolo26_coordinate" / "Yolo26_coordinate_revised.py"
DETECTOR_WEIGHTS = ROOT / "Yolo26_coordinate" / "YOLO26s_RDD_FRDC_Distilled_v2.pt"
COORDS = ROOT / "geotagged" / "precise_cam1_coords.csv"
DFPIR = ROOT / "weights" / "dfpir" / "DFPIR-5D-pn31.29-0.8889_pr37.62-0.9779_ph31.64-0.9794_pb28.82-0.8734_pl23.82-0.8428_avr30.64-0.9125.pth.tar"
DEMOE = ROOT / "weights" / "demoe" / "DeMoE.pt"
INSTRUCTIR_IMAGE = ROOT / "weights" / "instructir" / "im_instructir-7d.pt"
INSTRUCTIR_LM = ROOT / "weights" / "instructir" / "lm_instructir-7d.pt"
METHODS = ("nafnet", "dfpir", "demoe_auto", "demoe_scenario", "instructir_generic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--nafnet-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--atlas-max", type=int, default=46)
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


def safe_remove(path: Path, experiment: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(experiment.resolve()):
        raise RuntimeError(f"Refusing to remove output outside experiment: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def detector_complete(path: Path) -> bool:
    return (path / "detections.geojson").exists() and any(path.glob("*.csv"))


def detect(images: Path, output: Path, device: str, force: bool) -> None:
    if force:
        safe_remove(output, output.parents[1])
    if detector_complete(output):
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


def image_folder(root: Path, method: str) -> Path:
    return root / method / "images" / "test"


def main() -> None:
    args = parse_args()
    experiment = args.experiment.resolve()
    data = experiment / "native_exact" / "data.yaml"
    nafnet = args.nafnet_checkpoint.resolve()
    required = (data, nafnet, ANNOTATIONS, NATIVE, DETECTOR, DETECTOR_WEIGHTS, COORDS, DFPIR, DEMOE, INSTRUCTIR_IMAGE, INSTRUCTIR_LM)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    restored = experiment / "baseline_restored"
    if args.force:
        safe_remove(restored, experiment)
    command = [
        sys.executable,
        "tools/restore_native_yolo_split.py",
        "--data",
        str(data),
        "--split",
        "test",
        "--scenario",
        "native_real",
        "--out",
        str(restored),
        "--models",
        ",".join(METHODS),
        "--device",
        "cuda" if args.device != "cpu" else "cpu",
        "--tile",
        "768",
        "--overlap",
        "96",
        "--output-format",
        "png",
        "--nafnet-weights",
        str(nafnet),
        "--dfpir-weights",
        str(DFPIR),
        "--demoe-weights",
        str(DEMOE),
        "--instructir-image-weights",
        str(INSTRUCTIR_IMAGE),
        "--instructir-lm-weights",
        str(INSTRUCTIR_LM),
    ]
    if not args.force:
        command.append("--skip-existing")
    run(command)

    prediction_root = experiment / "detections"
    for method in METHODS:
        images = image_folder(restored, method)
        if len(list(images.glob("*.png"))) != 46:
            raise RuntimeError(f"{method} did not produce 46 lossless native-resolution images")
        detect(images, prediction_root / method, args.device, args.force)

    named_predictions: list[tuple[str, Path]] = []
    for path in sorted(prediction_root.iterdir()):
        if path.is_dir() and detector_complete(path):
            named_predictions.append((path.name, path))
    required_names = {"raw", *METHODS}
    missing = sorted(required_names - {name for name, _ in named_predictions})
    if missing:
        raise RuntimeError(f"Missing direct-view detector outputs: {missing}. Run run_ilx46_current_checkpoint_audit.py first.")

    evaluation = experiment / "direct_single_view_evaluation"
    if args.force:
        safe_remove(evaluation, experiment)
    cmd = [
        sys.executable,
        "tools/eval_yolo26_coordinate_gt46.py",
        "--annotations",
        str(ANNOTATIONS),
        "--images",
        str(NATIVE),
        "--out",
        str(evaluation),
        "--atlas-max",
        str(args.atlas_max),
        "--pred-root",
        *[str(path) for _name, path in named_predictions],
        "--pred-name",
        *[name for name, _path in named_predictions],
    ]
    run(cmd)

    manifest = {
        "scope": "direct single-view restoration; no detector fusion and no label-selected residual strength",
        "images": 46,
        "native_resolution": [4752, 3168],
        "synthetic_degradation_added": False,
        "restored_encoding": "lossless PNG",
        "detector_internal_imgsz": 1280,
        "detector_crop_mask_or_tiling": False,
        "detector_checkpoint": str(DETECTOR_WEIGHTS.relative_to(ROOT)),
        "detector_sha256": sha256(DETECTOR_WEIGHTS),
        "nafnet_checkpoint": str(nafnet),
        "nafnet_sha256": sha256(nafnet),
        "released_checkpoint_hashes": {
            "dfpir": sha256(DFPIR),
            "demoe": sha256(DEMOE),
            "instructir_image": sha256(INSTRUCTIR_IMAGE),
            "instructir_language": sha256(INSTRUCTIR_LM),
        },
        "methods": [name for name, _path in named_predictions],
        "evaluation": str(evaluation.relative_to(ROOT)),
    }
    (experiment / "direct_single_view_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
