from __future__ import annotations

"""Build an annotation-blind Sony field adaptation set for RMR-Net.

The source is the full native-resolution Cam1 sequence.  Every manually
annotated CRID-46 frame is excluded before split construction.  Validation
uses temporally held-out blocks with guard gaps; it is never used as training
data.  Patches are sampled directly from native pixels (no image resizing).

Native identity pairs teach conservative pass-through behavior under real
EXIF/INS/SBG packets.  Four synthetic corruptions provide paired restoration
targets in the same camera domain.  Their public metadata combines the real
capture context with an exposure-synchronized simulated sensor perturbation.
Exact renderer parameters are stored only in ``private_calibration`` as
training targets and are never model inputs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcadnet.practical_metadata import (  # noqa: E402
    ACCEL_END,
    CONTEXT_START,
    GYRO_END,
    PRACTICAL_SENSOR_DIM,
    SENSOR_PACKET_NAMES,
    observable_code_from_packet,
    sensor_packet_from_mapping,
)
from rcadnet.spatial_physics import encode_log_exposure_ms  # noqa: E402
from tools.build_parameterized_metadata_benchmark import degrade, sample_rng  # noqa: E402
from tools.build_practical_metadata_benchmark import (  # noqa: E402
    practical_sidecar,
    private_calibration_target,
)


SCENARIOS: dict[str, str | None] = {
    "native_identity": None,
    "motion_horizontal_medium": "motion",
    "defocus_medium": "defocus",
    "lowlight_medium": "lowlight",
    "mixed_motion_lowlight": "mixed",
}
CAPTURE_RE = re.compile(r"_capt(?P<capture>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "geotagged/cam1")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "experiments/geotagged_cam1_hybrid_sbg_ins_metadata_20260811/metadata",
    )
    parser.add_argument(
        "--annotated-images",
        type=Path,
        default=ROOT / "datasets/gt46_sony_classbalanced_20260801/images",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "data/crid_unlabelled_field_adaptation_v1",
    )
    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument("--patches-per-frame", type=int, default=1)
    parser.add_argument("--train-frames", type=int, default=960)
    parser.add_argument("--val-frames", type=int, default=192)
    parser.add_argument("--validation-block-width", type=int, default=180)
    parser.add_argument("--guard-frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_reset(path: Path, force: bool) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"Refusing to reset outside workspace: {resolved}")
    if resolved.exists():
        if not force:
            raise FileExistsError(f"Generated path exists; pass --force: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def capture_id(stem: str) -> int:
    match = CAPTURE_RE.search(stem)
    if match is None:
        raise ValueError(f"Could not parse capture id from {stem}")
    return int(match.group("capture"))


def evenly_sample(rows: list[Path], count: int) -> list[Path]:
    if count <= 0 or count >= len(rows):
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    return [rows[int(index)] for index in sorted(set(indices.tolist()))]


def packet_values(metadata: dict[str, Any]) -> np.ndarray:
    packet = sensor_packet_from_mapping(metadata).detach().cpu().numpy().astype(np.float32)
    if packet.shape != (PRACTICAL_SENSOR_DIM,):
        raise ValueError(packet.shape)
    return packet


def motion_record(metadata: dict[str, Any], stem: str) -> dict[str, Any]:
    packet = packet_values(metadata)
    return {
        "drive": "CRID-Sony-ILX-LR1-unlabelled",
        "frame": stem,
        "vf": float(metadata.get("speed_mps", 30.0 * packet[CONTEXT_START + 8]) or 0.0),
        "wu": float(metadata.get("raw_oxts_yaw_rate_radps", 0.15 * packet[CONTEXT_START + 9]) or 0.0),
        "af": float(metadata.get("raw_oxts_forward_accel_mps2", 0.0) or 0.0),
        "al": float(metadata.get("raw_oxts_lateral_accel_mps2", 0.0) or 0.0),
        "au": 0.0,
        "wl": 0.0,
        "wf": 0.0,
        "velacc": 0.0,
    }


def compact_native_sidecar(metadata: dict[str, Any]) -> dict[str, Any]:
    packet = packet_values(metadata)
    provenance = metadata.get("practical_sensor_provenance", {})
    exposure_ms = float(provenance.get("exposure_ms", 0.25) or 0.25)
    # Physics packet v2: exposure is encoded over 0.05--40 ms so the native
    # 0.25 ms Sony setting remains observable instead of collapsing to zero.
    packet[CONTEXT_START] = encode_log_exposure_ms(exposure_ms)
    return {
        "metadata_schema": "rmrnet_real_capture_packet_v1",
        "metadata_scope": "observable Sony camera/IMU/vehicle fields only",
        "metadata_source": metadata.get("metadata_source", "unknown"),
        "practical_sensor_packet": {
            name: float(value) for name, value in zip(SENSOR_PACKET_NAMES, packet)
        },
        "practical_sensor_provenance": provenance,
        "exposure_encoding": "log_0.05ms_40ms_v2",
        "scenario_family_is_model_input": False,
        "hidden_renderer_parameters_in_public_sidecar": False,
    }


def combined_corruption_sidecar(
    hidden: dict[str, Any],
    native_metadata: dict[str, Any],
    rng: np.random.Generator,
    stem: str,
) -> dict[str, Any]:
    synthetic = practical_sidecar(
        hidden,
        motion_record(native_metadata, stem),
        rng,
        profile="calibrated_v2",
    )
    real_packet = packet_values(native_metadata)
    synthetic_packet = packet_values(synthetic)
    packet = synthetic_packet.copy()
    synthetic_observation = synthetic.get("sensor_observation", {})
    packet[CONTEXT_START] = encode_log_exposure_ms(
        float(synthetic_observation.get("exposure_ms", 2.0) or 2.0)
    )

    # Exposure-synchronized simulated angular motion is added to the measured
    # trajectory.  Acceleration remains primarily measured and receives only
    # the simulated high-frequency component.  This preserves realistic sensor
    # statistics while retaining a known training-only corruption target.
    packet[:GYRO_END] = np.clip(
        synthetic_packet[:GYRO_END] + real_packet[:GYRO_END], -1.0, 1.0
    )
    packet[GYRO_END:ACCEL_END] = np.clip(
        0.75 * real_packet[GYRO_END:ACCEL_END]
        + 0.25 * synthetic_packet[GYRO_END:ACCEL_END],
        -1.0,
        1.0,
    )
    # The controlled renderer determines exposure, focal length, aperture and
    # focus evidence. Retaining those observable settings is necessary for the
    # calibrated spatial field. Native CRID keeps unsupported lens fields at
    # zero and therefore does not claim optical calibration that was not logged.
    packet[CONTEXT_START + 8 : CONTEXT_START + 10] = real_packet[
        CONTEXT_START + 8 : CONTEXT_START + 10
    ]
    packet[CONTEXT_START + 12] = min(
        real_packet[CONTEXT_START + 12], synthetic_packet[CONTEXT_START + 12]
    )
    packet[CONTEXT_START + 13] = min(
        real_packet[CONTEXT_START + 13], synthetic_packet[CONTEXT_START + 13]
    )
    packet[CONTEXT_START + 14] = real_packet[CONTEXT_START + 14]
    packet[CONTEXT_START + 15] = float(
        max(packet[CONTEXT_START + 12 : CONTEXT_START + 15]) > 0.0
    )
    packet = np.clip(packet, -1.0, 1.0)
    observable = observable_code_from_packet(
        torch.from_numpy(packet), gyro_full_scale=4.0
    ).tolist()
    return {
        "metadata_schema": "rmrnet_real_context_simulated_capture_perturbation_v1",
        "metadata_scope": "observable camera/IMU/vehicle fields only",
        "metadata_source": "real_capture_context_plus_training_simulated_perturbation",
        "practical_sensor_packet": {
            name: float(value) for name, value in zip(SENSOR_PACKET_NAMES, packet)
        },
        "observable_sensor_code": observable,
        "sensor_observation": synthetic_observation,
        "exposure_encoding": "log_0.05ms_40ms_v2",
        "native_sensor_provenance": native_metadata.get("practical_sensor_provenance", {}),
        "scenario_family_is_model_input": False,
        "hidden_renderer_parameters_in_public_sidecar": False,
    }


def identity_private_target() -> dict[str, Any]:
    return {
        "training_cause_target_code": [0.0] * 8,
        "training_target_is_model_input": False,
        "physical_target_available": False,
    }


def select_frames(args: argparse.Namespace) -> tuple[list[Path], list[Path], dict[str, Any]]:
    annotated = {path.stem for path in args.annotated_images.glob("*.jpg")}
    if len(annotated) != 46:
        raise RuntimeError(f"Expected 46 annotated CRID stems, found {len(annotated)}")
    images = sorted(args.images.glob("*.jpg"), key=lambda path: capture_id(path.stem))
    usable = [
        path
        for path in images
        if path.stem not in annotated and (args.metadata / f"{path.stem}.json").exists()
    ]
    if len(usable) < args.train_frames + args.val_frames:
        raise RuntimeError(f"Only {len(usable)} unlabelled frames are usable")

    complete_sensor_ids = []
    all_ids = [capture_id(path.stem) for path in usable]
    for path in usable:
        metadata = json.loads((args.metadata / f"{path.stem}.json").read_text(encoding="utf-8"))
        if metadata.get("metadata_source") in {
            "real_sony_exif_plus_direct_sbg",
            "real_sony_exif_plus_complete_sbg_ins",
        }:
            complete_sensor_ids.append(capture_id(path.stem))
    centres = []
    if complete_sensor_ids:
        centres.append(int(np.median(complete_sensor_ids)))
    centres.append(int(np.quantile(all_ids, 0.72)))
    half_width = max(1, args.validation_block_width // 2)
    validation_ranges = [(centre - half_width, centre + half_width) for centre in centres]

    val_candidates = [
        path
        for path in usable
        if any(low <= capture_id(path.stem) <= high for low, high in validation_ranges)
    ]
    validation = evenly_sample(val_candidates, args.val_frames)
    guarded_ranges = [
        (low - args.guard_frames, high + args.guard_frames)
        for low, high in validation_ranges
    ]
    train_candidates = [
        path
        for path in usable
        if not any(low <= capture_id(path.stem) <= high for low, high in guarded_ranges)
    ]
    training = evenly_sample(train_candidates, args.train_frames)
    overlap = {path.stem for path in training} & {path.stem for path in validation}
    annotation_overlap = ({path.stem for path in training + validation} & annotated)
    if overlap or annotation_overlap:
        raise RuntimeError(
            f"Split leakage: train/val={sorted(overlap)}, annotated={sorted(annotation_overlap)}"
        )
    audit = {
        "source_images": len(images),
        "annotated_stems_excluded_before_sampling": len(annotated),
        "usable_unlabelled_frames": len(usable),
        "complete_sensor_packet_usable_frames": len(complete_sensor_ids),
        "validation_ranges_capture_id": validation_ranges,
        "guarded_validation_ranges_capture_id": guarded_ranges,
        "training_frames": len(training),
        "validation_frames": len(validation),
        "train_validation_overlap": sorted(overlap),
        "annotated_overlap": sorted(annotation_overlap),
        "annotated_stems_sha256": hashlib.sha256(
            "\n".join(sorted(annotated)).encode("utf-8")
        ).hexdigest(),
    }
    return training, validation, audit


def patch_coordinates(
    image: np.ndarray, patch_size: int, rng: np.random.Generator
) -> tuple[int, int]:
    height, width = image.shape[:2]
    if height < patch_size or width < patch_size:
        raise ValueError(f"Image {width}x{height} is smaller than patch {patch_size}")
    # The lower 62% contains the surveyed road surface while retaining enough
    # context for lane markings and curb-adjacent defects.
    min_y = min(max(int(round(0.38 * height)), 0), height - patch_size)
    top = int(rng.integers(min_y, height - patch_size + 1))
    left = int(rng.integers(0, width - patch_size + 1))
    return top, left


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Could not write {path}")


def build_split(
    split: str,
    frames: list[Path],
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    def process_frame(frame: Path) -> tuple[list[dict[str, Any]], tuple[str, str]]:
        frame_records: list[dict[str, Any]] = []
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {frame}")
        metadata_path = args.metadata / f"{frame.stem}.json"
        native_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_hash = (frame.name, sha256(metadata_path))
        for patch_index in range(args.patches_per_frame):
            patch_rng = sample_rng(args.seed, split, frame.stem, str(patch_index), "crop")
            top, left = patch_coordinates(image, args.patch_size, patch_rng)
            clean = image[top : top + args.patch_size, left : left + args.patch_size]
            stem = f"{frame.stem}_p{patch_index:02d}"
            filename = f"{stem}.png"
            clean_master = output / "clean_cache" / filename
            write_png(clean_master, clean)

            for scenario, family in SCENARIOS.items():
                scenario_root = output / "scenarios" / scenario
                input_path = scenario_root / "input" / filename
                target_path = scenario_root / "gt" / filename
                metadata_out = scenario_root / "metadata" / f"{stem}.json"
                private_out = scenario_root / "private_calibration" / f"{stem}.json"
                hardlink_or_copy(clean_master, target_path)
                if family is None:
                    hardlink_or_copy(clean_master, input_path)
                    public = compact_native_sidecar(native_metadata)
                    private = identity_private_target()
                else:
                    rng = sample_rng(args.seed, split, frame.stem, str(patch_index), family)
                    corrupted, hidden = degrade(clean, family, rng)
                    write_png(input_path, corrupted)
                    public = combined_corruption_sidecar(
                        hidden, native_metadata, rng, frame.stem
                    )
                    private = private_calibration_target(hidden)
                metadata_out.parent.mkdir(parents=True, exist_ok=True)
                private_out.parent.mkdir(parents=True, exist_ok=True)
                metadata_out.write_text(json.dumps(public, indent=2), encoding="utf-8")
                private_out.write_text(json.dumps(private, indent=2), encoding="utf-8")
                frame_records.append(
                    {
                        "split": split,
                        "source_stem": frame.stem,
                        "patch": patch_index,
                        "crop_xywh": [left, top, args.patch_size, args.patch_size],
                        "scenario": scenario,
                        "metadata_source": native_metadata.get("metadata_source", "unknown"),
                    }
                )
        return frame_records, source_hash

    records: list[dict[str, Any]] = []
    source_hashes: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for frame_records, source_hash in executor.map(process_frame, frames):
            records.extend(frame_records)
            source_hashes.append(source_hash)
    manifest = {
        "split": split,
        "unique_source_frames": len(frames),
        "patches_per_frame": args.patches_per_frame,
        "paired_samples": len(records),
        "source_metadata_manifest_sha256": hashlib.sha256(
            "\n".join(f"{name},{digest}" for name, digest in source_hashes).encode("utf-8")
        ).hexdigest(),
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    args.images = args.images.resolve()
    args.metadata = args.metadata.resolve()
    args.annotated_images = args.annotated_images.resolve()
    if args.patch_size <= 0 or args.patches_per_frame <= 0:
        raise ValueError("patch size and patches per frame must be positive")
    for required in (args.images, args.metadata, args.annotated_images):
        if not required.exists():
            raise FileNotFoundError(required)

    train_frames, val_frames, split_audit = select_frames(args)
    outputs = {
        split: Path(f"{args.output_prefix}_{split}").resolve()
        for split in ("train", "val")
    }
    for output in outputs.values():
        safe_reset(output, args.force)

    manifests = {
        "train": build_split("train", train_frames, outputs["train"], args),
        "val": build_split("val", val_frames, outputs["val"], args),
    }
    audit = {
        "protocol": "annotation-blind native Sony field adaptation",
        "seed": args.seed,
        "native_resolution": [4752, 3168],
        "resizing": False,
        "patch_size": args.patch_size,
        "preparation_workers": args.workers,
        "public_metadata": "observable EXIF/INS/SBG/vehicle packet only",
        "private_targets_are_model_inputs": False,
        "test_annotations_used_for_training_or_selection": False,
        "split": split_audit,
        "manifests": {
            key: {
                "root": str(outputs[key]),
                "unique_source_frames": value["unique_source_frames"],
                "paired_samples": value["paired_samples"],
                "manifest_sha256": sha256(outputs[key] / "manifest.json"),
            }
            for key, value in manifests.items()
        },
    }
    for output in outputs.values():
        (output / "adaptation_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
