# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

"""Audit the practical sensor benchmark before training or test release."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FORBIDDEN_PUBLIC_KEYS = (
    "blur_length_px",
    "blur_angle_deg",
    "defocus_sigma",
    "noise_sigma",
    "renderer",
    "training_cause_target_code",
    "training_physical_target_code",
)
DATASETS = {
    "IVCNZ": ("pothole_restoration", 2924, 584),
    "PCM": ("pcm_restoration", 5608, 1412),
}
SCENARIOS = (
    "motion_horizontal_medium",
    "defocus_medium",
    "lowlight_medium",
    "mixed_motion_lowlight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["legacy", "calibrated_v2"],
        default="legacy",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "practical_metadata_build" / "audit_report.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset(
    name: str,
    prefix: str,
    expected_train: int,
    expected_val: int,
    profile: str,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    drive_sets: dict[str, set[str]] = {}
    tag = (
        "practical_sensor_calibrated_v2"
        if profile == "calibrated_v2"
        else "practical_sensor"
    )
    expected_schema = (
        "rmrnet_practical_sensor_trajectory_calibrated_v2"
        if profile == "calibrated_v2"
        else "rmrnet_practical_sensor_trajectory"
    )
    for split, expected in (("train", expected_train), ("val", expected_val)):
        root = ROOT / "data" / f"{prefix}_{tag}_{split}"
        if not root.exists():
            raise FileNotFoundError(root)
        manifest = json.loads(
            (root / "practical_metadata_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("sensor_profile") != profile:
            raise RuntimeError(
                f"{name}/{split}: expected profile {profile}, "
                f"got {manifest.get('sensor_profile')}"
            )
        if manifest.get("schema") != expected_schema:
            raise RuntimeError(
                f"{name}/{split}: expected schema {expected_schema}, "
                f"got {manifest.get('schema')}"
            )
        if int(manifest["sample_count"]) != expected:
            raise RuntimeError(
                f"{name}/{split}: expected {expected} samples, got {manifest['sample_count']}"
            )
        drive_sets[split] = set(manifest["telemetry_drives"])

        metadata_paths = sorted((root / "scenarios").glob("*/metadata/*.json"))
        if len(metadata_paths) != expected:
            raise RuntimeError(
                f"{name}/{split}: expected {expected} sidecars, got {len(metadata_paths)}"
            )
        packet_dimensions: set[int] = set()
        for path in metadata_paths:
            text = path.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_PUBLIC_KEYS:
                if f'"{forbidden}"' in text:
                    raise RuntimeError(f"Public leakage in {path}: {forbidden}")
            payload = json.loads(text)
            packet_dimensions.add(len(payload["practical_sensor_packet"]))

        private_paths = sorted(
            (root / "scenarios").glob("*/private_calibration/*.json")
        )
        if len(private_paths) != expected:
            raise RuntimeError(
                f"{name}/{split}: expected {expected} private calibration labels, "
                f"got {len(private_paths)}"
            )
        for path in private_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            required_targets = {
                "training_cause_target_code",
                "training_physical_target_code",
            }
            missing_targets = sorted(required_targets - set(payload))
            if missing_targets:
                raise RuntimeError(
                    f"Missing private supervision targets in {path}: "
                    f"{missing_targets}"
                )
            if payload.get("training_target_is_model_input") is not False:
                raise RuntimeError(f"Private target input contract missing: {path}")

        source_root = ROOT / "data" / f"{prefix}_parameterized_v2_{split}"
        source_sample = next(
            (source_root / "scenarios" / SCENARIOS[0] / "input").glob("*")
        )
        target_sample = (
            root / "scenarios" / SCENARIOS[0] / "input" / source_sample.name
        )
        same_content = sha256(source_sample) == sha256(target_sample)
        if not same_content:
            raise RuntimeError(f"Pixel mismatch: {source_sample} vs {target_sample}")
        try:
            same_file = os.path.samefile(source_sample, target_sample)
        except OSError:
            same_file = False

        records[split] = {
            "samples": len(metadata_paths),
            "private_calibration_labels": len(private_paths),
            "packet_dimensions": sorted(packet_dimensions),
            "telemetry_drives": sorted(drive_sets[split]),
            "sample_pixel_hash_equal": same_content,
            "sample_is_hardlink": same_file,
            "manifest_sha256": sha256(root / "practical_metadata_manifest.json"),
        }

    if drive_sets["train"] & drive_sets["val"]:
        raise RuntimeError(f"{name}: train/validation telemetry drives overlap")
    return records


def main() -> None:
    args = parse_args()
    tag = (
        "practical_sensor_calibrated_v2"
        if args.profile == "calibrated_v2"
        else "practical_sensor"
    )
    for prefix, _, _ in DATASETS.values():
        test_root = ROOT / "data" / f"{prefix}_{tag}_test"
        if test_root.exists():
            raise RuntimeError(f"Confirmatory test exists before freeze: {test_root}")

    datasets = {
        name: audit_dataset(
            name,
            prefix,
            expected_train,
            expected_val,
            args.profile,
        )
        for name, (prefix, expected_train, expected_val) in DATASETS.items()
    }
    report = {
        "status": "PASS",
        "sensor_profile": args.profile,
        "test_split_sealed": True,
        "public_sidecars_exclude_renderer_state": True,
        "public_sidecars_exclude_training_targets": True,
        "private_train_validation_calibration_targets_present": True,
        "training_target_is_not_model_input": True,
        "datasets": datasets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
