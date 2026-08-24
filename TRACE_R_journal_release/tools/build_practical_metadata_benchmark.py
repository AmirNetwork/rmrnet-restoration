from __future__ import annotations

"""Build the practical-sensor version of the PCM/IVCNZ benchmark.

The degraded pixels and clean targets are hard-linked from the frozen
parameterized-v2 benchmark. Exact generator parameters are read only inside
this builder and written to a private audit JSONL. Public per-image sidecars
contain only noisy camera/IMU/vehicle measurements that could exist in a
deployed road-monitoring system. Train/validation calibration labels are
stored separately under ``private_calibration`` and are never copied into YOLO
inference datasets. Test data contain no calibration labels.

The split-to-telemetry assignment is fixed:
  train -> KITTI drives 0001/0002
  val   -> KITTI drive 0005
  test  -> KITTI drive 0011

The test split is refused until a frozen validation-selection manifest exists.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcadnet.practical_metadata import (
    PRACTICAL_SENSOR_DIM,
    SENSOR_PACKET_NAMES,
    TRAJECTORY_STEPS,
    observable_code_from_packet,
)
from rcadnet import code_from_metadata
import torch


SCENARIO_MAP = {
    "motion": "motion_horizontal_medium",
    "defocus": "defocus_medium",
    "lowlight": "lowlight_medium",
    "mixed": "mixed_motion_lowlight",
}
SPLIT_DRIVES = {
    "train": ("2011_09_26_drive_0001_sync", "2011_09_26_drive_0002_sync"),
    "val": ("2011_09_26_drive_0005_sync",),
    "test": ("2011_09_26_drive_0011_sync",),
}
OXTS_FIELDS = (
    "lat", "lon", "alt", "roll", "pitch", "yaw", "vn", "ve", "vf", "vl", "vu",
    "ax", "ay", "az", "af", "al", "au", "wx", "wy", "wz", "wf", "wl", "wu",
    "posacc", "velacc", "navstat", "numsats", "posmode", "velmode", "orimode",
)
CALIBRATED_GYRO_FULL_SCALE = 4.0


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    restoration_prefix: str
    yolo_prefix: str


SPECS = (
    DatasetSpec("IVCNZ", "pothole_restoration", "pothole_yolo_sequence_disjoint"),
    DatasetSpec("PCM", "pcm_restoration", "road_damage_pcm_yolo_sequence_disjoint"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", choices=["train", "val", "test"], required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--profile",
        choices=["legacy", "calibrated_v2"],
        default="legacy",
        help=(
            "Sensor simulation profile. calibrated_v2 uses exposure-synchronized "
            "camera rotation with calibrated IMU/timestamp noise and writes to "
            "separate dataset paths."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--test-freeze",
        type=Path,
        help="Required when --split test is requested; must record validation-only selection.",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=ROOT / "experiments" / "practical_metadata_build" / "build_audit.json",
    )
    return parser.parse_args()


def safe_reset(path: Path, force: bool) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"Refusing to reset outside workspace: {resolved}")
    if path.exists():
        if not force:
            raise FileExistsError(f"Generated output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_oxts(split: str) -> list[dict[str, Any]]:
    base = ROOT / "datasets" / "raw" / "kitti" / "2011_09_26"
    rows: list[dict[str, Any]] = []
    for drive in SPLIT_DRIVES[split]:
        for path in sorted((base / drive / "oxts" / "data").glob("*.txt")):
            values = [float(value) for value in path.read_text(encoding="utf-8").split()]
            row = {name: values[index] for index, name in enumerate(OXTS_FIELDS[: len(values)])}
            row.update(drive=drive, frame=path.stem)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No KITTI OXTS rows found for {split}")
    return rows


def scenario_target(hidden: dict[str, Any]) -> list[float]:
    # The recovered strong RMR-Net checkpoints were trained with the historical
    # conditioning coordinates. The separate observable_sensor_code remains
    # axial and is used only by the physical motion prior.
    return code_from_metadata(hidden, encoding="legacy").tolist()


def normalized_log(value: float, minimum: float, maximum: float) -> float:
    value = max(float(value), minimum)
    return float(np.clip((math.log(value) - math.log(minimum)) / (math.log(maximum) - math.log(minimum)), 0.0, 1.0))


def sensor_packet(
    hidden: dict[str, Any],
    oxts: dict[str, Any],
    rng: np.random.Generator,
    *,
    profile: str = "legacy",
) -> tuple[list[float], dict[str, Any]]:
    """Simulate observable sensor measurements without exposing renderer state."""

    family = str(hidden["scenario_family"])
    low = float(hidden.get("low_light_score", 0.0) or 0.0)
    noise = float(hidden.get("noise_score", 0.0) or 0.0)
    defocus = float(hidden.get("defocus_score", 0.0) or 0.0)
    length = float(hidden.get("blur_length_px", 0.0) or 0.0)
    angle = math.radians(float(hidden.get("blur_angle_deg", 0.0) or 0.0))

    if family in {"lowlight", "mixed"}:
        exposure_ms = float(np.clip(5.0 + 25.0 * low + rng.normal(0.0, 1.2), 4.0, 32.0))
        iso = float(np.clip(100.0 * (2.0 ** (5.2 * low + rng.normal(0.0, 0.18))), 100.0, 6400.0))
    else:
        exposure_ms = float(np.clip(rng.uniform(3.0, 10.0), 2.0, 12.0))
        iso = float(np.clip(100.0 * (2.0 ** rng.uniform(0.0, 1.2)), 100.0, 400.0))

    focal_mm = float(rng.uniform(28.0, 50.0))
    aperture = float(rng.uniform(2.8, 8.0))
    if profile == "calibrated_v2":
        imu_noise = float(rng.uniform(0.002, 0.010))
        timestamp_offset_ms = float(rng.normal(0.0, 0.10))
    else:
        imu_noise = float(rng.uniform(0.01, 0.05))
        timestamp_offset_ms = float(rng.normal(0.0, 0.75))
    camera_reliability = float(np.clip(0.96 - abs(rng.normal(0.0, 0.025)), 0.75, 1.0))
    imu_reliability = float(np.clip(0.94 - 1.8 * imu_noise - abs(timestamp_offset_ms) / 20.0, 0.55, 1.0))
    vehicle_reliability = float(np.clip(0.93 - float(oxts.get("velacc", 0.0)) / 10.0, 0.55, 1.0))

    # The hidden image-plane displacement is converted into a noisy temporal
    # IMU observation. The public model sees this sampled sensor trajectory,
    # never the renderer's blur length or angle.
    dx = length * math.cos(angle)
    dy = length * math.sin(angle)
    displacement_noise = (
        max(0.08, 0.012 * max(length, 1.0))
        if profile == "calibrated_v2"
        else max(0.35, 0.06 * max(length, 1.0))
    )
    observed_dx = dx + float(rng.normal(0.0, displacement_noise))
    observed_dy = dy + float(rng.normal(0.0, displacement_noise))
    exposure_norm = normalized_log(exposure_ms, 2.0, 40.0)
    integration_scale = 0.20 + 0.80 * exposure_norm
    oxts_weight = 0.0 if profile == "calibrated_v2" else 0.08
    gyro_full_scale = (
        CALIBRATED_GYRO_FULL_SCALE
        if profile == "calibrated_v2"
        else 1.0
    )
    base_gyro_x = (
        observed_dx / 25.0 / integration_scale
        + oxts_weight * float(oxts.get("wl", 0.0))
    ) / gyro_full_scale
    base_gyro_y = (
        observed_dy / 25.0 / integration_scale
        + oxts_weight * float(oxts.get("wf", 0.0))
    ) / gyro_full_scale
    base_gyro_z = (
        0.2 * float(oxts.get("wu", 0.0)) / gyro_full_scale
    )
    accel_x = float(np.clip(float(oxts.get("af", 0.0)) / 5.0, -1.0, 1.0))
    accel_y = float(np.clip(float(oxts.get("al", 0.0)) / 5.0, -1.0, 1.0))
    accel_z = float(np.clip(float(oxts.get("au", 0.0)) / 5.0, -1.0, 1.0))
    vibration = float(
        np.clip(
            math.sqrt(accel_x * accel_x + accel_y * accel_y)
            + 0.20 * length / 25.0,
            0.0,
            1.0,
        )
    )

    time_axis = np.linspace(-1.0, 1.0, TRAJECTORY_STEPS, dtype=np.float64)
    curvature = float(rng.normal(0.0, 0.12 + 0.18 * length / 25.0))
    harmonic = np.sin(np.pi * time_axis)
    high_frequency = np.sin(3.0 * np.pi * time_axis + float(rng.uniform(-np.pi, np.pi)))
    gyro_sequence = np.stack(
        [
            base_gyro_x + curvature * harmonic,
            base_gyro_y - 0.65 * curvature * harmonic,
            np.full_like(time_axis, base_gyro_z) + 0.25 * curvature * high_frequency,
        ],
        axis=1,
    )
    gyro_sequence += rng.normal(0.0, imu_noise, size=gyro_sequence.shape)

    timestamp_shift = timestamp_offset_ms / max(exposure_ms, 1e-3)
    shifted_axis = np.clip(time_axis - timestamp_shift, -1.0, 1.0)
    gyro_sequence = np.stack(
        [
            np.interp(shifted_axis, time_axis, gyro_sequence[:, axis])
            for axis in range(3)
        ],
        axis=1,
    )
    accel_sequence = np.stack(
        [
            np.full_like(time_axis, accel_x) + vibration * 0.35 * high_frequency,
            np.full_like(time_axis, accel_y) + vibration * 0.25 * harmonic,
            np.full_like(time_axis, accel_z) + vibration * 0.12 * high_frequency,
        ],
        axis=1,
    )
    accel_sequence += rng.normal(0.0, 0.6 * imu_noise, size=accel_sequence.shape)
    gyro_sequence = np.clip(gyro_sequence, -1.0, 1.0)
    accel_sequence = np.clip(accel_sequence, -1.0, 1.0)

    focus_proxy = float(np.clip(defocus + rng.normal(0.0, 0.06), 0.0, 1.0))
    autofocus_confidence = float(np.clip(1.0 - 0.75 * defocus + rng.normal(0.0, 0.04), 0.0, 1.0))
    speed = float(max(oxts.get("vf", 0.0), 0.0))
    yaw_rate = float(oxts.get("wu", 0.0))

    packet = [
        *[float(value) for value in gyro_sequence.reshape(-1)],
        *[float(value) for value in accel_sequence.reshape(-1)],
        exposure_norm,
        normalized_log(iso, 100.0, 6400.0),
        normalized_log(iso, 100.0, 6400.0),
        float(np.clip((focal_mm - 20.0) / 50.0, 0.0, 1.0)),
        float(np.clip((aperture - 1.4) / 10.0, 0.0, 1.0)),
        focus_proxy,
        autofocus_confidence,
        float(np.clip(rng.uniform(8.0, 22.0) / 30.0, 0.0, 1.0)),
        float(np.clip(speed / 30.0, 0.0, 1.0)),
        float(np.clip(yaw_rate / 0.15, -1.0, 1.0)),
        float(np.clip(timestamp_offset_ms / 5.0, -1.0, 1.0)),
        float(np.clip(imu_noise / 0.10, 0.0, 1.0)),
        camera_reliability,
        imu_reliability,
        vehicle_reliability,
        1.0,
    ]
    if len(packet) != PRACTICAL_SENSOR_DIM:
        raise AssertionError(len(packet))
    observation = {
        "source": (
            "exposure-synchronized calibrated camera/IMU simulation with "
            "split-disjoint KITTI vehicle context"
            if profile == "calibrated_v2"
            else "telemetry-realistic simulation calibrated with split-disjoint KITTI OXTS distributions"
        ),
        "sensor_profile": profile,
        "gyro_full_scale": gyro_full_scale,
        "kitti_drive": oxts["drive"],
        "kitti_frame": oxts["frame"],
        "exposure_ms": exposure_ms,
        "iso": iso,
        "focal_length_mm": focal_mm,
        "aperture_f_number": aperture,
        "vehicle_speed_mps": speed,
        "vehicle_yaw_rate_radps": yaw_rate,
        "timestamp_offset_ms": timestamp_offset_ms,
        "camera_reliability": camera_reliability,
        "imu_reliability": imu_reliability,
        "vehicle_reliability": vehicle_reliability,
        "trajectory_samples": TRAJECTORY_STEPS,
    }
    return packet, observation


def practical_sidecar(
    hidden: dict[str, Any],
    oxts: dict[str, Any],
    rng: np.random.Generator,
    *,
    profile: str = "legacy",
) -> dict[str, Any]:
    packet, observation = sensor_packet(
        hidden,
        oxts,
        rng,
        profile=profile,
    )
    packet_tensor = torch.tensor(packet, dtype=torch.float32)
    gyro_full_scale = (
        CALIBRATED_GYRO_FULL_SCALE
        if profile == "calibrated_v2"
        else 1.0
    )
    observable_code = observable_code_from_packet(
        packet_tensor,
        gyro_full_scale=gyro_full_scale,
    ).tolist()
    sidecar: dict[str, Any] = {
        "metadata_schema": (
            "rmrnet_practical_sensor_trajectory_calibrated_v2"
            if profile == "calibrated_v2"
            else "rmrnet_practical_sensor_trajectory"
        ),
        "metadata_scope": "observable camera/IMU/vehicle fields only",
        "practical_sensor_packet": packet,
        "sensor_observation": observation,
        "observable_sensor_code": observable_code,
        "scenario_family_is_model_input": False,
        "hidden_renderer_parameters_in_public_sidecar": False,
    }
    return sidecar


def private_calibration_target(hidden: dict[str, Any]) -> dict[str, Any]:
    """Return train-time labels that must never enter an inference sidecar."""

    return {
        "training_cause_target_code": scenario_target(hidden),
        "training_physical_target_code": code_from_metadata(
            hidden,
            encoding="axial_v2",
        ).tolist(),
        "training_target_is_model_input": False,
    }


def source_yolo_root(spec: DatasetSpec, family: str, split: str) -> Path:
    return ROOT / "datasets" / f"{spec.yolo_prefix}_parameterized_v2_{family}_{split}"


def output_yolo_root(
    spec: DatasetSpec,
    family: str,
    split: str,
    profile: str,
) -> Path:
    tag = (
        "practical_sensor_calibrated_v2"
        if profile == "calibrated_v2"
        else "practical_sensor"
    )
    return ROOT / "datasets" / f"{spec.yolo_prefix}_{tag}_{family}_{split}"


def build_one(
    spec: DatasetSpec,
    split: str,
    seed: int,
    force: bool,
    profile: str,
) -> dict[str, Any]:
    source_restoration = ROOT / "data" / f"{spec.restoration_prefix}_parameterized_v2_{split}"
    output_tag = (
        "practical_sensor_calibrated_v2"
        if profile == "calibrated_v2"
        else "practical_sensor"
    )
    output_restoration = (
        ROOT / "data" / f"{spec.restoration_prefix}_{output_tag}_{split}"
    )
    safe_reset(output_restoration, force)
    hidden_rows: list[dict[str, Any]] = []
    public_hashes: list[dict[str, str]] = []
    oxts_rows = load_oxts(split)
    rng = np.random.default_rng(seed + sum(ord(c) for c in f"{spec.key}:{split}"))

    for family, scenario in SCENARIO_MAP.items():
        source_scenario = source_restoration / "scenarios" / scenario
        target_scenario = output_restoration / "scenarios" / scenario
        inputs = sorted((source_scenario / "input").glob("*"))
        for index, input_path in enumerate(inputs):
            gt_path = source_scenario / "gt" / input_path.name
            hidden_path = source_scenario / "metadata" / f"{input_path.stem}.json"
            hidden = json.loads(hidden_path.read_text(encoding="utf-8"))
            oxts = oxts_rows[(index * 17 + int(rng.integers(0, len(oxts_rows)))) % len(oxts_rows)]
            sidecar = practical_sidecar(
                hidden,
                oxts,
                rng,
                profile=profile,
            )

            link_or_copy(input_path, target_scenario / "input" / input_path.name)
            link_or_copy(gt_path, target_scenario / "gt" / gt_path.name)
            metadata_path = target_scenario / "metadata" / f"{input_path.stem}.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            if split in {"train", "val"}:
                private_path = (
                    target_scenario
                    / "private_calibration"
                    / f"{input_path.stem}.json"
                )
                private_path.parent.mkdir(parents=True, exist_ok=True)
                private_path.write_text(
                    json.dumps(private_calibration_target(hidden), indent=2),
                    encoding="utf-8",
                )
            hidden_rows.append(
                {
                    "dataset": spec.key,
                    "split": split,
                    "scenario": family,
                    "stem": input_path.stem,
                    "renderer": hidden,
                }
            )
            public_hashes.append({"path": str(metadata_path.relative_to(ROOT)), "sha256": sha256(metadata_path)})

        source_yolo = source_yolo_root(spec, family, split)
        target_yolo = output_yolo_root(spec, family, split, profile)
        safe_reset(target_yolo, force)
        source_yaml = yaml.safe_load((source_yolo / "data.yaml").read_text(encoding="utf-8"))
        for kind in ("images", "labels"):
            source_folder = source_yolo / kind / split
            for path in sorted(source_folder.glob("*")):
                if path.is_file():
                    link_or_copy(path, target_yolo / kind / split / path.name)
        source_meta = output_restoration / "scenarios" / scenario / "metadata"
        target_meta = target_yolo / "metadata" / split
        for path in sorted(source_meta.glob("*.json")):
            link_or_copy(path, target_meta / path.name)
        output_yaml = {
            "path": str(target_yolo.resolve()),
            "train": f"images/{split}",
            "val": f"images/{split}",
            "test": f"images/{split}",
            "names": source_yaml["names"],
            "nc": source_yaml.get("nc", len(source_yaml["names"])),
        }
        (target_yolo / "data.yaml").write_text(
            yaml.safe_dump(output_yaml, sort_keys=False),
            encoding="utf-8",
        )

    audit_dir = output_restoration / "private_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    hidden_path = audit_dir / "hidden_renderer_parameters.jsonl"
    with hidden_path.open("w", encoding="utf-8") as handle:
        for row in hidden_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "dataset": spec.key,
        "split": split,
        "schema": (
            "rmrnet_practical_sensor_trajectory_calibrated_v2"
            if profile == "calibrated_v2"
            else "rmrnet_practical_sensor_trajectory"
        ),
        "sensor_profile": profile,
        "sensor_gyro_full_scale": (
            CALIBRATED_GYRO_FULL_SCALE
            if profile == "calibrated_v2"
            else 1.0
        ),
        "sensor_packet_dimension": PRACTICAL_SENSOR_DIM,
        "sensor_packet_names": list(SENSOR_PACKET_NAMES),
        "source_pixels": str(source_restoration.relative_to(ROOT)),
        "public_sidecars_exclude_hidden_renderer_parameters": True,
        "public_sidecars_exclude_training_targets": True,
        "private_training_target_present": split in {"train", "val"},
        "test_cause_target_present": False,
        "telemetry_drives": list(SPLIT_DRIVES[split]),
        "sample_count": len(hidden_rows),
        "public_metadata_hashes": public_hashes,
        "private_renderer_audit": str(hidden_path.relative_to(ROOT)),
    }
    (output_restoration / "practical_metadata_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    splits = list(dict.fromkeys(args.split))
    test_freeze_verified = False
    if "test" in splits:
        if args.test_freeze is None or not args.test_freeze.exists():
            raise RuntimeError("--split test requires an existing --test-freeze manifest")
        freeze = json.loads(args.test_freeze.read_text(encoding="utf-8"))
        if not freeze.get("selection_frozen_before_test", False):
            raise RuntimeError("Test-freeze manifest does not confirm pre-test selection")
        test_freeze_verified = True
    records = []
    for split in splits:
        for spec in SPECS:
            records.append(
                build_one(
                    spec,
                    split,
                    args.seed,
                    args.force,
                    args.profile,
                )
            )
    payload = {
        "protocol": "practical camera/IMU/vehicle metadata over parameterized road degradations",
        "seed": args.seed,
        "sensor_profile": args.profile,
        "splits": splits,
        "selection_frozen_before_test": (
            test_freeze_verified if "test" in splits else None
        ),
        "test_freeze": str(args.test_freeze) if args.test_freeze else None,
        "records": records,
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"audit": str(args.audit_out), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
