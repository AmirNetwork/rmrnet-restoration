from __future__ import annotations

"""Build CRID practical metadata from the directly measured 200 Hz SBG log.

The builder is annotation-blind. It copies camera and vehicle context from an
existing CRID metadata sidecar, but replaces the inertial trajectory with the
direct accelerometer and angular-rate channels in ``ac1.csv``. Frames outside
the raw SBG time interval can either receive an explicitly unavailable IMU
modality (the strict missing-modality control) or retain the independently
synchronized attitude/velocity-derived INS trajectory from the source
sidecar. Direct channels are never extrapolated outside their recording.

The released 82-value packet is

    [11 x 3 angular-rate, 11 x 3 acceleration, 16 context values].

The 11 direct samples span a 50 ms capture-centred window. This is temporal
context, not a claim that a 200 Hz sensor measured eleven samples during the
Sony camera's 0.25 ms exposure.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcadnet.practical_metadata import (  # noqa: E402
    ACCEL_END,
    CONTEXT_START,
    GYRO_END,
    PRACTICAL_SENSOR_DIM,
    SENSOR_PACKET_NAMES,
    TRAJECTORY_STEPS,
)
from tools.prepare_sony_ins_metadata import exif_utc_timestamp  # noqa: E402


SBG_COLUMNS = (
    "UTC Date",
    "UTC Time",
    "Roll",
    "Pitch",
    "Accelerometer X",
    "Accelerometer Y",
    "Accelerometer Z",
    "Angular Rate X",
    "Angular Rate Y",
    "Angular Rate Z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbg", type=Path, default=ROOT / "ac1.csv")
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=(
            ROOT
            / "experiments"
            / "sony_ilx46_ins_synchronized_20260801"
            / "metadata"
            / "test"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--half-window-ms", type=float, default=25.0)
    parser.add_argument(
        "--gyro-full-scale-radps",
        type=float,
        default=4.0,
        help="Absolute full scale used to normalize direct angular rate.",
    )
    parser.add_argument(
        "--accel-full-scale-mps2",
        type=float,
        default=19.62,
        help="Absolute full scale used to normalize direct acceleration (2 g).",
    )
    parser.add_argument(
        "--accel-mode",
        choices=("gravity-compensated", "raw"),
        default="gravity-compensated",
        help=(
            "Use attitude-compensated dynamic acceleration for restoration "
            "conditioning, or retain the raw specific-force channels as a control."
        ),
    )
    parser.add_argument(
        "--outside-sbg-policy",
        choices=("unavailable", "ins-derived"),
        default="unavailable",
        help=(
            "How to represent frames outside the direct SBG interval. "
            "'unavailable' creates the strict missing-IMU control; "
            "'ins-derived' retains the synchronized attitude/velocity-derived "
            "trajectory already present in --source-metadata."
        ),
    )
    parser.add_argument(
        "--derived-imu-reliability-cap",
        type=float,
        default=0.65,
        help=(
            "Maximum IMU reliability assigned to attitude/velocity-derived "
            "trajectories; direct SBG measurements are not capped."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sbg(
    path: Path,
    *,
    gravity_compensate: bool = True,
) -> dict[str, np.ndarray]:
    """Read the two-header-row SBG Center export without positional guesses."""

    frame = pd.read_csv(
        path,
        sep="\t",
        skiprows=[0, 2],
        encoding="utf-8-sig",
        low_memory=False,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = sorted(set(SBG_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"SBG export is missing columns: {missing}")

    stamp = pd.to_datetime(
        frame["UTC Date"].astype(str).str.strip()
        + " "
        + frame["UTC Time"].astype(str).str.strip(),
        utc=True,
        errors="raise",
    )
    time = stamp.map(lambda value: value.timestamp()).to_numpy(dtype=np.float64)
    if len(time) < TRAJECTORY_STEPS or np.any(np.diff(time) <= 0.0):
        raise ValueError("SBG timestamps must be strictly increasing")

    def values(names: tuple[str, str, str]) -> np.ndarray:
        return np.stack(
            [pd.to_numeric(frame[name], errors="raise").to_numpy(float) for name in names],
            axis=1,
        )

    # The SBG vehicle/body frame is x-forward, y-right, z-down. RMR-Net's
    # camera-like packet is x-image-right, y-image-down, z-optical-forward.
    # Therefore (camera x, y, z) = (body y, body z, body x). This fixed mounting
    # convention is recorded in every output sidecar.
    gyro_body_deg = values(("Angular Rate X", "Angular Rate Y", "Angular Rate Z"))
    accel_body = values(("Accelerometer X", "Accelerometer Y", "Accelerometer Z"))
    if gravity_compensate:
        roll = np.deg2rad(pd.to_numeric(frame["Roll"], errors="raise").to_numpy(float))
        pitch = np.deg2rad(pd.to_numeric(frame["Pitch"], errors="raise").to_numpy(float))
        gravity = 9.80665
        # SBG body axes are x-forward, y-right, z-down. The stationary
        # specific-force component observed in ac1.csv is therefore
        # [g sin(pitch), -g sin(roll) cos(pitch),
        #  -g cos(roll) cos(pitch)]. Removing it isolates translational
        # acceleration and vibration, which are the quantities relevant to
        # exposure-time image motion.
        gravity_body = np.stack(
            [
                gravity * np.sin(pitch),
                -gravity * np.sin(roll) * np.cos(pitch),
                -gravity * np.cos(roll) * np.cos(pitch),
            ],
            axis=1,
        )
        accel_body = accel_body - gravity_body
    gyro_body = np.deg2rad(gyro_body_deg)
    gyro_camera = gyro_body[:, [1, 2, 0]]
    accel_camera = accel_body[:, [1, 2, 0]]
    return {
        "time": time,
        "gyro": gyro_camera,
        "accel": accel_camera,
        "accel_mode": "gravity-compensated" if gravity_compensate else "raw",
    }


def packet_values(raw_packet: Any) -> list[float]:
    if isinstance(raw_packet, dict):
        return [float(raw_packet.get(name, 0.0)) for name in SENSOR_PACKET_NAMES]
    values = [float(value) for value in raw_packet]
    if len(values) != PRACTICAL_SENSOR_DIM:
        raise ValueError(f"Expected {PRACTICAL_SENSOR_DIM} packet values, got {len(values)}")
    return values


def interpolate_trajectory(
    sbg: dict[str, np.ndarray],
    capture_seconds: float,
    half_window_seconds: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    support = np.linspace(
        capture_seconds - half_window_seconds,
        capture_seconds + half_window_seconds,
        TRAJECTORY_STEPS,
    )
    time = sbg["time"]
    if support[0] < time[0] or support[-1] > time[-1]:
        raise ValueError("Capture-centred window lies outside the SBG recording")
    gyro = np.stack(
        [np.interp(support, time, sbg["gyro"][:, axis]) for axis in range(3)],
        axis=1,
    )
    accel = np.stack(
        [np.interp(support, time, sbg["accel"][:, axis]) for axis in range(3)],
        axis=1,
    )
    nearest = float(time[np.argmin(np.abs(time - capture_seconds))] - capture_seconds)
    return gyro, accel, nearest * 1000.0


def build_packet(
    metadata: dict[str, Any],
    sbg: dict[str, np.ndarray],
    *,
    half_window_seconds: float,
    gyro_full_scale: float,
    accel_full_scale: float,
    outside_sbg_policy: str = "unavailable",
    derived_imu_reliability_cap: float = 0.65,
) -> tuple[list[float], dict[str, Any], bool]:
    values = packet_values(metadata["practical_sensor_packet"])
    context = values[CONTEXT_START:]
    capture = exif_utc_timestamp(metadata["exif"])
    capture_seconds = capture.timestamp()
    covered = (
        capture_seconds - half_window_seconds >= sbg["time"][0]
        and capture_seconds + half_window_seconds <= sbg["time"][-1]
    )

    if covered:
        gyro, accel, offset_ms = interpolate_trajectory(
            sbg,
            capture_seconds,
            half_window_seconds,
        )
        normalized_gyro = np.clip(gyro / gyro_full_scale, -1.0, 1.0)
        normalized_accel = np.clip(accel / accel_full_scale, -1.0, 1.0)
        imu_variation = float(
            np.sqrt(
                np.mean(np.var(normalized_gyro, axis=0))
                + np.mean(np.var(normalized_accel, axis=0))
            )
        )
        imu_reliability = float(np.clip(0.98 - abs(offset_ms) / 10.0, 0.60, 0.98))
        context[10] = float(np.clip(offset_ms / 5.0, -1.0, 1.0))
        context[11] = float(np.clip(imu_variation / 0.25, 0.0, 1.0))
        context[13] = imu_reliability
        context[15] = max(context[12], context[13], context[14])
        packet = [
            *normalized_gyro.reshape(-1).tolist(),
            *normalized_accel.reshape(-1).tolist(),
            *context,
        ]
        provenance = {
            "source": "direct 200 Hz SBG angular-rate and accelerometer channels",
            "capture_timestamp_utc": capture.isoformat(),
            "trajectory_samples": TRAJECTORY_STEPS,
            "capture_context_support_ms": 2000.0 * half_window_seconds,
            "nearest_sample_offset_ms": offset_ms,
            "angular_rate_source": "direct SBG measurement",
            "acceleration_source": (
                "direct SBG measurement with roll/pitch gravity compensation"
                if sbg.get("accel_mode") == "gravity-compensated"
                else "direct SBG measurement including gravity"
            ),
            "coordinate_mapping": "camera_xyz = body_yzx (fixed mounting convention)",
            "gyro_full_scale_radps": gyro_full_scale,
            "accel_full_scale_mps2": accel_full_scale,
            "imu_reliability": imu_reliability,
            "annotation_blind": True,
        }
    elif outside_sbg_policy == "ins-derived":
        # The full INS log spans every CRID-46 capture. Its attitude and
        # velocity trajectories were synchronized independently of defect
        # annotations by prepare_sony_ins_metadata.py. Retain those temporal
        # measurements when the shorter raw-channel export is unavailable,
        # but state their derived provenance explicitly. This gives every
        # frame a useful partial sensor packet without calling derived rates
        # direct gyroscope/accelerometer observations.
        source_provenance = metadata.get("practical_sensor_provenance") or {}
        imu_reliability = float(
            min(
                np.clip(context[13], 0.0, 1.0),
                np.clip(derived_imu_reliability_cap, 0.0, 1.0),
            )
        )
        context[13] = imu_reliability
        context[15] = max(context[12], context[13], context[14])
        values[CONTEXT_START:] = context
        packet = values
        provenance = {
            "source": "synchronized 200 Hz INS attitude/velocity trajectory",
            "capture_timestamp_utc": capture.isoformat(),
            "trajectory_samples": TRAJECTORY_STEPS,
            "capture_context_support_ms": 2000.0 * half_window_seconds,
            "angular_rate_source": source_provenance.get(
                "angular_rate_source",
                "derived from locally fitted roll/pitch/yaw",
            ),
            "acceleration_source": source_provenance.get(
                "acceleration_source",
                "derived from locally fitted NED velocity",
            ),
            "coordinate_mapping": source_provenance.get(
                "coordinate_mapping",
                "INS navigation frame to the released camera-like packet axes",
            ),
            "imu_reliability": imu_reliability,
            "direct_sbg_available": False,
            "annotation_blind": True,
        }
    else:
        # Preserve the genuinely observed camera and vehicle fields but never
        # extrapolate the direct IMU beyond its recording interval.
        context[10] = 0.0
        context[11] = 0.0
        context[13] = 0.0
        context[15] = max(context[12], context[14])
        packet = [0.0] * ACCEL_END + context
        nearest_boundary = min(
            abs(capture_seconds - sbg["time"][0]),
            abs(capture_seconds - sbg["time"][-1]),
        )
        provenance = {
            "source": "camera/vehicle context only; direct SBG interval unavailable",
            "capture_timestamp_utc": capture.isoformat(),
            "nearest_recording_boundary_seconds": float(nearest_boundary),
            "angular_rate_source": "unavailable",
            "acceleration_source": "unavailable",
            "imu_reliability": 0.0,
            "annotation_blind": True,
        }

    if len(packet) != PRACTICAL_SENSOR_DIM:
        raise AssertionError((len(packet), PRACTICAL_SENSOR_DIM))
    return [float(value) for value in packet], provenance, covered


def main() -> None:
    args = parse_args()
    source = args.source_metadata.resolve()
    output = args.out.resolve()
    sbg_path = args.sbg.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not sbg_path.exists():
        raise FileNotFoundError(sbg_path)
    if args.half_window_ms <= 0.0:
        raise ValueError("half-window must be positive")
    if not 0.0 <= args.derived_imu_reliability_cap <= 1.0:
        raise ValueError("--derived-imu-reliability-cap must be in [0, 1]")
    if args.gyro_full_scale_radps <= 0.0 or args.accel_full_scale_mps2 <= 0.0:
        raise ValueError("normalization full scales must be positive")
    if args.force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    sbg = load_sbg(
        sbg_path,
        gravity_compensate=args.accel_mode == "gravity-compensated",
    )
    records: list[dict[str, Any]] = []
    source_files = sorted(source.glob("*.json"))
    if not source_files:
        raise FileNotFoundError(f"No metadata JSON files found under {source}")
    for source_path in source_files:
        metadata = json.loads(source_path.read_text(encoding="utf-8"))
        packet, provenance, covered = build_packet(
            metadata,
            sbg,
            half_window_seconds=args.half_window_ms / 1000.0,
            gyro_full_scale=args.gyro_full_scale_radps,
            accel_full_scale=args.accel_full_scale_mps2,
            outside_sbg_policy=args.outside_sbg_policy,
            derived_imu_reliability_cap=args.derived_imu_reliability_cap,
        )
        if covered:
            metadata_source = "real_sony_exif_plus_direct_sbg"
            inertial_source = "direct_sbg"
        elif args.outside_sbg_policy == "ins-derived":
            metadata_source = "real_sony_exif_plus_ins_derived"
            inertial_source = "ins_derived"
        else:
            metadata_source = "real_sony_exif_vehicle_only"
            inertial_source = "unavailable"
        metadata["metadata_source"] = metadata_source
        metadata["practical_sensor_packet"] = {
            name: value for name, value in zip(SENSOR_PACKET_NAMES, packet)
        }
        metadata["practical_sensor_provenance"] = provenance
        destination = output / source_path.name
        destination.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(
            {
                "file": source_path.name,
                "original_name": metadata.get("original_name", ""),
                "capture_timestamp_utc": provenance["capture_timestamp_utc"],
                "direct_sbg_available": covered,
                "inertial_source": inertial_source,
                "inertial_available": inertial_source != "unavailable",
                "imu_reliability": provenance["imu_reliability"],
                "nearest_sample_offset_ms": provenance.get("nearest_sample_offset_ms", ""),
            }
        )

    report = pd.DataFrame(records)
    report.to_csv(output.parent / "crid_raw_sbg_alignment.csv", index=False)
    covered_count = int(report["direct_sbg_available"].sum())
    derived_count = int((report["inertial_source"] == "ins_derived").sum())
    unavailable_count = int((report["inertial_source"] == "unavailable").sum())
    summary = {
        "records": len(records),
        "direct_sbg_covered": covered_count,
        "ins_derived_covered": derived_count,
        "inertial_unavailable": unavailable_count,
        "direct_sbg_unavailable": len(records) - covered_count,
        "outside_sbg_policy": args.outside_sbg_policy,
        "derived_imu_reliability_cap": args.derived_imu_reliability_cap,
        "source_metadata": str(source),
        "sbg_file": str(sbg_path),
        "sbg_sha256": sha256(sbg_path),
        "sbg_rows": int(len(sbg["time"])),
        "sbg_rate_hz_median": float(1.0 / np.median(np.diff(sbg["time"]))),
        "sbg_interval_utc": [
            datetime.fromtimestamp(sbg["time"][0], tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(sbg["time"][-1], tz=timezone.utc).isoformat(),
        ],
        "capture_context_support_ms": 2.0 * args.half_window_ms,
        "direct_acceleration_mode": args.accel_mode,
        "annotation_blind": True,
        "no_extrapolation": True,
        "privacy_note": "The public packet omits latitude and longitude.",
    }
    (output.parent / "crid_raw_sbg_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
