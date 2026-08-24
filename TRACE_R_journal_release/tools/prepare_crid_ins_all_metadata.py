#!/usr/bin/env python3
# TRACE-R release integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Build all-frame CRID practical metadata from the complete SBG INS export.

``ins_all.txt`` contains the 200 Hz navigation solution, attitude and velocity
uncertainties, and body-frame acceleration. It does not contain angular rate,
so angular velocity is obtained by differentiating the smoothed attitude
trajectory. The overlapping direct-rate log in ``ac1.csv`` is used only to
estimate a fixed sensor-to-camera calibration. This calibration is independent
of images and defect annotations.

The resulting public sidecar keeps RMR-Net's deployed 82-value schema:
11x3 angular-rate samples, 11x3 acceleration samples, and 16 capture-context
values. Latitude and longitude are deliberately omitted from the packet.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import ExifTags, Image
from scipy.signal import savgol_filter

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
from tools.prepare_crid_raw_sbg_metadata import load_sbg  # noqa: E402
from tools.prepare_sony_ins_metadata import (  # noqa: E402
    clamp,
    exif_utc_timestamp,
    normalized_log,
)


INS_ALL_COLUMNS = (
    "time",
    "latitude",
    "longitude",
    "altitude_ellipsoid",
    "north_velocity",
    "east_velocity",
    "down_velocity",
    "roll",
    "pitch",
    "yaw",
    "latitude_std",
    "longitude_std",
    "altitude_std",
    "north_velocity_std",
    "east_velocity_std",
    "down_velocity_std",
    "roll_std",
    "pitch_std",
    "yaw_std",
    "body_accel_x",
    "body_accel_y",
    "body_accel_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ins-all", type=Path, default=ROOT / "ins_all.txt")
    parser.add_argument("--direct-sbg", type=Path, default=ROOT / "ac1.csv")
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=ROOT / "experiments/geotagged_cam1_ins_metadata_20260811/metadata",
    )
    parser.add_argument(
        "--native-root",
        type=Path,
        default=ROOT / "geotagged/cam1",
        help="Native Sony frames used to recover nested EXIF and subsecond timing.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--half-window-ms", type=float, default=25.0)
    parser.add_argument("--gyro-full-scale-radps", type=float, default=4.0)
    parser.add_argument("--accel-full-scale-mps2", type=float, default=5.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packet_values(raw_packet: Any) -> list[float]:
    if isinstance(raw_packet, dict):
        return [float(raw_packet.get(name, 0.0)) for name in SENSOR_PACKET_NAMES]
    values = [float(value) for value in raw_packet]
    if len(values) != PRACTICAL_SENSOR_DIM:
        raise ValueError(f"Expected {PRACTICAL_SENSOR_DIM} values, got {len(values)}")
    return values


EXIF_KEEP = {
    "Make",
    "Model",
    "Software",
    "ExposureTime",
    "BrightnessValue",
    "DigitalZoomRatio",
    "FNumber",
    "ISOSpeedRatings",
    "RecommendedExposureIndex",
    "FocalLength",
    "FocalLengthIn35mmFilm",
    "DateTime",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "OffsetTime",
    "OffsetTimeOriginal",
    "OffsetTimeDigitized",
    "SubsecTime",
    "SubsecTimeOriginal",
    "SubsecTimeDigitized",
    "ExposureMode",
    "ExposureProgram",
    "Sharpness",
}


def json_scalar(value: Any) -> Any:
    """Convert Pillow EXIF values without retaining maker notes or GPS."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        numerator = float(value.numerator)
        denominator = float(value.denominator)
        if denominator != 0.0:
            return numerator / denominator
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        pass
    if isinstance(value, (tuple, list)):
        return [json_scalar(item) for item in value]
    return str(value)


def read_complete_sony_exif(path: Path) -> dict[str, Any]:
    """Flatten top-level and nested Sony EXIF while excluding geolocation."""

    with Image.open(path) as image:
        raw = image.getexif()
        width, height = image.size
    output: dict[str, Any] = {"width": width, "height": height}

    def retain(items: Any) -> None:
        for key, value in items:
            name = str(ExifTags.TAGS.get(key, key))
            if name in EXIF_KEEP:
                output[name] = json_scalar(value)

    retain(raw.items())
    if hasattr(raw, "get_ifd") and hasattr(ExifTags, "IFD"):
        exif_ifd = getattr(ExifTags.IFD, "Exif", None)
        if exif_ifd is not None:
            try:
                retain(raw.get_ifd(exif_ifd).items())
            except (KeyError, TypeError, ValueError):
                pass
    required = ("DateTimeOriginal", "OffsetTimeOriginal", "SubsecTimeOriginal")
    missing = [name for name in required if not output.get(name)]
    if missing:
        raise KeyError(f"{path.name} is missing synchronization EXIF fields: {missing}")
    return output


def apply_camera_context(context: list[float], exif: dict[str, Any]) -> dict[str, float]:
    """Map observed Sony settings to the declared 16-value context schema."""

    exposure_seconds = float(exif.get("ExposureTime") or 0.0)
    exposure_ms = max(1000.0 * exposure_seconds, 0.0)
    iso = float(
        exif.get("ISOSpeedRatings")
        or exif.get("RecommendedExposureIndex")
        or 100.0
    )
    focal_mm = float(exif.get("FocalLength") or 0.0)
    aperture = float(exif.get("FNumber") or 0.0)
    iso_code = normalized_log(max(iso, 100.0), 100.0, 6400.0)
    # Preserve the native 0.25 ms Sony exposure for physical integration.
    # Earlier packets used a 2 ms lower bound and collapsed every CRID frame to
    # zero, which made exposure-time motion semantically unidentifiable.
    context[0] = normalized_log(max(exposure_ms, 0.05), 0.05, 40.0)
    context[1] = iso_code
    context[2] = iso_code
    context[3] = clamp((focal_mm - 20.0) / 50.0) if focal_mm > 0.0 else 0.0
    context[4] = clamp((aperture - 1.4) / 10.0) if aperture > 0.0 else 0.0
    context[6] = 0.75
    context[12] = 0.95 if exposure_seconds > 0.0 and iso > 0.0 else 0.60
    return {
        "exposure_seconds": exposure_seconds,
        "exposure_ms": exposure_ms,
        "iso": iso,
        "brightness_value": float(exif.get("BrightnessValue") or 0.0),
        "camera_reliability": float(context[12]),
    }


def load_ins_all(path: Path) -> dict[str, np.ndarray]:
    """Read the 22-column export, with or without a textual header row."""

    frame = pd.read_csv(
        path,
        sep="\t",
        names=INS_ALL_COLUMNS,
        header=None,
        na_values=["N/A", "nan", "NaN", ""],
        low_memory=False,
    )
    stamp = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame = frame.loc[stamp.notna()].copy()
    stamp = stamp.loc[stamp.notna()]
    if len(frame) < 100:
        raise ValueError("ins_all contains too few valid rows")
    seconds = stamp.map(lambda value: value.timestamp()).to_numpy(np.float64)
    if np.any(np.diff(seconds) <= 0.0):
        raise ValueError("ins_all timestamps must be strictly increasing")

    output: dict[str, np.ndarray] = {"time": seconds}
    for name in INS_ALL_COLUMNS[1:]:
        output[name] = pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float64)
    required = (
        "north_velocity",
        "east_velocity",
        "down_velocity",
        "roll",
        "pitch",
        "yaw",
        "body_accel_x",
        "body_accel_y",
        "body_accel_z",
    )
    for name in required:
        if not np.isfinite(output[name]).all():
            raise ValueError(f"Required ins_all field contains missing values: {name}")
    # Uncertainty fields contain a few startup/transition N/A records. Fill
    # those once along the time axis rather than scanning the full log during
    # every frame lookup.
    for name in (
        "north_velocity_std",
        "east_velocity_std",
        "down_velocity_std",
        "roll_std",
        "pitch_std",
        "yaw_std",
    ):
        values = output[name].copy()
        valid = np.isfinite(values)
        if valid.sum() < 2:
            raise ValueError(f"Uncertainty field has insufficient support: {name}")
        if not valid.all():
            values[~valid] = np.interp(seconds[~valid], seconds[valid], values[valid])
        output[name] = values

    dt = float(np.median(np.diff(seconds)))
    if not 0.004 <= dt <= 0.006:
        raise ValueError(f"Expected a 200 Hz stream; median interval is {dt:.6f} s")

    roll = np.unwrap(np.deg2rad(output["roll"]))
    pitch = np.unwrap(np.deg2rad(output["pitch"]))
    yaw = np.unwrap(np.deg2rad(output["yaw"]))
    # A short Savitzky-Golay derivative suppresses quantization while retaining
    # the exposure-scale motion. These are the standard ZYX body-rate equations.
    derivative = lambda values: savgol_filter(  # noqa: E731
        values, window_length=11, polyorder=2, deriv=1, delta=dt, mode="interp"
    )
    roll_dot, pitch_dot, yaw_dot = map(derivative, (roll, pitch, yaw))
    body_rate = np.stack(
        [
            roll_dot - yaw_dot * np.sin(pitch),
            pitch_dot * np.cos(roll) + yaw_dot * np.sin(roll) * np.cos(pitch),
            -pitch_dot * np.sin(roll) + yaw_dot * np.cos(roll) * np.cos(pitch),
        ],
        axis=1,
    )
    output["body_rate"] = body_rate
    output["body_accel"] = np.stack(
        [output["body_accel_x"], output["body_accel_y"], output["body_accel_z"]],
        axis=1,
    )
    output["sample_interval_seconds"] = np.asarray(dt)
    return output


def fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit target = scale * source @ rotation + bias with det(rotation)=+1."""

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = source - source_mean
    y = target - target_mean
    left, _, right = np.linalg.svd(x.T @ y)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    projected = x @ rotation
    scale = float(np.sum(projected * y) / max(np.sum(projected * projected), 1e-12))
    bias = target_mean - scale * source_mean @ rotation
    return rotation, scale, bias


def apply_similarity(
    values: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    rotation = np.asarray(calibration["rotation"], dtype=np.float64)
    bias = np.asarray(calibration["bias"], dtype=np.float64)
    return float(calibration["scale"]) * values @ rotation + bias


def calibrate_to_camera(
    ins: dict[str, np.ndarray],
    direct_sbg_path: Path,
) -> dict[str, Any]:
    """Use unlabelled overlapping logs to calibrate INS body axes to camera axes."""

    direct = load_sbg(direct_sbg_path, gravity_compensate=True)
    time = ins["time"]
    covered = (time >= direct["time"][0]) & (time <= direct["time"][-1])
    overlap_time = time[covered]
    if len(overlap_time) < 1000:
        raise RuntimeError("Insufficient overlap between ins_all and direct SBG")
    direct_gyro = np.stack(
        [np.interp(overlap_time, direct["time"], direct["gyro"][:, axis]) for axis in range(3)],
        axis=1,
    )
    direct_accel = np.stack(
        [np.interp(overlap_time, direct["time"], direct["accel"][:, axis]) for axis in range(3)],
        axis=1,
    )
    source_gyro = ins["body_rate"][covered]
    source_accel = ins["body_accel"][covered]

    # Calibration uses the first temporal half; the second half is an untouched
    # sensor-only audit. Extreme acceleration outliers are excluded before fit.
    midpoint = len(overlap_time) // 2
    accel_limit = float(np.quantile(np.linalg.norm(source_accel[:midpoint], axis=1), 0.99))
    accel_keep = np.linalg.norm(source_accel[:midpoint], axis=1) <= accel_limit
    gyro_limit = float(np.quantile(np.linalg.norm(source_gyro[:midpoint], axis=1), 0.99))
    gyro_keep = np.linalg.norm(source_gyro[:midpoint], axis=1) <= gyro_limit

    ar, asc, ab = fit_similarity(source_accel[:midpoint][accel_keep], direct_accel[:midpoint][accel_keep])
    gr, gsc, gb = fit_similarity(source_gyro[:midpoint][gyro_keep], direct_gyro[:midpoint][gyro_keep])
    accel_cal = {"rotation": ar.tolist(), "scale": asc, "bias": ab.tolist()}
    gyro_cal = {"rotation": gr.tolist(), "scale": gsc, "bias": gb.tolist()}

    accel_prediction = apply_similarity(source_accel[midpoint:], accel_cal)
    gyro_prediction = apply_similarity(source_gyro[midpoint:], gyro_cal)
    accel_rmse = np.sqrt(np.mean(np.square(accel_prediction - direct_accel[midpoint:]), axis=0))
    gyro_rmse = np.sqrt(np.mean(np.square(gyro_prediction - direct_gyro[midpoint:]), axis=0))
    accel_corr = [
        float(np.corrcoef(accel_prediction[:, axis], direct_accel[midpoint:, axis])[0, 1])
        for axis in range(3)
    ]
    gyro_corr = [
        float(np.corrcoef(gyro_prediction[:, axis], direct_gyro[midpoint:, axis])[0, 1])
        for axis in range(3)
    ]
    return {
        "method": "first-half unlabelled log similarity calibration; second-half audit",
        "overlap_samples": int(len(overlap_time)),
        "fit_samples": int(midpoint),
        "audit_samples": int(len(overlap_time) - midpoint),
        "acceleration": accel_cal,
        "angular_rate": gyro_cal,
        "audit": {
            "acceleration_rmse_mps2": accel_rmse.tolist(),
            "acceleration_correlation": accel_corr,
            "angular_rate_rmse_radps": gyro_rmse.tolist(),
            "angular_rate_correlation": gyro_corr,
        },
        "annotation_blind": True,
    }


def interpolate_rows(time: np.ndarray, values: np.ndarray, support: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(support, time, values[:, axis]) for axis in range(values.shape[1])],
        axis=1,
    )


def interpolate_scalar(time: np.ndarray, values: np.ndarray, target: float) -> float:
    return float(np.interp(target, time, values))


def nearest_offset_ms(time: np.ndarray, target: float) -> float:
    """Return the nearest timestamp offset with O(log N) indexed lookup."""

    right = int(np.searchsorted(time, target, side="left"))
    candidates = [min(max(right, 0), len(time) - 1)]
    if right > 0:
        candidates.append(right - 1)
    index = min(candidates, key=lambda item: abs(float(time[item]) - target))
    return 1000.0 * float(time[index] - target)


def reliability_at(ins: dict[str, np.ndarray], target: float, offset_ms: float, calibration: dict[str, Any]) -> tuple[float, float, float]:
    time = ins["time"]
    orientation_std = np.nanmedian(
        [
            interpolate_scalar(time, ins[name], target)
            for name in ("roll_std", "pitch_std", "yaw_std")
        ]
    )
    velocity_std = np.nanmedian(
        [
            interpolate_scalar(time, ins[name], target)
            for name in ("north_velocity_std", "east_velocity_std", "down_velocity_std")
        ]
    )
    if not np.isfinite(orientation_std):
        orientation_std = 5.0
    if not np.isfinite(velocity_std):
        velocity_std = 1.0
    timing_quality = math.exp(-abs(offset_ms) / 5.0)
    attitude_quality = math.exp(-float(orientation_std) / 3.0)
    calibration_rmse = float(np.mean(calibration["audit"]["angular_rate_rmse_radps"]))
    calibration_quality = math.exp(-calibration_rmse / 0.25)
    imu_reliability = float(np.clip(0.98 * timing_quality * math.sqrt(attitude_quality * calibration_quality), 0.10, 0.98))
    vehicle_reliability = float(np.clip(math.exp(-float(velocity_std) / 0.5), 0.10, 0.98))
    normalized_uncertainty = float(np.clip(0.5 * orientation_std / 5.0 + 0.5 * velocity_std / 1.0, 0.0, 1.0))
    return imu_reliability, vehicle_reliability, normalized_uncertainty


def build_packet(
    metadata: dict[str, Any],
    ins: dict[str, np.ndarray],
    calibration: dict[str, Any],
    *,
    half_window_seconds: float,
    gyro_full_scale: float,
    accel_full_scale: float,
) -> tuple[list[float], dict[str, Any], bool]:
    original = packet_values(metadata["practical_sensor_packet"])
    context = list(original[CONTEXT_START:])
    camera = apply_camera_context(context, metadata["exif"])
    capture = exif_utc_timestamp(metadata["exif"])
    target = capture.timestamp()
    support = np.linspace(target - half_window_seconds, target + half_window_seconds, TRAJECTORY_STEPS)
    covered = support[0] >= ins["time"][0] and support[-1] <= ins["time"][-1]
    if not covered:
        original[CONTEXT_START:] = context
        provenance = {
            "source": "original synchronized INS fallback at recording boundary",
            "capture_timestamp_utc": capture.isoformat(),
            "camera_source": "native Sony nested EXIF",
            **camera,
            "annotation_blind": True,
        }
        return original, provenance, False

    gyro = interpolate_rows(ins["time"], ins["gyro_camera"], support)
    accel = interpolate_rows(ins["time"], ins["accel_camera"], support)
    offset_ms = nearest_offset_ms(ins["time"], target)
    imu_rel, vehicle_rel, uncertainty = reliability_at(ins, target, offset_ms, calibration)

    speed = math.sqrt(sum(interpolate_scalar(ins["time"], ins[name], target) ** 2 for name in ("north_velocity", "east_velocity", "down_velocity")))
    yaw_rate = float(
        interpolate_rows(
            ins["time"], ins["gyro_camera"], np.asarray([target])
        )[0, 1]
    )
    context[8] = float(np.clip(speed / 30.0, 0.0, 1.0))
    context[9] = float(np.clip(yaw_rate / 0.15, -1.0, 1.0))
    context[10] = float(np.clip(offset_ms / 5.0, -1.0, 1.0))
    context[11] = uncertainty
    context[13] = imu_rel
    context[14] = vehicle_rel
    context[15] = max(context[12], context[13], context[14])
    packet = [
        *np.clip(gyro / gyro_full_scale, -1.0, 1.0).reshape(-1).tolist(),
        *np.clip(accel / accel_full_scale, -1.0, 1.0).reshape(-1).tolist(),
        *context,
    ]
    provenance = {
        "source": "complete 200 Hz SBG INS solution",
        "capture_timestamp_utc": capture.isoformat(),
        "trajectory_samples": TRAJECTORY_STEPS,
        "capture_context_support_ms": 2000.0 * half_window_seconds,
        "nearest_sample_offset_ms": offset_ms,
        "angular_rate_source": "smoothed attitude derivative calibrated to overlapping direct SBG rate",
        "acceleration_source": "measured SBG body acceleration calibrated to camera axes",
        "uncertainty_source": "reported attitude and velocity standard deviations",
        "coordinate_calibration": "annotation-independent first-half overlapping sensor logs",
        "gyro_full_scale_radps": gyro_full_scale,
        "accel_full_scale_mps2": accel_full_scale,
        "imu_reliability": imu_rel,
        "vehicle_reliability": vehicle_rel,
        "camera_source": "native Sony nested EXIF",
        **camera,
        "annotation_blind": True,
        "geolocation_in_public_packet": False,
    }
    return [float(value) for value in packet], provenance, True


def main() -> None:
    args = parse_args()
    source = args.source_metadata.resolve()
    native_root = args.native_root.resolve()
    output = args.out.resolve()
    ins_path = args.ins_all.resolve()
    direct_path = args.direct_sbg.resolve()
    for path in (source, native_root, ins_path, direct_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.half_window_ms <= 0.0 or args.gyro_full_scale_radps <= 0.0 or args.accel_full_scale_mps2 <= 0.0:
        raise ValueError("Window and normalization scales must be positive")
    output.mkdir(parents=True)

    ins = load_ins_all(ins_path)
    calibration = calibrate_to_camera(ins, direct_path)
    # Fixed sensor calibration is applied once to the complete trajectory.
    # Per-frame packet construction then performs only local interpolation.
    ins["gyro_camera"] = apply_similarity(ins["body_rate"], calibration["angular_rate"])
    ins["accel_camera"] = apply_similarity(ins["body_accel"], calibration["acceleration"])
    records: list[dict[str, Any]] = []
    source_files = sorted(source.glob("*.json"))
    if not source_files:
        raise FileNotFoundError(f"No source metadata under {source}")
    native_by_name = {
        path.name: path
        for path in native_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    for source_path in source_files:
        metadata = json.loads(source_path.read_text(encoding="utf-8"))
        original_name = str(metadata.get("original_name") or "")
        native_path = native_by_name.get(original_name)
        if native_path is None:
            raise FileNotFoundError(
                f"Native Sony frame for {source_path.name} is missing: {original_name}"
            )
        metadata["exif"] = read_complete_sony_exif(native_path)
        pose = dict(metadata.get("pose_csv") or {})
        for private_name in ("lat", "lon", "ht"):
            pose.pop(private_name, None)
        metadata["pose_csv"] = pose
        packet, provenance, covered = build_packet(
            metadata,
            ins,
            calibration,
            half_window_seconds=args.half_window_ms / 1000.0,
            gyro_full_scale=args.gyro_full_scale_radps,
            accel_full_scale=args.accel_full_scale_mps2,
        )
        metadata["metadata_source"] = (
            "real_sony_exif_plus_complete_sbg_ins" if covered else "real_sony_exif_plus_ins_boundary_fallback"
        )
        metadata["practical_sensor_packet"] = dict(zip(SENSOR_PACKET_NAMES, packet))
        metadata["practical_sensor_provenance"] = provenance
        (output / source_path.name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        records.append(
            {
                "file": source_path.name,
                "original_name": metadata.get("original_name", ""),
                "capture_timestamp_utc": provenance["capture_timestamp_utc"],
                "complete_sbg_ins_available": covered,
                "imu_reliability": provenance.get("imu_reliability", 0.0),
                "vehicle_reliability": provenance.get("vehicle_reliability", 0.0),
                "nearest_sample_offset_ms": provenance.get("nearest_sample_offset_ms", ""),
            }
        )

    report = pd.DataFrame(records)
    report.to_csv(output.parent / "crid_ins_all_alignment.csv", index=False)
    summary = {
        "records": len(records),
        "complete_sbg_ins_covered": int(report["complete_sbg_ins_available"].sum()),
        "boundary_fallback": int((~report["complete_sbg_ins_available"]).sum()),
        "ins_all": str(ins_path),
        "ins_all_sha256": sha256(ins_path),
        "direct_sbg_calibration_log": str(direct_path),
        "direct_sbg_sha256": sha256(direct_path),
        "source_metadata": str(source),
        "native_exif_root": str(native_root),
        "camera_fields": (
            "native nested Sony EXIF including exposure, ISO, brightness and "
            "subsecond UTC synchronization"
        ),
        "rows": int(len(ins["time"])),
        "rate_hz": float(1.0 / float(ins["sample_interval_seconds"])),
        "interval_utc": [
            datetime.fromtimestamp(ins["time"][0], tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(ins["time"][-1], tz=timezone.utc).isoformat(),
        ],
        "capture_context_support_ms": 2.0 * args.half_window_ms,
        "calibration": calibration,
        "annotation_blind": True,
        "privacy_note": "Latitude, longitude and altitude are retained only for synchronization audit and are not model inputs.",
    }
    (output.parent / "crid_ins_all_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
