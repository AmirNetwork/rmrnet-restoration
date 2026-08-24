# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcadnet import code_from_metadata
from rcadnet.practical_metadata import (
    PRACTICAL_SENSOR_DIM,
    TRAJECTORY_STEPS,
    observable_code_from_packet,
)


KITTI_OXTS_FIELDS = (
    "lat",
    "lon",
    "alt",
    "roll",
    "pitch",
    "yaw",
    "vn",
    "ve",
    "vf",
    "vl",
    "vu",
    "ax",
    "ay",
    "az",
    "af",
    "al",
    "au",
    "wx",
    "wy",
    "wz",
    "wf",
    "wl",
    "wu",
    "pos_accuracy",
    "vel_accuracy",
    "navstat",
    "numsats",
    "posmode",
    "velmode",
    "orimode",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a controlled road-restoration split from KITTI raw images and real "
            "OXTS GPS/IMU metadata. The clean KITTI frames are degraded with a telemetry-"
            "calibrated blur model, and the same real telemetry record is saved as metadata."
        )
    )
    parser.add_argument("--kitti-date-root", default="datasets/raw/kitti/2011_09_26")
    parser.add_argument("--train-sequence", action="append", default=None)
    parser.add_argument("--test-sequence", action="append", default=None)
    parser.add_argument("--train-out", default="data/kitti_realmeta_restoration_train")
    parser.add_argument("--test-out", default="data/kitti_realmeta_restoration_test")
    parser.add_argument("--scenario", default="kitti_realmeta_motion")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--exposure-ms", type=float, default=12.0)
    parser.add_argument("--blur-scale", type=float, default=1.0, help="Scale the telemetry-estimated blur length for exposure/camera studies.")
    parser.add_argument("--calibration-file", default=None, help="KITTI calib_cam_to_cam.txt; defaults to the date root.")
    parser.add_argument(
        "--road-depth-m",
        type=float,
        default=4.5,
        help="Representative depth used to convert camera translation to road-region optical flow.",
    )
    parser.add_argument(
        "--road-roi-y",
        type=float,
        default=0.92,
        help="Representative vertical image coordinate as a fraction of image height.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=96)
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Prepare only the held-out test sequences; useful for exposure sensitivity sweeps.",
    )
    return parser.parse_args()


def list_frames(sequence_root: Path) -> list[Path]:
    return sorted((sequence_root / "image_02" / "data").glob("*.png"))


def read_oxts(path: Path) -> dict[str, float]:
    values = [float(item) for item in path.read_text(encoding="utf-8").split()]
    return {name: values[index] for index, name in enumerate(KITTI_OXTS_FIELDS[: len(values)])}


def resize_max(image: Image.Image, max_side: int) -> Image.Image:
    if not max_side or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.Resampling.BICUBIC)


def read_camera_calibration(path: Path) -> dict[str, float]:
    records: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, values = line.split(":", 1)
        try:
            records[key] = [float(value) for value in values.split()]
        except ValueError:
            continue
    projection = records.get("P_rect_02")
    size = records.get("S_rect_02")
    if projection is None or len(projection) != 12 or size is None or len(size) != 2:
        raise ValueError(f"Missing P_rect_02/S_rect_02 in {path}")
    return {
        "fx": projection[0],
        "fy": projection[5],
        "cx": projection[2],
        "cy": projection[6],
        "calibration_width": size[0],
        "calibration_height": size[1],
    }


def telemetry_blur(
    oxts: dict[str, float],
    exposure_ms: float,
    blur_scale: float,
    calibration: dict[str, float],
    image_size: tuple[int, int],
    road_depth_m: float,
    road_roi_y: float,
) -> tuple[int, float, dict[str, float]]:
    """Map raw OXTS motion to one representative road-region image trajectory.

    The camera is approximated as a pinhole camera with KITTI camera-02
    intrinsics. Vehicle-local forward/left/up velocity and angular rate are
    transformed to camera right/down/forward axes. Optical flow is evaluated
    at the horizontal centre and a declared lower-image road point. A single
    line-spread function is therefore an approximation to spatially varying
    depth-dependent blur, which is recorded explicitly in every metadata file.
    """

    if road_depth_m <= 0:
        raise ValueError("road_depth_m must be positive")
    if not 0.0 <= road_roi_y <= 1.0:
        raise ValueError("road_roi_y must be in [0, 1]")
    speed_mps = math.sqrt(oxts["vf"] ** 2 + oxts["vl"] ** 2 + oxts["vu"] ** 2)
    lateral_accel = abs(oxts["al"])
    forward_accel = abs(oxts["af"])
    vibration = math.sqrt(forward_accel**2 + lateral_accel**2)
    yaw_rate = oxts["wu"]
    pitch_roll_rate = math.sqrt(oxts["wl"] ** 2 + oxts["wf"] ** 2)

    width, height = image_size
    sx = width / calibration["calibration_width"]
    sy = height / calibration["calibration_height"]
    fx = calibration["fx"] * sx
    fy = calibration["fy"] * sy
    cx = calibration["cx"] * sx
    cy = calibration["cy"] * sy
    u = cx
    v = road_roi_y * height
    x = (u - cx) / fx
    y = (v - cy) / fy

    # Vehicle local axes (forward, left, up) -> camera axes (right, down, forward).
    tx, ty, tz = -oxts["vl"], -oxts["vu"], oxts["vf"]
    wx, wy, wz = -oxts["wl"], -oxts["wu"], oxts["wf"]
    inv_depth = 1.0 / road_depth_m
    xdot_translation = (-tx + x * tz) * inv_depth
    ydot_translation = (-ty + y * tz) * inv_depth
    xdot_rotation = x * y * wx - (1.0 + x * x) * wy + y * wz
    ydot_rotation = (1.0 + y * y) * wx - x * y * wy - x * wz
    exposure_s = exposure_ms / 1000.0
    du_translation = fx * xdot_translation * exposure_s
    dv_translation = fy * ydot_translation * exposure_s
    du_rotation = fx * xdot_rotation * exposure_s
    dv_rotation = fy * ydot_rotation * exposure_s
    du = (du_translation + du_rotation) * blur_scale
    dv = (dv_translation + dv_rotation) * blur_scale
    displacement = math.hypot(du, dv)
    angle = math.degrees(math.atan2(dv, du)) if displacement > 1e-9 else 0.0
    length = max(1, int(round(displacement)))
    if length % 2 == 0:
        length += 1

    strength = min(displacement / 21.0, 1.0)

    derived = {
        "speed_mps": speed_mps,
        "vibration_mps2": vibration,
        "yaw_rate_radps": yaw_rate,
        "pitch_roll_rate_radps": pitch_roll_rate,
        "telemetry_strength": strength,
        "blur_scale": blur_scale,
        "blur_length_px": float(length),
        "blur_angle_deg": float(angle),
        "exposure_ms": exposure_ms,
        "translation_displacement_x_px": du_translation * blur_scale,
        "translation_displacement_y_px": dv_translation * blur_scale,
        "rotation_displacement_x_px": du_rotation * blur_scale,
        "rotation_displacement_y_px": dv_rotation * blur_scale,
        "road_depth_m": road_depth_m,
        "road_roi_y_fraction": road_roi_y,
        "camera_fx_px": fx,
        "camera_fy_px": fy,
        "camera_cx_px": cx,
        "camera_cy_px": cy,
    }
    return length, angle, derived


def motion_kernel(length: int, angle: float) -> np.ndarray:
    size = max(3, length)
    if size % 2 == 0:
        size += 1
    arr = np.zeros((size, size), dtype=np.float32)
    arr[size // 2, :] = 1.0
    kernel_image = Image.fromarray(np.uint8(arr * 255.0), mode="L")
    rotated = kernel_image.rotate(angle, resample=Image.Resampling.BICUBIC)
    kernel = np.asarray(rotated, dtype=np.float32)
    kernel = np.maximum(kernel, 0)
    total = float(kernel.sum())
    if total <= 0:
        kernel[size // 2, :] = 1.0
        total = float(kernel.sum())
    kernel /= total
    return kernel


def degrade_with_metadata(
    image: Image.Image,
    oxts: dict[str, float],
    exposure_ms: float,
    blur_scale: float,
    calibration: dict[str, float],
    road_depth_m: float,
    road_roi_y: float,
) -> tuple[Image.Image, dict[str, float]]:
    length, angle, derived = telemetry_blur(
        oxts,
        exposure_ms,
        blur_scale,
        calibration,
        image.size,
        road_depth_m,
        road_roi_y,
    )
    arr = np.asarray(image, dtype=np.uint8)
    blurred_arr = cv2.filter2D(arr, ddepth=-1, kernel=motion_kernel(length, angle), borderType=cv2.BORDER_REFLECT101)
    blurred = Image.fromarray(blurred_arr, mode="RGB")
    return blurred, derived


def normalized_log(value: float, minimum: float, maximum: float) -> float:
    value = max(float(value), minimum)
    return float(
        np.clip(
            (math.log(value) - math.log(minimum))
            / (math.log(maximum) - math.log(minimum)),
            0.0,
            1.0,
        )
    )


def kitti_sensor_packet(
    oxts: dict[str, float],
    derived: dict[str, float],
    *,
    full_prior: bool,
) -> list[float]:
    """Build the deployed 82-value packet from real OXTS measurements.

    The raw packet contains measured OXTS angular rate, acceleration, speed,
    and declared exposure. The full-prior packet replaces the two camera-plane
    gyro coordinates with the telemetry renderer's derived displacement and is
    therefore an explicitly labelled generator-aligned upper bound.
    """

    gyro_full_scale = 4.0
    exposure_norm = normalized_log(float(derived["exposure_ms"]), 2.0, 40.0)
    integration_scale = 0.20 + 0.80 * exposure_norm
    if full_prior:
        displacement_x = float(derived["translation_displacement_x_px"]) + float(
            derived["rotation_displacement_x_px"]
        )
        displacement_y = float(derived["translation_displacement_y_px"]) + float(
            derived["rotation_displacement_y_px"]
        )
        gyro_x = displacement_x / 25.0 / integration_scale / gyro_full_scale
        gyro_y = displacement_y / 25.0 / integration_scale / gyro_full_scale
        gyro_z = 0.20 * float(oxts["wu"]) / gyro_full_scale
    else:
        # Vehicle axes (forward, left, up) are mapped to camera-like axes. One
        # OXTS record is available per frame, so the measurement is held
        # constant over the declared exposure rather than interpolated from GT.
        gyro_x = -float(oxts["wl"]) / gyro_full_scale
        gyro_y = -float(oxts["wu"]) / gyro_full_scale
        gyro_z = float(oxts["wf"]) / gyro_full_scale
    gyro = np.tile(
        np.asarray([gyro_x, gyro_y, gyro_z], dtype=np.float64),
        (TRAJECTORY_STEPS, 1),
    )
    accel = np.tile(
        np.asarray(
            [
                np.clip(-float(oxts["al"]) / 5.0, -1.0, 1.0),
                np.clip(-float(oxts["au"]) / 5.0, -1.0, 1.0),
                np.clip(float(oxts["af"]) / 5.0, -1.0, 1.0),
            ],
            dtype=np.float64,
        ),
        (TRAJECTORY_STEPS, 1),
    )
    velocity_accuracy = max(float(oxts.get("vel_accuracy", 0.0)), 0.0)
    imu_reliability = float(np.clip(0.98 - velocity_accuracy / 10.0, 0.55, 0.98))
    vehicle_reliability = float(
        np.clip(0.96 - velocity_accuracy / 8.0, 0.55, 0.96)
    )
    context = [
        exposure_norm,
        0.0,  # ISO is not supplied by KITTI raw.
        0.0,  # Analog gain is not supplied by KITTI raw.
        0.10,  # Fixed camera-02 lens identity, not a fitted blur parameter.
        0.0,  # Aperture unavailable.
        0.0,  # Focus-error proxy unavailable.
        0.75,  # Fixed autofocus confidence for the controlled sharp source.
        0.0,  # Rolling readout unavailable.
        float(np.clip(float(derived["speed_mps"]) / 30.0, 0.0, 1.0)),
        float(np.clip(float(oxts["wu"]) / 0.15, -1.0, 1.0)),
        0.0,  # Timestamp offset unavailable.
        float(np.clip(velocity_accuracy / 5.0, 0.0, 1.0)),
        0.70,
        imu_reliability,
        vehicle_reliability,
        1.0,
    ]
    packet = [
        *[float(value) for value in gyro.reshape(-1)],
        *[float(value) for value in accel.reshape(-1)],
        *context,
    ]
    if len(packet) != PRACTICAL_SENSOR_DIM:
        raise AssertionError(
            f"Expected {PRACTICAL_SENSOR_DIM} sensor values, got {len(packet)}"
        )
    return [float(np.clip(value, -1.0, 1.0)) for value in packet]


def make_dirs(root: Path, scenario: str) -> dict[str, Path]:
    base = root / "scenarios" / scenario
    folders = {
        "input": base / "input",
        "gt": base / "gt",
        "metadata": base / "metadata",
        "private_calibration": base / "private_calibration",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def write_split(
    *,
    date_root: Path,
    sequences: Iterable[str],
    out_root: Path,
    scenario: str,
    max_side: int,
    exposure_ms: float,
    blur_scale: float,
    calibration: dict[str, float],
    calibration_file: Path,
    road_depth_m: float,
    road_roi_y: float,
    jpeg_quality: int,
    write_private_calibration: bool,
) -> list[dict[str, str | float]]:
    folders = make_dirs(out_root, scenario)
    rows: list[dict[str, str | float]] = []
    for sequence in sequences:
        sequence_root = date_root / sequence
        for frame_path in list_frames(sequence_root):
            frame_id = frame_path.stem
            oxts_path = sequence_root / "oxts" / "data" / f"{frame_id}.txt"
            if not oxts_path.exists():
                continue
            oxts = read_oxts(oxts_path)
            with Image.open(frame_path) as raw:
                clean = resize_max(raw.convert("RGB"), max_side)
            degraded, derived = degrade_with_metadata(
                clean,
                oxts,
                exposure_ms,
                blur_scale,
                calibration,
                road_depth_m,
                road_roi_y,
            )

            out_name = f"{sequence.replace('_sync', '')}_{frame_id}.jpg"
            clean.save(folders["gt"] / out_name, quality=jpeg_quality)
            degraded.save(folders["input"] / out_name, quality=jpeg_quality)

            metadata = {
                "metadata_source": "real_kitti_oxts_calibrated_blur",
                "dataset": "KITTI raw",
                "sequence": sequence,
                "frame_id": frame_id,
                "real_metadata_fields_used": "vehicle-local velocity, angular rates, acceleration, exposure, camera-02 intrinsics",
                "calibration_file": str(calibration_file),
                "calibration_sha256": sha256(calibration_file),
                "calibration_note": (
                    "A pinhole optical-flow proxy uses KITTI camera-02 intrinsics, raw vehicle-local OXTS "
                    "velocity/angular rate, declared exposure, and the recorded representative road depth/ROI. "
                    "The resulting global line-spread function approximates spatially varying scene blur; no "
                    "clean target or detector output is used."
                ),
                "gyro_x": abs(oxts["wl"]),
                "gyro_y": abs(oxts["wu"]),
                "accel_norm": derived["vibration_mps2"],
                "speed_mps": derived["speed_mps"],
                "exposure_ms": derived["exposure_ms"],
                "defocus_score": 0.0,
                "noise_score": 0.0,
                "blur_angle_deg": derived["blur_angle_deg"],
                "blur_length_px": derived["blur_length_px"],
                "low_light_score": 0.0,
                "jpeg_quality": float(jpeg_quality),
                "raw_oxts_yaw_rad": oxts["yaw"],
                "raw_oxts_yaw_rate_radps": derived["yaw_rate_radps"],
                "raw_oxts_forward_accel_mps2": oxts["af"],
                "raw_oxts_lateral_accel_mps2": oxts["al"],
                "raw_oxts_up_accel_mps2": oxts["au"],
                "telemetry_strength": derived["telemetry_strength"],
                "blur_scale": derived["blur_scale"],
                "translation_displacement_x_px": derived["translation_displacement_x_px"],
                "translation_displacement_y_px": derived["translation_displacement_y_px"],
                "rotation_displacement_x_px": derived["rotation_displacement_x_px"],
                "rotation_displacement_y_px": derived["rotation_displacement_y_px"],
                "road_depth_m": derived["road_depth_m"],
                "road_roi_y_fraction": derived["road_roi_y_fraction"],
                "camera_fx_px": derived["camera_fx_px"],
                "camera_fy_px": derived["camera_fy_px"],
                "camera_cx_px": derived["camera_cx_px"],
                "camera_cy_px": derived["camera_cy_px"],
            }
            raw_packet = kitti_sensor_packet(
                oxts,
                derived,
                full_prior=False,
            )
            full_packet = kitti_sensor_packet(
                oxts,
                derived,
                full_prior=True,
            )
            metadata.update(
                {
                    "metadata_schema": "rmrnet_kitti_oxts_practical_sensor_v1",
                    "metadata_scope": (
                        "real OXTS/camera fields plus a separately labelled "
                        "generator-aligned upper-bound packet"
                    ),
                    "practical_sensor_packet": full_packet,
                    "raw_practical_sensor_packet": raw_packet,
                    "sensor_packet_variant": "generator_aligned_upper_bound",
                    "raw_sensor_packet_uses_clean_target": False,
                    "raw_sensor_packet_uses_renderer_kernel": False,
                    "full_sensor_packet_is_upper_bound": True,
                    "observable_sensor_code": observable_code_from_packet(
                        torch.tensor(raw_packet, dtype=torch.float32),
                        gyro_full_scale=4.0,
                    ).tolist(),
                }
            )
            (folders["metadata"] / f"{Path(out_name).stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if write_private_calibration:
                private = {
                    "training_cause_target_code": code_from_metadata(
                        metadata,
                        encoding="legacy",
                    ).tolist(),
                    "training_physical_target_code": code_from_metadata(
                        metadata,
                        encoding="axial_v2",
                    ).tolist(),
                    "training_target_is_model_input": False,
                }
                (
                    folders["private_calibration"]
                    / f"{Path(out_name).stem}.json"
                ).write_text(json.dumps(private, indent=2), encoding="utf-8")
            rows.append(
                {
                    "file": out_name,
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "speed_mps": derived["speed_mps"],
                    "yaw_rate_radps": derived["yaw_rate_radps"],
                    "vibration_mps2": derived["vibration_mps2"],
                    "blur_length_px": derived["blur_length_px"],
                    "blur_angle_deg": derived["blur_angle_deg"],
                    "telemetry_strength": derived["telemetry_strength"],
                    "blur_scale": derived["blur_scale"],
                }
            )
    summary_path = out_root / "metadata_summary.csv"
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_root / "preparation_protocol.json").write_text(
        json.dumps(
            {
                "scenario": scenario,
                "sequences": list(sequences),
                "frames": len(rows),
                "max_side": max_side,
                "exposure_ms": exposure_ms,
                "blur_scale": blur_scale,
                "road_depth_m": road_depth_m,
                "road_roi_y_fraction": road_roi_y,
                "jpeg_quality": jpeg_quality,
                "calibration_file": str(calibration_file),
                "calibration_sha256": sha256(calibration_file),
                "camera_02_intrinsics": calibration,
                "target_or_detector_used": False,
                "model": "representative-road-point pinhole optical flow followed by one global line-spread function",
                "limitation": "The true blur is depth dependent and spatially varying; this is a controlled proxy.",
                "sensor_packet_schema": "rmrnet_kitti_oxts_practical_sensor_v1",
                "raw_packet_uses_renderer_kernel": False,
                "full_packet_role": "generator-aligned upper bound",
                "private_calibration_targets_written": write_private_calibration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    args = parse_args()
    if args.train_sequence is None:
        args.train_sequence = ["2011_09_26_drive_0001_sync", "2011_09_26_drive_0002_sync"]
    if args.test_sequence is None:
        args.test_sequence = ["2011_09_26_drive_0005_sync"]
    date_root = Path(args.kitti_date_root)
    calibration_file = Path(args.calibration_file) if args.calibration_file else date_root / "calib_cam_to_cam.txt"
    calibration = read_camera_calibration(calibration_file)
    train_rows = []
    if not args.test_only:
        train_rows = write_split(
            date_root=date_root,
            sequences=args.train_sequence,
            out_root=Path(args.train_out),
            scenario=args.scenario,
            max_side=args.max_side,
            exposure_ms=args.exposure_ms,
            blur_scale=args.blur_scale,
            calibration=calibration,
            calibration_file=calibration_file,
            road_depth_m=args.road_depth_m,
            road_roi_y=args.road_roi_y,
            jpeg_quality=args.jpeg_quality,
            write_private_calibration=True,
        )
    test_rows = write_split(
        date_root=date_root,
        sequences=args.test_sequence,
        out_root=Path(args.test_out),
        scenario=args.scenario,
        max_side=args.max_side,
        exposure_ms=args.exposure_ms,
        blur_scale=args.blur_scale,
        calibration=calibration,
        calibration_file=calibration_file,
        road_depth_m=args.road_depth_m,
        road_roi_y=args.road_roi_y,
        jpeg_quality=args.jpeg_quality,
        write_private_calibration=False,
    )
    print(
        json.dumps(
            {
                "dataset": "KITTI raw real OXTS metadata",
                "scenario": args.scenario,
                "train_images": len(train_rows),
                "test_images": len(test_rows),
                "train_sequences": args.train_sequence,
                "test_sequences": args.test_sequence,
                "train_out": args.train_out,
                "test_out": args.test_out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
