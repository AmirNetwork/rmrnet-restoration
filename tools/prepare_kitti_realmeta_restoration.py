from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


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
    parser.add_argument("--jpeg-quality", type=int, default=96)
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


def telemetry_blur(oxts: dict[str, float], exposure_ms: float, blur_scale: float) -> tuple[int, float, dict[str, float]]:
    speed_mps = math.sqrt(oxts["vf"] ** 2 + oxts["vl"] ** 2 + oxts["vu"] ** 2)
    lateral_accel = abs(oxts["al"])
    forward_accel = abs(oxts["af"])
    vibration = math.sqrt(forward_accel**2 + lateral_accel**2)
    yaw_rate = oxts["wz"]
    pitch_roll_rate = math.sqrt(oxts["wx"] ** 2 + oxts["wy"] ** 2)

    speed_score = min(speed_mps / 18.0, 1.0)
    yaw_score = min(abs(yaw_rate) / 0.09, 1.0)
    vib_score = min(vibration / 4.5, 1.0)
    strength = min(max(0.22 + 0.42 * speed_score + 0.24 * yaw_score + 0.12 * vib_score, 0.18), 1.0)

    # Direction is a calibrated camera-plane proxy from real vehicle telemetry.
    # Forward motion mostly smears horizontally; yaw/lateral acceleration tilt the kernel.
    angle = math.degrees(math.atan2(24.0 * yaw_rate + 0.18 * oxts["al"], max(abs(oxts["vf"]), 1.0)))
    angle = max(min(angle * 2.0, 48.0), -48.0)
    exposure_scale = max(exposure_ms / 12.0, 0.25)
    length = int(round(3 + 20 * strength * blur_scale * math.sqrt(exposure_scale)))
    if length % 2 == 0:
        length += 1

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


def degrade_with_metadata(image: Image.Image, oxts: dict[str, float], exposure_ms: float, blur_scale: float) -> tuple[Image.Image, dict[str, float]]:
    length, angle, derived = telemetry_blur(oxts, exposure_ms, blur_scale)
    arr = np.asarray(image, dtype=np.uint8)
    blurred_arr = cv2.filter2D(arr, ddepth=-1, kernel=motion_kernel(length, angle), borderType=cv2.BORDER_REFLECT101)
    blurred = Image.fromarray(blurred_arr, mode="RGB")
    return blurred, derived


def make_dirs(root: Path, scenario: str) -> dict[str, Path]:
    base = root / "scenarios" / scenario
    folders = {
        "input": base / "input",
        "gt": base / "gt",
        "metadata": base / "metadata",
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
    jpeg_quality: int,
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
            degraded, derived = degrade_with_metadata(clean, oxts, exposure_ms, blur_scale)

            out_name = f"{sequence.replace('_sync', '')}_{frame_id}.jpg"
            clean.save(folders["gt"] / out_name, quality=jpeg_quality)
            degraded.save(folders["input"] / out_name, quality=jpeg_quality)

            metadata = {
                "metadata_source": "real_kitti_oxts_calibrated_blur",
                "dataset": "KITTI raw",
                "sequence": sequence,
                "frame_id": frame_id,
                "real_metadata_fields_used": "speed_mps, angular rates, forward/lateral acceleration",
                "calibration_note": (
                    "blur_length_px and blur_angle_deg are deterministic estimates from real OXTS "
                    "speed/gyro/acceleration plus the stated camera exposure and blur-scale setting; no clean target or "
                    "detector output is used."
                ),
                "gyro_x": abs(oxts["wx"]),
                "gyro_y": abs(oxts["wy"]),
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
            }
            (folders["metadata"] / f"{Path(out_name).stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
    return rows


def main() -> None:
    args = parse_args()
    if args.train_sequence is None:
        args.train_sequence = ["2011_09_26_drive_0001_sync", "2011_09_26_drive_0002_sync"]
    if args.test_sequence is None:
        args.test_sequence = ["2011_09_26_drive_0005_sync"]
    date_root = Path(args.kitti_date_root)
    train_rows = write_split(
        date_root=date_root,
        sequences=args.train_sequence,
        out_root=Path(args.train_out),
        scenario=args.scenario,
        max_side=args.max_side,
        exposure_ms=args.exposure_ms,
        blur_scale=args.blur_scale,
        jpeg_quality=args.jpeg_quality,
    )
    test_rows = write_split(
        date_root=date_root,
        sequences=args.test_sequence,
        out_root=Path(args.test_out),
        scenario=args.scenario,
        max_side=args.max_side,
        exposure_ms=args.exposure_ms,
        blur_scale=args.blur_scale,
        jpeg_quality=args.jpeg_quality,
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
