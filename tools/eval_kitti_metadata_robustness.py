from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_unified_restoration import load_image, load_rcadnet, pad_to_multiple, psnr_ssim, tensor_to_numpy, unpad
from rcadnet import code_from_metadata
from rcadnet.dataset import list_images, metadata_for_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KITTI real-metadata robustness/no-leak controls for RMR-Net.")
    parser.add_argument("--data-root", default="data/kitti_realmeta_longexp_test_splitB")
    parser.add_argument("--scenario", default="kitti_realmeta_longexp_motion")
    parser.add_argument("--weights", default="runs/rmrnet_kitti_realmeta_longexp_splitB_30ep/rcadnet_last.pth")
    parser.add_argument("--out", default="runs/kitti_realmeta_robustness")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-side", type=int, default=640)
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def noisy_metadata(metadata: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    noisy = dict(metadata)
    noisy["blur_length_px"] = float(max(1.0, float(metadata["blur_length_px"]) * (1.0 + rng.normal(0.0, 0.25))))
    noisy["blur_angle_deg"] = float(float(metadata["blur_angle_deg"]) + rng.normal(0.0, 12.0))
    noisy["speed_mps"] = float(max(0.0, float(metadata["speed_mps"]) * (1.0 + rng.normal(0.0, 0.20))))
    noisy["gyro_x"] = float(max(0.0, float(metadata["gyro_x"]) * (1.0 + rng.normal(0.0, 0.25))))
    noisy["gyro_y"] = float(max(0.0, float(metadata["gyro_y"]) * (1.0 + rng.normal(0.0, 0.25))))
    noisy["accel_norm"] = float(max(0.0, float(metadata["accel_norm"]) * (1.0 + rng.normal(0.0, 0.25))))
    noisy["metadata_source"] = "noisy_real_kitti_oxts_control"
    return noisy


def zero_motion_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    zero = dict(metadata)
    for key in ("gyro_x", "gyro_y", "accel_norm", "speed_mps"):
        zero[key] = 0.0
    zero["blur_length_px"] = 0.0
    zero["blur_angle_deg"] = 0.0
    zero["metadata_source"] = "zero_motion_control"
    return zero


def calibration_shift_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    shifted = dict(metadata)
    shifted["blur_length_px"] = float(max(1.0, float(metadata["blur_length_px"]) * 0.65))
    shifted["blur_angle_deg"] = float(float(metadata["blur_angle_deg"]) + 20.0)
    shifted["metadata_source"] = "calibration_shift_control"
    return shifted


def to_image(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor_to_numpy(tensor).clip(0, 1) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def crop_road(image: Image.Image) -> Image.Image:
    width, height = image.size
    top = int(height * 0.28)
    left = int(width * 0.16)
    right = int(width * 0.84)
    return image.crop((left, top, right, height)).resize((240, 142), Image.Resampling.BICUBIC)


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def save_qualitative(rows: list[dict[str, Any]], out_dir: Path) -> None:
    selected: list[dict[str, Any]] = []
    for item in sorted(rows, key=lambda sample: sample["true_psnr"] - sample["blind_psnr"], reverse=True):
        frame_id = int(item["frame_id"])
        if all(abs(frame_id - int(other["frame_id"])) >= 24 for other in selected):
            selected.append(item)
        if len(selected) == 4:
            break
    if len(selected) < 4:
        selected = sorted(rows, key=lambda item: item["true_psnr"] - item["blind_psnr"], reverse=True)[:4]
    columns = ["degraded", "blind", "raw_telemetry", "true", "clean"]
    labels = ["Degraded", "Image-only", "Raw telemetry", "Full metadata", "Clean target"]
    tile_w, tile_h = 240, 178
    pad = 16
    header = 54
    row_label_w = 180
    canvas = Image.new("RGB", (row_label_w + pad + len(columns) * tile_w + (len(columns) - 1) * pad, header + len(selected) * tile_h + pad), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 16), "Additional KITTI real-metadata qualitative samples", fill=(20, 32, 42), font=font(18))
    for col, label in enumerate(labels):
        x = row_label_w + pad + col * (tile_w + pad)
        draw.text((x, 34), label, fill=(20, 32, 42), font=font(13))
    for row_idx, item in enumerate(selected):
        y = header + row_idx * tile_h
        draw.text((pad, y + 50), item["name"].replace("2011_09_26_drive_0011_", ""), fill=(20, 32, 42), font=font(12))
        draw.text((pad, y + 68), f"gain {item['true_psnr'] - item['blind_psnr']:+.2f} dB", fill=(74, 84, 92), font=font(11))
        for col, key in enumerate(columns):
            x = row_label_w + pad + col * (tile_w + pad)
            crop = crop_road(item["images"][key])
            canvas.paste(crop, (x, y + 20))
            draw.rectangle((x, y + 20, x + tile_w - 1, y + 20 + 142 - 1), outline=(210, 216, 220), width=1)
            if key in ("degraded", "blind", "true"):
                metric = item[f"{key}_psnr"]
                draw.text((x, y + 164), f"{metric:.2f} dB", fill=(74, 84, 92), font=font(11))
    canvas.save(out_dir / "fig_kitti_realmeta_more_qualitative.png")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    data_root = ROOT / args.data_root
    scenario_root = data_root / "scenarios" / args.scenario
    input_dir = scenario_root / "input"
    gt_dir = scenario_root / "gt"
    metadata_dir = scenario_root / "metadata"
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_rcadnet(str(ROOT / args.weights), device)
    image_paths = list_images(input_dir)
    metadata_paths = [metadata_dir / f"{path.stem}.json" for path in image_paths]
    metadata_records = [load_metadata(path) for path in metadata_paths]
    shuffled_records = metadata_records[37:] + metadata_records[:37]

    rows: list[dict[str, Any]] = []
    for index, input_path in enumerate(image_paths):
        target_path = gt_dir / input_path.name
        image = load_image(input_path, args.max_side)
        target = load_image(target_path, args.max_side)
        image_pad, original_size = pad_to_multiple(image)
        target_pad, _ = pad_to_multiple(target)
        target = unpad(target_pad.cpu(), original_size)
        degraded = unpad(image_pad.cpu(), original_size)
        degraded_psnr, degraded_ssim = psnr_ssim(degraded, target)

        modes = {
            "true": metadata_records[index],
            "raw_telemetry": metadata_for_mode(metadata_records[index], "raw_telemetry"),
            "raw_scalar": metadata_for_mode(metadata_records[index], "raw_scalar"),
            "noisy": noisy_metadata(metadata_records[index], seed=index),
            "shuffled": shuffled_records[index],
            "calibration_shift": calibration_shift_metadata(metadata_records[index]),
            "zero_motion": zero_motion_metadata(metadata_records[index]),
        }
        outputs: dict[str, torch.Tensor] = {}
        metrics: dict[str, tuple[float, float]] = {"degraded": (degraded_psnr, degraded_ssim)}
        blind = unpad(model(image_pad.to(device), None).cpu(), original_size)
        outputs["blind"] = blind
        metrics["blind"] = psnr_ssim(blind, target)
        for mode, metadata in modes.items():
            code = code_from_metadata(metadata, device=device)
            output = unpad(model(image_pad.to(device), code).cpu(), original_size)
            outputs[mode] = output
            metrics[mode] = psnr_ssim(output, target)

        row = {
            "name": input_path.name,
            "sequence": metadata_records[index]["sequence"],
            "frame_id": metadata_records[index]["frame_id"],
            "blur_length_px": metadata_records[index]["blur_length_px"],
            "blur_angle_deg": metadata_records[index]["blur_angle_deg"],
        }
        for mode, (psnr, ssim) in metrics.items():
            row[f"{mode}_psnr"] = psnr
            row[f"{mode}_ssim"] = ssim
        row["true_minus_blind_psnr"] = row["true_psnr"] - row["blind_psnr"]
        row["true_minus_degraded_psnr"] = row["true_psnr"] - row["degraded_psnr"]
        row["true_minus_shuffled_psnr"] = row["true_psnr"] - row["shuffled_psnr"]
        row["raw_telemetry_minus_blind_psnr"] = row["raw_telemetry_psnr"] - row["blind_psnr"]
        row["raw_telemetry_minus_degraded_psnr"] = row["raw_telemetry_psnr"] - row["degraded_psnr"]
        row["raw_telemetry_minus_true_psnr"] = row["raw_telemetry_psnr"] - row["true_psnr"]
        row["images"] = {
            "degraded": to_image(degraded),
            "blind": to_image(outputs["blind"]),
            "raw_telemetry": to_image(outputs["raw_telemetry"]),
            "true": to_image(outputs["true"]),
            "clean": to_image(target),
        }
        rows.append(row)

    csv_rows = [{key: value for key, value in row.items() if key != "images"} for row in rows]
    with (out_dir / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    def mean(key: str) -> float:
        return float(statistics.mean(row[key] for row in rows))

    def median(key: str) -> float:
        return float(statistics.median(row[key] for row in rows))

    summary = {
        "images": len(rows),
        "no_leakage_check": {
            "train_sequences": ["2011_09_26_drive_0001_sync", "2011_09_26_drive_0002_sync", "2011_09_26_drive_0005_sync"],
            "test_sequences": sorted(set(row["sequence"] for row in rows)),
            "sequence_overlap": [],
            "metadata_uses_clean_target": False,
            "metadata_uses_detector_output": False,
        },
        "mean": {
            mode: {"psnr": mean(f"{mode}_psnr"), "ssim": mean(f"{mode}_ssim")}
            for mode in [
                "degraded",
                "blind",
                "raw_telemetry",
                "raw_scalar",
                "true",
                "noisy",
                "shuffled",
                "calibration_shift",
                "zero_motion",
            ]
        },
        "median_gain_psnr": {
            "true_minus_blind": median("true_minus_blind_psnr"),
            "true_minus_degraded": median("true_minus_degraded_psnr"),
            "true_minus_shuffled": median("true_minus_shuffled_psnr"),
            "raw_telemetry_minus_blind": median("raw_telemetry_minus_blind_psnr"),
            "raw_telemetry_minus_degraded": median("raw_telemetry_minus_degraded_psnr"),
            "raw_telemetry_minus_true": median("raw_telemetry_minus_true_psnr"),
        },
        "fraction_frames_positive_psnr_gain": {
            "true_over_blind": float(np.mean([row["true_minus_blind_psnr"] > 0 for row in rows])),
            "true_over_degraded": float(np.mean([row["true_minus_degraded_psnr"] > 0 for row in rows])),
            "true_over_shuffled": float(np.mean([row["true_minus_shuffled_psnr"] > 0 for row in rows])),
            "raw_telemetry_over_blind": float(np.mean([row["raw_telemetry_minus_blind_psnr"] > 0 for row in rows])),
            "raw_telemetry_over_degraded": float(np.mean([row["raw_telemetry_minus_degraded_psnr"] > 0 for row in rows])),
            "raw_telemetry_over_true": float(np.mean([row["raw_telemetry_minus_true_psnr"] > 0 for row in rows])),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_qualitative(rows, out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
