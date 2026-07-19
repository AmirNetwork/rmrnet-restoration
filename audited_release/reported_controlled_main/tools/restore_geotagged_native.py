# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.demoe_adapter import DeMoEAdapter
from baselines.dfpir_adapter import DFPIRAdapter
from baselines.instructir_adapter import InstructIRAdapter, generic_road_prompt, metadata_prompt
from baselines.nafnet_road import NAFNetRoad
from rcadnet import RCADNet, code_from_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore geotagged cam1 images at native output resolution. Models "
            "may process overlapping tiles internally, but saved images keep the "
            "original width/height."
        )
    )
    parser.add_argument("--metadata-jsonl", type=Path, default=Path("experiments/geotagged_cam1_native/geotagged_metadata.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("experiments/geotagged_cam1_native/restored"))
    parser.add_argument(
        "--models",
        default="raw,rmr_blind,rmr_metadata,nafnet,dfpir,demoe_auto,demoe_scenario,instructir_generic,instructir_metadata",
        help="Comma-separated models to run.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. 0 means all records.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--rmr-weights", type=Path, default=Path("runs/fresh_final_rmr_task_pcm_mixed/rcadnet_best.pth"))
    parser.add_argument("--nafnet-weights", type=Path, default=Path("runs/nafnet_road_combined_12ep/nafnet_last.pth"))
    parser.add_argument("--dfpir-weights", type=Path, default=Path("weights/dfpir/DFPIR-5D-pn31.29-0.8889_pr37.62-0.9779_ph31.64-0.9794_pb28.82-0.8734_pl23.82-0.8428_avr30.64-0.9125.pth.tar"))
    parser.add_argument("--demoe-weights", type=Path, default=Path("weights/demoe/DeMoE.pt"))
    parser.add_argument("--instructir-image-weights", type=Path, default=Path("weights/instructir/im_instructir-7d.pt"))
    parser.add_argument("--instructir-lm-weights", type=Path, default=Path("weights/instructir/lm_instructir-7d.pt"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_records(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if limit > 0:
        records = records[:limit]
    return records


def load_rcadnet(weights: Path, device: torch.device) -> RCADNet:
    checkpoint = torch.load(weights, map_location=device)
    arch = checkpoint.get("arch", {})
    model = RCADNet(
        width=arch.get("width", 32),
        code_dim=arch.get("code_dim", 8),
        use_defect_attention=arch.get("use_defect_attention", True),
        use_estimated_code=arch.get("use_estimated_code", False),
        code_fusion=arch.get("code_fusion", "scenario"),
        block_type=arch.get("block_type", "simple"),
        attention_type=arch.get("attention_type", "edge"),
        conditioning=arch.get("conditioning", "film"),
        use_tdac_head=arch.get("use_tdac_head", False),
        detail_preserve=arch.get("detail_preserve", False),
        detail_gain=arch.get("detail_gain", 0.20),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def load_nafnet(weights: Path, device: torch.device) -> NAFNetRoad:
    checkpoint = torch.load(weights, map_location=device)
    arch = checkpoint.get("arch", {})
    model = NAFNetRoad(width=arch.get("width", 32)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:  # type: ignore[comparison-overlap]
            return default
        return float(value)
    except Exception:
        return default


def parse_exposure_ms(value: Any, default: float = 2.0) -> float:
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return v * 1000.0 if v < 1.0 else v
    text = str(value)
    if "/" in text:
        a, b = text.split("/", 1)
        return 1000.0 * parse_float(a, 0.0) / max(parse_float(b, 1.0), 1e-6)
    v = parse_float(text, default)
    return v * 1000.0 if v < 1.0 else v


def rmr_metadata_from_record(record: dict[str, Any]) -> dict[str, float]:
    csv = record.get("csv", {}) or {}
    exif = record.get("exif", {}) or {}
    exposure_ms = parse_exposure_ms(exif.get("ExposureTime", ""), default=2.0)
    brightness = parse_float(exif.get("BrightnessValue", ""), default=3.0)
    std_e = parse_float(csv.get("std_east", 0.0))
    std_n = parse_float(csv.get("std_north", 0.0))
    std_h = parse_float(csv.get("std_ht", 0.0))
    std_roll = parse_float(csv.get("std_roll", 0.0))
    std_pitch = parse_float(csv.get("std_pitch", 0.0))
    std_yaw = parse_float(csv.get("std_yaw", 0.0))
    pose_uncertainty = math.sqrt(std_e * std_e + std_n * std_n + std_h * std_h)
    angular_uncertainty = math.sqrt(std_roll * std_roll + std_pitch * std_pitch + std_yaw * std_yaw)
    low_light = max(0.0, min((1.0 - brightness) / 4.0, 1.0))
    return {
        "gyro_x": min(std_pitch / 0.25, 1.0),
        "gyro_y": min(std_roll / 0.25, 1.0),
        "accel_norm": min((pose_uncertainty * 10.0 + angular_uncertainty) / 5.0, 1.0),
        "speed_mps": 0.0,
        "exposure_ms": exposure_ms,
        "defocus_score": 0.0,
        "noise_score": low_light,
        "low_light_score": low_light,
        "jpeg_quality": 95.0,
    }


def starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(tile - overlap, 1)
    values = list(range(0, max(length - tile, 0), step))
    values.append(length - tile)
    return sorted(set(values))


def restore_tiled(
    tensor: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    tile: int,
    overlap: int,
) -> torch.Tensor:
    _, _, height, width = tensor.shape
    if height <= tile and width <= tile:
        return fn(tensor).clamp(0.0, 1.0)
    device = tensor.device
    accum = torch.zeros_like(tensor)
    weight = torch.zeros((1, 1, height, width), device=device, dtype=tensor.dtype)
    for y in starts(height, tile, overlap):
        for x in starts(width, tile, overlap):
            patch = tensor[..., y : y + tile, x : x + tile]
            restored = fn(patch).clamp(0.0, 1.0)
            accum[..., y : y + restored.shape[-2], x : x + restored.shape[-1]] += restored
            weight[..., y : y + restored.shape[-2], x : x + restored.shape[-1]] += 1.0
    return (accum / weight.clamp_min(1.0)).clamp(0.0, 1.0)


def save_tensor(path: Path, tensor: torch.Tensor, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = TF.to_pil_image(tensor[0].detach().cpu())
    image.save(path, quality=quality, subsampling=0)


def copy_raw(record: dict[str, Any], out_path: Path, skip_existing: bool) -> float:
    if skip_existing and out_path.exists():
        return 0.0
    t0 = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record["image"], out_path)
    return time.perf_counter() - t0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    records = load_records(args.metadata_jsonl, args.limit)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, Any] = {}
    rows = []

    for model_name in model_names:
        if model_name == "raw":
            loaded[model_name] = None
        elif model_name.startswith("rmr"):
            loaded.setdefault("rmr", load_rcadnet(args.rmr_weights, device))
        elif model_name == "nafnet":
            loaded[model_name] = load_nafnet(args.nafnet_weights, device)
        elif model_name == "dfpir":
            loaded[model_name] = DFPIRAdapter(args.dfpir_weights, device=str(device), use_clip=False)
        elif model_name.startswith("demoe"):
            task = "scenario" if model_name == "demoe_scenario" else "auto"
            loaded[model_name] = DeMoEAdapter(args.demoe_weights, device=device, task=task)
        elif model_name.startswith("instructir"):
            loaded.setdefault(
                "instructir",
                InstructIRAdapter(args.instructir_image_weights, args.instructir_lm_weights, device=device),
            )
        else:
            raise ValueError(f"Unknown model '{model_name}'")

    with torch.inference_mode():
        for model_name in model_names:
            method_dir = args.out / model_name
            image_dir = method_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            for index, record in enumerate(records):
                src = Path(record["image"])
                out_path = image_dir / src.name
                if model_name == "raw":
                    elapsed = copy_raw(record, out_path, args.skip_existing)
                    width = record["exif"]["width"]
                    height = record["exif"]["height"]
                elif args.skip_existing and out_path.exists():
                    elapsed = 0.0
                    width = record["exif"]["width"]
                    height = record["exif"]["height"]
                else:
                    with Image.open(src) as image:
                        width, height = image.size
                        tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
                    t0 = time.perf_counter()
                    if model_name == "rmr_blind":
                        model = loaded["rmr"]
                        restored = restore_tiled(tensor, lambda patch: model(patch, None), tile=args.tile, overlap=args.overlap)
                    elif model_name == "rmr_metadata":
                        model = loaded["rmr"]
                        meta = rmr_metadata_from_record(record)
                        code = code_from_metadata(meta, device=device).unsqueeze(0)
                        restored = restore_tiled(tensor, lambda patch: model(patch, code), tile=args.tile, overlap=args.overlap)
                    elif model_name == "nafnet":
                        model = loaded[model_name]
                        restored = restore_tiled(tensor, lambda patch: model(patch), tile=args.tile, overlap=args.overlap)
                    elif model_name == "dfpir":
                        model = loaded[model_name]
                        restored = restore_tiled(tensor, lambda patch: model(patch, "native_real"), tile=args.tile, overlap=args.overlap)
                    elif model_name.startswith("demoe"):
                        model = loaded[model_name]
                        restored = restore_tiled(tensor, lambda patch: model(patch, scenario="native_real"), tile=args.tile, overlap=args.overlap)
                    elif model_name == "instructir_generic":
                        model = loaded["instructir"]
                        prompt = generic_road_prompt()
                        restored = restore_tiled(tensor, lambda patch: model(patch, prompt), tile=args.tile, overlap=args.overlap)
                    elif model_name == "instructir_metadata":
                        model = loaded["instructir"]
                        prompt = metadata_prompt(record)
                        restored = restore_tiled(tensor, lambda patch: model(patch, prompt), tile=args.tile, overlap=args.overlap)
                    else:
                        raise ValueError(model_name)
                    elapsed = time.perf_counter() - t0
                    save_tensor(out_path, restored, args.jpeg_quality)
                    del tensor, restored
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                rows.append(
                    {
                        "model": model_name,
                        "image": src.name,
                        "output": str(out_path),
                        "width": width,
                        "height": height,
                        "native_resolution_preserved": True,
                        "runtime_s": elapsed,
                    }
                )
                print(json.dumps(rows[-1]), flush=True)

            report_path = method_dir / "restore_manifest.jsonl"
            with report_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    if row["model"] == model_name:
                        handle.write(json.dumps(row) + "\n")

    summary_path = args.out / "restore_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
