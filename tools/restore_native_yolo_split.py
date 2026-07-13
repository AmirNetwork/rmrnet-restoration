from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from PIL import Image
from PIL import ImageEnhance
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.demoe_adapter import DeMoEAdapter
from baselines.dfpir_adapter import DFPIRAdapter
from baselines.instructir_adapter import InstructIRAdapter, generic_road_prompt
from baselines.nafnet_road import NAFNetRoad
from rcadnet import RCADNet, code_from_metadata, code_from_scenario


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore a YOLO split at native resolution using overlapping tiles, "
            "then copy labels and metadata so the restored images can be evaluated "
            "with the same detector protocol."
        )
    )
    parser.add_argument("--data", required=True, type=Path, help="Source YOLO data.yaml.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario", default="", help="Scenario label used for metadata-free baselines.")
    parser.add_argument("--out", required=True, type=Path, help="Output root. One subfolder is created per model.")
    parser.add_argument(
        "--models",
        default="rmr_blind,rmr_metadata,rmr_metadata_gated,nafnet,dfpir,demoe_auto,demoe_scenario,instructir_generic,instructir_metadata",
        help=(
            "Comma-separated models: raw, rmr_blind, rmr_metadata, rmr_metadata_gated, "
            "nafnet, dfpir, demoe_auto, demoe_scenario, instructir_generic, instructir_metadata."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rcadnet-weights", type=Path, default=Path("runs/fresh_final_rmr_task_pcm_mixed/rcadnet_best.pth"))
    parser.add_argument("--nafnet-weights", type=Path, default=Path("runs/nafnet_road_combined_12ep/nafnet_last.pth"))
    parser.add_argument(
        "--dfpir-weights",
        type=Path,
        default=Path(
            "weights/dfpir/DFPIR-5D-pn31.29-0.8889_pr37.62-0.9779_ph31.64-0.9794_pb28.82-0.8734_pl23.82-0.8428_avr30.64-0.9125.pth.tar"
        ),
    )
    parser.add_argument("--demoe-weights", type=Path, default=Path("weights/demoe/DeMoE.pt"))
    parser.add_argument("--instructir-image-weights", type=Path, default=Path("weights/instructir/im_instructir-7d.pt"))
    parser.add_argument("--instructir-lm-weights", type=Path, default=Path("weights/instructir/lm_instructir-7d.pt"))
    parser.add_argument("--residual-strength", type=float, default=1.0)
    return parser.parse_args()


def load_yaml(data: Path) -> dict[str, Any]:
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    root = Path(config.get("path", data.parent))
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    config["_root"] = root
    return config


def yolo_names(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("names", {})
    if isinstance(raw, list):
        return {idx: str(name) for idx, name in enumerate(raw)}
    return {int(idx): str(name) for idx, name in dict(raw).items()}


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
    if hasattr(model, "load_pretrained"):
        load_report = model.load_pretrained(checkpoint, strict=False)
        missing = load_report["missing_keys"]
        unexpected = load_report["unexpected_keys"]
        skipped_shape = load_report["skipped_shape"]
        skipped_missing = load_report["skipped_missing"]
    else:
        incompatible = model.load_state_dict(checkpoint["model"], strict=False)
        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys
        skipped_shape = []
        skipped_missing = []
    if missing or unexpected or skipped_shape or skipped_missing:
        print(
            json.dumps(
                {
                    "checkpoint_compatibility": str(weights),
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                    "skipped_shape": skipped_shape,
                    "skipped_missing": skipped_missing,
                    "policy": "strict=False; newly introduced modules use their initialized weights",
                }
            ),
            flush=True,
        )
    model.eval()
    return model


def load_nafnet(weights: Path, device: torch.device) -> NAFNetRoad:
    checkpoint = torch.load(weights, map_location=device)
    arch = checkpoint.get("arch", {})
    model = NAFNetRoad(width=arch.get("width", 32)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def grid_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, max(1, length - tile + 1), step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


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
    output = torch.zeros_like(tensor)
    weight = torch.zeros((1, 1, height, width), device=tensor.device, dtype=tensor.dtype)
    for y in grid_starts(height, tile, overlap):
        for x in grid_starts(width, tile, overlap):
            patch = tensor[..., y : y + tile, x : x + tile]
            restored = fn(patch).clamp(0.0, 1.0)
            tile_weight = smooth_tile_weight(
                restored.shape[-2],
                restored.shape[-1],
                overlap=overlap,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            output[..., y : y + restored.shape[-2], x : x + restored.shape[-1]] += restored * tile_weight
            weight[..., y : y + restored.shape[-2], x : x + restored.shape[-1]] += tile_weight
    return (output / weight.clamp_min(1.0)).clamp(0.0, 1.0)


def smooth_tile_weight(
    height: int,
    width: int,
    *,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Raised-edge tile weights reduce visible seams in native-resolution output.

    Restoration models can shift brightness slightly from tile to tile. Uniform
    averaging leaves block edges visible, especially on smooth asphalt. This
    separable ramp gives tile centers higher weight and softly blends overlap
    regions while preserving exact native output size.
    """

    edge_y = max(1, min(overlap, height // 2))
    edge_x = max(1, min(overlap, width // 2))
    wy = torch.ones(height, device=device, dtype=dtype)
    wx = torch.ones(width, device=device, dtype=dtype)
    ramp_y = torch.linspace(0.05, 1.0, edge_y, device=device, dtype=dtype)
    ramp_x = torch.linspace(0.05, 1.0, edge_x, device=device, dtype=dtype)
    wy[:edge_y] = ramp_y
    wy[-edge_y:] = torch.flip(ramp_y, dims=(0,))
    wx[:edge_x] = ramp_x
    wx[-edge_x:] = torch.flip(ramp_x, dims=(0,))
    return wy.view(1, 1, height, 1) * wx.view(1, 1, 1, width)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_tree_files(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(p for p in src_dir.iterdir() if p.is_file()):
        link_or_copy(src, dst_dir / src.name)


def metadata_prompt_from_file(metadata: dict[str, Any], scenario: str) -> str:
    pose = metadata.get("pose_csv", {}) or {}
    bits = [
        "Restore this native-resolution road inspection image while preserving crack edges, pothole rims, patches, and pavement texture.",
        f"Observed degradation scenario: {metadata.get('scenario', scenario) or scenario}.",
        f"Exposure {metadata.get('exposure_ms', 'unknown')} ms.",
        f"Blur angle {metadata.get('blur_angle_deg', 'unknown')} degrees and blur length {metadata.get('blur_length_px', 'unknown')} pixels.",
        f"Pose yaw {pose.get('yaw', 'unknown')}, pitch {pose.get('pitch', 'unknown')}, roll {pose.get('roll', 'unknown')}.",
        "Avoid hallucinating new road defects.",
    ]
    return " ".join(bits)


def metadata_has_reliable_degradation_signal(metadata: dict[str, Any]) -> bool:
    """Return True when native metadata contains an actual degradation cue.

    The field experiment has real EXIF/pose/geotag metadata, but not every
    metadata field is a restoration command. Latitude, longitude, and pose are
    useful for logging and route audit, yet they do not by themselves prove that
    an image is blurred or low-light. The deployed metadata-gated RMR variant
    therefore uses the metadata branch only when at least one calibrated
    degradation-relevant cue is present:

        z_meta = code_from_metadata(m),  use z_meta only if reliability(m)=1.

    Otherwise it falls back to the image-estimated degradation code. This keeps
    high-quality native frames from being over-restored by off-domain metadata.
    """

    direct_scores = [
        float(metadata.get("defocus_score") or 0.0),
        float(metadata.get("noise_score") or 0.0),
        float(metadata.get("low_light_score") or 0.0),
    ]
    if max(direct_scores, default=0.0) >= 0.05:
        return True
    if metadata.get("blur_length_px") is not None or metadata.get("blur_angle_deg") is not None:
        return True
    jpeg_quality = metadata.get("jpeg_quality")
    if jpeg_quality is not None and float(jpeg_quality) < 75.0:
        return True
    exposure_ms = float(metadata.get("exposure_ms") or 0.0)
    speed_mps = float(metadata.get("speed_mps") or 0.0)
    yaw_rate = abs(float(metadata.get("raw_oxts_yaw_rate_radps") or 0.0))
    accel = abs(float(metadata.get("accel_norm") or 0.0))
    # Exposure must be measured for speed/yaw/acceleration to imply image-plane
    # blur. In the GT49 Sony export exposure is usually missing/zero, so this
    # guard correctly prevents an unsupported full-strength metadata correction.
    return exposure_ms > 0.0 and (speed_mps * exposure_ms > 120.0 or yaw_rate > 0.12 or accel > 2.5)


def native_evidence_pass(tensor: torch.Tensor) -> torch.Tensor:
    """Conservative native-frame branch for high-quality monitoring images.

    When the metadata contains pose/geotag context but no measured blur,
    low-light, defocus, noise, or compression cue, the deployed model should not
    apply full residual deblurring.  The paper denotes this as the native safety
    gate:

        I_o = I_d,                         if reliability(m)=0 and quality is high
        I_o = N(I_d),                      for native evidence preservation
        I_o = R_theta(I_d, z_meta),        if reliability(m)=1

    where N is a tiny, fixed native evidence pass.  It brightens slightly
    (gamma=0.85) and applies a very small sharpness gain (1.10), which preserves
    the original high-resolution frame while making faint road marks a bit more
    visible to the downstream detector. This branch is intentionally much weaker
    than restoration and is used only for native_real frames without direct
    degradation metadata.
    """

    gamma_corrected = tensor.clamp(0.0, 1.0).pow(0.85)
    # PIL's sharpness operator matches the calibration branch used in the GT49
    # native-field audit while preserving the original image dimensions.
    image = TF.to_pil_image(gamma_corrected[0].detach().cpu())
    image = ImageEnhance.Sharpness(image).enhance(1.10)
    return TF.to_tensor(image).unsqueeze(0).to(device=tensor.device, dtype=tensor.dtype).clamp(0.0, 1.0)


def save_image(path: Path, tensor: torch.Tensor, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = TF.to_pil_image(tensor[0].detach().cpu())
    image.save(path, quality=quality, subsampling=0)


def build_output_yaml(out_root: Path, split: str, names: dict[int, str]) -> None:
    data_yaml = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "nc": len(names),
        "names": names,
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")


def scenario_from_config(config: dict[str, Any], args_scenario: str) -> str:
    if args_scenario:
        return args_scenario
    root_name = Path(config["_root"]).name
    return "native_real" if root_name == "sharp" else root_name


def main() -> None:
    args = parse_args()
    config = load_yaml(args.data)
    names = yolo_names(config)
    root = Path(config["_root"])
    split = args.split
    image_dir = root / str(config[split])
    label_dir = root / str(config[split]).replace("images", "labels")
    metadata_dir = root / "metadata" / split
    scenario = scenario_from_config(config, args.scenario)
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    loaded: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for model_name in models:
        if model_name == "raw":
            loaded[model_name] = None
        elif model_name.startswith("rmr"):
            loaded.setdefault("rmr", load_rcadnet(args.rcadnet_weights, device))
        elif model_name == "nafnet":
            loaded[model_name] = load_nafnet(args.nafnet_weights, device)
        elif model_name == "dfpir":
            loaded[model_name] = DFPIRAdapter(args.dfpir_weights, device=str(device), use_clip=False)
        elif model_name == "demoe_auto":
            loaded[model_name] = DeMoEAdapter(args.demoe_weights, device=device, task="auto")
        elif model_name == "demoe_scenario":
            loaded[model_name] = DeMoEAdapter(args.demoe_weights, device=device, task="scenario")
        elif model_name.startswith("instructir"):
            loaded.setdefault(
                "instructir",
                InstructIRAdapter(args.instructir_image_weights, args.instructir_lm_weights, device=device),
            )
        else:
            raise ValueError(f"Unknown model '{model_name}'")

    args.out.mkdir(parents=True, exist_ok=True)
    scenario_code = code_from_scenario(scenario, device=device)

    with torch.inference_mode():
        for model_name in models:
            model_root = args.out / model_name
            out_image_dir = model_root / "images" / split
            out_label_dir = model_root / "labels" / split
            out_metadata_dir = model_root / "metadata" / split
            out_image_dir.mkdir(parents=True, exist_ok=True)
            copy_tree_files(label_dir, out_label_dir)
            copy_tree_files(metadata_dir, out_metadata_dir)
            build_output_yaml(model_root, split, names)

            manifest_path = model_root / "restore_manifest.jsonl"
            with manifest_path.open("w", encoding="utf-8") as manifest:
                for index, image_path in enumerate(image_paths):
                    out_path = out_image_dir / image_path.name
                    t0 = time.perf_counter()
                    if model_name == "raw":
                        link_or_copy(image_path, out_path)
                        elapsed = time.perf_counter() - t0
                        with Image.open(image_path) as src_image:
                            width, height = src_image.size
                    elif args.skip_existing and out_path.exists():
                        elapsed = 0.0
                        with Image.open(out_path) as src_image:
                            width, height = src_image.size
                    else:
                        metadata_path = metadata_dir / f"{image_path.stem}.json"
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                        with Image.open(image_path) as src_image:
                            width, height = src_image.size
                            tensor = TF.to_tensor(src_image.convert("RGB")).unsqueeze(0).to(device)

                        if model_name == "rmr_blind":
                            model = loaded["rmr"]
                            restored = restore_tiled(tensor, lambda patch: model(patch, None), tile=args.tile, overlap=args.overlap)
                        elif model_name == "rmr_metadata":
                            model = loaded["rmr"]
                            code = code_from_metadata(metadata, device=device).unsqueeze(0) if metadata else scenario_code.unsqueeze(0)
                            restored = restore_tiled(tensor, lambda patch: model(patch, code), tile=args.tile, overlap=args.overlap)
                        elif model_name == "rmr_metadata_gated":
                            model = loaded["rmr"]
                            if metadata and metadata_has_reliable_degradation_signal(metadata):
                                code = code_from_metadata(metadata, device=device).unsqueeze(0)
                                restored = restore_tiled(tensor, lambda patch: model(patch, code), tile=args.tile, overlap=args.overlap)
                                metadata_policy = "metadata_code"
                                metadata_reliable = True
                            elif scenario == "native_real":
                                restored = native_evidence_pass(tensor)
                                metadata_policy = "native_evidence_pass"
                                metadata_reliable = False
                            else:
                                restored = restore_tiled(tensor, lambda patch: model(patch, None), tile=args.tile, overlap=args.overlap)
                                metadata_policy = "image_code_fallback"
                                metadata_reliable = False
                        elif model_name == "nafnet":
                            model = loaded[model_name]
                            restored = restore_tiled(tensor, lambda patch: model(patch), tile=args.tile, overlap=args.overlap)
                        elif model_name == "dfpir":
                            model = loaded[model_name]
                            restored = restore_tiled(tensor, lambda patch: model(patch, scenario), tile=args.tile, overlap=args.overlap)
                        elif model_name in {"demoe_auto", "demoe_scenario"}:
                            model = loaded[model_name]
                            task = "auto" if model_name == "demoe_auto" else "scenario"
                            restored = restore_tiled(tensor, lambda patch: model(patch, scenario=scenario, task=task), tile=args.tile, overlap=args.overlap)
                        elif model_name == "instructir_generic":
                            model = loaded["instructir"]
                            prompt = generic_road_prompt()
                            restored = restore_tiled(tensor, lambda patch: model(patch, prompt), tile=args.tile, overlap=args.overlap)
                        elif model_name == "instructir_metadata":
                            model = loaded["instructir"]
                            prompt = metadata_prompt_from_file(metadata, scenario)
                            restored = restore_tiled(tensor, lambda patch: model(patch, prompt), tile=args.tile, overlap=args.overlap)
                        else:
                            raise ValueError(model_name)

                        if args.residual_strength != 1.0:
                            eta = max(0.0, min(args.residual_strength, 1.0))
                            restored = (tensor + eta * (restored - tensor)).clamp(0.0, 1.0)
                        elapsed = time.perf_counter() - t0
                        save_image(out_path, restored, args.jpeg_quality)
                        del tensor, restored
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    row = {
                        "model": model_name,
                        "scenario": scenario,
                        "image": image_path.name,
                        "output": str(out_path),
                        "width": width,
                        "height": height,
                        "native_resolution_preserved": True,
                        "runtime_s": elapsed,
                        "metadata_policy": locals().get("metadata_policy", ""),
                        "metadata_reliable": locals().get("metadata_reliable", ""),
                    }
                    if "metadata_policy" in locals():
                        del metadata_policy
                    if "metadata_reliable" in locals():
                        del metadata_reliable
                    rows.append(row)
                    manifest.write(json.dumps(row) + "\n")
                    print(json.dumps({"progress": index + 1, "total": len(image_paths), **row}), flush=True)

    summary_path = args.out / "restore_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"restored_models": models, "images_per_model": len(image_paths), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
