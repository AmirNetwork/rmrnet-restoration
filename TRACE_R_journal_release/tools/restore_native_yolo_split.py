from __future__ import annotations

import argparse
import csv
import json
import math
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
from models.rmrp_metadata_demoe import RMRPMetadataDeMoE
from models.rmrp_prompted_dfpir import RMRPPromptedDFPIR
from models.tracer_sensor_adapter import TRACESensorAdapterDeMoE
from models.rmrnet import RMRNet
from rcadnet import code_from_metadata, code_from_scenario
from rcadnet.practical_metadata import (
    ACCEL_END,
    CONTEXT_START,
    GYRO_END,
    PRACTICAL_SENSOR_DIM,
    sensor_packet_from_mapping,
)


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
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=None,
        help=(
            "Optional explicit metadata-sidecar directory. This keeps image/label "
            "splits fixed while comparing separately audited sensor packets."
        ),
    )
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
    parser.add_argument(
        "--output-format",
        choices=["source", "png"],
        default="source",
        help="Use png for lossless restored outputs; raw pass-through keeps its source encoding.",
    )
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
    parser.add_argument(
        "--dfpir-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use DFPIR's official CLIP degradation prompt.",
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


def load_rcadnet(weights: Path, device: torch.device) -> torch.nn.Module:
    # TRACE-R checkpoints include architecture metadata (including Paths), so
    # PyTorch >=2.6 cannot use its tensor-only default for this trusted local
    # artifact.  This matches the standard-resolution restoration loader.
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", {})
    saved_args = checkpoint.get("args", {})
    if hasattr(saved_args, "__dict__"):
        saved_args = vars(saved_args)
    if not isinstance(saved_args, dict):
        saved_args = {}

    if arch.get("backbone") == "demoe_sensor_low_rank":
        adapter = DeMoEAdapter(None, device=device, smoke=True)
        model = TRACESensorAdapterDeMoE(
            adapter.model,
            sensor_gyro_full_scale=float(arch.get("sensor_gyro_full_scale", 4.0)),
            top_k=int(arch.get("top_k", 1)),
            use_refiner=bool(arch.get("use_refiner", False)),
            backbone_route_mode=str(
                arch.get("backbone_route_mode", "sensor_task")
            ),
            sensor_task_thresholds=tuple(
                arch.get("sensor_task_thresholds", (0.18, 0.20, 0.385))
            ),
            sensor_task_mixed_expert=arch.get("sensor_task_mixed_expert"),
            feature_rank=int(arch.get("feature_rank", 16)),
            feature_max_gain=float(arch.get("feature_max_gain", 0.25)),
            use_cause_feature_adapters=bool(
                arch.get("use_cause_feature_adapters", False)
            ),
            cause_feature_max_gain=float(
                arch.get("cause_feature_max_gain", 0.18)
            ),
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        return model

    if arch.get("backbone") == "demoe_sensor_router":
        adapter = DeMoEAdapter(None, device=device, smoke=True)
        model = RMRPMetadataDeMoE(
            adapter.model,
            sensor_gyro_full_scale=float(arch.get("sensor_gyro_full_scale", 4.0)),
            top_k=int(arch.get("top_k", 2)),
            refiner_gain=float(arch.get("refiner_gain", 0.12)),
            use_refiner=bool(arch.get("use_refiner", True)),
            use_compound_blend_gate=bool(
                arch.get("use_compound_blend_gate", False)
            ),
            compound_blend_init=float(arch.get("compound_blend_init", 0.65)),
            compound_metadata_acceptance=float(
                arch.get("compound_metadata_acceptance", 1.0)
            ),
            cause_route_acceptance=tuple(
                arch.get("cause_route_acceptance", (1.0, 1.0, 1.0))
            ),
            use_cause_refiners=bool(arch.get("use_cause_refiners", False)),
            cause_refiner_gain=float(arch.get("cause_refiner_gain", 0.08)),
            backbone_route_mode=str(
                arch.get("backbone_route_mode", "metadata")
            ),
            sensor_task_thresholds=tuple(
                arch.get("sensor_task_thresholds", (0.18, 0.10, 0.385))
            ),
            sensor_task_mixed_expert=arch.get("sensor_task_mixed_expert"),
            use_semantic_adapters=bool(
                arch.get("use_semantic_adapters", False)
            ),
            semantic_adapter_gain=float(
                arch.get("semantic_adapter_gain", 0.25)
            ),
            semantic_adapter_acceptance=tuple(
                arch.get("semantic_adapter_acceptance", (1.0, 1.0, 1.0, 1.0))
            ),
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        return model

    if arch.get("backbone") == "dfpir_sensor_prompt":
        adapter = DFPIRAdapter(None, device=str(device), use_clip=False)
        state = checkpoint.get("model", checkpoint)
        basis_count = int(
            arch.get(
                "prompt_basis_count",
                state.get("prompt_embeddings", torch.zeros(3, 512)).shape[0],
            )
        )
        model = RMRPPromptedDFPIR(
            adapter.model,
            torch.zeros(basis_count, 512, device=device),
            sensor_gyro_full_scale=float(arch.get("sensor_gyro_full_scale", 4.0)),
            prompt_residual_scale=float(arch.get("prompt_residual_scale", 0.10)),
            refiner_gain=float(arch.get("refiner_gain", 0.12)),
            prompt_router=str(arch.get("prompt_router", "hard")),
            sensor_route_mode=str(arch.get("sensor_route_mode", "posterior")),
            use_refiner=bool(arch.get("use_refiner", True)),
            compound_motion_blend=float(arch.get("compound_motion_blend", 0.0)),
            use_compound_refiner=bool(arch.get("use_compound_refiner", False)),
            compound_refiner_gain=float(arch.get("compound_refiner_gain", 0.18)),
            use_cause_refiners=bool(arch.get("use_cause_refiners", False)),
            cause_refiner_gain=float(arch.get("cause_refiner_gain", 0.08)),
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        return model

    # Checkpoints produced during the first spatial-physics experiment stored
    # these values in ``args`` but not ``arch``. Recovering from the same saved
    # checkpoint is deterministic and retains strict state-dict loading; it is
    # not a permissive architecture guess.
    def arch_or_arg(arch_key: str, arg_key: str, default):
        return arch.get(arch_key, saved_args.get(arg_key, default))

    has_physics_weights = any(
        str(key).startswith("physics_feature_encoder.")
        for key in checkpoint.get("model", {})
    )
    model = RMRNet(
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
        use_cause_experts=arch.get("use_cause_experts", False),
        cause_expert_gain=arch.get("cause_expert_gain", 0.35),
        cause_compound_boost=arch.get("cause_compound_boost", 1.0),
        exact_metadata_mode=arch.get("exact_metadata_mode"),
        use_motion_prior=arch.get("use_motion_prior", False),
        motion_prior_k=arch.get("motion_prior_k", 0.0075),
        motion_prior_blend=arch.get("motion_prior_blend", 0.90),
        motion_prior_nuisance_decay=arch.get(
            "motion_prior_nuisance_decay",
            30.0,
        ),
        motion_prior_compound_floor=arch.get(
            "motion_prior_compound_floor",
            0.0,
        ),
        motion_prior_adaptive_k=arch.get("motion_prior_adaptive_k", 0.0),
        motion_prior_compound_source=arch.get(
            "motion_prior_compound_source",
            False,
        ),
        practical_nuisance_deadzone=arch.get(
            "practical_nuisance_deadzone",
            0.0,
        ),
        use_practical_sensor_encoder=arch.get(
            "use_practical_sensor_encoder",
            False,
        ),
        sensor_dim=arch.get("sensor_dim", PRACTICAL_SENSOR_DIM),
        sensor_gyro_full_scale=arch.get("sensor_gyro_full_scale", 1.0),
        sensor_residual_scale=arch.get("sensor_residual_scale", 0.25),
        use_sensor_prior_fusion=arch.get("use_sensor_prior_fusion", False),
        use_sensor_image_psf_refiner=arch.get(
            "use_sensor_image_psf_refiner",
            False,
        ),
        use_post_prior_evidence_refiner=arch.get(
            "use_post_prior_evidence_refiner",
            False,
        ),
        post_prior_refiner_gain=arch.get("post_prior_refiner_gain", 0.12),
        post_prior_refiner_support=arch.get(
            "post_prior_refiner_support",
            "all",
        ),
        practical_prior_source=arch.get("practical_prior_source", "joint"),
        use_spatial_physics=arch_or_arg(
            "use_spatial_physics",
            "spatial_physics",
            has_physics_weights,
        ),
        physics_samples=arch_or_arg("physics_samples", "physics_samples", 5),
        physics_exposure_min_ms=arch_or_arg(
            "physics_exposure_min_ms", "physics_exposure_min_ms", 0.05
        ),
        physics_exposure_max_ms=arch_or_arg(
            "physics_exposure_max_ms", "physics_exposure_max_ms", 40.0
        ),
        physics_focal_ratio=arch_or_arg(
            "physics_focal_ratio", "physics_focal_ratio", 0.75
        ),
        physics_calibration_reliability=arch_or_arg(
            "physics_calibration_reliability",
            "physics_calibration_reliability",
            0.50,
        ),
        physics_activation_motion_px=arch_or_arg(
            "physics_activation_motion_px",
            "physics_activation_motion_px",
            0.10,
        ),
        physics_exclusive_trajectory=arch_or_arg(
            "physics_exclusive_trajectory",
            "physics_exclusive_trajectory",
            False,
        ),
        use_physics_inverse_candidate=arch_or_arg(
            "use_physics_inverse_candidate", "physics_inverse_candidate", False
        ),
        physics_inverse_iterations=arch_or_arg(
            "physics_inverse_iterations", "physics_inverse_iterations", 3
        ),
        physics_inverse_blend=arch_or_arg(
            "physics_inverse_blend", "physics_inverse_blend", 1.0
        ),
        physics_decoder_motion_threshold_px=arch_or_arg(
            "physics_decoder_motion_threshold_px",
            "physics_decoder_motion_threshold_px",
            1.5,
        ),
        physics_decoder_motion_transition_px=arch_or_arg(
            "physics_decoder_motion_transition_px",
            "physics_decoder_motion_transition_px",
            0.35,
        ),
        enable_aux_contour=False,
    ).to(device)
    model.metadata_encoding = arch.get("metadata_encoding", "legacy")
    # Reported inference must use an architecture-exact checkpoint. Partial
    # transfer remains available inside the trainer only, where every newly
    # initialized parameter is subsequently optimized.
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"RMR-Net checkpoint/model mismatch for inference: {weights}. "
            "Retrain the current architecture or use its exact archived definition."
        ) from exc
    model.eval()
    return model


def normalized_log(value: float, minimum: float, maximum: float) -> float:
    value = max(float(value), minimum)
    return max(
        0.0,
        min(
            (
                math.log(value) - math.log(minimum)
            )
            / (
                math.log(maximum) - math.log(minimum)
            ),
            1.0,
        ),
    )


def native_partial_sensor_packet(
    metadata: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Load an audited practical packet or construct a camera-only fallback.

    CRID sidecars can contain synchronized gyro/accelerometer trajectories,
    vehicle context, EXIF, and per-modality reliability. Those 82 values are
    forwarded unchanged. For older sidecars with EXIF only, missing inertial
    and vehicle channels remain zero so the in-network cause-wise reliability
    gate falls back to image evidence. No annotation, detector output, or
    synthetic degradation parameter enters either path.
    """

    if (
        "practical_sensor_packet" in metadata
        or "sensor_packet" in metadata
    ):
        return sensor_packet_from_mapping(metadata, device=device)
    exif = metadata.get("exif", {})
    exposure_seconds = float(
        exif.get(
            "ExposureTime",
            float(metadata.get("exposure_ms", 0.0)) / 1000.0,
        )
        or 0.0
    )
    exposure_ms = max(exposure_seconds * 1000.0, 0.05)
    iso = float(
        exif.get(
            "ISOSpeedRatings",
            exif.get("RecommendedExposureIndex", 100.0),
        )
        or 100.0
    )
    focus_proxy = float(metadata.get("defocus_score", 0.0) or 0.0)
    packet = torch.zeros(
        PRACTICAL_SENSOR_DIM,
        dtype=torch.float32,
        device=device,
    )
    context = packet[CONTEXT_START:]
    context[0] = normalized_log(exposure_ms, 0.05, 40.0)
    context[1] = normalized_log(iso, 100.0, 25600.0)
    context[2] = context[1]
    context[5] = max(0.0, min(focus_proxy, 1.0))
    context[6] = 0.75 if exif else 0.0
    context[12] = 0.90 if exif else 0.0
    context[13] = 0.0  # no synchronized IMU
    context[14] = 0.0  # no vehicle-motion packet
    context[15] = context[12]
    if torch.any(packet[:GYRO_END]) or torch.any(
        packet[GYRO_END:ACCEL_END]
    ):
        raise AssertionError("Sony partial packet must not invent inertial data")
    return packet


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
    # Every accumulated pixel must be normalized by its actual blending
    # weight.  Clamping the denominator to 1.0 darkens image-border pixels,
    # where the raised-cosine tile weight is intentionally below one and only
    # one tile contributes.  A dtype-aware epsilon prevents division by zero
    # without changing valid sub-unit weights.
    epsilon = max(float(torch.finfo(weight.dtype).eps), 1e-8)
    return (output / weight.clamp_min(epsilon)).clamp(0.0, 1.0)


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
    if path.suffix.lower() == ".png":
        image.save(path, format="PNG", compress_level=3)
    else:
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
    metadata_dir = (
        args.metadata_dir.resolve()
        if args.metadata_dir is not None
        else root / "metadata" / split
    )
    if args.metadata_dir is not None and not metadata_dir.exists():
        raise FileNotFoundError(f"Explicit metadata directory does not exist: {metadata_dir}")
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
            loaded[model_name] = DFPIRAdapter(
                args.dfpir_weights,
                device=str(device),
                use_clip=args.dfpir_clip,
            )
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
            existing_manifest: dict[str, dict[str, Any]] = {}
            if args.skip_existing and manifest_path.exists():
                for line in manifest_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    image_name = str(record.get("image", ""))
                    if image_name:
                        existing_manifest[image_name] = record
            with manifest_path.open("w", encoding="utf-8") as manifest:
                for index, image_path in enumerate(image_paths):
                    out_path = out_image_dir / (
                        f"{image_path.stem}.png"
                        if args.output_format == "png" and model_name != "raw"
                        else image_path.name
                    )
                    preserved = existing_manifest.get(image_path.name)
                    if args.skip_existing and out_path.exists() and preserved is not None:
                        with Image.open(image_path) as source_image:
                            source_size = source_image.size
                        with Image.open(out_path) as saved_image:
                            width, height = saved_image.size
                        if (width, height) != source_size:
                            raise AssertionError(
                                f"Resumed native output changed dimensions: {out_path} "
                                f"has {(width, height)}, expected {source_size}"
                            )
                        rows.append(preserved)
                        manifest.write(json.dumps(preserved) + "\n")
                        print(
                            json.dumps(
                                {
                                    "progress": index + 1,
                                    "total": len(image_paths),
                                    "resumed_existing": True,
                                    **preserved,
                                }
                            ),
                            flush=True,
                        )
                        continue
                    t0 = time.perf_counter()
                    if model_name == "raw":
                        link_or_copy(image_path, out_path)
                        elapsed = time.perf_counter() - t0
                        with Image.open(image_path) as src_image:
                            width, height = src_image.size
                    elif args.skip_existing and out_path.exists():
                        # The pixels are reusable, but no timing provenance was
                        # recovered. Leave runtime blank instead of recording a
                        # misleading zero-second inference.
                        elapsed = None
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
                            if (
                                metadata
                                and getattr(
                                    model,
                                    "use_practical_sensor_encoder",
                                    False,
                                )
                            ):
                                code = native_partial_sensor_packet(
                                    metadata,
                                    device,
                                ).unsqueeze(0)
                            else:
                                code = (
                                    code_from_metadata(
                                        metadata,
                                        device=device,
                                        encoding=getattr(
                                            model,
                                            "metadata_encoding",
                                            "legacy",
                                        ),
                                    ).unsqueeze(0)
                                    if metadata
                                    else scenario_code.unsqueeze(0)
                                )
                            restored = restore_tiled(tensor, lambda patch: model(patch, code), tile=args.tile, overlap=args.overlap)
                        elif model_name == "rmr_metadata_gated":
                            model = loaded["rmr"]
                            if metadata and metadata_has_reliable_degradation_signal(metadata):
                                if getattr(
                                    model,
                                    "use_practical_sensor_encoder",
                                    False,
                                ):
                                    code = native_partial_sensor_packet(
                                        metadata,
                                        device,
                                    ).unsqueeze(0)
                                else:
                                    code = code_from_metadata(
                                        metadata,
                                        device=device,
                                        encoding=getattr(
                                            model,
                                            "metadata_encoding",
                                            "legacy",
                                        ),
                                    ).unsqueeze(0)
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
