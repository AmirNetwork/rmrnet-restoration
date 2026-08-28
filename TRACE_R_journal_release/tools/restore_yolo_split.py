"""Restore a YOLO split with TRACE-R or a matched standalone restorer."""

from __future__ import annotations

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.demoe_adapter import DeMoEAdapter
from baselines.dfpir_adapter import DFPIRAdapter
from baselines.instructir_adapter import InstructIRAdapter
from baselines.nafnet_metadata import MetadataNAFNetRoad
from baselines.nafnet_road import NAFNetRoad, build_nafnet_from_payload
from models.rmrnet import RMRNet
from models.tracer import TRACERExpertFusion, TRACERPolicy
from models.rmrp_metadata_demoe import RMRPMetadataDeMoE
from models.rmrp_prompted_dfpir import RMRPPromptedDFPIR
from models.tracer_sensor_adapter import TRACESensorAdapterDeMoE
from models.tracer_sparse_wavelet import TRACERSparseDeMoE
from rcadnet import code_from_metadata, code_from_scenario
from rcadnet.practical_metadata import (
    ACCEL_END,
    CONTEXT_START,
    GYRO_END,
    apply_sensor_modality_mask,
    conditioning_code_from_packet,
    perturb_sensor_packet,
    sensor_packet_from_mapping,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a YOLO image split and copy labels for detection evaluation.")
    parser.add_argument("--data", required=True, help="Degraded YOLO data.yaml.")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--model",
        choices=[
            "rcadnet",
            "trace_r",
            "rmrp_fusion",
            "dfpir",
            "nafnet",
            "nafnet_meta",
            "demoe",
            "instructir",
        ],
        required=True,
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help=(
            "Deterministic development-only image cap. Zero processes the full "
            "split; positive values sample with --image-sample-seed."
        ),
    )
    parser.add_argument("--image-sample-seed", type=int, default=2026)
    parser.add_argument("--rcadnet-weights")
    parser.add_argument(
        "--rcadnet-image-route-weights",
        help=(
            "Optional matched DeMoE checkpoint for the conservative image-evidence "
            "route of dual-route RMR-P. The deployed output is blended with the "
            "metadata route using only the observable sensor state."
        ),
    )
    parser.add_argument(
        "--rcadnet-dual-route-coefficients",
        nargs=4,
        type=float,
        metavar=("MOTION", "DEFOCUS", "LOWLIGHT", "COMPOUND"),
        help=(
            "Validation-selected acceptance of the metadata correction for motion, "
            "defocus, low-light, and compound sensor states. Requires "
            "--rcadnet-image-route-weights."
        ),
    )
    parser.add_argument(
        "--rcadnet-dual-route-support-threshold",
        type=float,
        default=0.50,
        help=(
            "Minimum reliability for the selected observable cause. Weak or missing "
            "metadata falls back to the image-evidence route."
        ),
    )
    parser.add_argument(
        "--rcadnet-lowlight-specialist-weights",
        help=(
            "Optional validation-selected low-light/mixed specialist. "
            "The shared RMR-Net remains active for other causes."
        ),
    )
    parser.add_argument(
        "--rcadnet-lowlight-route-threshold",
        type=float,
        default=0.39,
        help=(
            "Route to the specialist when the observable/fused low-light "
            "coordinate reaches this validation-locked threshold."
        ),
    )
    parser.add_argument(
        "--rcadnet-lowlight-image-mean-threshold",
        type=float,
        default=0.29,
        help=(
            "When camera fields are unavailable, route dark images below this "
            "validation-locked mean intensity through the image-only safety "
            "fallback instead of applying an IMU-only motion prior."
        ),
    )
    parser.add_argument(
        "--rcadnet-specialist-support",
        choices=["both", "lowlight", "mixed"],
        default="both",
        help=(
            "Observable-cause support for the optional specialist. 'mixed' "
            "requires both low-light and motion evidence; 'lowlight' excludes "
            "that compound region; 'both' preserves the historical route."
        ),
    )
    parser.add_argument(
        "--rcadnet-motion-route-threshold",
        type=float,
        default=0.12,
        help="Minimum motion/vibration support for a mixed specialist route.",
    )
    parser.add_argument("--dfpir-weights")
    parser.add_argument("--nafnet-weights")
    parser.add_argument("--demoe-weights")
    parser.add_argument("--demoe-task", choices=["auto", "scenario", "defocus", "global_motion", "local_motion", "synth_global_motion", "low_light"], default="auto")
    parser.add_argument("--demoe-smoke", action="store_true", help="Use random DeMoE weights for wiring tests only; never report these as benchmark results.")
    parser.add_argument("--instructir-image-weights", default="weights/instructir/im_instructir-7d.pt")
    parser.add_argument("--instructir-lm-weights", default="weights/instructir/lm_instructir-7d.pt")
    parser.add_argument(
        "--trace-r-motion-threshold", "--rmrp-motion-threshold",
        dest="trace_r_motion_threshold", type=float, default=0.18,
    )
    parser.add_argument(
        "--trace-r-defocus-threshold", "--rmrp-defocus-threshold",
        dest="trace_r_defocus_threshold", type=float, default=0.20,
    )
    parser.add_argument(
        "--trace-r-lowlight-threshold", "--rmrp-lowlight-threshold",
        dest="trace_r_lowlight_threshold", type=float, default=0.385,
    )
    parser.add_argument(
        "--trace-r-support-threshold", "--rmrp-expert-support-threshold",
        dest="trace_r_support_threshold", type=float, default=0.50,
    )
    parser.add_argument(
        "--trace-r-lowlight-dfpir-weight", "--rmrp-lowlight-dfpir-weight",
        dest="trace_r_lowlight_dfpir_weight", type=float, default=0.40,
    )
    parser.add_argument(
        "--trace-r-mixed-dfpir-weight", "--rmrp-mixed-dfpir-weight",
        dest="trace_r_mixed_dfpir_weight", type=float, default=0.075,
    )
    parser.add_argument(
        "--trace-r-gyro-full-scale", "--rmrp-sensor-gyro-full-scale",
        dest="trace_r_gyro_full_scale", type=float, default=4.0,
    )
    parser.add_argument(
        "--instructir-prompt-mode",
        choices=["scenario", "generic"],
        default="scenario",
        help="scenario discloses the controlled degradation family; generic supplies no family label.",
    )
    parser.add_argument(
        "--dfpir-clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use DFPIR's official CLIP degradation prompt.",
    )
    parser.add_argument("--gate-threshold", type=float, default=-1.0, help="RMR clean-frame pass-through threshold; disabled when negative.")
    parser.add_argument(
        "--rcadnet-output-stage",
        choices=(
            "restored",
            "neural_restored",
            "physics_restoration_input",
            "physics_inverse_candidate",
        ),
        default="restored",
        help=(
            "Audited RMR-P output stage. The default is the deployed final output; "
            "other stages are intended for validation-only architecture ablations."
        ),
    )
    parser.add_argument(
        "--rcadnet-prompt-router-override",
        choices=("hard", "sparse_blend"),
        default=None,
        help=(
            "Validation-only architecture audit for prompted RMR-P checkpoints. "
            "Omit this option for checkpoint-declared inference."
        ),
    )
    parser.add_argument(
        "--rcadnet-prompt-delta-scale-override",
        type=float,
        default=None,
        help=(
            "Validation-only override for the bounded learned prompt residual. "
            "Use 0 to audit the fixed task basis alone."
        ),
    )
    parser.add_argument(
        "--rcadnet-sensor-route-mode-override",
        choices=("posterior", "physical_fused"),
        default=None,
        help=(
            "Validation-only audit of whether discrete task routing uses the "
            "learned posterior or reliability-fused calibrated sensor state."
        ),
    )
    parser.add_argument(
        "--rcadnet-compound-motion-blend-override",
        type=float,
        default=None,
        help=(
            "Validation-only override for the fixed motion contribution used "
            "when reliable metadata identifies compound motion and low light."
        ),
    )
    parser.add_argument(
        "--rcadnet-demoe-route-acceptance",
        type=float,
        nargs=3,
        metavar=("MOTION", "DEFOCUS", "LOWLIGHT"),
        default=None,
        help=(
            "Validation-only metadata-routing compatibility for DeMoE-based "
            "RMR-P. Values must lie in [0, 1]."
        ),
    )
    parser.add_argument(
        "--rcadnet-backbone-route-mode-override",
        choices=("metadata", "image", "sensor_task"),
        default=None,
        help="Validation-only DeMoE backbone routing audit.",
    )
    parser.add_argument(
        "--rcadnet-semantic-adapter-gain-override",
        type=float,
        default=None,
        help="Validation-only absolute gain for RMR-P residual adapters.",
    )
    parser.add_argument(
        "--rcadnet-semantic-adapter-acceptance",
        type=float,
        nargs=4,
        metavar=("MOTION", "DEFOCUS", "LOWLIGHT", "COMPOUND"),
        default=None,
        help="Validation-only trust in the four semantic residual adapters.",
    )
    parser.add_argument(
        "--rcadnet-compound-metadata-acceptance-override",
        type=float,
        default=None,
        help=(
            "Validation-only blend between the low-light fallback and the "
            "joint metadata route for compound captures."
        ),
    )
    parser.add_argument("--gate-softness", type=float, default=0.03, help="Set to 0 for hard bypass of clean/low-severity images.")
    parser.add_argument("--residual-strength", type=float, default=1.0, help="Blend restored output as input + eta * (restored - input). Use eta<1 for residual perception policy.")
    parser.add_argument(
        "--rcadnet-motion-prior-compound-floor",
        type=float,
        default=None,
        help=(
            "Validation-audit override for the minimum exact-motion-prior gate "
            "under compound degradation. Omit to use checkpoint provenance."
        ),
    )
    parser.add_argument(
        "--rcadnet-motion-prior-regularization",
        type=float,
        default=None,
        help="Validation-audit override for Wiener regularization K.",
    )
    parser.add_argument(
        "--rcadnet-motion-prior-blend",
        type=float,
        default=None,
        help="Validation-audit override for the bounded Wiener residual blend.",
    )
    parser.add_argument(
        "--rcadnet-motion-prior-solver",
        choices=["wiener", "richardson_lucy"],
        default=None,
        help="Validation-audit override for the metadata motion solver.",
    )
    parser.add_argument(
        "--rcadnet-richardson-lucy-iterations",
        type=int,
        default=None,
        help="Positive iteration count for the bounded Richardson-Lucy audit.",
    )
    parser.add_argument("--debug-every", type=int, default=0, help="Print JSON restoration stats every N images. 0 disables per-image debug.")
    parser.add_argument(
        "--rcadnet-code-source",
        choices=["scenario", "metadata", "blind"],
        default="scenario",
        help="RMR/RCAD conditioning: scenario label, metadata JSON, or blind image-estimated code only.",
    )
    parser.add_argument(
        "--rcadnet-metadata-control",
        choices=[
            "correct",
            "zero",
            "unavailable",
            "shuffled",
            "counterfactual",
            "correct_family_wrong_severity",
            "wrong_family_matched_severity",
            "adversarial",
            "noisy",
            "camera_only",
            "imu_only",
            "vehicle_only",
            "camera_imu",
            "camera_vehicle",
            "imu_vehicle",
        ],
        default="correct",
        help=(
            "Reviewer-control mode for metadata-conditioned RMR. Applies only when "
            "--rcadnet-code-source metadata. correct uses the image metadata, zero "
            "uses an all-zero metadata code, shuffled assigns another image's "
            "metadata code, unavailable removes the external code, the explicit "
            "family/severity modes isolate those factors, adversarial changes both, "
            "noisy adds Gaussian measurement perturbation, and the modality controls "
            "retain only the named practical sensor groups."
        ),
    )
    parser.add_argument("--metadata-noise-std", type=float, default=0.10)
    parser.add_argument("--metadata-control-seed", type=int, default=2026)
    parser.add_argument(
        "--metadata-dir-override",
        type=Path,
        default=None,
        help=(
            "Read per-image JSON records from this directory while leaving the "
            "images and labels unchanged. This supports cross-condition metadata "
            "controls without copying image data."
        ),
    )
    parser.add_argument(
        "--require-metadata",
        action="store_true",
        help="Fail instead of using a scenario fallback when per-image metadata are missing.",
    )
    return parser.parse_args()


def load_rcadnet(weights: str, device: torch.device) -> RMRNet:
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", {})
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
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"TRACE-R sensor-adapter checkpoint/model mismatch: {weights}."
            ) from exc
        model.eval()
        return model
    if arch.get("backbone") == "demoe_sparse_wavelet":
        adapter = DeMoEAdapter(None, device=device, smoke=True)
        model = TRACERSparseDeMoE(
            adapter.model,
            sensor_gyro_full_scale=float(arch.get("sensor_gyro_full_scale", 4.0)),
            top_k=int(arch.get("top_k", 1)),
            use_refiner=False,
            backbone_route_mode=str(
                arch.get("backbone_route_mode", "sensor_task")
            ),
            sensor_task_thresholds=tuple(
                arch.get("sensor_task_thresholds", (0.18, 0.20, 0.385))
            ),
            sensor_task_mixed_expert=arch.get("sensor_task_mixed_expert"),
            wavelet_hidden_channels=int(
                arch.get("wavelet_hidden_channels", 48)
            ),
            wavelet_stages=int(arch.get("wavelet_stages", 3)),
            wavelet_max_residual=float(
                arch.get("wavelet_max_residual", 0.16)
            ),
        ).to(device)
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"TRACE-R sparse-wavelet checkpoint/model mismatch: {weights}."
            ) from exc
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
                arch.get("sensor_task_thresholds", (0.18, 0.20, 0.385))
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
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"RMR-P metadata-DeMoE checkpoint/model mismatch: {weights}."
            ) from exc
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
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"RMR-P prompted-DFPIR checkpoint/model mismatch: {weights}."
            ) from exc
        model.eval()
        return model
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
        cause_expert_gain=arch.get("cause_expert_gain", 0.20),
        cause_compound_boost=arch.get("cause_compound_boost", 1.0),
        exact_metadata_mode=arch.get("exact_metadata_mode", arch.get("use_cause_experts", False)),
        use_motion_prior=arch.get("use_motion_prior", False),
        motion_prior_k=arch.get("motion_prior_k", 0.0075),
        motion_prior_blend=arch.get("motion_prior_blend", 0.90),
        motion_prior_nuisance_decay=arch.get("motion_prior_nuisance_decay", 30.0),
        motion_prior_compound_floor=arch.get("motion_prior_compound_floor", 0.0),
        motion_prior_adaptive_k=arch.get("motion_prior_adaptive_k", 0.0),
        motion_prior_compound_source=arch.get(
            "motion_prior_compound_source",
            False,
        ),
        practical_nuisance_deadzone=arch.get(
            "practical_nuisance_deadzone",
            0.0,
        ),
        use_practical_sensor_encoder=arch.get("use_practical_sensor_encoder", False),
        sensor_dim=arch.get("sensor_dim", 32),
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
        post_prior_refiner_gain=arch.get(
            "post_prior_refiner_gain",
            0.12,
        ),
        post_prior_refiner_support=arch.get(
            "post_prior_refiner_support",
            "all",
        ),
        practical_prior_source=arch.get(
            "practical_prior_source",
            "joint",
        ),
        enable_aux_contour=False,
    ).to(device)
    # The sidecar-to-code mapping is checkpoint provenance. Historical weights
    # keep legacy semantics; exact-kernel runs opt into axial_v2 explicitly.
    model.metadata_encoding = arch.get("metadata_encoding", "legacy")
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"RMR-Net checkpoint/model mismatch for inference: {weights}. "
            "Reported evaluation forbids partially initialized restoration models."
        ) from exc
    model.eval()
    return model


def metadata_tensor(metadata: dict, model: RMRNet, device: torch.device) -> torch.Tensor:
    """Load the checkpoint-declared metadata interface without silent fallback."""

    if getattr(model, "use_practical_sensor_encoder", False):
        return sensor_packet_from_mapping(metadata, device=device)
    return code_from_metadata(metadata, device=device, encoding=model.metadata_encoding)


def practical_motion_prior_code(metadata: dict, device: torch.device) -> torch.Tensor | None:
    """Return an observable sensor-derived code, never a hidden renderer code."""

    values = metadata.get("observable_sensor_code")
    if values is None:
        values = metadata.get("observable_motion_code")
    if values is None:
        return None
    code = torch.tensor(values, dtype=torch.float32, device=device)
    if code.numel() != 8:
        raise ValueError(f"observable sensor code must have eight values, got {code.numel()}")
    return code


def instructir_prompt(scenario: str, mode: str) -> str:
    """Return the fixed prompt declared for the controlled transfer baseline."""

    if mode == "generic":
        return "Restore this road inspection image while preserving cracks, pothole rims, and pavement texture."
    family = scenario.lower()
    if "motion" in family and "lowlight" in family:
        corruption = "combined camera-motion blur and low illumination with noise"
    elif "motion" in family or "vibration" in family:
        corruption = "camera-motion blur"
    elif "defocus" in family:
        corruption = "defocus blur"
    elif "lowlight" in family:
        corruption = "low illumination with sensor noise"
    else:
        corruption = "unknown image degradation"
    return (
        f"Remove {corruption} from this road inspection image. Preserve thin cracks, "
        "pothole boundaries, lane markings, and natural pavement texture without hallucinating defects."
    )


def load_nafnet(weights: str, device: torch.device) -> NAFNetRoad:
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    model, _, _ = build_nafnet_from_payload(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def load_metadata_nafnet(weights: str, device: torch.device) -> MetadataNAFNetRoad:
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", {})
    model = MetadataNAFNetRoad(
        width=arch.get("width", 32),
        code_dim=arch.get("code_dim", 8),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def pad_to_multiple(image: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image, (height, width)


def counterfactual_metadata_code(code: torch.Tensor) -> torch.Tensor:
    """Create the same hard negative metadata code used during training.

    Controlled scenario splits often have one shared metadata code per
    degradation; shuffling within such a split may not change the code at all.
    The counterfactual control changes only the metadata supplied to RMR-Net:

        z_m^- = remove_low_light(z_m)       if motion(z_m) and low_light(z_m)
              = swap_to_motion(z_m)         if low_light-only/defocus
              = swap_to_defocus(z_m)        if motion-only
              = [roll(z_m[0:7]), 1-.5s]     otherwise.

    Labels and image pixels remain unchanged, so this directly tests whether
    aligned metadata is genuinely used.
    """

    original = code.detach().clone()
    squeeze = False
    if original.ndim == 1:
        original = original.unsqueeze(0)
        squeeze = True
    if original.shape[1] > 8:
        wrong = original.clone()
        wrong[:, :GYRO_END] = -wrong[:, :GYRO_END]
        wrong[:, GYRO_END:ACCEL_END] = -wrong[:, GYRO_END:ACCEL_END]
        wrong[:, CONTEXT_START : CONTEXT_START + 3] = (
            1.0 - wrong[:, CONTEXT_START : CONTEXT_START + 3].clamp(0.0, 1.0)
        )
        wrong[:, CONTEXT_START + 5 : CONTEXT_START + 7] = (
            1.0
            - wrong[:, CONTEXT_START + 5 : CONTEXT_START + 7].clamp(0.0, 1.0)
        )
        return wrong.squeeze(0) if squeeze else wrong
    original = original.clamp(0.0, 1.0)
    if original.shape[1] <= 1:
        wrong = 1.0 - original
        return wrong.squeeze(0) if squeeze else wrong

    motion = original[:, :3].amax(dim=1, keepdim=True)
    low = original[:, 5:6] if original.shape[1] > 5 else torch.zeros_like(motion)
    defocus = original[:, 3:4] if original.shape[1] > 3 else torch.zeros_like(motion)
    noise = original[:, 4:5] if original.shape[1] > 4 else torch.zeros_like(motion)
    severity = original[:, -1:]

    rotated = original.clone()
    if rotated.shape[1] > 2:
        rotated[:, :-1] = torch.roll(rotated[:, :-1], shifts=1, dims=1)
    rotated[:, -1:] = (1.0 - 0.5 * severity).clamp(0.0, 1.0)
    wrong = rotated

    mixed_motion_low = (motion > 0.05) & (low > 0.05)
    motion_only = (motion > 0.05) & (low <= 0.05) & (defocus <= 0.05)
    low_or_defocus = ((low > 0.05) | (defocus > 0.05)) & (motion <= 0.05)

    if original.shape[1] > 5:
        keep_motion_drop_low = original.clone()
        keep_motion_drop_low[:, 5] = 0.0
        keep_motion_drop_low[:, -1:] = torch.maximum(motion, torch.maximum(defocus, noise))
        wrong = torch.where(mixed_motion_low.expand_as(wrong), keep_motion_drop_low, wrong)

    swap_to_defocus = torch.zeros_like(original)
    if swap_to_defocus.shape[1] > 3:
        swap_to_defocus[:, 3:4] = torch.maximum(motion, severity * 0.6)
    swap_to_defocus[:, -1:] = torch.maximum(swap_to_defocus[:, 3:4], 1.0 - 0.5 * severity)
    wrong = torch.where(motion_only.expand_as(wrong), swap_to_defocus, wrong)

    swap_to_motion = torch.zeros_like(original)
    swap_to_motion[:, 2:3] = torch.maximum(low, defocus).clamp(0.0, 1.0)
    swap_to_motion[:, -1:] = torch.maximum(swap_to_motion[:, 2:3], 1.0 - 0.5 * severity)
    wrong = torch.where(low_or_defocus.expand_as(wrong), swap_to_motion, wrong)

    wrong = wrong.clamp(0.0, 1.0)
    return wrong.squeeze(0) if squeeze else wrong


def metadata_factor_control(code: torch.Tensor, mode: str) -> torch.Tensor:
    """Isolate family and severity information in the eight-coordinate code.

    Coordinates 0:7 represent corruption family/effects and coordinate 7 is
    global severity. These controls alter metadata only; pixels, labels, the
    image-estimated code, and model weights remain fixed.
    """
    original = code.detach().clone()
    squeeze = original.ndim == 1
    if squeeze:
        original = original.unsqueeze(0)
    if original.shape[1] > 8:
        # These family/severity controls are defined for the causal code, not
        # raw sensor packets. Use shuffled, noisy, unavailable, or the practical
        # counterfactual control for sensor-conditioned checkpoints.
        raise ValueError(
            f"{mode} is not defined for practical sensor packets; "
            "use shuffled/noisy/unavailable/counterfactual"
        )
    original = original.clamp(0.0, 1.0)
    if original.shape[1] < 2:
        changed = 1.0 - original
        return changed.squeeze(0) if squeeze else changed

    changed = original.clone()
    severity = original[:, -1:].clone()
    if mode == "correct_family_wrong_severity":
        changed[:, -1:] = 1.0 - severity
    elif mode in {"wrong_family_matched_severity", "adversarial"}:
        family = original[:, :-1]
        # Rotate by half the family vector so motion, defocus, noise, low-light,
        # and compression evidence cannot remain in its original coordinate.
        shift = max(1, family.shape[1] // 2)
        changed[:, :-1] = torch.roll(family, shifts=shift, dims=1)
        changed[:, -1:] = severity if mode == "wrong_family_matched_severity" else 1.0 - severity
    else:
        raise ValueError(f"Unknown metadata factor control: {mode}")
    changed = changed.clamp(0.0, 1.0)
    return changed.squeeze(0) if squeeze else changed




def image_stats(name: str, tensor: torch.Tensor) -> dict[str, float]:
    t = tensor.detach()
    return {
        f"{name}_mean": float(t.mean().cpu()),
        f"{name}_std": float(t.std(unbiased=False).cpu()),
        f"{name}_min": float(t.min().cpu()),
        f"{name}_max": float(t.max().cpu()),
    }


def rmrp_dual_route_policy(
    stage_result: dict[str, torch.Tensor | None],
    coefficients: list[float],
    support_threshold: float,
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Convert observable sensor evidence into a conservative route policy.

    The benchmark scenario name is deliberately absent from this function.  The
    three single-cause scores are motion (maximum axial motion/vibration),
    defocus, and low-light.  A simultaneous motion/low-light state selects the
    compound coefficient.  Missing, negligible, or unreliable metadata yields
    alpha=0 and therefore the image-evidence candidate.

    Paper equation (dual-route deployment):

        I_out = I_img + alpha(z_obs) * (I_meta - I_img).
    """

    if len(coefficients) != 4 or any(not 0.0 <= value <= 1.0 for value in coefficients):
        raise ValueError("Dual-route coefficients must contain four values in [0, 1].")
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("Dual-route support threshold must lie in [0, 1].")

    direct = stage_result.get("sensor_direct_physical_code")
    support = stage_result.get("sensor_cause_reliability")
    metadata_used = stage_result.get("metadata_used")
    if direct is None or support is None or metadata_used is None:
        raise RuntimeError(
            "Dual-route RMR-P requires observable sensor state, cause reliability, "
            "and metadata-availability outputs from the metadata model."
        )

    motion = direct[:, :3].amax(dim=1)
    cause_scores = torch.stack((motion, direct[:, 3], direct[:, 5]), dim=1)
    motion_support = support[:, :3].amax(dim=1)
    cause_support = torch.stack((motion_support, support[:, 3], support[:, 5]), dim=1)
    dominant = cause_scores.argmax(dim=1)
    dominant_score = cause_scores.gather(1, dominant[:, None]).squeeze(1)
    dominant_support = cause_support.gather(1, dominant[:, None]).squeeze(1)
    compound = (motion > 0.12) & (direct[:, 5] > 0.40)
    compound_support = torch.minimum(motion_support, support[:, 5])

    single_coefficients = direct.new_tensor(coefficients[:3])
    alpha = single_coefficients[dominant]
    alpha = torch.where(
        compound,
        alpha.new_full(alpha.shape, float(coefficients[3])),
        alpha,
    )
    selected_support = torch.where(compound, compound_support, dominant_support)
    eligible = (
        (metadata_used > 0.5)
        & (dominant_score > 0.05)
        & (selected_support >= float(support_threshold))
    )
    alpha = torch.where(eligible, alpha, torch.zeros_like(alpha))

    cause_names = ("motion", "defocus", "lowlight")
    task_names = ("synth_global_motion", "defocus", "low_light")
    causes = [cause_names[int(index)] for index in dominant.detach().cpu()]

    # The image-route expert is selected from the joint corruption posterior,
    # not from a benchmark scenario label.  In practical packets this posterior
    # equals reliable sensor evidence where available and falls back coordinate
    # by coordinate to the image estimate where measurements are weak/missing:
    #
    #     z = q_M * z_M + (1 - q_M) * z_I.
    #
    # Previously weak packets were sent to DeMoE's generic auto-router.  That
    # discarded RMR-P's learned image estimate and caused avoidable defocus
    # misrouting.  The metadata residual gate below remains based on direct,
    # trustworthy sensor support; only expert selection uses the joint state.
    routing_code = stage_result.get("prompt_routing_code")
    if routing_code is None:
        routing_code = direct
    routing_motion = routing_code[:, :3].amax(dim=1)
    routing_scores = torch.stack(
        (routing_motion, routing_code[:, 3], routing_code[:, 5]), dim=1
    )
    routing_dominant = routing_scores.argmax(dim=1)
    routing_strength = routing_scores.gather(
        1, routing_dominant[:, None]
    ).squeeze(1)
    routing_compound = (routing_motion > 0.12) & (routing_code[:, 5] > 0.40)
    tasks = [task_names[int(index)] for index in routing_dominant.detach().cpu()]
    for index, is_compound in enumerate(routing_compound.detach().cpu().tolist()):
        if is_compound:
            tasks[index] = "low_light"
        elif float(routing_strength[index]) <= 0.05:
            tasks[index] = "auto"
    eligible_values = eligible.detach().cpu().tolist()
    for index, is_compound in enumerate(compound.detach().cpu().tolist()):
        if is_compound:
            causes[index] = "compound"
            # The matched DeMoE protocol maps mixed motion/low-light captures
            # to its low-light expert.  Here that choice is made from z_obs.
            tasks[index] = "low_light"
        if not eligible_values[index]:
            causes[index] = "image_fallback"
    return alpha[:, None, None, None], tasks, causes


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    source_root = Path(config["path"])
    split_image_path = str(config[args.split])
    image_dir = source_root / split_image_path
    label_dir = source_root / split_image_path.replace("images", "labels", 1)
    # Follow the path named by the data YAML rather than assuming that its
    # directory name equals the logical split key. This supports immutable
    # holdouts exposed as ``test: images/holdout`` while keeping their paired
    # records under ``metadata/holdout``.
    metadata_dir = (
        args.metadata_dir_override.resolve()
        if args.metadata_dir_override is not None
        else source_root / split_image_path.replace("images", "metadata", 1)
    )
    out = Path(args.out)
    out_image_dir = out / "images" / args.split
    out_label_dir = out / "labels" / args.split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_route_model = None
    if args.model in {"trace_r", "rmrp_fusion"}:
        missing = [
            name
            for name, value in (
                ("--demoe-weights", args.demoe_weights),
                ("--dfpir-weights", args.dfpir_weights),
                ("--instructir-image-weights", args.instructir_image_weights),
                ("--instructir-lm-weights", args.instructir_lm_weights),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "TRACE-R expert fusion requires " + ", ".join(missing)
            )
        policy = TRACERPolicy(
            motion_threshold=args.trace_r_motion_threshold,
            defocus_threshold=args.trace_r_defocus_threshold,
            lowlight_threshold=args.trace_r_lowlight_threshold,
            support_threshold=args.trace_r_support_threshold,
            lowlight_dfpir_weight=args.trace_r_lowlight_dfpir_weight,
            mixed_dfpir_weight=args.trace_r_mixed_dfpir_weight,
            gyro_full_scale=args.trace_r_gyro_full_scale,
        )
        model = TRACERExpertFusion(
            DeMoEAdapter(
                args.demoe_weights,
                device=device,
                task="auto",
                strict=True,
            ),
            DFPIRAdapter(
                args.dfpir_weights,
                device=str(device),
                use_clip=args.dfpir_clip,
            ),
            InstructIRAdapter(
                args.instructir_image_weights,
                args.instructir_lm_weights,
                device=device,
            ),
            policy=policy,
        ).eval()
        scenario_code = None
    elif args.model == "rcadnet":
        if not args.rcadnet_weights:
            raise ValueError("--rcadnet-weights is required")
        model = load_rcadnet(args.rcadnet_weights, device)
        if (args.rcadnet_image_route_weights is None) != (
            args.rcadnet_dual_route_coefficients is None
        ):
            raise ValueError(
                "Dual-route RMR-P requires both --rcadnet-image-route-weights "
                "and --rcadnet-dual-route-coefficients."
            )
        if args.rcadnet_image_route_weights is not None:
            if not isinstance(model, RMRPMetadataDeMoE):
                raise ValueError(
                    "The dual-route policy requires a DeMoE-based RMR-P checkpoint."
                )
            if args.rcadnet_output_stage != "restored":
                raise ValueError(
                    "The dual-route policy is defined only for the final restored output."
                )
            image_route_model = DeMoEAdapter(
                args.rcadnet_image_route_weights,
                device=device,
                task="auto",
                strict=True,
            )
        if args.rcadnet_prompt_router_override is not None:
            if not isinstance(model, RMRPPromptedDFPIR):
                raise ValueError(
                    "Prompt-router override requires a prompted RMR-P checkpoint"
                )
            model.prompt_router = args.rcadnet_prompt_router_override
        if args.rcadnet_prompt_delta_scale_override is not None:
            if not isinstance(model, RMRPPromptedDFPIR):
                raise ValueError(
                    "Prompt-delta override requires a prompted RMR-P checkpoint"
                )
            scale = float(args.rcadnet_prompt_delta_scale_override)
            if not 0.0 <= scale <= 0.5:
                raise ValueError("Prompt-delta scale must be in [0, 0.5]")
            model.prompt_residual_scale = scale
        if args.rcadnet_sensor_route_mode_override is not None:
            if not isinstance(model, RMRPPromptedDFPIR):
                raise ValueError(
                    "Sensor-route override requires a prompted RMR-P checkpoint"
                )
            model.sensor_route_mode = args.rcadnet_sensor_route_mode_override
        if args.rcadnet_compound_motion_blend_override is not None:
            if not isinstance(model, RMRPPromptedDFPIR):
                raise ValueError(
                    "Compound-motion override requires a prompted RMR-P checkpoint"
                )
            blend = float(args.rcadnet_compound_motion_blend_override)
            if not 0.0 <= blend <= 1.0:
                raise ValueError("Compound-motion blend must be in [0, 1]")
            model.compound_motion_blend = blend
        if args.rcadnet_demoe_route_acceptance is not None:
            if not isinstance(model, RMRPMetadataDeMoE):
                raise ValueError(
                    "DeMoE route acceptance requires a DeMoE-based RMR-P checkpoint"
                )
            model.set_cause_route_acceptance(
                args.rcadnet_demoe_route_acceptance
            )
        if args.rcadnet_backbone_route_mode_override is not None:
            if not isinstance(model, RMRPMetadataDeMoE):
                raise ValueError(
                    "Backbone-route override requires a DeMoE-based RMR-P checkpoint"
                )
            model.backbone_route_mode = args.rcadnet_backbone_route_mode_override
        if args.rcadnet_semantic_adapter_gain_override is not None:
            if not isinstance(model, RMRPMetadataDeMoE):
                raise ValueError(
                    "Semantic-adapter gain requires a DeMoE-based RMR-P checkpoint"
                )
            model.set_semantic_adapter_gain(
                args.rcadnet_semantic_adapter_gain_override
            )
        if args.rcadnet_semantic_adapter_acceptance is not None:
            if not isinstance(model, RMRPMetadataDeMoE):
                raise ValueError(
                    "Semantic-adapter trust requires a DeMoE-based RMR-P checkpoint"
                )
            model.set_semantic_adapter_acceptance(
                args.rcadnet_semantic_adapter_acceptance
            )
        if args.rcadnet_compound_metadata_acceptance_override is not None:
            if not isinstance(model, RMRPMetadataDeMoE):
                raise ValueError(
                    "Compound metadata acceptance requires a DeMoE-based RMR-P checkpoint"
                )
            acceptance = float(args.rcadnet_compound_metadata_acceptance_override)
            if not 0.0 <= acceptance <= 1.0:
                raise ValueError(
                    "compound metadata acceptance must be in [0, 1]"
                )
            model.compound_metadata_acceptance = acceptance
        if args.rcadnet_motion_prior_compound_floor is not None:
            floor = float(args.rcadnet_motion_prior_compound_floor)
            if not 0.0 <= floor <= 1.0:
                raise ValueError(
                    "--rcadnet-motion-prior-compound-floor must be in [0, 1]"
                )
            if model.motion_prior is None:
                raise ValueError(
                    "Compound-floor override requires a checkpoint with "
                    "the metadata motion prior enabled"
                )
            model.motion_prior.compound_gate_floor = floor
        if args.rcadnet_motion_prior_regularization is not None:
            regularization = float(args.rcadnet_motion_prior_regularization)
            if regularization <= 0.0 or model.motion_prior is None:
                raise ValueError(
                    "Motion-prior regularization must be positive and requires "
                    "an enabled metadata motion prior"
                )
            model.motion_prior.regularization = regularization
        if args.rcadnet_motion_prior_blend is not None:
            blend = float(args.rcadnet_motion_prior_blend)
            if not 0.0 <= blend <= 1.0 or model.motion_prior is None:
                raise ValueError(
                    "Motion-prior blend must be in [0, 1] and requires an "
                    "enabled metadata motion prior"
                )
            model.motion_prior.blend = blend
        if args.rcadnet_motion_prior_solver is not None:
            if model.motion_prior is None:
                raise ValueError(
                    "Motion-prior solver override requires an enabled prior"
                )
            model.motion_prior.solver = args.rcadnet_motion_prior_solver
        if args.rcadnet_richardson_lucy_iterations is not None:
            iterations = int(args.rcadnet_richardson_lucy_iterations)
            if iterations < 1 or model.motion_prior is None:
                raise ValueError(
                    "Richardson-Lucy iterations must be positive and require "
                    "an enabled metadata motion prior"
                )
            model.motion_prior.richardson_lucy_iterations = iterations
        specialist_model = (
            load_rcadnet(args.rcadnet_lowlight_specialist_weights, device)
            if args.rcadnet_lowlight_specialist_weights
            else None
        )
        scenario_code = code_from_scenario(args.scenario, device=device)
    elif args.model in {"nafnet", "nafnet_meta"}:
        if not args.nafnet_weights:
            raise ValueError("--nafnet-weights is required")
        model = (
            load_metadata_nafnet(args.nafnet_weights, device)
            if args.model == "nafnet_meta"
            else load_nafnet(args.nafnet_weights, device)
        )
        scenario_code = code_from_scenario(args.scenario, device=device) if args.model == "nafnet_meta" else None
    elif args.model == "demoe":
        if not args.demoe_weights and not args.demoe_smoke:
            raise ValueError("--demoe-weights is required for real DeMoE; use --demoe-smoke only for pipeline tests")
        model = DeMoEAdapter(
            args.demoe_weights,
            device=device,
            task=args.demoe_task,
            smoke=args.demoe_smoke,
            strict=not args.demoe_smoke,
        )
        scenario_code = None
    elif args.model == "instructir":
        model = InstructIRAdapter(
            args.instructir_image_weights,
            args.instructir_lm_weights,
            device=device,
        )
        scenario_code = None
    else:
        if not args.dfpir_weights:
            raise ValueError("--dfpir-weights is required")
        model = DFPIRAdapter(args.dfpir_weights, device=str(device), use_clip=args.dfpir_clip)
        scenario_code = None

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.max_images < 0:
        raise ValueError("--max-images cannot be negative")
    if args.max_images and len(paths) > args.max_images:
        sampler = random.Random(args.image_sample_seed)
        paths = sorted(sampler.sample(paths, args.max_images))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metadata_conditioned = (
        args.model in {"trace_r", "rmrp_fusion"}
        or (args.model == "rcadnet" and args.rcadnet_code_source == "metadata")
    )
    if metadata_conditioned and args.require_metadata:
        missing_metadata = [path.name for path in paths if not (metadata_dir / f"{path.stem}.json").exists()]
        if missing_metadata:
            examples = ", ".join(missing_metadata[:5])
            raise FileNotFoundError(
                f"Required per-image metadata are missing for {len(missing_metadata)} images under "
                f"{metadata_dir}. Examples: {examples}"
            )
    shuffled_codes: dict[str, torch.Tensor] = {}
    if metadata_conditioned and args.rcadnet_metadata_control == "shuffled":
        rng = random.Random(args.metadata_control_seed)
        metadata_codes: list[torch.Tensor] = []
        for path in paths:
            metadata_path = metadata_dir / f"{path.stem}.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata_codes.append(metadata_tensor(metadata, model, device))
            else:
                metadata_codes.append(scenario_code)
        shuffled = list(metadata_codes)
        rng.shuffle(shuffled)
        if len(shuffled) > 1 and all(torch.equal(a, b) for a, b in zip(metadata_codes, shuffled)):
            shuffled = shuffled[1:] + shuffled[:1]
        shuffled_codes = {path.name: code for path, code in zip(paths, shuffled)}
    inference_times_ms: list[float] = []
    specialist_route_count = 0
    specialist_image_fallback_count = 0
    specialist_route_scores: list[float] = []
    prompted_weights: list[list[float]] = []
    prompted_codes: list[list[float]] = []
    prompted_route_codes: list[list[float]] = []
    prompted_support: list[list[float]] = []
    dual_route_alphas: list[float] = []
    dual_route_causes: list[str] = []
    dual_route_tasks: list[str] = []
    fusion_routes: list[str] = []
    with torch.inference_mode():
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
            tensor, original_size = pad_to_multiple(
                tensor,
                multiple=(
                    16
                    if args.model == "instructir"
                    else (1 if args.model in {"trace_r", "rmrp_fusion"} else 8)
                ),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_started = time.perf_counter()
            if args.model in {"trace_r", "rmrp_fusion"}:
                metadata_path = metadata_dir / f"{path.stem}.json"
                if args.rcadnet_metadata_control == "unavailable":
                    code = None
                elif args.rcadnet_metadata_control == "shuffled":
                    code = shuffled_codes[path.name]
                elif metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    code = sensor_packet_from_mapping(metadata, device=device)
                    if args.rcadnet_metadata_control == "zero":
                        code = torch.zeros_like(code)
                    elif args.rcadnet_metadata_control == "noisy":
                        code = perturb_sensor_packet(
                            code,
                            float(args.metadata_noise_std),
                        )
                    elif args.rcadnet_metadata_control in {
                        "camera_only",
                        "imu_only",
                        "vehicle_only",
                        "camera_imu",
                        "camera_vehicle",
                        "imu_vehicle",
                    }:
                        keep_camera = args.rcadnet_metadata_control in {
                            "camera_only",
                            "camera_imu",
                            "camera_vehicle",
                        }
                        keep_imu = args.rcadnet_metadata_control in {
                            "imu_only",
                            "camera_imu",
                            "imu_vehicle",
                        }
                        keep_vehicle = args.rcadnet_metadata_control in {
                            "vehicle_only",
                            "camera_vehicle",
                            "imu_vehicle",
                        }
                        code = apply_sensor_modality_mask(
                            code,
                            camera_available=keep_camera,
                            imu_available=keep_imu,
                            vehicle_available=keep_vehicle,
                        )
                    elif args.rcadnet_metadata_control != "correct":
                        raise ValueError(
                            "TRACE-R expert fusion supports correct, zero, unavailable, "
                            "shuffled, noisy, and modality metadata controls; received "
                            f"{args.rcadnet_metadata_control!r}."
                        )
                else:
                    code = None
                fusion_result = model(tensor, code, return_dict=True)
                restored = fusion_result["restored"]
                fusion_routes.extend(fusion_result["route_names"])
            elif args.model == "rcadnet":
                motion_prior_code = None
                if args.rcadnet_code_source == "blind":
                    code = None
                elif args.rcadnet_code_source == "metadata":
                    metadata_path = metadata_dir / f"{path.stem}.json"
                    if args.rcadnet_metadata_control == "unavailable":
                        # True missing-metadata control. Passing None causes the
                        # fusion module to use the image-estimated degradation
                        # code and an unavailable external-metadata indicator.
                        code = None
                    elif args.rcadnet_metadata_control == "zero":
                        # Metadata ablation: z_m is replaced by an all-zero
                        # degradation code while the learned image-estimated
                        # code remains available inside metadata_fused models.
                        code = (
                            torch.zeros(model.sensor_dim, device=device)
                            if getattr(model, "use_practical_sensor_encoder", False)
                            else torch.zeros_like(scenario_code)
                        )
                    elif args.rcadnet_metadata_control == "shuffled" and path.name in shuffled_codes:
                        # Reviewer control: correct labels are preserved, but
                        # the metadata code comes from another image. This tests
                        # whether metadata gains depend on aligned metadata.
                        code = shuffled_codes[path.name]
                    elif metadata_path.exists():
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if (
                            getattr(model, "use_motion_prior", False)
                            and not getattr(model, "use_practical_sensor_encoder", False)
                        ):
                            # Historical controlled-data checkpoints receive
                            # their explicitly declared axial code. Practical
                            # sensor checkpoints must leave this argument unset:
                            # RMR-Net then uses the training-calibrated
                            # camera/IMU-to-PSF head rather than bypassing it
                            # with the raw, under-scaled observable trajectory.
                            motion_prior_code = practical_motion_prior_code(metadata, device)
                            if motion_prior_code is None:
                                motion_prior_code = code_from_metadata(
                                    metadata,
                                    device=device,
                                    encoding="axial_v2",
                                )
                        code = metadata_tensor(metadata, model, device)
                        if args.rcadnet_metadata_control == "noisy":
                            # Reviewer control for imperfect sensors:
                            # z_m' = clamp(z_m + eps), eps ~ N(0,sigma^2).
                            if getattr(model, "use_practical_sensor_encoder", False):
                                code = perturb_sensor_packet(code, float(args.metadata_noise_std))
                            else:
                                noise = torch.randn_like(code) * float(args.metadata_noise_std)
                                code = (code + noise).clamp(0.0, 1.0)
                            if motion_prior_code is not None:
                                motion_prior_code = (
                                    motion_prior_code + torch.randn_like(motion_prior_code) * float(args.metadata_noise_std)
                                ).clamp(0.0, 1.0)
                        elif args.rcadnet_metadata_control in {
                            "camera_only",
                            "imu_only",
                            "vehicle_only",
                            "camera_imu",
                            "camera_vehicle",
                            "imu_vehicle",
                        }:
                            if not getattr(model, "use_practical_sensor_encoder", False):
                                raise ValueError(
                                    f"{args.rcadnet_metadata_control} requires a "
                                    "practical sensor-conditioned checkpoint"
                                )
                            keep_camera = args.rcadnet_metadata_control in {
                                "camera_only",
                                "camera_imu",
                                "camera_vehicle",
                            }
                            keep_imu = args.rcadnet_metadata_control in {
                                "imu_only",
                                "camera_imu",
                                "imu_vehicle",
                            }
                            keep_vehicle = args.rcadnet_metadata_control in {
                                "vehicle_only",
                                "camera_vehicle",
                                "imu_vehicle",
                            }
                            code = apply_sensor_modality_mask(
                                code,
                                camera_available=keep_camera,
                                imu_available=keep_imu,
                                vehicle_available=keep_vehicle,
                            )
                        elif args.rcadnet_metadata_control == "counterfactual":
                            # Hard negative for scenario-constant metadata:
                            # remove or swap one causal degradation factor.
                            code = counterfactual_metadata_code(code)
                            if motion_prior_code is not None:
                                motion_prior_code = counterfactual_metadata_code(motion_prior_code)
                        elif args.rcadnet_metadata_control in {
                            "correct_family_wrong_severity",
                            "wrong_family_matched_severity",
                            "adversarial",
                        }:
                            code = metadata_factor_control(code, args.rcadnet_metadata_control)
                    else:
                        if args.rcadnet_metadata_control == "counterfactual":
                            code = counterfactual_metadata_code(scenario_code)
                        elif args.rcadnet_metadata_control in {
                            "correct_family_wrong_severity",
                            "wrong_family_matched_severity",
                            "adversarial",
                        }:
                            code = metadata_factor_control(scenario_code, args.rcadnet_metadata_control)
                        else:
                            code = scenario_code
                else:
                    code = scenario_code
                model_kwargs = {
                    "motion_prior_code": motion_prior_code,
                    "gate_threshold": (
                        args.gate_threshold
                        if args.gate_threshold >= 0
                        else None
                    ),
                    "gate_softness": args.gate_softness,
                }
                if image_route_model is not None:
                    stage_result = model(
                        tensor,
                        code,
                        return_dict=True,
                        **model_kwargs,
                    )
                    metadata_restored = stage_result["restored"]
                    alpha, image_tasks, route_causes = rmrp_dual_route_policy(
                        stage_result,
                        list(args.rcadnet_dual_route_coefficients),
                        float(args.rcadnet_dual_route_support_threshold),
                    )
                    if tensor.shape[0] == 1:
                        image_restored = image_route_model(
                            tensor,
                            task=image_tasks[0],
                        )
                    else:
                        image_restored = torch.cat(
                            [
                                image_route_model(
                                    tensor[item : item + 1],
                                    task=image_tasks[item],
                                )
                                for item in range(tensor.shape[0])
                            ],
                            dim=0,
                        )
                    restored = image_restored + alpha * (
                        metadata_restored - image_restored
                    )
                    dual_route_alphas.extend(
                        float(value) for value in alpha[:, 0, 0, 0].detach().cpu()
                    )
                    dual_route_causes.extend(route_causes)
                    dual_route_tasks.extend(image_tasks)
                    prompted_codes.extend(stage_result["code"].detach().cpu().tolist())
                    prompted_route_codes.extend(
                        stage_result["prompt_routing_code"].detach().cpu().tolist()
                    )
                    prompted_support.extend(
                        stage_result["sensor_cause_reliability"].detach().cpu().tolist()
                    )
                elif specialist_model is None:
                    if isinstance(model, RMRPPromptedDFPIR):
                        stage_result = model(
                            tensor,
                            code,
                            return_dict=True,
                            **model_kwargs,
                        )
                        restored = stage_result.get(args.rcadnet_output_stage)
                        if restored is None:
                            raise RuntimeError(
                                f"RMR-P output stage {args.rcadnet_output_stage!r} "
                                "is unavailable for this checkpoint/input."
                            )
                        prompted_weights.extend(
                            stage_result["prompt_weights"].detach().cpu().tolist()
                        )
                        prompted_codes.extend(
                            stage_result["code"].detach().cpu().tolist()
                        )
                        prompted_route_codes.extend(
                            stage_result["prompt_routing_code"].detach().cpu().tolist()
                        )
                        prompted_support.extend(
                            stage_result["sensor_cause_reliability"].detach().cpu().tolist()
                        )
                    elif args.rcadnet_output_stage == "restored":
                        restored = model(tensor, code, **model_kwargs)
                    else:
                        stage_result = model(
                            tensor,
                            code,
                            return_dict=True,
                            **model_kwargs,
                        )
                        restored = stage_result.get(args.rcadnet_output_stage)
                        if restored is None:
                            raise RuntimeError(
                                f"RMR-P output stage {args.rcadnet_output_stage!r} "
                                "is unavailable for this checkpoint/input."
                            )
                else:
                    shared_result = model(
                        tensor,
                        code,
                        return_dict=True,
                        **model_kwargs,
                    )
                    restored = shared_result["restored"]
                    fused_code = shared_result["code"]
                    route_score = fused_code[:, 5].clamp(0.0, 1.0)
                    motion_route_score = fused_code[:, :3].amax(dim=1).clamp(
                        0.0, 1.0
                    )
                    camera_available = torch.zeros(
                        tensor.shape[0],
                        dtype=torch.bool,
                        device=tensor.device,
                    )
                    if (
                        code is not None
                        and getattr(model, "use_practical_sensor_encoder", False)
                        and code.shape[-1] == model.sensor_dim
                    ):
                        packet = code.unsqueeze(0) if code.ndim == 1 else code
                        direct_code = conditioning_code_from_packet(
                            packet,
                            gyro_full_scale=model.sensor_gyro_full_scale,
                        )
                        camera_available = (
                            packet[:, CONTEXT_START + 12] > 0.0
                        )
                        route_score = torch.where(
                            camera_available,
                            direct_code[:, 5].clamp(0.0, 1.0),
                            route_score,
                        )
                        motion_route_score = torch.where(
                            camera_available,
                            direct_code[:, :3].amax(dim=1).clamp(0.0, 1.0),
                            motion_route_score,
                        )
                    image_mean = tensor.mean(dim=(1, 2, 3))
                    image_lowlight_fallback = (
                        ~camera_available
                    ) & (
                        image_mean
                        <= float(
                            args.rcadnet_lowlight_image_mean_threshold
                        )
                    )
                    route_score = torch.where(
                        image_lowlight_fallback,
                        torch.ones_like(route_score),
                        route_score,
                    )
                    route_mask = (
                        route_score
                        >= float(args.rcadnet_lowlight_route_threshold)
                    )
                    motion_present = (
                        motion_route_score
                        >= float(args.rcadnet_motion_route_threshold)
                    )
                    if args.rcadnet_specialist_support == "mixed":
                        route_mask = route_mask & motion_present
                    elif args.rcadnet_specialist_support == "lowlight":
                        route_mask = route_mask & ~motion_present
                    specialist_route_scores.extend(
                        float(value) for value in route_score.detach().cpu()
                    )
                    if bool(route_mask.any()):
                        # Without exposure/ISO, an IMU-only trajectory can
                        # over-deconvolve a dark compound frame. Fall back to
                        # the specialist's image-estimated state in that case.
                        specialist_code = (
                            None
                            if bool(image_lowlight_fallback.all())
                            else code
                        )
                        specialist_restored = specialist_model(
                            tensor,
                            specialist_code,
                            motion_prior_code=(
                                None
                                if specialist_code is None
                                else motion_prior_code
                            ),
                            gate_threshold=model_kwargs["gate_threshold"],
                            gate_softness=model_kwargs["gate_softness"],
                        )
                        restored = torch.where(
                            route_mask[:, None, None, None],
                            specialist_restored,
                            restored,
                        )
                        specialist_route_count += int(route_mask.sum().item())
                        specialist_image_fallback_count += int(
                            image_lowlight_fallback.sum().item()
                        )
            elif args.model == "nafnet":
                restored = model(tensor)
            elif args.model == "nafnet_meta":
                metadata_path = metadata_dir / f"{path.stem}.json"
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if model.code_dim == 82:
                        # The matched FiLM-NAFNet control receives the same
                        # public camera/IMU/vehicle packet as RMR-P. Hidden
                        # renderer targets and scenario labels are not used.
                        code = sensor_packet_from_mapping(metadata, device=device)
                    else:
                        code = code_from_metadata(
                            metadata,
                            device=device,
                            encoding=getattr(model, "metadata_encoding", "legacy"),
                        )
                else:
                    code = (
                        torch.zeros(model.code_dim, device=device)
                        if model.code_dim == 82
                        else scenario_code
                    )
                restored = model(tensor, code)
            elif args.model == "demoe":
                restored = model(tensor, scenario=args.scenario, task=args.demoe_task)
            elif args.model == "instructir":
                restored = model(tensor, instructir_prompt(args.scenario, args.instructir_prompt_mode))
            else:
                restored = model(tensor, args.scenario)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_times_ms.append((time.perf_counter() - inference_started) * 1000.0)
            if args.residual_strength != 1.0:
                eta = max(0.0, min(float(args.residual_strength), 1.0))
                restored = (tensor + eta * (restored - tensor)).clamp(0.0, 1.0)
            restored = restored[..., : original_size[0], : original_size[1]]
            if args.debug_every > 0 and index % args.debug_every == 0:
                residual = restored - tensor[..., : original_size[0], : original_size[1]]
                debug = {
                    "tag": "restore_yolo_split_debug",
                    "index": index,
                    "image": path.name,
                    "model": args.model,
                    "scenario": args.scenario,
                    "residual_strength": float(args.residual_strength),
                    "residual_abs_mean": float(residual.abs().mean().cpu()),
                    "residual_abs_max": float(residual.abs().max().cpu()),
                }
                debug.update(image_stats("input", tensor[..., : original_size[0], : original_size[1]]))
                debug.update(image_stats("restored", restored))
                print(json.dumps(debug), flush=True)
            # Restoration must not be followed by an uncontrolled second JPEG
            # degradation.  Export lossless PNG while retaining the source stem
            # so the unchanged YOLO label remains aligned.
            # PNG compression level changes storage time/size only; decoded
            # restoration pixels are identical. A low level keeps large,
            # restartable candidate-cache runs from becoming CPU-bound.
            TF.to_pil_image(restored[0].detach().cpu()).save(
                out_image_dir / f"{path.stem}.png",
                compress_level=1,
            )
            label_path = label_dir / path.with_suffix(".txt").name
            if label_path.exists():
                shutil.copy2(label_path, out_label_dir / label_path.name)

    data_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": config.get("train", "images/train"),
        "val": f"images/{args.split}",
        "test": f"images/{args.split}",
        "names": config["names"],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    measured = inference_times_ms[2:] if len(inference_times_ms) > 2 else inference_times_ms
    runtime = {
        "model": args.model,
        "scenario": args.scenario,
        "rcadnet_output_stage": (
            args.rcadnet_output_stage if args.model == "rcadnet" else None
        ),
        "rcadnet_prompt_router": (
            getattr(model, "prompt_router", None)
            if args.model == "rcadnet"
            else None
        ),
        "rcadnet_prompt_router_override": (
            args.rcadnet_prompt_router_override
            if args.model == "rcadnet"
            else None
        ),
        "rcadnet_prompt_delta_scale": (
            float(getattr(model, "prompt_residual_scale", 0.0))
            if args.model == "rcadnet"
            and isinstance(model, RMRPPromptedDFPIR)
            else None
        ),
        "rcadnet_prompt_delta_scale_override": (
            args.rcadnet_prompt_delta_scale_override
            if args.model == "rcadnet"
            else None
        ),
        "rcadnet_sensor_route_mode": (
            getattr(model, "sensor_route_mode", None)
            if args.model == "rcadnet"
            else None
        ),
        "rcadnet_sensor_route_mode_override": (
            args.rcadnet_sensor_route_mode_override
            if args.model == "rcadnet"
            else None
        ),
        "rcadnet_compound_motion_blend": (
            float(getattr(model, "compound_motion_blend", 0.0))
            if args.model == "rcadnet" and isinstance(model, RMRPPromptedDFPIR)
            else None
        ),
        "rcadnet_compound_motion_blend_override": (
            args.rcadnet_compound_motion_blend_override
            if args.model == "rcadnet"
            else None
        ),
        "rcadnet_demoe_route_acceptance": (
            list(getattr(model, "cause_route_acceptance", ()))
            if args.model == "rcadnet" and isinstance(model, RMRPMetadataDeMoE)
            else None
        ),
        "rcadnet_backbone_route_mode": (
            getattr(model, "backbone_route_mode", None)
            if args.model == "rcadnet" and isinstance(model, RMRPMetadataDeMoE)
            else None
        ),
        "rcadnet_semantic_adapter_gain": (
            float(getattr(model, "semantic_adapter_gain", 0.0))
            if args.model == "rcadnet" and isinstance(model, RMRPMetadataDeMoE)
            else None
        ),
        "rcadnet_semantic_adapter_acceptance": (
            list(getattr(model, "semantic_adapter_acceptance", ()))
            if args.model == "rcadnet" and isinstance(model, RMRPMetadataDeMoE)
            else None
        ),
        "rcadnet_compound_metadata_acceptance": (
            float(getattr(model, "compound_metadata_acceptance", 1.0))
            if args.model == "rcadnet" and isinstance(model, RMRPMetadataDeMoE)
            else None
        ),
        "dual_route_enabled": bool(image_route_model is not None),
        "dual_route_image_weights": (
            str(Path(args.rcadnet_image_route_weights).resolve())
            if args.rcadnet_image_route_weights
            else None
        ),
        "dual_route_coefficients": (
            list(args.rcadnet_dual_route_coefficients)
            if args.rcadnet_dual_route_coefficients is not None
            else None
        ),
        "dual_route_support_threshold": (
            float(args.rcadnet_dual_route_support_threshold)
            if image_route_model is not None
            else None
        ),
        "dual_route_alpha_mean": (
            float(np.mean(dual_route_alphas)) if dual_route_alphas else None
        ),
        "dual_route_alpha_values": dual_route_alphas or None,
        "dual_route_cause_counts": (
            {
                cause: dual_route_causes.count(cause)
                for cause in sorted(set(dual_route_causes))
            }
            if dual_route_causes
            else None
        ),
        "dual_route_task_counts": (
            {
                task: dual_route_tasks.count(task)
                for task in sorted(set(dual_route_tasks))
            }
            if dual_route_tasks
            else None
        ),
        "trace_r_expert_fusion_policy": (
            {
                "motion_threshold": model.policy.motion_threshold,
                "defocus_threshold": model.policy.defocus_threshold,
                "lowlight_threshold": model.policy.lowlight_threshold,
                "support_threshold": model.policy.support_threshold,
                "lowlight_dfpir_weight": model.policy.lowlight_dfpir_weight,
                "mixed_dfpir_weight": model.policy.mixed_dfpir_weight,
                "gyro_full_scale": model.policy.gyro_full_scale,
                "scenario_family_is_model_input": False,
            }
            if isinstance(model, TRACERExpertFusion)
            else None
        ),
        "trace_r_expert_fusion_route_counts": (
            {
                route: fusion_routes.count(route)
                for route in sorted(set(fusion_routes))
            }
            if fusion_routes
            else None
        ),
        "images": len(paths),
        "device": str(device),
        "backend": "GPU-confirmed" if device.type == "cuda" else "CPU",
        "warmup_images_excluded": min(2, len(inference_times_ms)),
        "timing_scope": "model inference only; image decode, tensor transfer, label copy, and image encoding excluded",
        "mean_inference_ms": float(np.mean(measured)) if measured else None,
        "median_inference_ms": float(np.median(measured)) if measured else None,
        "p95_inference_ms": float(np.percentile(measured, 95)) if measured else None,
        "peak_cuda_MiB": (
            float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            if device.type == "cuda"
            else 0.0
        ),
        "per_image_inference_ms": inference_times_ms,
        "prompt_weight_mean": (
            np.asarray(prompted_weights, dtype=np.float64).mean(axis=0).tolist()
            if prompted_weights
            else None
        ),
        "prompt_route_counts": (
            np.bincount(
                np.asarray(prompted_weights, dtype=np.float64).argmax(axis=1),
                minlength=3,
            ).tolist()
            if prompted_weights
            else None
        ),
        "posterior_code_mean": (
            np.asarray(prompted_codes, dtype=np.float64).mean(axis=0).tolist()
            if prompted_codes
            else None
        ),
        "prompt_routing_code_mean": (
            np.asarray(prompted_route_codes, dtype=np.float64).mean(axis=0).tolist()
            if prompted_route_codes
            else None
        ),
        "sensor_cause_reliability_mean": (
            np.asarray(prompted_support, dtype=np.float64).mean(axis=0).tolist()
            if prompted_support
            else None
        ),
        "motion_prior_compound_floor_override": (
            float(args.rcadnet_motion_prior_compound_floor)
            if args.rcadnet_motion_prior_compound_floor is not None
            else None
        ),
        "motion_prior_regularization_override": (
            float(args.rcadnet_motion_prior_regularization)
            if args.rcadnet_motion_prior_regularization is not None
            else None
        ),
        "motion_prior_blend_override": (
            float(args.rcadnet_motion_prior_blend)
            if args.rcadnet_motion_prior_blend is not None
            else None
        ),
        "motion_prior_solver_override": args.rcadnet_motion_prior_solver,
        "richardson_lucy_iterations_override": (
            int(args.rcadnet_richardson_lucy_iterations)
            if args.rcadnet_richardson_lucy_iterations is not None
            else None
        ),
        "lowlight_specialist_enabled": bool(
            args.rcadnet_lowlight_specialist_weights
        ),
        "lowlight_specialist_weights": (
            str(Path(args.rcadnet_lowlight_specialist_weights).resolve())
            if args.rcadnet_lowlight_specialist_weights
            else None
        ),
        "lowlight_route_threshold": float(
            args.rcadnet_lowlight_route_threshold
        ),
        "lowlight_image_mean_threshold": float(
            args.rcadnet_lowlight_image_mean_threshold
        ),
        "lowlight_specialist_route_count": specialist_route_count,
        "lowlight_specialist_image_fallback_count": (
            specialist_image_fallback_count
        ),
        "lowlight_specialist_route_fraction": (
            specialist_route_count / len(paths) if paths else 0.0
        ),
        "lowlight_route_score_mean": (
            float(np.mean(specialist_route_scores))
            if specialist_route_scores
            else None
        ),
    }
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print({"model": args.model, "scenario": args.scenario, "images": len(paths), "out": str(out)})


if __name__ == "__main__":
    main()
