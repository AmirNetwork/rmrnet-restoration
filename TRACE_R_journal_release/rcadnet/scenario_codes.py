from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch


CODE_NAMES = (
    "motion_x",
    "motion_y",
    "motion_random",
    "defocus",
    "gaussian_noise",
    "low_light",
    "jpeg",
    "strength",
)


@dataclass(frozen=True)
class DegradationMetadata:
    """Real or proxy sensor/context metadata used by the RMR-Net conditioning path."""

    gyro_x: float = 0.0
    gyro_y: float = 0.0
    accel_norm: float = 0.0
    speed_mps: float = 0.0
    exposure_ms: float = 0.0
    defocus_score: float = 0.0
    noise_score: float = 0.0
    blur_angle_deg: float | None = None
    blur_length_px: float | None = None
    low_light_score: float = 0.0
    jpeg_quality: float | None = None
    raw_oxts_yaw_rate_radps: float | None = None
    raw_oxts_forward_accel_mps2: float | None = None
    raw_oxts_lateral_accel_mps2: float | None = None


def _strength_from_name(name: str) -> float:
    if "mild" in name or "sigma1" in name:
        return 0.30
    if "medium" in name or "sigma2" in name or "sigma3" in name:
        return 0.60
    if "strong" in name or "sigma5" in name:
        return 0.90
    return 0.50


def code_from_scenario(scenario: str, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Map benchmark scenario names to an 8-D degradation-conditioning code.

    This keeps RCAD-Net usable before real IMU is available. The code is soft,
    not one-hot, because mixed scenarios should activate multiple priors.
    """

    name = scenario.lower()
    code = torch.zeros(len(CODE_NAMES), dtype=torch.float32, device=device)
    if name in {"clean", "identity", "native_identity", "clean_native", "native_clean"}:
        return code
    strength = _strength_from_name(name)

    if "horizontal" in name:
        code[0] = strength
    if "vertical" in name:
        code[1] = strength
    if "diagonal" in name:
        code[0] = strength * 0.7
        code[1] = strength * 0.7
    if "random" in name or "vibration" in name:
        code[2] = strength
    if "motion" in name and not torch.any(code[:3] > 0):
        code[2] = strength * 0.7
    if "defocus" in name:
        code[3] = strength
    if "gaussian" in name or "noise" in name:
        code[4] = strength
    if "lowlight" in name or "low_light" in name:
        code[5] = strength
    if "jpeg" in name:
        code[6] = 0.7 if "40" in name else 0.4

    code[7] = strength
    return code


def code_from_metadata(
    metadata: DegradationMetadata | Mapping[str, float | int | None],
    *,
    device: torch.device | str | None = None,
    encoding: str = "legacy",
) -> torch.Tensor:
    """Convert real or estimated vehicle/camera metadata into an RMR code.

    ``legacy`` preserves historical checkpoint semantics. ``axial_v2`` is the
    corrected exact-kernel representation: a line-spread function is axial, so
    cos(2 theta) and sin(2 theta) preserve its signed orientation modulo pi.
    """

    if encoding not in {"legacy", "axial_v2"}:
        raise ValueError(f"Unknown metadata encoding: {encoding}")

    if not isinstance(metadata, DegradationMetadata):
        metadata = DegradationMetadata(**{k: v for k, v in metadata.items() if k in DegradationMetadata.__annotations__})

    code = torch.zeros(len(CODE_NAMES), dtype=torch.float32, device=device)
    gyro_x = abs(float(metadata.gyro_x))
    gyro_y = abs(float(metadata.gyro_y))
    accel = abs(float(metadata.accel_norm))
    exposure = max(float(metadata.exposure_ms), 0.0)
    speed = max(float(metadata.speed_mps), 0.0)
    yaw_rate = 0.0 if metadata.raw_oxts_yaw_rate_radps is None else float(metadata.raw_oxts_yaw_rate_radps)
    forward_accel = 0.0 if metadata.raw_oxts_forward_accel_mps2 is None else float(metadata.raw_oxts_forward_accel_mps2)
    lateral_accel = 0.0 if metadata.raw_oxts_lateral_accel_mps2 is None else float(metadata.raw_oxts_lateral_accel_mps2)
    raw_vibration = math.sqrt(forward_accel**2 + lateral_accel**2)
    accel = max(accel, raw_vibration)

    if metadata.blur_length_px is not None:
        # If a blur-length estimate is available, treat it as the calibrated
        # physics-derived motion severity.  Earlier versions also let the
        # synthetic speed/exposure proxy increase this value, which saturated
        # mixed low-light motion codes and made correct metadata too aggressive.
        #
        # Paper notation: z_m is the metadata degradation coordinate; this line
        # maps the measured blur magnitude directly into the motion component
        # of z_m rather than letting a secondary proxy dominate it.
        motion_scale = min(float(metadata.blur_length_px) / 25.0, 1.0)
    else:
        # Raw telemetry mode: no derived blur length or angle is available. We
        # project speed, exposure, yaw rate, and vibration to a coarse motion
        # prior. This is intentionally less privileged than passing a blur
        # kernel estimate, but still represents metadata that a vehicle can log.
        base_motion = min((gyro_x + gyro_y + 0.02 * speed * exposure) / 5.0, 1.0)
        speed_score = min(speed / 18.0, 1.0)
        yaw_score = min(abs(yaw_rate) / 0.09, 1.0)
        vib_score = min(accel / 4.5, 1.0)
        exposure_score = min(exposure / 24.0, 1.0) if exposure > 0 else 0.0
        raw_motion = 0.46 * speed_score * math.sqrt(max(exposure_score, 0.0)) + 0.25 * yaw_score + 0.17 * vib_score
        if encoding == "legacy":
            # Retain the old weak-motion prior only for old checkpoints. In the
            # corrected encoding, a record with no motion evidence stays zero.
            raw_motion += 0.12
        motion_scale = max(base_motion, min(raw_motion, 1.0))

    if metadata.blur_angle_deg is not None:
        # A calibrated telemetry blur estimate gives direction as well as severity.
        angle = math.radians(float(metadata.blur_angle_deg))
        if encoding == "axial_v2":
            code[0] = min(0.5 * (1.0 + math.cos(2.0 * angle)) * motion_scale, 1.0)
            code[1] = min(0.5 * (1.0 + math.sin(2.0 * angle)) * motion_scale, 1.0)
            code[2] = min(motion_scale, 1.0)
        else:
            code[0] = min(abs(math.cos(angle)) * motion_scale, 1.0)
            code[1] = min(abs(math.sin(angle)) * motion_scale, 1.0)
    elif abs(yaw_rate) > 1e-8 or abs(lateral_accel) > 1e-8:
        # Camera-plane direction proxy from raw vehicle motion. It uses only
        # speed, yaw rate, and lateral acceleration, not the generated blur
        # kernel parameters saved in the controlled KITTI split.
        angle = math.atan2(24.0 * yaw_rate + 0.18 * lateral_accel, max(speed, 1.0))
        angle = max(min(angle * 2.0, math.radians(48.0)), math.radians(-48.0))
        if encoding == "axial_v2":
            code[0] = min(0.5 * (1.0 + math.cos(2.0 * angle)) * motion_scale, 1.0)
            code[1] = min(0.5 * (1.0 + math.sin(2.0 * angle)) * motion_scale, 1.0)
            code[2] = min(motion_scale, 1.0)
        else:
            code[0] = min(abs(math.cos(angle)) * motion_scale, 1.0)
            code[1] = min(abs(math.sin(angle)) * motion_scale, 1.0)
    else:
        horizontal = min(0.65 * motion_scale + gyro_x / 3.0, 1.0)
        vertical = min(gyro_y / 3.0, 1.0)
        code[0] = horizontal
        code[1] = vertical
    if encoding != "axial_v2" or metadata.blur_angle_deg is None:
        code[2] = min(accel / 5.0, 1.0)
    code[3] = min(float(metadata.defocus_score), 1.0)
    code[4] = min(float(metadata.noise_score), 1.0)
    code[5] = min(float(metadata.low_light_score), 1.0)
    if metadata.jpeg_quality is not None:
        code[6] = max(0.0, min((80.0 - float(metadata.jpeg_quality)) / 60.0, 1.0))
    code[7] = max(motion_scale, code[2], code[3], code[4], code[5], code[6])
    return code
