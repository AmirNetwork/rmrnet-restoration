from __future__ import annotations

import math
from typing import Any

import numpy as np


def synthetic_metadata_from_scenario(scenario: str, seed: int = 0) -> dict[str, Any]:
    """Create sensor-like metadata matching the synthetic degradation.

    The values are not real IMU readings; they are controlled proxies for
    speed, exposure, vibration, blur length, and low-light severity. This lets
    the paper demonstrate metadata-conditioned restoration now, while keeping
    the same schema ready for real vehicle/camera metadata later.
    """

    name = scenario.lower()
    rng = np.random.default_rng(seed)
    metadata: dict[str, Any] = {
        "gyro_x": 0.0,
        "gyro_y": 0.0,
        "accel_norm": 0.0,
        "speed_mps": float(rng.uniform(7.0, 16.0)),
        "exposure_ms": float(rng.uniform(6.0, 14.0)),
        "defocus_score": 0.0,
        "noise_score": 0.0,
        "blur_angle_deg": None,
        "blur_length_px": None,
        "low_light_score": 0.0,
        "jpeg_quality": None,
        "metadata_source": "synthetic_scenario_proxy",
    }

    if "mild" in name or "sigma1" in name:
        strength = 0.30
    elif "strong" in name or "sigma5" in name:
        strength = 0.90
    else:
        strength = 0.60

    blur_length = 5.0 + 22.0 * strength + float(rng.normal(0, 1.2))
    if "horizontal" in name:
        metadata.update(
            {
                "gyro_x": float(2.2 * strength + rng.normal(0, 0.05)),
                "gyro_y": float(0.15 + rng.normal(0, 0.02)),
                "blur_angle_deg": float(rng.normal(0, 2.5)),
                "blur_length_px": float(max(1.0, blur_length)),
            }
        )
    if "vertical" in name:
        metadata.update(
            {
                "gyro_x": float(0.15 + rng.normal(0, 0.02)),
                "gyro_y": float(2.2 * strength + rng.normal(0, 0.05)),
                "blur_angle_deg": float(90.0 + rng.normal(0, 2.5)),
                "blur_length_px": float(max(1.0, blur_length)),
            }
        )
    if "diagonal" in name:
        angle = 35.0 + float(rng.normal(0, 3.0))
        metadata.update(
            {
                "gyro_x": float(1.6 * strength * abs(math.cos(math.radians(angle)))),
                "gyro_y": float(1.6 * strength * abs(math.sin(math.radians(angle)))),
                "blur_angle_deg": angle,
                "blur_length_px": float(max(1.0, blur_length)),
            }
        )
    if "random" in name or "vibration" in name:
        metadata["accel_norm"] = float(3.8 * strength + rng.normal(0, 0.2))
        metadata["blur_length_px"] = float(max(float(metadata.get("blur_length_px") or 0.0), blur_length * 0.7))
    if "motion" in name and metadata["blur_length_px"] is None:
        metadata["accel_norm"] = float(2.4 * strength + rng.normal(0, 0.15))
        metadata["blur_angle_deg"] = float(rng.uniform(-45, 45))
        metadata["blur_length_px"] = float(max(1.0, blur_length))
    if "defocus" in name:
        metadata["defocus_score"] = float(strength)
    if "lowlight" in name or "low_light" in name:
        metadata["low_light_score"] = float(strength)
        metadata["exposure_ms"] = float(rng.uniform(18.0, 38.0))
        metadata["noise_score"] = float(max(metadata["noise_score"], 0.25 * strength))
    if "gaussian" in name or "noise" in name:
        metadata["noise_score"] = float(max(metadata["noise_score"], strength))
    if "jpeg" in name:
        metadata["jpeg_quality"] = 40.0 if "40" in name else 70.0

    return metadata
