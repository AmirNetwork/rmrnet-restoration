"""Telemetry-routed restoration expert bank used by TRACE-R.

The router consumes only the public 82-value camera/IMU/vehicle packet. It does
not receive a dataset name, corruption label, renderer parameter, or test label.
The selected routes and blend coefficients are validation-locked.

Journal method: TRACE-R (Telemetry-Routed Adaptive Corruption-Expert
Restoration). The historical RMR-P names remain aliases at the bottom of this
module so previously executed experiment manifests stay reproducible.
"""

from __future__ import annotations

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rcadnet.practical_metadata import (
    CONTEXT_START,
    PRACTICAL_SENSOR_DIM,
    observable_code_from_packet,
)


@dataclass(frozen=True)
class TRACERPolicy:
    """Validation-selected physical thresholds and convex expert weights."""

    motion_threshold: float = 0.18
    defocus_threshold: float = 0.20
    lowlight_threshold: float = 0.385
    support_threshold: float = 0.50
    lowlight_dfpir_weight: float = 0.40
    mixed_dfpir_weight: float = 0.075
    gyro_full_scale: float = 4.0

    def __post_init__(self) -> None:
        probability_fields = (
            "motion_threshold",
            "defocus_threshold",
            "lowlight_threshold",
            "support_threshold",
            "lowlight_dfpir_weight",
            "mixed_dfpir_weight",
        )
        for field in probability_fields:
            value = float(getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must lie in [0, 1], received {value}")
        if self.gyro_full_scale <= 0.0:
            raise ValueError("gyro_full_scale must be positive")


class TRACERExpertFusion(nn.Module):
    """Route each capture to matched restoration experts using sensor evidence.

    Routes:
      * motion: DFPIR;
      * defocus: InstructIR;
      * low light: convex DFPIR/DeMoE blend;
      * motion plus low light: conservative DFPIR/DeMoE blend;
      * missing or weak metadata: DeMoE image router.

    Camera reliability supports defocus and low-light decisions, while IMU
    reliability supports motion. This permits partial metadata: an unavailable
    modality disables only the causes that depend on it.
    """

    ROUTE_NAMES = ("fallback", "motion", "defocus", "lowlight", "mixed")

    def __init__(
        self,
        demoe: nn.Module,
        dfpir: nn.Module,
        instructir: nn.Module,
        policy: TRACERPolicy | None = None,
    ) -> None:
        super().__init__()
        self.demoe = demoe
        self.dfpir = dfpir
        self.instructir = instructir
        self.policy = policy or TRACERPolicy()
        self.sensor_dim = PRACTICAL_SENSOR_DIM
        self.sensor_gyro_full_scale = self.policy.gyro_full_scale
        self.use_practical_sensor_encoder = True

    @staticmethod
    def _instructir_prompt(cause: str) -> str:
        corruption = {
            "defocus": "defocus blur",
            "motion": "camera-motion blur",
            "lowlight": "low illumination with sensor noise",
            "mixed": "combined camera-motion blur and low illumination with noise",
        }.get(cause, "unknown image degradation")
        return (
            f"Remove {corruption} from this road inspection image. Preserve thin "
            "cracks, pothole boundaries, lane markings, and natural pavement "
            "texture without hallucinating defects."
        )

    def route(self, sensor_packet: torch.Tensor | None, batch: int) -> dict[str, Any]:
        """Derive one physical cause per image without a scenario identifier."""

        if sensor_packet is None:
            return {
                "route_index": torch.zeros(batch, dtype=torch.long),
                "route_names": ["fallback"] * batch,
                "sensor_code": None,
                "camera_support": None,
                "imu_support": None,
            }
        if sensor_packet.ndim == 1:
            sensor_packet = sensor_packet.unsqueeze(0)
        if sensor_packet.shape != (batch, PRACTICAL_SENSOR_DIM):
            raise ValueError(
                "Expected sensor packet shape "
                f"({batch}, {PRACTICAL_SENSOR_DIM}), got {tuple(sensor_packet.shape)}"
            )

        code = observable_code_from_packet(
            sensor_packet,
            gyro_full_scale=self.policy.gyro_full_scale,
        )
        camera_support = sensor_packet[:, CONTEXT_START + 12].clamp(0.0, 1.0)
        imu_support = sensor_packet[:, CONTEXT_START + 13].clamp(0.0, 1.0)
        available = sensor_packet[:, CONTEXT_START + 15] > 0.0

        motion = code[:, :3].amax(dim=1)
        defocus = code[:, 3]
        lowlight = code[:, 5]
        motion_active = (
            available
            & (imu_support >= self.policy.support_threshold)
            & (motion >= self.policy.motion_threshold)
        )
        defocus_active = (
            available
            & (camera_support >= self.policy.support_threshold)
            & (defocus >= self.policy.defocus_threshold)
        )
        lowlight_active = (
            available
            & (camera_support >= self.policy.support_threshold)
            & (lowlight >= self.policy.lowlight_threshold)
        )

        # Priority is physical: focus error is independent, then the joint
        # motion/illumination state, followed by its single-cause components.
        route_index = torch.zeros(batch, dtype=torch.long, device=sensor_packet.device)
        route_index = torch.where(motion_active, torch.ones_like(route_index), route_index)
        route_index = torch.where(lowlight_active, torch.full_like(route_index, 3), route_index)
        route_index = torch.where(
            motion_active & lowlight_active,
            torch.full_like(route_index, 4),
            route_index,
        )
        route_index = torch.where(defocus_active, torch.full_like(route_index, 2), route_index)
        route_names = [self.ROUTE_NAMES[int(index)] for index in route_index.detach().cpu()]
        return {
            "route_index": route_index,
            "route_names": route_names,
            "sensor_code": code,
            "camera_support": camera_support,
            "imu_support": imu_support,
        }

    @staticmethod
    def _run_padded(
        expert: Any,
        image: torch.Tensor,
        multiple: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h or pad_w:
            mode = "reflect" if pad_h < height and pad_w < width else "replicate"
            padded = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
        else:
            padded = image
        output = expert(padded, *args, **kwargs)
        return output[..., :height, :width]

    def _restore_route(self, image: torch.Tensor, route: str) -> torch.Tensor:
        if route == "motion":
            return self._run_padded(self.dfpir, image, 8, "motion")
        if route == "defocus":
            return self._run_padded(
                self.instructir,
                image,
                16,
                self._instructir_prompt("defocus"),
            )
        if route == "lowlight":
            dfpir = self._run_padded(self.dfpir, image, 8, "lowlight")
            demoe = self._run_padded(self.demoe, image, 8, task="low_light")
            weight = self.policy.lowlight_dfpir_weight
            return weight * dfpir + (1.0 - weight) * demoe
        if route == "mixed":
            dfpir = self._run_padded(
                self.dfpir,
                image,
                8,
                "mixed_motion_lowlight",
            )
            demoe = self._run_padded(self.demoe, image, 8, task="low_light")
            weight = self.policy.mixed_dfpir_weight
            return weight * dfpir + (1.0 - weight) * demoe
        return self._run_padded(self.demoe, image, 8, task="auto")

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        sensor_packet: torch.Tensor | None = None,
        *,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        routing = self.route(sensor_packet, image.shape[0])
        route_names: list[str] = routing["route_names"]
        restored = torch.empty_like(image)
        for route in self.ROUTE_NAMES:
            selected = [index for index, name in enumerate(route_names) if name == route]
            if not selected:
                continue
            index = torch.tensor(selected, device=image.device, dtype=torch.long)
            restored[index] = self._restore_route(image.index_select(0, index), route)
        restored = restored.clamp(0.0, 1.0)
        if not return_dict:
            return restored
        return {
            "restored": restored,
            "input": image,
            "metadata_used": image.new_tensor(
                [name != "fallback" for name in route_names], dtype=torch.float32
            ),
            **routing,
        }


# Backward-compatible experiment identifiers. These aliases deliberately keep
# old validation commands readable without changing the TRACE-R implementation.
ExpertFusionPolicy = TRACERPolicy
RMRPExpertFusion = TRACERExpertFusion

__all__ = [
    "TRACERPolicy",
    "TRACERExpertFusion",
    "ExpertFusionPolicy",
    "RMRPExpertFusion",
]
