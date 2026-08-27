"""Sensor-conditioned low-rank feature adaptation for TRACE-R.

TRACE-R starts from a matched DeMoE restorer and inserts lightweight residual
adapters throughout its encoder, bottleneck, and decoder. Each adapter combines
image-estimated corruption, observable sensor state, image--sensor disagreement,
and modality reliability. The result is one restored image; neither detector
ensembling nor a second restoration output is used at inference.

For feature ``x_l`` and joint state ``s``, one adapter implements

    x'_l = x_l + a_l(s, q) P_l sigma(D_l(Q_l N(x_l); s)),

where ``Q_l`` and ``P_l`` are low-rank 1x1 projections, ``D_l`` is a depthwise
spatial operator, and ``a_l`` is a bounded image--sensor compatibility gate.
The final projection is zero initialized, so wrapping a matched checkpoint is
exactly function preserving before training.
"""

from __future__ import annotations

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from models.rmrp_metadata_demoe import RMRPMetadataDeMoE


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SensorConditionedLowRankAdapter(nn.Module):
    """A bounded spatial adapter driven by image and sensor evidence."""

    def __init__(
        self,
        channels: int,
        rank: int,
        *,
        conditioning_dim: int = 32,
        max_gain: float = 0.25,
    ) -> None:
        super().__init__()
        if channels < 1 or rank < 1:
            raise ValueError("adapter channels and rank must be positive")
        if not 0.0 < max_gain <= 0.5:
            raise ValueError("adapter max_gain must lie in (0, 0.5]")
        self.channels = int(channels)
        self.rank = int(rank)
        self.conditioning_dim = int(conditioning_dim)
        self.max_gain = float(max_gain)

        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.reduce = nn.Conv2d(channels, rank, 1)
        self.spatial = nn.Sequential(
            nn.Conv2d(rank, rank, 3, padding=1, groups=rank),
            nn.GELU(),
            nn.Conv2d(rank, rank, 3, padding=2, dilation=2, groups=rank),
            nn.GELU(),
        )
        self.condition_affine = nn.Linear(conditioning_dim, 2 * rank)
        self.compatibility = nn.Sequential(
            nn.Linear(conditioning_dim, rank),
            nn.GELU(),
            nn.Linear(rank, 1),
        )
        self.spatial_gate = nn.Conv2d(rank, 1, 3, padding=1)
        self.expand = nn.Conv2d(rank, channels, 1)

        # Exact function-preserving initialization. The expand layer receives a
        # gradient immediately; the state pathway begins learning on the next
        # optimizer step without perturbing the matched DeMoE initialization.
        nn.init.zeros_(self.condition_affine.weight)
        nn.init.zeros_(self.condition_affine.bias)
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.constant_(self.compatibility[-1].bias, -1.0)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(
        self,
        feature: torch.Tensor,
        conditioning: torch.Tensor,
        cause_reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if conditioning.ndim != 2 or conditioning.shape[1] != self.conditioning_dim:
            raise ValueError(
                f"Expected conditioning shape (B, {self.conditioning_dim}), "
                f"received {tuple(conditioning.shape)}"
            )
        hidden = self.reduce(self.norm(feature))
        scale, shift = self.condition_affine(conditioning).chunk(2, dim=1)
        hidden = hidden * (1.0 + 0.25 * torch.tanh(scale)[:, :, None, None])
        hidden = hidden + 0.25 * torch.tanh(shift)[:, :, None, None]
        hidden = self.spatial(hidden)

        # The learned compatibility gate considers both image and telemetry.
        # Available measurements raise adaptation capacity but are never a hard
        # switch; partial packets remain valid and missing packets fall back to
        # the image-conditioned quarter-gain path.
        available = cause_reliability.amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        global_gate = torch.sigmoid(self.compatibility(conditioning))
        global_gate = global_gate * (0.25 + 0.75 * available)
        local_gate = torch.sigmoid(self.spatial_gate(hidden))
        correction = self.max_gain * global_gate[:, :, None, None]
        correction = correction * local_gate * self.expand(hidden)
        return feature + correction, global_gate[:, 0]


class TRACESensorAdapterDeMoE(RMRPMetadataDeMoE):
    """Single-output TRACE-R restorer with hierarchy-wide sensor adapters."""

    feature_channels = {
        "encoder_0": 32,
        "encoder_1": 64,
        "encoder_2": 128,
        "encoder_3": 256,
        "bottleneck": 512,
        "decoder_0": 256,
        "decoder_1": 128,
        "decoder_2": 64,
        "decoder_3": 32,
    }

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_rank: int = 16,
        feature_max_gain: float = 0.25,
        use_cause_feature_adapters: bool = False,
        cause_feature_max_gain: float = 0.18,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("top_k", 1)
        kwargs.setdefault("use_refiner", False)
        kwargs.setdefault("backbone_route_mode", "sensor_task")
        kwargs.setdefault("use_cause_refiners", False)
        kwargs.setdefault("use_semantic_adapters", False)
        super().__init__(backbone, **kwargs)
        if feature_rank < 4:
            raise ValueError("feature_rank must be at least four")
        self.feature_rank = int(feature_rank)
        self.feature_max_gain = float(feature_max_gain)
        self.use_cause_feature_adapters = bool(use_cause_feature_adapters)
        self.cause_feature_max_gain = float(cause_feature_max_gain)
        self.feature_adapters = nn.ModuleDict()
        for stage, channels in self.feature_channels.items():
            stage_rank = min(channels, max(4, feature_rank * channels // 128))
            self.feature_adapters[stage] = SensorConditionedLowRankAdapter(
                channels,
                stage_rank,
                max_gain=feature_max_gain,
            )
        self._adapter_gates: dict[str, torch.Tensor] = {}
        self.cause_feature_adapters = nn.ModuleDict()
        if self.use_cause_feature_adapters:
            self._build_cause_feature_adapters()
        self._cause_adapter_weights: torch.Tensor | None = None
        self._cause_adapter_gates: dict[str, torch.Tensor] = {}

    def _build_cause_feature_adapters(self) -> None:
        """Create four identity-initialized physical-cause branches per stage."""

        if self.cause_feature_adapters:
            return
        for stage, channels in self.feature_channels.items():
            stage_rank = min(channels, max(4, self.feature_rank * channels // 128))
            self.cause_feature_adapters[stage] = nn.ModuleList(
                [
                    SensorConditionedLowRankAdapter(
                        channels,
                        stage_rank,
                        max_gain=self.cause_feature_max_gain,
                    )
                    for _ in range(4)
                ]
            )

    def enable_cause_feature_adapters(self, max_gain: float | None = None) -> None:
        """Enable motion/defocus/low-light/compound hierarchy adapters."""

        if max_gain is not None:
            if not 0.0 < max_gain <= 0.5:
                raise ValueError("cause feature max_gain must lie in (0, 0.5]")
            self.cause_feature_max_gain = float(max_gain)
        self.use_cause_feature_adapters = True
        self._build_cause_feature_adapters()
        reference = next(self.parameters())
        self.cause_feature_adapters.to(
            device=reference.device,
            dtype=reference.dtype,
        )

    @staticmethod
    def _physical_cause_weights(
        sensor_direct: torch.Tensor,
        cause_reliability: torch.Tensor,
    ) -> torch.Tensor:
        """Return continuous motion/defocus/low-light/compound support.

        The weights are computed only from observable measurements and their
        availability. They are not benchmark scenario labels. Missing
        modalities contribute zero, while a partial packet can still activate
        every physically supported branch.
        """

        motion = sensor_direct[:, :3].amax(dim=1)
        motion = motion * cause_reliability[:, :3].amax(dim=1)
        defocus = sensor_direct[:, 3] * cause_reliability[:, 3]
        lowlight = sensor_direct[:, 5] * cause_reliability[:, 5]
        compound = torch.minimum(motion, lowlight)
        weights = torch.stack(
            [
                (motion - compound).clamp_min(0.0),
                defocus,
                (lowlight - compound).clamp_min(0.0),
                compound,
            ],
            dim=1,
        )
        # Preserve physical magnitude below one and bound simultaneous causes.
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)

    def _sensor_state(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        state = super()._sensor_state(image, metadata)
        self._cause_adapter_weights = self._physical_cause_weights(
            state[5], state[6]
        )
        return state

    def _adapt_backbone_feature(
        self,
        stage: str,
        feature: torch.Tensor,
        conditioning: torch.Tensor,
        cause_reliability: torch.Tensor,
    ) -> torch.Tensor:
        adapted, gate = self.feature_adapters[stage](
            feature, conditioning, cause_reliability
        )
        self._adapter_gates[stage] = gate
        if not self.use_cause_feature_adapters:
            return adapted
        if self._cause_adapter_weights is None:
            raise RuntimeError("cause-adapter weights were not initialized")

        correction = torch.zeros_like(adapted)
        branch_gates = []
        for index, branch in enumerate(self.cause_feature_adapters[stage]):
            branch_output, branch_gate = branch(
                adapted, conditioning, cause_reliability
            )
            weight = self._cause_adapter_weights[:, index, None, None, None]
            correction = correction + weight * (branch_output - adapted)
            branch_gates.append(branch_gate)
        self._cause_adapter_gates[stage] = torch.stack(branch_gates, dim=1)
        return adapted + correction

    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        return_dict: bool = False,
        return_aux: bool = False,
        **kwargs: object,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        self._adapter_gates = {}
        self._cause_adapter_gates = {}
        self._cause_adapter_weights = None
        result = super().forward(
            image,
            metadata,
            return_dict=return_dict,
            return_aux=return_aux,
            **kwargs,
        )
        if not return_dict:
            return result
        assert isinstance(result, dict)
        if self._adapter_gates:
            result["feature_adapter_gates"] = torch.stack(
                [self._adapter_gates[name] for name in self.feature_channels], dim=1
            )
        else:
            result["feature_adapter_gates"] = image.new_zeros((image.shape[0], 0))
        if self._cause_adapter_gates:
            result["cause_feature_adapter_gates"] = torch.stack(
                [
                    self._cause_adapter_gates[name]
                    for name in self.feature_channels
                ],
                dim=1,
            )
        else:
            result["cause_feature_adapter_gates"] = image.new_zeros(
                (image.shape[0], 0, 4)
            )
        if self._cause_adapter_weights is not None:
            result["cause_feature_adapter_weights"] = self._cause_adapter_weights
        else:
            result["cause_feature_adapter_weights"] = image.new_zeros(
                (image.shape[0], 4)
            )
        return result


__all__ = ["SensorConditionedLowRankAdapter", "TRACESensorAdapterDeMoE"]
