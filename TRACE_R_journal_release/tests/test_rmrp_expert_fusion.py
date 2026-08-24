from __future__ import annotations

import torch
from torch import nn

from models.rmrp_expert_fusion import ExpertFusionPolicy, RMRPExpertFusion
from rcadnet.practical_metadata import CONTEXT_START, PRACTICAL_SENSOR_DIM


class FakeDeMoE(nn.Module):
    def forward(self, image: torch.Tensor, task: str = "auto") -> torch.Tensor:
        values = {"auto": 0.1, "low_light": 0.6}
        return torch.full_like(image, values[task])


class FakeDFPIR(nn.Module):
    def forward(self, image: torch.Tensor, scenario: str) -> torch.Tensor:
        return torch.full_like(image, 0.8 if "lowlight" in scenario else 0.4)


class FakeInstructIR(nn.Module):
    def forward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        assert "defocus blur" in prompt
        return torch.full_like(image, 0.3)


def model() -> RMRPExpertFusion:
    return RMRPExpertFusion(FakeDeMoE(), FakeDFPIR(), FakeInstructIR())


def packet(*, motion: bool = False, defocus: bool = False, lowlight: bool = False) -> torch.Tensor:
    value = torch.zeros(PRACTICAL_SENSOR_DIM)
    value[CONTEXT_START + 12 : CONTEXT_START + 16] = 1.0
    if motion:
        value[:33:3] = 1.0
    if defocus:
        value[CONTEXT_START + 5] = 0.7
    if lowlight:
        value[CONTEXT_START] = 0.8
        value[CONTEXT_START + 1] = 0.8
    return value


def test_routes_observable_single_and_compound_causes() -> None:
    packets = torch.stack(
        [
            packet(motion=True),
            packet(defocus=True),
            packet(lowlight=True),
            packet(motion=True, lowlight=True),
        ]
    )
    result = model()(torch.zeros(4, 3, 8, 8), packets, return_dict=True)
    assert result["route_names"] == ["motion", "defocus", "lowlight", "mixed"]
    assert torch.allclose(result["restored"][0], torch.tensor(0.4))
    assert torch.allclose(result["restored"][1], torch.tensor(0.3))
    assert torch.allclose(result["restored"][2], torch.tensor(0.68))
    assert torch.allclose(result["restored"][3], torch.tensor(0.615))


def test_missing_metadata_uses_image_router() -> None:
    result = model()(torch.zeros(2, 3, 8, 8), None, return_dict=True)
    assert result["route_names"] == ["fallback", "fallback"]
    assert torch.allclose(result["restored"], torch.tensor(0.1))
    assert result["metadata_used"].sum() == 0


def test_partial_metadata_disables_only_unsupported_causes() -> None:
    motion_packet = packet(motion=True)
    motion_packet[CONTEXT_START + 12] = 0.0
    lowlight_packet = packet(lowlight=True)
    lowlight_packet[CONTEXT_START + 12] = 0.0
    result = model()(
        torch.zeros(2, 3, 8, 8),
        torch.stack((motion_packet, lowlight_packet)),
        return_dict=True,
    )
    assert result["route_names"] == ["motion", "fallback"]


def test_policy_rejects_invalid_values() -> None:
    try:
        ExpertFusionPolicy(lowlight_dfpir_weight=1.1)
    except ValueError as exc:
        assert "lowlight_dfpir_weight" in str(exc)
    else:
        raise AssertionError("invalid blend coefficient was accepted")
