from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.tracer_sensor_adapter import TRACESensorAdapterDeMoE
from rcadnet.practical_metadata import CONTEXT_START, PRACTICAL_SENSOR_DIM


class _ToyExpert(nn.Module):
    """Minimal DeMoE expert interface for dependency-free adapter tests."""

    def __init__(self) -> None:
        super().__init__()
        self.used = 1

    def forward(
        self, feature: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = torch.zeros(5, dtype=torch.long, device=feature.device)
        return feature, counts, weights


class _ToyRouter(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(channels, 5)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.classifier(F.adaptive_avg_pool2d(feature, 1).flatten(1))


class _ToyDeMoE(nn.Module):
    """Interface-compatible hierarchy without third-party source or weights."""

    def __init__(self) -> None:
        super().__init__()
        channels = (32, 64, 128, 256)
        self.intro = nn.Conv2d(3, channels[0], 3, padding=1)
        self.encoders = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU()) for c in channels]
        )
        self.downs = nn.ModuleList(
            [nn.Conv2d(c, 2 * c, 2, stride=2) for c in channels]
        )
        self.middle_blks = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.GELU())
        self.ups = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.PixelShuffle(2))
                for c in (512, 256, 128, 64)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU())
                for c in (256, 128, 64, 32)
            ]
        )
        self.experts = nn.ModuleList([_ToyExpert() for _ in range(5)])
        self.mlp_branch = _ToyRouter(512)
        self.ending = nn.Conv2d(32, 3, 3, padding=1)
        self.padder_size = 16

    def check_image_size(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(image, (0, pad_w, 0, pad_h))


def model() -> TRACESensorAdapterDeMoE:
    return TRACESensorAdapterDeMoE(_ToyDeMoE(), feature_rank=8).eval()


def packet() -> torch.Tensor:
    value = torch.zeros(1, PRACTICAL_SENSOR_DIM)
    value[:, 0:33:3] = 0.6
    value[:, CONTEXT_START : CONTEXT_START + 4] = 1.0
    value[:, CONTEXT_START + 5] = 0.6
    value[:, CONTEXT_START + 12 : CONTEXT_START + 16] = 1.0
    return value


def test_identity_initialization_preserves_matched_backbone() -> None:
    trace = model()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        result = trace(image, packet(), return_dict=True)
    # Every low-rank expansion starts at zero, so the deployed result equals
    # the routed DeMoE result exactly before adaptation training.
    assert torch.allclose(result["restored"], result["neural_restored"], atol=1e-7)
    assert result["feature_adapter_gates"].shape == (1, 9)


def test_all_feature_adapters_receive_gradients() -> None:
    trace = model()
    image = torch.rand(1, 3, 32, 32)
    result = trace(image, packet(), return_dict=True)
    result["restored"].mean().backward()
    for adapter in trace.feature_adapters.values():
        assert adapter.expand.weight.grad is not None
        assert torch.isfinite(adapter.expand.weight.grad).all()


def test_partial_or_missing_metadata_remains_valid() -> None:
    trace = model()
    image = torch.rand(1, 3, 32, 32)
    partial = packet()
    partial[:, 33:] = 0.0
    with torch.no_grad():
        partial_output = trace(image, partial)
        missing_output = trace(image, None)
    assert partial_output.shape == image.shape
    assert missing_output.shape == image.shape
    assert torch.isfinite(partial_output).all()
    assert torch.isfinite(missing_output).all()


def test_cause_feature_expansion_is_function_preserving() -> None:
    baseline = model()
    expanded = model()
    expanded.load_state_dict(baseline.state_dict(), strict=True)
    expanded.enable_cause_feature_adapters(max_gain=0.18)
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        expected = baseline(image, packet())
        actual = expanded(image, packet(), return_dict=True)
    assert torch.equal(expected, actual["restored"])
    assert actual["cause_feature_adapter_weights"].shape == (1, 4)
    assert actual["cause_feature_adapter_gates"].shape == (1, 9, 4)


def test_active_cause_feature_branches_receive_gradients() -> None:
    trace = model()
    trace.enable_cause_feature_adapters(max_gain=0.18)
    image = torch.rand(1, 3, 32, 32)
    result = trace(image, packet(), return_dict=True)
    result["restored"].mean().backward()
    for stage in trace.cause_feature_adapters.values():
        active = [
            branch.expand.weight.grad is not None
            and bool(torch.isfinite(branch.expand.weight.grad).all())
            and float(branch.expand.weight.grad.abs().sum()) > 0.0
            for branch in stage
        ]
        assert any(active)
