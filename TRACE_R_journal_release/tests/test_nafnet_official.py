# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Regression tests for the paper's faithful NAFNet baseline."""

from __future__ import annotations

import torch

from baselines.nafnet_road import (
    CompactNAFNetRoad,
    NAFNetRoad,
    build_nafnet_from_payload,
)


def test_official_width32_architecture_and_shape() -> None:
    model = NAFNetRoad()
    assert model.architecture == {
        "variant": "official_eccv2022_gopro_width32",
        "width": 32,
        "enc_blk_nums": [1, 1, 1, 28],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
    }
    assert sum(parameter.numel() for parameter in model.parameters()) == 17_111_907

    image = torch.rand(1, 3, 65, 73)
    with torch.no_grad():
        restored = model(image)
    assert restored.shape == image.shape
    assert torch.isfinite(restored).all()


def test_payload_loader_distinguishes_official_and_legacy_models() -> None:
    official = NAFNetRoad()
    loaded_official, _, is_official = build_nafnet_from_payload(
        {"params": official.state_dict()}
    )
    assert is_official
    assert isinstance(loaded_official, NAFNetRoad)

    compact = CompactNAFNetRoad()
    loaded_compact, _, is_official = build_nafnet_from_payload(
        {"model": compact.state_dict(), "arch": {"width": 32}}
    )
    assert not is_official
    assert isinstance(loaded_compact, CompactNAFNetRoad)
