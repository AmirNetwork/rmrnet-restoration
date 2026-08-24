"""Regression tests for the journal-facing TRACE-R API."""

from __future__ import annotations

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

import torch

from models.rmrp_expert_fusion import ExpertFusionPolicy, RMRPExpertFusion
from models.tracer import TRACERExpertFusion, TRACERPolicy
from rcadnet.practical_metadata import PRACTICAL_SENSOR_DIM


def test_historical_names_are_exact_aliases() -> None:
    assert RMRPExpertFusion is TRACERExpertFusion
    assert ExpertFusionPolicy is TRACERPolicy


def test_missing_packet_uses_image_only_fallback() -> None:
    model = object.__new__(TRACERExpertFusion)
    routing = TRACERExpertFusion.route(model, None, batch=2)
    assert routing["route_names"] == ["fallback", "fallback"]


def test_packet_contract_remains_82_values() -> None:
    assert PRACTICAL_SENSOR_DIM == 82
    packet = torch.zeros(3, PRACTICAL_SENSOR_DIM)
    assert packet.shape == (3, 82)
