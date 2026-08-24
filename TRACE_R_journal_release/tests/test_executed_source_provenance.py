# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Verify the frozen source snapshot and public TRACE-R router equivalence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn

from models.tracer import TRACERExpertFusion


ROOT = Path(__file__).resolve().parents[1]
LEDGER = (
    ROOT
    / "experiments"
    / "final_rmrp_v50_validation_ledger_20260824"
    / "provenance_ledger.json"
)
SNAPSHOT = ROOT / "provenance" / "executed_source"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_executed_router():
    path = SNAPSHOT / "models" / "rmrp_expert_fusion.py"
    spec = importlib.util.spec_from_file_location("executed_rmrp_expert_fusion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ConstantExpert(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, image: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return image + self.delta


def test_executed_source_hashes_match_frozen_ledger() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    for relative, expected in ledger["code_sha256"].items():
        path = SNAPSHOT / Path(relative)
        assert path.exists(), relative
        assert sha256(path) == expected


def test_public_router_matches_executed_route_equations() -> None:
    executed_module = load_executed_router()
    experts = (ConstantExpert(1.0), ConstantExpert(2.0), ConstantExpert(3.0))
    executed = executed_module.RMRPExpertFusion(*experts)
    public = TRACERExpertFusion(*experts)
    image = torch.rand(2, 3, 17, 19)
    for route in ("fallback", "motion", "defocus", "lowlight", "mixed"):
        torch.testing.assert_close(
            executed._restore_route(image, route),
            public._restore_route(image, route),
            rtol=0.0,
            atol=1e-6,
        )
