# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from baselines.dfpir_adapter import _load_dfpir_model_class, load_official_dfpir_state


ROOT = Path(__file__).resolve().parents[1]
DFPIR_SOURCE = ROOT / "third_party" / "DFPIR-main"


def test_publication_dfpir_loader_is_strict() -> None:
    """The final source-native load must remain strict after validation."""

    source = ROOT / "baselines" / "dfpir_adapter.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    load_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_state_dict"
    ]
    assert load_calls
    assert all(
        any(
            keyword.arg == "strict"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in load_calls
    )


@pytest.mark.skipif(
    not DFPIR_SOURCE.is_dir(),
    reason="official DFPIR source is not bundled; install third_party/DFPIR-main",
)
def test_source_native_filter_requires_full_current_model_coverage() -> None:
    model = _load_dfpir_model_class()(
        dim=8,
        num_blocks=[1, 1, 1, 1],
        num_refinement_blocks=1,
        heads=[1, 1, 1, 1],
        device="cpu",
    )
    state = dict(model.state_dict())
    state["encoder_shuffle_channel1.select_attn.attn1"] = torch.tensor([0.1])
    report = load_official_dfpir_state(model, state)
    assert report["coverage"] == 1.0
    assert report["ignored_legacy_keys"] == [
        "encoder_shuffle_channel1.select_attn.attn1"
    ]

    partial = dict(state)
    partial.pop(next(iter(model.state_dict())))
    try:
        load_official_dfpir_state(model, partial)
    except RuntimeError as error:
        assert "missing=" in str(error)
    else:
        raise AssertionError("A partial DFPIR checkpoint must be rejected")
