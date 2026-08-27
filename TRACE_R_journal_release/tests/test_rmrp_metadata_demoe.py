from __future__ import annotations

import torch

from baselines.demoe_adapter import DeMoEAdapter
from models.rmrp_metadata_demoe import RMRPMetadataDeMoE
from rcadnet.practical_metadata import CONTEXT_START, PRACTICAL_SENSOR_DIM


def build_model(
    *,
    refiner: bool = True,
    compound_gate: bool = False,
    metadata_acceptance: float = 1.0,
    semantic_adapters: bool = False,
    backbone_route_mode: str = "metadata",
) -> RMRPMetadataDeMoE:
    adapter = DeMoEAdapter(None, device="cpu", smoke=True)
    return RMRPMetadataDeMoE(
        adapter.model,
        top_k=2,
        use_refiner=refiner,
        use_compound_blend_gate=compound_gate,
        compound_blend_init=0.65,
        compound_metadata_acceptance=metadata_acceptance,
        use_semantic_adapters=semantic_adapters,
        backbone_route_mode=backbone_route_mode,
    )


def full_packet(batch: int = 1) -> torch.Tensor:
    packet = torch.zeros(batch, PRACTICAL_SENSOR_DIM)
    packet[:, CONTEXT_START + 12 : CONTEXT_START + 16] = 1.0
    return packet


def cause_packet(cause: str) -> torch.Tensor:
    """Construct an observable packet that supports one declared task."""

    packet = full_packet()
    if cause in {"motion", "compound"}:
        packet[:, 0:33:3] = 0.7
        packet[:, CONTEXT_START] = 1.0
    if cause == "defocus":
        packet[:, CONTEXT_START + 5] = 0.6
        packet[:, CONTEXT_START + 6] = 1.0
    if cause in {"lowlight", "compound"}:
        packet[:, CONTEXT_START] = 0.75
        packet[:, CONTEXT_START + 1] = 0.75
        packet[:, CONTEXT_START + 2] = 0.75
    return packet


def test_missing_metadata_uses_image_fallback() -> None:
    model = build_model(refiner=False).eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        output = model(image, None, return_dict=True)
    assert output["restored"].shape == image.shape
    assert torch.isfinite(output["restored"]).all()
    assert torch.count_nonzero(output["sensor_cause_reliability"]) == 0
    assert torch.allclose(
        output["prompt_routing_code"], output["code"], atol=1e-6
    )
    assert all(block.used == 1 for block in model.backbone.experts)


def test_compound_teacher_activates_motion_and_lowlight_experts() -> None:
    model = build_model(refiner=False).eval()
    image = torch.rand(1, 3, 32, 32)
    teacher = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    with torch.no_grad():
        output = model(
            image,
            full_packet(),
            return_dict=True,
            prompt_teacher_weights=teacher,
        )
    weights = output["expert_weights"][0]
    assert torch.allclose(weights[[3, 4]], torch.tensor([0.5, 0.5]))
    assert torch.count_nonzero(weights) == 2
    assert all(block.used == 2 for block in model.backbone.experts)


def test_bounded_refiner_is_identity_initialized() -> None:
    model = build_model(refiner=True).eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        output = model(image, full_packet(), return_dict=True)
    assert torch.allclose(output["restored"], output["neural_restored"], atol=1e-7)
    assert torch.count_nonzero(output["post_prior_correction"]) == 0


def test_reliable_defocus_is_not_overridden_by_motion_biased_image_router() -> None:
    model = build_model(refiner=False).eval()
    route_code = torch.tensor(
        [[0.01, 0.01, 0.01, 0.55, 0.05, 0.25, 0.0, 0.55]]
    )
    cause_reliability = torch.ones_like(route_code)
    auto_weights = torch.tensor([[0.02, 0.02, 0.02, 0.90, 0.04]])
    zeros = torch.zeros(1, 8)
    with torch.no_grad():
        weights = model._metadata_expert_weights(
            route_code,
            zeros,
            zeros,
            cause_reliability,
            zeros,
            auto_weights,
        )
    assert int(weights.argmax(dim=1).item()) == 0


def test_sensor_and_route_paths_receive_gradients() -> None:
    model = build_model(refiner=False).train()
    model.backbone.mlp_branch.eval()
    image = torch.rand(1, 3, 32, 32)
    output = model(image, full_packet(), return_dict=True)
    loss = output["restored"].mean() + output["code"].mean()
    loss.backward()
    assert model.sensor_encoder.fuse[-1].weight.grad is not None
    assert model.route_residual[-1].weight.grad is not None
    assert torch.isfinite(model.route_residual[-1].weight.grad).all()


def test_compound_gate_is_initialized_to_audited_global_blend() -> None:
    model = build_model(refiner=False, compound_gate=True).eval()
    image = torch.rand(1, 3, 32, 32)
    teacher = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    with torch.no_grad():
        output = model(
            image,
            full_packet(),
            return_dict=True,
            prompt_teacher_weights=teacher,
        )
    assert bool(output["compound_mask"].item())
    assert torch.allclose(output["compound_blend"], torch.tensor([0.65]), atol=1e-6)


def test_compound_gate_only_receives_gradients_on_compound_samples() -> None:
    # Evaluation mode avoids the DeMoE router's batch-normalization constraint
    # for this one-sample fixture; autograd remains enabled for the gate.
    model = build_model(refiner=False, compound_gate=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.compound_blend_head is not None
    for parameter in model.compound_blend_head.parameters():
        parameter.requires_grad_(True)
    image = torch.rand(1, 3, 32, 32)
    teacher = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    output = model(
        image,
        full_packet(),
        return_dict=True,
        prompt_teacher_weights=teacher,
    )
    output["restored"].mean().backward()
    assert model.compound_blend_head.weight.grad is not None
    assert torch.isfinite(model.compound_blend_head.weight.grad).all()


def test_compound_compatibility_blends_metadata_and_fallback_candidates() -> None:
    model = build_model(
        refiner=False, compound_gate=True, metadata_acceptance=0.5
    ).eval()
    image = torch.rand(1, 3, 32, 32)
    teacher = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    with torch.no_grad():
        output = model(
            image,
            full_packet(),
            return_dict=True,
            prompt_teacher_weights=teacher,
        )
    expected = output["compound_fallback_restored"] + 0.5 * (
        output["metadata_candidate_restored"]
        - output["compound_fallback_restored"]
    )
    assert torch.allclose(output["restored"], expected, atol=1e-6)
    assert torch.allclose(
        output["compound_metadata_acceptance"], torch.tensor([0.5])
    )


def test_semantic_adapters_preserve_shared_restorer_at_initialization() -> None:
    model = build_model(
        refiner=False,
        semantic_adapters=True,
        backbone_route_mode="image",
    ).eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        output = model(image, full_packet(), return_dict=True)
    assert torch.allclose(output["restored"], output["neural_restored"], atol=1e-7)
    assert torch.count_nonzero(output["semantic_adapter_correction"]) == 0


def test_semantic_adapters_receive_gradients_but_missing_metadata_is_identity() -> None:
    model = build_model(
        refiner=False,
        semantic_adapters=True,
        backbone_route_mode="image",
    ).eval()
    image = torch.rand(1, 3, 32, 32)
    available = model(image, full_packet(), return_dict=True)
    available["restored"].mean().backward()
    assert model.semantic_adapters is not None
    assert model.semantic_adapters[0].output.weight.grad is not None
    with torch.no_grad():
        missing = model(image, None, return_dict=True)
    assert torch.allclose(missing["restored"], missing["neural_restored"], atol=1e-7)
    assert torch.count_nonzero(missing["semantic_adapter_weights"]) == 0


def test_sensor_task_route_exactly_reproduces_declared_demoe_tasks() -> None:
    adapter = DeMoEAdapter(None, device="cpu", smoke=True)
    model = RMRPMetadataDeMoE(
        adapter.model,
        top_k=2,
        use_refiner=False,
        backbone_route_mode="sensor_task",
    ).eval()
    image = torch.rand(1, 3, 32, 32)
    cases = {
        "motion": "synth_global_motion",
        "defocus": "defocus",
        "lowlight": "low_light",
        "compound": "low_light",
    }
    with torch.no_grad():
        for cause, task in cases.items():
            actual = model(image, cause_packet(cause), return_dict=True)
            expected = adapter.model(image, task=task)["output"].clamp(0.0, 1.0)
            assert torch.allclose(actual["restored"], expected, atol=1e-6)
            assert all(block.used == 1 for block in model.backbone.experts)


def test_sensor_task_route_can_reserve_internal_expert_for_compound_cause() -> None:
    adapter = DeMoEAdapter(None, device="cpu", smoke=True)
    model = RMRPMetadataDeMoE(
        adapter.model,
        top_k=1,
        use_refiner=False,
        backbone_route_mode="sensor_task",
        sensor_task_mixed_expert=1,
    ).eval()
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        compound = model(image, cause_packet("compound"), return_dict=True)
        lowlight = model(image, cause_packet("lowlight"), return_dict=True)
    assert int(compound["expert_weights"].argmax(dim=1).item()) == 1
    assert int(lowlight["expert_weights"].argmax(dim=1).item()) == 4
    assert int(compound["sensor_task_mixed_expert"].item()) == 1
