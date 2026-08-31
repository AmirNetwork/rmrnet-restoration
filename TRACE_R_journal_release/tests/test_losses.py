# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

import pytest
import torch

from rcadnet.losses import RCADLoss, frequency_l1, gradient_map, visibility_map


def test_simple_loss_is_exactly_charbonnier_plus_gradient() -> None:
    pred = torch.tensor(
        [[[[0.1, 0.4], [0.7, 0.2]]]], dtype=torch.float32, requires_grad=True
    )
    target = torch.tensor([[[[0.2, 0.3], [0.6, 0.1]]]], dtype=torch.float32)
    epsilon = 1e-3
    edge_weight = 0.15
    criterion = RCADLoss(
        profile="simple",
        edge_weight=edge_weight,
        freq_weight=9.0,
        defect_weight=9.0,
        visibility_weight=9.0,
        charbonnier_epsilon=epsilon,
    )

    observed = criterion(pred, target)
    expected = torch.sqrt((pred - target).square() + epsilon**2).mean()
    expected = expected + edge_weight * torch.nn.functional.l1_loss(
        gradient_map(pred), gradient_map(target)
    )

    torch.testing.assert_close(observed, expected)
    observed.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_legacy_loss_retains_explicit_auxiliary_terms() -> None:
    pred = torch.rand(2, 3, 8, 8)
    target = torch.rand(2, 3, 8, 8)
    weights = dict(edge_weight=0.2, freq_weight=0.3, defect_weight=0.4, visibility_weight=0.5)
    criterion = RCADLoss(profile="legacy", **weights)
    target_grad = gradient_map(target)
    defect_mask = target_grad.mean(dim=1, keepdim=True)
    defect_mask = defect_mask / (defect_mask.amax(dim=(2, 3), keepdim=True) + 1e-6)
    expected = torch.nn.functional.l1_loss(pred, target)
    expected = expected + weights["edge_weight"] * torch.nn.functional.l1_loss(
        gradient_map(pred), target_grad
    )
    expected = expected + weights["freq_weight"] * frequency_l1(pred, target)
    expected = expected + weights["defect_weight"] * torch.mean(
        torch.abs(pred - target) * (1.0 + defect_mask)
    )
    expected = expected + weights["visibility_weight"] * torch.nn.functional.l1_loss(
        visibility_map(pred), visibility_map(target)
    )
    torch.testing.assert_close(criterion(pred, target), expected)


def test_loss_profile_validation() -> None:
    with pytest.raises(ValueError, match="profile"):
        RCADLoss(profile="unknown")
    with pytest.raises(ValueError, match="positive"):
        RCADLoss(charbonnier_epsilon=0.0)
