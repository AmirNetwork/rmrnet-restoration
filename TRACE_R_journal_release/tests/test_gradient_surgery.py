# TRACE-R release integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

import torch

from train_rcadnet import apply_primary_protected_gradients


def test_primary_protected_gradients_remove_opposing_component() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    primary = parameter.sum()
    task = -2.0 * parameter.sum()

    stats = apply_primary_protected_gradients(primary, task, [parameter])

    assert stats["gradient_conflict"] == 1.0
    assert torch.allclose(parameter.grad, torch.ones_like(parameter), atol=1e-6)


def test_primary_protected_gradients_keep_orthogonal_task() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    primary = parameter[0]
    task = parameter[1]

    stats = apply_primary_protected_gradients(primary, task, [parameter])

    assert stats["gradient_conflict"] == 0.0
    assert torch.allclose(parameter.grad, torch.tensor([1.0, 1.0]), atol=1e-6)
