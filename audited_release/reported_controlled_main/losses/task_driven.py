# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

"""Compatibility helpers for task-driven RMR-Net training.

The full training pipeline uses ``train_rcadnet.py`` plus
``rcadnet.task_losses.CompositeTaskLoss``. This module is kept as a lightweight
notebook/experiment shim so older imports still work, but it delegates to the
paper-facing implementations in ``rcadnet.task_losses``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcadnet.task_losses import (
    ActiveContourGeometryLoss,
    FrozenDetectorFeatureExtractor,
    TaskDrivenPerceptualLoss,
    hutchinson_jacobian_penalty,
)


FrozenYOLOFeatureExtractor = FrozenDetectorFeatureExtractor
TrainableDeepActiveContourLoss = ActiveContourGeometryLoss
SpatiallyVaryingTDACLoss = ActiveContourGeometryLoss


class HutchinsonJacobianPenalty(nn.Module):
    """Small module wrapper around the manuscript Jacobian penalty.

    Equation:
        L_J = E_v ||grad_{I_r} <Phi(I_r), v>||_2^2
    """

    def __init__(self, feature_extractor: FrozenDetectorFeatureExtractor, *, num_probes: int = 1) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_probes = int(num_probes)

    def forward(self, restored: torch.Tensor) -> torch.Tensor:
        return hutchinson_jacobian_penalty(
            self.feature_extractor,
            restored,
            num_probes=self.num_probes,
        )


CascadedJacobianPenalty = HutchinsonJacobianPenalty


def train_step_task_driven(
    rmr_net: torch.nn.Module,
    frozen_detector: FrozenDetectorFeatureExtractor,
    optimizer: torch.optim.Optimizer,
    degraded_img: torch.Tensor,
    clean_img: torch.Tensor,
    metadata: torch.Tensor | None,
    *,
    w_l1: float = 1.0,
    w_tdp: float = 0.001,
    w_jac: float = 0.00002,
    w_ac: float = 0.0,
    tdp_loss: Optional[TaskDrivenPerceptualLoss] = None,
    tdac_loss: Optional[ActiveContourGeometryLoss] = None,
    jacobian_loss: Optional[HutchinsonJacobianPenalty] = None,
) -> dict[str, float]:
    """Single compact training step for notebooks.

    Manuscript objective represented here:
        L = L1 + lambda_TDP L_TDP + lambda_J L_J + lambda_AC L_AC

    The final paper configuration uses ``w_ac=0``. Active contours are normally
    run after detection for measurement, not as a train-time loss.
    """

    rmr_net.train()
    optimizer.zero_grad(set_to_none=True)

    result = rmr_net(
        degraded_img,
        metadata,
        return_dict=True,
        return_aux=w_ac > 0.0,
    )
    if not isinstance(result, dict):
        raise TypeError("RMR-Net must return a dict when return_dict=True")
    restored = result["restored"]

    loss_l1 = F.l1_loss(restored, clean_img)

    if tdp_loss is None:
        tdp_loss = TaskDrivenPerceptualLoss(frozen_detector)
    loss_tdp = tdp_loss(restored, clean_img)

    if jacobian_loss is None:
        jacobian_loss = HutchinsonJacobianPenalty(frozen_detector)
    loss_jac = jacobian_loss(restored)

    loss_ac = restored.new_tensor(0.0)
    if w_ac > 0.0:
        if tdac_loss is None:
            tdac_loss = ActiveContourGeometryLoss()
        required = ("phi", "lambda1", "lambda2")
        if any(result.get(key) is None for key in required):
            raise RuntimeError("Active-contour loss requested but phi/lambda maps are missing.")
        loss_ac = tdac_loss(restored, result["phi"], result["lambda1"], result["lambda2"])

    total = (w_l1 * loss_l1) + (w_tdp * loss_tdp) + (w_jac * loss_jac) + (w_ac * loss_ac)
    total.backward()
    optimizer.step()

    return {
        "loss": float(total.detach().cpu()),
        "loss_l1": float(loss_l1.detach().cpu()),
        "loss_tdp": float(loss_tdp.detach().cpu()),
        "loss_jacobian": float(loss_jac.detach().cpu()),
        "loss_active_contour": float(loss_ac.detach().cpu()),
    }


__all__ = [
    "FrozenYOLOFeatureExtractor",
    "FrozenDetectorFeatureExtractor",
    "TaskDrivenPerceptualLoss",
    "CascadedJacobianPenalty",
    "HutchinsonJacobianPenalty",
    "SpatiallyVaryingTDACLoss",
    "TrainableDeepActiveContourLoss",
    "ActiveContourGeometryLoss",
    "train_step_task_driven",
]
