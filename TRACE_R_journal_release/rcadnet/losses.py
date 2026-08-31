# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def gradient_map(image: torch.Tensor) -> torch.Tensor:
    dx = F.pad(image[:, :, :, 1:] - image[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[:, :, 1:, :] - image[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-6)


def frequency_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def visibility_map(image: torch.Tensor) -> torch.Tensor:
    """Detector-oriented proxy for thin defect visibility.

    The map is not a detector loss; it is a differentiable saliency proxy built
    from gradients and local contrast. It helps the restorer preserve crack
    edges, pothole rims, patches, and lane markings that detectors often rely on.
    """

    gray = image.mean(dim=1, keepdim=True)
    grad = gradient_map(gray)
    local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    contrast = torch.abs(gray - local_mean)
    visibility = grad + contrast
    return visibility / (visibility.amax(dim=(2, 3), keepdim=True) + 1e-6)


class RCADLoss(nn.Module):
    """Restoration-fidelity group used by TRACE-R.

    The reported controlled checkpoint uses the ``simple`` profile:

        L_fid = rho(I_r - I_c) + lambda_g |grad I_r - grad I_c|_1,
        rho(x) = sqrt(x^2 + epsilon^2).

    Charbonnier reconstruction is robust to a few annotation/capture outliers,
    while the gradient term protects narrow cracks and pothole rims. Detector
    feature agreement is added by ``train_matched_restorer.py`` rather than
    hidden inside this image-fidelity module. The ``legacy`` profile is retained
    only for loading earlier checkpoint families and can additionally evaluate

        L_rest = L1 + lambda_g L_grad + lambda_f L_fft
                 + lambda_d L_defect + lambda_v L_visibility.

    Audit configurations record the selected profile and every coefficient, so
    a run cannot silently change the objective associated with a checkpoint.
    """

    def __init__(
        self,
        edge_weight: float = 0.15,
        freq_weight: float = 0.0,
        defect_weight: float = 0.0,
        visibility_weight: float = 0.0,
        profile: str = "simple",
        charbonnier_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        if profile not in {"simple", "legacy"}:
            raise ValueError("profile must be 'simple' or 'legacy'")
        if charbonnier_epsilon <= 0.0:
            raise ValueError("charbonnier_epsilon must be positive")
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight
        self.defect_weight = defect_weight
        self.visibility_weight = visibility_weight
        self.profile = profile
        self.charbonnier_epsilon = float(charbonnier_epsilon)

    def _reconstruction(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.profile == "legacy":
            return F.l1_loss(pred, target)
        residual = pred - target
        return torch.sqrt(residual.square() + self.charbonnier_epsilon**2).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        degraded: torch.Tensor | None = None,
        code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        recon = self._reconstruction(pred, target)
        pred_grad = gradient_map(pred)
        target_grad = gradient_map(target)
        edge = F.l1_loss(pred_grad, target_grad)

        total = recon + self.edge_weight * edge
        if self.profile == "simple":
            return total

        defect_mask = target_grad.mean(dim=1, keepdim=True)
        defect_mask = defect_mask / (defect_mask.amax(dim=(2, 3), keepdim=True) + 1e-6)
        defect = torch.mean(torch.abs(pred - target) * (1.0 + defect_mask))

        freq = frequency_l1(pred, target)
        visibility = F.l1_loss(visibility_map(pred), visibility_map(target))
        return (
            total
            + self.freq_weight * freq
            + self.defect_weight * defect
            + self.visibility_weight * visibility
        )
