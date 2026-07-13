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
    """Balanced restoration loss with explicit defect-edge preservation.

    Manuscript Eq. (base loss):
        L_base = ||I_r-I_c||_1
               + lambda_e ||grad I_r - grad I_c||_1
               + lambda_f ||F(I_r)-F(I_c)||_1
               + lambda_d ||(1+M_c)(I_r-I_c)||_1
               + lambda_v ||V(I_r)-V(I_c)||_1

    ``frequency_l1`` implements the Fourier magnitude term F(.), the normalized
    target-gradient map is M_c, and ``visibility_map`` is V(.).
    """

    def __init__(
        self,
        edge_weight: float = 0.15,
        freq_weight: float = 0.05,
        defect_weight: float = 0.10,
        visibility_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight
        self.defect_weight = defect_weight
        self.visibility_weight = visibility_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        degraded: torch.Tensor | None = None,
        code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        recon = F.l1_loss(pred, target)
        pred_grad = gradient_map(pred)
        target_grad = gradient_map(target)
        edge = F.l1_loss(pred_grad, target_grad)

        defect_mask = target_grad.mean(dim=1, keepdim=True)
        defect_mask = defect_mask / (defect_mask.amax(dim=(2, 3), keepdim=True) + 1e-6)
        defect = torch.mean(torch.abs(pred - target) * (1.0 + defect_mask))

        freq = frequency_l1(pred, target)
        visibility = F.l1_loss(visibility_map(pred), visibility_map(target))
        total = (
            recon
            + self.edge_weight * edge
            + self.freq_weight * freq
            + self.defect_weight * defect
            + self.visibility_weight * visibility
        )
        return total
