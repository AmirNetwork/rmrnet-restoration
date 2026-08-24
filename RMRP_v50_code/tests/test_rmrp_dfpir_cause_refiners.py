from __future__ import annotations

import torch
from torch import nn

from models.rmrp_prompted_dfpir import RMRPPromptedDFPIR


class _IdentityPromptBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def enable_continuous_conditioning(self, code_dim: int) -> None:
        self.condition_affine = nn.Linear(code_dim, 2)
        nn.init.zeros_(self.condition_affine.weight)
        nn.init.zeros_(self.condition_affine.bias)

    def forward(
        self,
        image: torch.Tensor,
        prompt: torch.Tensor,
        *,
        condition_code: torch.Tensor,
    ) -> torch.Tensor:
        del prompt, condition_code
        return image + 0.0 * self.anchor


def test_cause_refiner_bank_is_identity_at_initialization() -> None:
    model = RMRPPromptedDFPIR(
        _IdentityPromptBackbone(),
        torch.zeros(6, 512),
        use_refiner=False,
    )
    model.enable_cause_refiners(max_gain=0.08)
    image = torch.rand(2, 3, 24, 32)
    code = torch.rand(2, 8)
    prompt_weights = torch.zeros(2, 6)
    prompt_weights[0, 1] = 1.0
    prompt_weights[1, 5] = 1.0

    restored, _, correction, weights = model._apply_cause_refiners(
        image,
        image,
        code,
        prompt_weights,
    )

    torch.testing.assert_close(restored, image)
    torch.testing.assert_close(correction, torch.zeros_like(image))
    torch.testing.assert_close(weights, prompt_weights[:, 1:])


def test_cause_refiner_bank_has_five_physical_operators() -> None:
    model = RMRPPromptedDFPIR(
        _IdentityPromptBackbone(),
        torch.zeros(6, 512),
        use_refiner=False,
        use_cause_refiners=True,
    )

    assert model.cause_refiners is not None
    assert len(model.cause_refiners) == 5

