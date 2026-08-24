from __future__ import annotations

import unittest

import torch
from torch import nn

from models.rmrp_prompted_dfpir import RMRPPromptedDFPIR
from rcadnet.practical_metadata import CONTEXT_START, PRACTICAL_SENSOR_DIM


class PromptAwareIdentityBackbone(nn.Module):
    """Small differentiable stand-in for the expensive official DFPIR model."""

    def __init__(self) -> None:
        super().__init__()
        self.condition_affine: nn.Linear | None = None
        self.forward_batch_sizes: list[int] = []

    def enable_continuous_conditioning(self, code_dim: int = 8):
        self.condition_affine = nn.Linear(code_dim, 3)
        nn.init.zeros_(self.condition_affine.weight)
        nn.init.zeros_(self.condition_affine.bias)
        return self

    def forward(
        self,
        image: torch.Tensor,
        prompt: torch.Tensor,
        condition_code: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.forward_batch_sizes.append(int(image.shape[0]))
        # Preserve the image while retaining a gradient path through the prompt.
        output = image + 0.0 * prompt.mean(dim=1)[:, None, None, None]
        if self.condition_affine is not None and condition_code is not None:
            output = output + self.condition_affine(condition_code)[:, :, None, None]
        return output


def full_sensor_packet(batch: int) -> torch.Tensor:
    packet = torch.zeros(batch, PRACTICAL_SENSOR_DIM)
    packet[:, :33] = 0.10
    packet[:, CONTEXT_START] = 0.5
    packet[:, CONTEXT_START + 1] = 0.3
    packet[:, CONTEXT_START + 12 : CONTEXT_START + 16] = 1.0
    return packet


class RMRPPromptedDFPIRTest(unittest.TestCase):
    def build_model(
        self,
        *,
        prompt_router: str = "hard",
        prompt_basis_count: int = 3,
    ) -> RMRPPromptedDFPIR:
        prompts = torch.randn(prompt_basis_count, 512)
        return RMRPPromptedDFPIR(
            PromptAwareIdentityBackbone(),
            prompts,
            prompt_router=prompt_router,
        )

    def test_identity_initialization_and_sensor_availability(self) -> None:
        torch.manual_seed(7)
        model = self.build_model().eval()
        image = torch.rand(2, 3, 32, 40)

        full = model(image, full_sensor_packet(2), return_dict=True)
        missing = model(image, None, return_dict=True)
        partial_packet = full_sensor_packet(2)
        partial_packet[:, CONTEXT_START + 13] = 0.0
        partial = model(image, partial_packet, return_dict=True)

        for result in (full, missing, partial):
            self.assertTrue(torch.isfinite(result["restored"]).all())
            self.assertTrue(torch.allclose(result["restored"], image, atol=1e-7))
            self.assertEqual(tuple(result["sensor_cause_reliability"].shape), (2, 8))

        # A missing packet must reset support instead of reusing the preceding
        # sample's cached PracticalSensorEncoder state.
        self.assertEqual(float(missing["sensor_cause_reliability"].abs().max()), 0.0)
        self.assertLessEqual(
            float(partial["sensor_cause_reliability"].mean()),
            float(full["sensor_cause_reliability"].mean()),
        )

    def test_task_losses_can_reach_new_metadata_modules(self) -> None:
        torch.manual_seed(11)
        model = self.build_model().train()
        image = torch.rand(2, 3, 24, 24)
        result = model(image, full_sensor_packet(2), return_dict=True)
        objective = result["code"].mean() + result["prompt_weights"].square().mean()
        objective.backward()

        self.assertIsNotNone(model.image_state.head[-1].weight.grad)
        self.assertIsNotNone(model.sensor_encoder.fuse[-1].weight.grad)
        self.assertIsNotNone(model.posterior_refine[-1].weight.grad)

    def test_restoration_gradient_reaches_continuous_state_adapter(self) -> None:
        model = self.build_model(prompt_basis_count=6).train()
        image = torch.rand(2, 3, 24, 24)
        result = model(image, full_sensor_packet(2), return_dict=True)
        (result["restored"] - 0.25).square().mean().backward()

        affine = model.backbone.condition_affine
        self.assertIsNotNone(affine)
        assert affine is not None
        self.assertIsNotNone(affine.weight.grad)
        self.assertGreater(float(affine.weight.grad.abs().sum()), 0.0)

    def test_calibrated_prompt_routing_rejects_secondary_camera_cues(self) -> None:
        codes = torch.tensor(
            [
                # Defocus with a weak illumination cue remains deblur.
                [0.01, 0.02, 0.02, 0.22, 0.06, 0.26, 0.05, 0.26],
                # Strong low light is routed to the low-light prompt.
                [0.02, 0.02, 0.02, 0.01, 0.48, 0.63, 0.05, 0.63],
                # Compound motion and low light also uses low light.
                [0.37, 0.10, 0.07, 0.04, 0.37, 0.51, 0.05, 0.51],
                # Dominant noise uses denoise.
                [0.02, 0.02, 0.02, 0.01, 0.80, 0.10, 0.05, 0.80],
            ]
        )
        routes = self.build_model()._prompt_weights(codes).argmax(dim=1)
        self.assertEqual(routes.tolist(), [0, 2, 2, 1])

    def test_four_prompt_basis_selects_compound_only_for_joint_evidence(self) -> None:
        codes = torch.tensor(
            [
                [0.02, 0.02, 0.02, 0.01, 0.30, 0.63, 0.05, 0.63],
                [0.37, 0.10, 0.07, 0.04, 0.37, 0.51, 0.05, 0.51],
                [0.45, 0.05, 0.02, 0.02, 0.08, 0.15, 0.02, 0.45],
            ]
        )
        routes = self.build_model(prompt_basis_count=4)._prompt_weights(codes).argmax(dim=1)
        self.assertEqual(routes.tolist(), [2, 3, 0])

    def test_five_prompt_basis_separates_motion_defocus_and_compound(self) -> None:
        codes = torch.tensor(
            [
                [0.02, 0.02, 0.02, 0.70, 0.05, 0.15, 0.02, 0.70],
                [0.45, 0.05, 0.02, 0.02, 0.08, 0.15, 0.02, 0.45],
                [0.37, 0.10, 0.07, 0.04, 0.37, 0.51, 0.05, 0.51],
            ]
        )
        routes = self.build_model(prompt_basis_count=5)._prompt_weights(codes).argmax(dim=1)
        self.assertEqual(routes.tolist(), [1, 0, 4])

    def test_six_prompt_basis_routes_clean_and_joint_evidence(self) -> None:
        codes = torch.tensor(
            [
                [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0, 0.01],
                [0.45, 0.05, 0.02, 0.02, 0.08, 0.15, 0.02, 0.45],
                [0.37, 0.10, 0.07, 0.04, 0.37, 0.51, 0.05, 0.51],
            ]
        )
        routes = self.build_model(prompt_basis_count=6)._prompt_weights(codes).argmax(dim=1)
        self.assertEqual(routes.tolist(), [0, 1, 5])

    def test_training_teacher_can_override_predicted_prompt(self) -> None:
        model = self.build_model(prompt_basis_count=6).eval()
        image = torch.rand(1, 3, 16, 16)
        teacher = torch.nn.functional.one_hot(torch.tensor([4]), 6).float()
        result = model(
            image,
            full_sensor_packet(1),
            return_dict=True,
            prompt_teacher_weights=teacher,
            prompt_teacher_mask=torch.ones(1),
        )
        self.assertTrue(torch.equal(result["prompt_weights"], teacher))

    def test_sparse_router_retains_both_compound_causes(self) -> None:
        model = self.build_model(prompt_router="sparse_blend")
        code = torch.tensor(
            [[0.37, 0.10, 0.07, 0.04, 0.37, 0.51, 0.05, 0.51]]
        )
        weights = model._prompt_weights(code)
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(1)))
        self.assertGreater(float(weights[0, 0]), 0.0)
        self.assertGreater(float(weights[0, 2]), 0.0)

    def test_refiner_free_deployment_is_exact_backbone_output(self) -> None:
        prompts = torch.randn(3, 512)
        model = RMRPPromptedDFPIR(
            PromptAwareIdentityBackbone(),
            prompts,
            sensor_route_mode="physical_fused",
            use_refiner=False,
        ).eval()
        image = torch.rand(1, 3, 20, 24)
        result = model(image, full_sensor_packet(1), return_dict=True)
        self.assertIsNone(model.refiner)
        self.assertTrue(torch.allclose(result["restored"], result["neural_restored"]))
        self.assertEqual(float(result["post_prior_correction"].abs().max()), 0.0)

    def test_compound_second_pass_only_processes_selected_samples(self) -> None:
        backbone = PromptAwareIdentityBackbone()
        model = RMRPPromptedDFPIR(
            backbone,
            torch.randn(6, 512),
            use_refiner=False,
            compound_motion_blend=0.25,
        ).train()
        image = torch.rand(2, 3, 20, 24, requires_grad=True)
        teacher = torch.nn.functional.one_hot(torch.tensor([5, 1]), 6).float()
        result = model(
            image,
            full_sensor_packet(2),
            return_dict=True,
            prompt_teacher_weights=teacher,
            prompt_teacher_mask=torch.ones(2),
        )
        result["restored"].square().mean().backward()

        self.assertEqual(backbone.forward_batch_sizes, [2, 1])
        self.assertIsNotNone(image.grad)
        assert backbone.condition_affine is not None
        self.assertIsNotNone(backbone.condition_affine.weight.grad)

    def test_compound_refiner_is_identity_initialized_and_route_gated(self) -> None:
        model = RMRPPromptedDFPIR(
            PromptAwareIdentityBackbone(),
            torch.randn(6, 512),
            use_refiner=False,
            use_compound_refiner=True,
        ).train()
        image = torch.rand(2, 3, 20, 24)
        teacher = torch.nn.functional.one_hot(torch.tensor([5, 1]), 6).float()
        result = model(
            image,
            full_sensor_packet(2),
            return_dict=True,
            prompt_teacher_weights=teacher,
            prompt_teacher_mask=torch.ones(2),
        )

        self.assertTrue(torch.allclose(result["restored"], image, atol=1e-7))
        self.assertEqual(
            float(result["compound_refiner_correction"][1].detach().abs().max()),
            0.0,
        )
        result["restored"][0].square().mean().backward()
        assert model.compound_refiner is not None
        self.assertIsNotNone(model.compound_refiner.output.weight.grad)
        self.assertGreater(
            float(model.compound_refiner.output.weight.grad.abs().sum()), 0.0
        )


if __name__ == "__main__":
    unittest.main()
