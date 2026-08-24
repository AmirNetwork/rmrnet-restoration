# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Sensor-conditioned DFPIR backbone used by the final RMR-P model.

The image backbone is deliberately shared with the strongest matched baseline.
RMR-P adds only an observable-sensor state path, an image fallback, and a
bounded residual head. This makes the controlled comparison interpretable:
both methods start from the same DFPIR checkpoint and receive the same target
domain optimizer-step budget.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rcadnet.model import PostPriorEvidenceRefiner
from rcadnet.practical_metadata import PRACTICAL_SENSOR_DIM, PracticalSensorEncoder


class ImageCorruptionState(nn.Module):
    """Estimate the eight-coordinate corruption state from image evidence."""

    def __init__(self, code_dim: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 40, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(40, 56, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(56, 48), nn.GELU(), nn.Linear(48, code_dim))
        # An untrained sigmoid head otherwise emits about 0.5 for every cause.
        # That is not "unknown" in a multi-cause state: it can incorrectly
        # outweigh a reliable observed sensor cause. Start from low evidence and
        # let the supervised image branch learn positive causes from training.
        final = self.head[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, -3.0)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        code = torch.sigmoid(self.head(self.features(image)))
        severity = code[:, :7].amax(dim=1, keepdim=True)
        return torch.cat([code[:, :7], severity], dim=1)


class RMRPPromptedDFPIR(nn.Module):
    """Road Metadata-aware Restoration for Pavement Inspection.

    Let ``z_I`` be the image-estimated state, ``z_M`` the state decoded from
    camera/IMU/vehicle measurements, and ``q_M`` the coordinate-wise sensor
    reliability. The deployed state is

        z = q_M * z_M + (1 - q_M) * z_I.

    The state selects a physically meaningful DFPIR task prompt (deblur,
    defocus, denoise, low light, or joint motion--low-light) and continuously
    modulates four feature scales through zero-initialized FiLM adapters.
    Missing or partial metadata therefore falls back per coordinate rather
    than switching the complete metadata path on or off.
    """

    def __init__(
        self,
        backbone: nn.Module,
        prompt_embeddings: torch.Tensor,
        *,
        sensor_gyro_full_scale: float = 4.0,
        prompt_residual_scale: float = 0.10,
        refiner_gain: float = 0.12,
        prompt_router: str = "hard",
        sensor_route_mode: str = "posterior",
        use_refiner: bool = True,
        compound_motion_blend: float = 0.0,
        use_compound_refiner: bool = False,
        compound_refiner_gain: float = 0.18,
        use_cause_refiners: bool = False,
        cause_refiner_gain: float = 0.08,
    ) -> None:
        super().__init__()
        if prompt_embeddings.ndim != 2 or prompt_embeddings.shape[1] != 512:
            raise ValueError("prompt_embeddings must have shape [N, 512]")
        if prompt_embeddings.shape[0] not in {3, 4, 5, 6}:
            raise ValueError(
                "RMR-P supports three-, four-, five-, or six-prompt task bases"
            )
        self.backbone = backbone
        self.sensor_dim = PRACTICAL_SENSOR_DIM
        self.use_practical_sensor_encoder = True
        self.use_motion_prior = False
        self.metadata_encoding = "practical_sensor_v1"
        self.sensor_gyro_full_scale = float(sensor_gyro_full_scale)
        self.prompt_residual_scale = float(prompt_residual_scale)
        self.prompt_basis_count = int(prompt_embeddings.shape[0])
        if prompt_router not in {"hard", "sparse_blend"}:
            raise ValueError(f"Unsupported prompt router: {prompt_router}")
        self.prompt_router = prompt_router
        if sensor_route_mode not in {"posterior", "physical_fused"}:
            raise ValueError(f"Unsupported sensor route mode: {sensor_route_mode}")
        self.sensor_route_mode = sensor_route_mode
        self.use_refiner = bool(use_refiner)
        if not 0.0 <= compound_motion_blend <= 1.0:
            raise ValueError("compound_motion_blend must be in [0, 1]")
        self.compound_motion_blend = float(compound_motion_blend)
        self.use_compound_refiner = bool(use_compound_refiner)
        self.sensor_encoder = PracticalSensorEncoder(
            sensor_dim=self.sensor_dim,
            code_dim=8,
            gyro_full_scale=self.sensor_gyro_full_scale,
        )
        self.image_state = ImageCorruptionState(code_dim=8)
        self.posterior_refine = nn.Sequential(
            nn.Linear(24, 48),
            nn.GELU(),
            nn.Linear(48, 8),
        )
        self.prompt_delta = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, 512),
        )
        self.refiner = (
            PostPriorEvidenceRefiner(code_dim=8, hidden_channels=32, max_gain=refiner_gain)
            if self.use_refiner
            else None
        )
        self.refiner_gain = float(refiner_gain)
        self.compound_refiner = (
            PostPriorEvidenceRefiner(
                code_dim=8,
                hidden_channels=48,
                max_gain=compound_refiner_gain,
            )
            if self.use_compound_refiner
            else None
        )
        self.compound_refiner_gain = float(compound_refiner_gain)
        self.cause_refiners: nn.ModuleList | None = None
        self.use_cause_refiners = bool(use_cause_refiners)
        self.cause_refiner_gain = float(cause_refiner_gain)
        if self.use_cause_refiners:
            self.enable_cause_refiners(self.cause_refiner_gain)
        self.register_buffer("prompt_embeddings", prompt_embeddings.detach().float().clone())
        if not hasattr(self.backbone, "enable_continuous_conditioning"):
            raise TypeError("DFPIR backbone does not expose continuous conditioning")
        self.backbone.enable_continuous_conditioning(code_dim=8)
        self._init_identity_paths()

    def _init_identity_paths(self) -> None:
        # New checkpoints initially reproduce the sensor-selected DFPIR output.
        for branch in (self.posterior_refine, self.prompt_delta):
            final = branch[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        # The official DFPIR text code affects a discrete top-k permutation,
        # so a learned text-vector delta has no usable gradient. Keep these
        # tensors only for backward-compatible loading of audited checkpoints;
        # continuous state learning is performed by the feature FiLM adapters.
        for parameter in self.prompt_delta.parameters():
            parameter.requires_grad_(False)

    def enable_bounded_refiner(self, max_gain: float = 0.12) -> None:
        """Enable the zero-initialized state-conditioned output correction."""
        if self.refiner is None:
            self.refiner = PostPriorEvidenceRefiner(
                code_dim=8,
                hidden_channels=32,
                max_gain=max_gain,
            ).to(device=self.prompt_embeddings.device)
        self.use_refiner = True
        self.refiner_gain = float(max_gain)

    def enable_compound_refiner(self, max_gain: float = 0.18) -> None:
        """Enable the identity-initialized motion/low-light residual expert."""
        if self.compound_refiner is None:
            self.compound_refiner = PostPriorEvidenceRefiner(
                code_dim=8,
                hidden_channels=48,
                max_gain=max_gain,
            ).to(device=self.prompt_embeddings.device)
        self.use_compound_refiner = True
        self.compound_refiner_gain = float(max_gain)

    def enable_cause_refiners(self, max_gain: float = 0.08) -> None:
        """Enable metadata-selected bounded residual specialists.

        The five heads correspond to the non-clean prompt operators: motion,
        defocus, noise/compression, low light, and compound motion--low-light.
        They are selected by ``prompt_weights`` inferred from observable image
        and sensor evidence, never by a benchmark scenario identifier.  Every
        output projection is zero initialized, so enabling the bank preserves
        the validated backbone output exactly before optimization.
        """

        if self.prompt_basis_count != 6:
            raise ValueError("cause refiners require the six-prompt DFPIR basis")
        if not 0.0 < float(max_gain) <= 0.5:
            raise ValueError("cause refiner gain must be in (0, 0.5]")
        if self.cause_refiners is None:
            self.cause_refiners = nn.ModuleList(
                [
                    PostPriorEvidenceRefiner(
                        code_dim=8,
                        hidden_channels=32,
                        max_gain=max_gain,
                    )
                    for _ in range(5)
                ]
            ).to(device=next(self.backbone.parameters()).device)
        self.use_cause_refiners = True
        self.cause_refiner_gain = float(max_gain)

    def _apply_cause_refiners(
        self,
        image: torch.Tensor,
        candidate: torch.Tensor,
        code: torch.Tensor,
        prompt_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply only the specialists supported by the inferred physical state."""

        if not self.use_cause_refiners or self.cause_refiners is None:
            zero_gate = image.new_zeros(
                (image.shape[0], 1, image.shape[-2], image.shape[-1])
            )
            return candidate, zero_gate, torch.zeros_like(image), prompt_weights[:, 1:]

        cause_weights = prompt_weights[:, 1:6]
        corrections: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for index, refiner in enumerate(self.cause_refiners):
            _, gate, correction = refiner(
                image,
                candidate,
                candidate,
                code,
                cause_weights[:, index],
            )
            corrections.append(correction)
            gates.append(gate * cause_weights[:, index, None, None, None])
        correction = torch.stack(corrections, dim=1).sum(dim=1)
        gate = torch.stack(gates, dim=1).sum(dim=1)
        restored = torch.clamp(candidate + correction, 0.0, 1.0)
        return restored, gate, correction, cause_weights

    def _prompt_weights(self, code: torch.Tensor) -> torch.Tensor:
        motion = code[:, :3].amax(dim=1)
        defocus = code[:, 3]
        motion_or_defocus = torch.maximum(motion, defocus)
        # Camera exposure can contain weak noise/illumination evidence even in
        # a defocus capture. Calibrated nuisance floors stop those secondary
        # cues from changing the DFPIR task, while the compound term ensures
        # that synchronized motion plus substantial low light selects the
        # low-light expert used by the official mixed-scenario protocol.
        noise_or_compression = torch.maximum(code[:, 4], code[:, 6]) - 0.30
        # The practical code is normalized so 0.35 is the onset of material
        # illumination loss. A steeper calibrated logit separates that state
        # from modest exposure variation, while motion reinforces compound
        # captures without turning ordinary motion blur into low light.
        low_light = 4.0 * (code[:, 5] - 0.35) + 0.50 * motion
        if self.prompt_basis_count == 6:
            severity = code[:, :7].amax(dim=1)
            clean = 0.20 - 2.0 * severity
            score_terms = [
                clean,
                motion,
                defocus,
                noise_or_compression,
                low_light,
            ]
        elif self.prompt_basis_count == 5:
            score_terms = [motion, defocus, noise_or_compression, low_light]
        else:
            score_terms = [motion_or_defocus, noise_or_compression, low_light]
        if self.prompt_basis_count in {4, 5, 6}:
            # A compound operator is activated only when *both* observable
            # motion and illumination-loss evidence exceed their calibrated
            # onsets. This reproduces the semantics of the public DFPIR mixed
            # prompt without exposing a scenario label or a synthetic kernel.
            joint_support = torch.relu(motion - 0.12) * torch.relu(code[:, 5] - 0.35)
            compound = low_light + 8.0 * joint_support - 0.05
            score_terms.append(compound)
        scores = torch.stack(score_terms, dim=1)
        if self.prompt_router == "hard":
            soft = torch.softmax(8.0 * scores, dim=1)
            hard = F.one_hot(
                scores.argmax(dim=1), num_classes=self.prompt_basis_count
            ).to(soft.dtype)
            return hard + soft - soft.detach()

        # Mixed physical causes should retain every supported task basis. This
        # sparse convex combination lets motion and low-light evidence coexist
        # without introducing a hidden scenario label or an arbitrary prompt.
        if self.prompt_basis_count == 6:
            severity = code[:, :7].amax(dim=1)
            strength_terms = [
                torch.relu(0.12 - severity),
                motion,
                defocus,
                torch.relu(torch.maximum(code[:, 4], code[:, 6]) - 0.20),
                torch.relu(code[:, 5] - 0.20),
            ]
        elif self.prompt_basis_count == 5:
            strength_terms = [
                motion,
                defocus,
                torch.relu(torch.maximum(code[:, 4], code[:, 6]) - 0.20),
                torch.relu(code[:, 5] - 0.20),
            ]
        else:
            strength_terms = [
                motion_or_defocus,
                torch.relu(torch.maximum(code[:, 4], code[:, 6]) - 0.20),
                torch.relu(code[:, 5] - 0.20),
            ]
        if self.prompt_basis_count in {4, 5, 6}:
            strength_terms.append(
                4.0
                * torch.relu(motion - 0.12)
                * torch.relu(code[:, 5] - 0.35)
            )
        strengths = torch.stack(strength_terms, dim=1)
        active = torch.relu(strengths - 0.05)
        fallback = F.one_hot(
            strengths.argmax(dim=1), num_classes=self.prompt_basis_count
        ).to(active.dtype)
        normalized = active / active.sum(dim=1, keepdim=True).clamp_min(1e-6)
        has_support = (active.sum(dim=1, keepdim=True) > 0).to(active.dtype)
        return has_support * normalized + (1.0 - has_support) * fallback

    def _sensor_state(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        image_code = self.image_state(image)
        if metadata is None:
            sensor_code = torch.zeros_like(image_code)
            support = torch.zeros_like(image_code)
            sensor_physical = torch.zeros_like(image_code)
            sensor_direct = torch.zeros_like(image_code)
        else:
            if metadata.ndim == 1:
                metadata = metadata.unsqueeze(0)
            sensor_code = self.sensor_encoder(metadata.to(device=image.device, dtype=image.dtype))
            support = self.sensor_encoder.last_cause_reliability
            sensor_physical = self.sensor_encoder.last_calibrated_physical_code
            sensor_direct = self.sensor_encoder.last_direct_code
            if support is None or sensor_physical is None or sensor_direct is None:
                raise RuntimeError("PracticalSensorEncoder did not expose its audited state")
            support = support.to(device=image.device, dtype=image.dtype)
            sensor_physical = sensor_physical.to(device=image.device, dtype=image.dtype)
            sensor_direct = sensor_direct.to(device=image.device, dtype=image.dtype)

        fused = support * sensor_code + (1.0 - support) * image_code
        disagreement = (sensor_code - image_code).abs()
        correction = 0.15 * torch.tanh(
            self.posterior_refine(torch.cat([image_code, sensor_code, disagreement], dim=1))
        )
        posterior = (fused + correction).clamp(0.0, 1.0)
        posterior = torch.cat(
            [posterior[:, :7], posterior[:, :7].amax(dim=1, keepdim=True)],
            dim=1,
        )
        return posterior, image_code, sensor_code, sensor_physical, support, sensor_direct

    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        return_dict: bool = False,
        return_aux: bool = False,
        prompt_teacher_weights: torch.Tensor | None = None,
        prompt_teacher_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        del return_aux
        (
            code,
            image_code,
            sensor_code,
            sensor_physical,
            support,
            sensor_direct,
        ) = self._sensor_state(image, metadata)
        if self.sensor_route_mode == "physical_fused":
            # Discrete task selection should remain tied to observable capture
            # physics when that coordinate is reliable. The learned posterior
            # still conditions the continuous prompt residual and refiner.
            route_code = support * sensor_direct + (1.0 - support) * code
            route_code = torch.cat(
                [route_code[:, :7], route_code[:, :7].amax(dim=1, keepdim=True)],
                dim=1,
            )
        else:
            route_code = code
        predicted_prompt_weights = self._prompt_weights(route_code)
        prompt_weights = predicted_prompt_weights
        if prompt_teacher_weights is not None:
            teacher = prompt_teacher_weights.to(
                device=image.device, dtype=predicted_prompt_weights.dtype
            )
            if teacher.shape != predicted_prompt_weights.shape:
                raise ValueError(
                    "prompt teacher weights must match predicted prompt weights"
                )
            if prompt_teacher_mask is None:
                prompt_weights = teacher
            else:
                mask = prompt_teacher_mask.to(
                    device=image.device, dtype=predicted_prompt_weights.dtype
                ).reshape(-1, 1)
                prompt_weights = (
                    mask * teacher + (1.0 - mask) * predicted_prompt_weights
                )
        base_prompt = prompt_weights @ self.prompt_embeddings.to(dtype=image.dtype)
        # The text prompt remains a fixed, auditable task selector. Continuous
        # image/sensor state enters through differentiable feature adapters.
        backbone_restored = self.backbone(
            image, base_prompt, condition_code=route_code
        ).clamp(0.0, 1.0)
        if self.prompt_basis_count == 6 and self.compound_motion_blend > 0.0:
            compound_weight = prompt_weights[:, 5]
            compound_indices = torch.nonzero(
                compound_weight > 0.5, as_tuple=False
            ).flatten()
            if compound_indices.numel() > 0:
                motion_prompt = self.prompt_embeddings[1].to(
                    device=image.device, dtype=image.dtype
                ).unsqueeze(0).expand(compound_indices.numel(), -1)
                motion_restored = self.backbone(
                    image.index_select(0, compound_indices),
                    motion_prompt,
                    condition_code=route_code.index_select(0, compound_indices),
                ).clamp(0.0, 1.0)
                selected_backbone = backbone_restored.index_select(
                    0, compound_indices
                )
                compound_candidate = (
                    self.compound_motion_blend * motion_restored
                    + (1.0 - self.compound_motion_blend) * selected_backbone
                )
                # Eq. (compound expert fusion): update only samples for which
                # observable motion and exposure evidence selected the compound
                # route. index_copy is differentiable with respect to both the
                # standard and motion-conditioned restoration passes.
                backbone_restored = backbone_restored.index_copy(
                    0, compound_indices, compound_candidate
                )
        if self.compound_refiner is None:
            compound_gate = image.new_zeros(
                (image.shape[0], 1, image.shape[-2], image.shape[-1])
            )
            compound_correction = torch.zeros_like(image)
        else:
            # The compound route is inferred from observable image/sensor state,
            # not from an evaluation scenario label. The support multiplier makes
            # the expert an exact identity for all other corruption causes.
            compound_support = (
                prompt_weights[:, 5]
                if self.prompt_basis_count == 6
                else image.new_zeros(image.shape[0])
            )
            backbone_restored, compound_gate, compound_correction = (
                self.compound_refiner(
                    image,
                    backbone_restored,
                    backbone_restored,
                    code,
                    compound_support,
                )
            )
        if self.use_cause_refiners and self.cause_refiners is not None:
            restored, refiner_gate, correction, cause_refiner_weights = (
                self._apply_cause_refiners(
                    image,
                    backbone_restored,
                    code,
                    prompt_weights,
                )
            )
        elif self.refiner is None:
            restored = backbone_restored
            refiner_gate = image.new_zeros((image.shape[0], 1, image.shape[-2], image.shape[-1]))
            correction = torch.zeros_like(image)
            cause_refiner_weights = prompt_weights[:, 1:]
        else:
            restored, refiner_gate, correction = self.refiner(
                image,
                backbone_restored,
                backbone_restored,
                code,
                code[:, -1],
            )
            cause_refiner_weights = prompt_weights[:, 1:]
        if not return_dict:
            return restored
        return {
            "restored": restored,
            "input": image,
            "metadata_used": image.new_full((image.shape[0],), 1.0 if metadata is not None else 0.0),
            "code": code,
            "severity": code[:, -1],
            "z_severity": code[:, -1],
            "image_degradation_code": image_code,
            "sensor_code": sensor_code,
            "sensor_cause_reliability": support,
            "sensor_only_physical_code": sensor_physical,
            "sensor_calibrated_physical_code": sensor_physical,
            "sensor_direct_physical_code": sensor_direct,
            "image_physical_code": image_code,
            "posterior_degradation_code": code,
            "degradation_state": code,
            "prompt_routing_code": route_code,
            "neural_restored": backbone_restored,
            "post_prior_gate": refiner_gate,
            "post_prior_correction": correction,
            "compound_refiner_gate": compound_gate,
            "compound_refiner_correction": compound_correction,
            "cause_refiners_enabled": image.new_full(
                (image.shape[0],), float(self.use_cause_refiners)
            ),
            "cause_refiner_weights": cause_refiner_weights,
            "prompt_weights": prompt_weights,
            "predicted_prompt_weights": predicted_prompt_weights,
            "phi": None,
            "lambda1": None,
            "lambda2": None,
        }
