"""Checkpoint-compatible metadata-routed DeMoE base used by TRACE-R.

This module keeps the released DeMoE image-restoration function and adds a
small, observable-sensor controller.  Camera, IMU, and vehicle measurements
are converted to an eight-coordinate corruption state and fused with an image
estimate according to coordinate-wise sensor reliability.  The fused state
routes DeMoE experts and conditions a bounded, identity-initialized output
correction.  No benchmark scenario name or hidden synthetic blur kernel is an
inference input.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.rmrp_prompted_dfpir import ImageCorruptionState
from rcadnet.model import PostPriorEvidenceRefiner
from rcadnet.practical_metadata import PRACTICAL_SENSOR_DIM, PracticalSensorEncoder


class MetadataResidualAdapter(nn.Module):
    """Identity-initialized, detector-aware correction for one physical cause.

    The shared DeMoE output remains the generic restoration.  This adapter sees
    the degraded image, that restoration, their residual, and local high-pass
    evidence.  A physical corruption state modulates the features through FiLM.
    Its final projection is zero initialized, so adding the module cannot change
    an existing checkpoint until supervised adaptation learns a correction.

    For cause ``c`` the adapter implements

        Delta_c = eta * sigmoid(g_c) * tanh(r_c),

    and the sensor controller later forms a reliability-weighted sum of the
    cause corrections.  This is a residual specialist, not a scenario lookup.
    """

    def __init__(
        self,
        code_dim: int = 8,
        hidden_channels: int = 48,
        max_gain: float = 0.25,
    ) -> None:
        super().__init__()
        if not 0.0 < float(max_gain) <= 0.5:
            raise ValueError("metadata residual adapter gain must be in (0, 0.5]")
        self.max_gain = float(max_gain)
        self.input_projection = nn.Sequential(
            nn.Conv2d(12, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        groups=hidden_channels,
                    ),
                    nn.GroupNorm(4, hidden_channels),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, hidden_channels, 1),
                    nn.GELU(),
                )
                for dilation in (1, 2, 4, 1)
            ]
        )
        self.code_affine = nn.Linear(code_dim, 2 * hidden_channels)
        self.output = nn.Conv2d(hidden_channels, 4, 3, padding=1)
        nn.init.zeros_(self.code_affine.weight)
        nn.init.zeros_(self.code_affine.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        degraded: torch.Tensor,
        restored: torch.Tensor,
        code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        highpass = degraded - F.avg_pool2d(
            degraded, kernel_size=5, stride=1, padding=2
        )
        evidence = torch.cat(
            [degraded, restored, restored - degraded, highpass], dim=1
        )
        features = self.input_projection(evidence)
        scale, shift = self.code_affine(code).chunk(2, dim=1)
        features = features * (1.0 + 0.2 * torch.tanh(scale)[:, :, None, None])
        features = features + 0.2 * torch.tanh(shift)[:, :, None, None]
        for block in self.blocks:
            features = features + block(features)
        raw = self.output(features)
        gate = torch.sigmoid(raw[:, 3:4])
        correction = self.max_gain * gate * torch.tanh(raw[:, :3])
        return correction, gate


class RMRPMetadataDeMoE(nn.Module):
    """Checkpoint-compatible metadata-aware DeMoE controller for TRACE-R.

    For image state ``z_I``, sensor state ``z_M``, and coordinate reliability
    ``q_M``, the deployed corruption state is

        z = q_M * z_M + (1 - q_M) * z_I.

    Motion, defocus, and low-light coordinates form physically interpretable
    DeMoE routing evidence.  A learned bounded residual calibrates that route,
    while DeMoE's image router remains the fallback when sensor support is
    weak or absent.  Two experts can be active for compound motion/low-light
    captures; single-cause states naturally assign negligible weight to the
    second expert.
    """

    prompt_basis_count = 6

    def __init__(
        self,
        backbone: nn.Module,
        *,
        sensor_gyro_full_scale: float = 4.0,
        top_k: int = 2,
        refiner_gain: float = 0.12,
        use_refiner: bool = True,
        use_compound_blend_gate: bool = False,
        compound_blend_init: float = 0.65,
        compound_metadata_acceptance: float = 1.0,
        cause_route_acceptance: tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_cause_refiners: bool = False,
        cause_refiner_gain: float = 0.08,
        backbone_route_mode: str = "metadata",
        use_semantic_adapters: bool = False,
        semantic_adapter_gain: float = 0.25,
        sensor_task_thresholds: tuple[float, float, float] = (0.18, 0.20, 0.385),
        sensor_task_mixed_expert: int | None = None,
        semantic_adapter_acceptance: tuple[float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            1.0,
        ),
    ) -> None:
        super().__init__()
        if top_k not in {1, 2}:
            raise ValueError("TRACE-R DeMoE top_k must be 1 or 2")
        self.backbone = backbone
        self.sensor_dim = PRACTICAL_SENSOR_DIM
        self.use_practical_sensor_encoder = True
        self.metadata_encoding = "practical_sensor_v1"
        self.sensor_gyro_full_scale = float(sensor_gyro_full_scale)
        self.top_k = int(top_k)
        self.use_refiner = bool(use_refiner)
        self.refiner_gain = float(refiner_gain)
        self.use_compound_blend_gate = bool(use_compound_blend_gate)
        self.compound_blend_init = float(compound_blend_init)
        if not 0.0 < self.compound_blend_init < 1.0:
            raise ValueError("compound_blend_init must be strictly between 0 and 1")
        self.compound_metadata_acceptance = float(compound_metadata_acceptance)
        if not 0.0 <= self.compound_metadata_acceptance <= 1.0:
            raise ValueError("compound_metadata_acceptance must be in [0, 1]")
        self.set_cause_route_acceptance(cause_route_acceptance)
        self.use_cause_refiners = bool(use_cause_refiners)
        self.cause_refiner_gain = float(cause_refiner_gain)
        if backbone_route_mode not in {"metadata", "image", "sensor_task"}:
            raise ValueError(
                "backbone_route_mode must be 'metadata', 'image', or 'sensor_task'"
            )
        self.backbone_route_mode = str(backbone_route_mode)
        if len(sensor_task_thresholds) != 3:
            raise ValueError("sensor_task_thresholds requires motion, defocus, and low-light values")
        self.sensor_task_thresholds = tuple(float(value) for value in sensor_task_thresholds)
        if any(value < 0.0 or value > 1.0 for value in self.sensor_task_thresholds):
            raise ValueError("sensor_task_thresholds values must be in [0, 1]")
        if sensor_task_mixed_expert is not None and not 0 <= sensor_task_mixed_expert <= 4:
            raise ValueError("sensor_task_mixed_expert must be None or an expert index in [0, 4]")
        # None preserves the historical matched-DeMoE route (mixed -> low light).
        # New checkpoints may reserve one internal restoration expert for a
        # compound cause without combining downstream detector outputs.
        self.sensor_task_mixed_expert = sensor_task_mixed_expert
        self.use_semantic_adapters = bool(use_semantic_adapters)
        self.semantic_adapter_gain = float(semantic_adapter_gain)
        self.set_semantic_adapter_acceptance(semantic_adapter_acceptance)

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
        self.route_residual = nn.Sequential(
            nn.Linear(24, 32),
            nn.GELU(),
            nn.Linear(32, 5),
        )
        self.refiner = (
            PostPriorEvidenceRefiner(
                code_dim=8,
                hidden_channels=32,
                max_gain=self.refiner_gain,
            )
            if self.use_refiner
            else None
        )
        self.cause_refiners = None
        if self.use_cause_refiners:
            self.enable_cause_refiners(self.cause_refiner_gain)
        self.semantic_adapters = None
        if self.use_semantic_adapters:
            self.enable_semantic_adapters(self.semantic_adapter_gain)
        self.compound_blend_head = (
            nn.Linear(32, 1) if self.use_compound_blend_gate else None
        )
        self._init_identity_paths()
        self._init_compound_blend_head()
        self._set_top_k(self.top_k)

    def _init_identity_paths(self) -> None:
        # New modules initially preserve physical fusion and the released
        # DeMoE output.  Improvements must therefore be learned by training.
        for branch in (self.posterior_refine, self.route_residual):
            final = branch[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def _init_compound_blend_head(self) -> None:
        if self.compound_blend_head is None:
            return
        nn.init.zeros_(self.compound_blend_head.weight)
        nn.init.constant_(
            self.compound_blend_head.bias,
            math.log(self.compound_blend_init / (1.0 - self.compound_blend_init)),
        )

    def _set_top_k(self, top_k: int) -> None:
        experts = getattr(self.backbone, "experts", None)
        if experts is None:
            raise TypeError("DeMoE backbone does not expose its expert blocks")
        for block in experts:
            block.used = int(top_k)

    def set_cause_route_acceptance(
        self,
        values: tuple[float, float, float] | list[float],
    ) -> None:
        """Set bounded metadata-to-image routing trust.

        The entries correspond to motion, defocus, and low light.  They do not
        encode a benchmark scenario.  Instead, they calibrate how strongly an
        observable physical cause may move DeMoE away from its image-derived
        router.  A value of one reproduces the original metadata route; zero
        retains the image route.  Missing modalities still fall back through
        the coordinate reliability mask before this calibration is applied.
        """

        if len(values) != 3:
            raise ValueError("cause_route_acceptance requires three values")
        calibrated = tuple(float(value) for value in values)
        if any(value < 0.0 or value > 1.0 for value in calibrated):
            raise ValueError("cause_route_acceptance values must be in [0, 1]")
        self.cause_route_acceptance = calibrated

    def enable_bounded_refiner(self, max_gain: float = 0.12) -> None:
        if self.refiner is None:
            self.refiner = PostPriorEvidenceRefiner(
                code_dim=8,
                hidden_channels=32,
                max_gain=max_gain,
            ).to(next(self.backbone.parameters()).device)
        self.use_refiner = True
        self.refiner_gain = float(max_gain)

    def enable_cause_refiners(self, max_gain: float = 0.08) -> None:
        """Enable motion, defocus, and low-light correction experts.

        When a validated shared refiner exists, each cause expert is an exact
        copy.  Enabling the bank therefore preserves the current degraded-image
        function before optimization while allowing each physical cause to
        learn a different bounded correction afterwards.
        """

        if not 0.0 < float(max_gain) <= 0.5:
            raise ValueError("cause refiner gain must be in (0, 0.5]")
        if self.refiner is None:
            self.enable_bounded_refiner(max_gain)
        assert self.refiner is not None
        if self.cause_refiners is None:
            self.cause_refiners = nn.ModuleList(
                [copy.deepcopy(self.refiner) for _ in range(3)]
            ).to(next(self.backbone.parameters()).device)
        self.use_cause_refiners = True
        self.cause_refiner_gain = float(max_gain)

    def enable_semantic_adapters(self, max_gain: float = 0.25) -> None:
        """Enable motion, defocus, low-light, and compound residual experts."""

        if not 0.0 < float(max_gain) <= 0.5:
            raise ValueError("semantic adapter gain must be in (0, 0.5]")
        if self.semantic_adapters is None:
            self.semantic_adapters = nn.ModuleList(
                [
                    MetadataResidualAdapter(max_gain=max_gain)
                    for _ in range(4)
                ]
            ).to(next(self.backbone.parameters()).device)
        self.use_semantic_adapters = True
        self.semantic_adapter_gain = float(max_gain)

    def set_semantic_adapter_acceptance(
        self,
        values: tuple[float, float, float, float] | list[float],
    ) -> None:
        """Set trust in motion, defocus, low-light, and compound adapters."""

        calibrated = tuple(float(value) for value in values)
        if len(calibrated) != 4:
            raise ValueError("semantic_adapter_acceptance requires four values")
        if any(value < 0.0 or value > 1.0 for value in calibrated):
            raise ValueError(
                "semantic_adapter_acceptance values must be in [0, 1]"
            )
        self.semantic_adapter_acceptance = calibrated

    def set_semantic_adapter_gain(self, max_gain: float) -> None:
        """Apply one bounded validation-selected gain to all residual experts."""

        if not 0.0 <= float(max_gain) <= 0.5:
            raise ValueError("semantic adapter gain must be in [0, 0.5]")
        self.semantic_adapter_gain = float(max_gain)
        if self.semantic_adapters is not None:
            for adapter in self.semantic_adapters:
                adapter.max_gain = float(max_gain)

    def enable_compound_blend_gate(self, initial_blend: float = 0.65) -> None:
        """Enable the image-and-sensor gate without changing old checkpoints.

        The gate estimates how much top-2 expert evidence to add to the sparse
        top-1 restoration for motion-plus-low-light captures.  Its inputs are
        the image state, sensor state, their disagreement, and per-cause sensor
        reliability.  A zero-weight initialization exactly reproduces the
        validation-selected global blend before gate-only calibration.
        """

        if not 0.0 < initial_blend < 1.0:
            raise ValueError("initial_blend must be strictly between 0 and 1")
        if self.compound_blend_head is None:
            self.compound_blend_head = nn.Linear(32, 1).to(
                next(self.backbone.parameters()).device
            )
        self.use_compound_blend_gate = True
        self.compound_blend_init = float(initial_blend)
        self._init_compound_blend_head()

    def _sensor_state(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        image_code = self.image_state(image)
        if metadata is None:
            sensor_code = torch.zeros_like(image_code)
            sensor_physical = torch.zeros_like(image_code)
            sensor_direct = torch.zeros_like(image_code)
            support = torch.zeros_like(image_code)
        else:
            if metadata.ndim == 1:
                metadata = metadata.unsqueeze(0)
            sensor_code = self.sensor_encoder(
                metadata.to(device=image.device, dtype=image.dtype)
            )
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
            self.posterior_refine(
                torch.cat([image_code, sensor_code, disagreement], dim=1)
            )
        )
        code = (fused + correction).clamp(0.0, 1.0)
        code = torch.cat(
            [code[:, :7], code[:, :7].amax(dim=1, keepdim=True)], dim=1
        )
        route_code = support * sensor_direct + (1.0 - support) * code
        route_code = torch.cat(
            [
                route_code[:, :7],
                route_code[:, :7].amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        return (
            code,
            route_code,
            image_code,
            sensor_code,
            sensor_physical,
            sensor_direct,
            support,
            disagreement,
        )

    @staticmethod
    def _teacher_to_experts(
        teacher: torch.Tensor,
        auto_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Map clean/motion/defocus/noise/low-light/compound to DeMoE."""

        if teacher.shape[1] != 6:
            raise ValueError("TRACE-R route teacher must have six cause columns")
        motion = teacher[:, 1] + 0.5 * teacher[:, 5]
        defocus = teacher[:, 2]
        lowlight = teacher[:, 4] + 0.5 * teacher[:, 5]
        explicit = torch.stack(
            [defocus, torch.zeros_like(motion), torch.zeros_like(motion), motion, lowlight],
            dim=1,
        )
        explicit_sum = explicit.sum(dim=1, keepdim=True)
        explicit = explicit / explicit_sum.clamp_min(1e-6)
        has_explicit = (explicit_sum > 0).to(explicit.dtype)
        return has_explicit * explicit + (1.0 - has_explicit) * auto_weights

    def _metadata_expert_weights(
        self,
        route_code: torch.Tensor,
        image_code: torch.Tensor,
        sensor_code: torch.Tensor,
        cause_reliability: torch.Tensor,
        disagreement: torch.Tensor,
        auto_weights: torch.Tensor,
    ) -> torch.Tensor:
        motion = route_code[:, :3].amax(dim=1)
        defocus = route_code[:, 3]
        lowlight = route_code[:, 5]
        # DeMoE expert order: defocus, real global motion, local motion,
        # synthetic/global vibration, and low light.  Noise/compression and
        # weak evidence retain DeMoE's own image-router distribution.
        evidence = torch.stack(
            [
                defocus,
                0.05 * motion,
                0.10 * route_code[:, 2],
                motion,
                lowlight,
            ],
            dim=1,
        )
        # Reliability is not degradation magnitude.  The physical code above
        # determines which corruption is present, whereas camera/IMU quality
        # determines whether that route should override DeMoE's image router:
        #
        #   w = q_M w_M(z) + (1 - q_M) w_I.
        #
        # Using max(motion, defocus, lowlight) as q_M made mild but reliable
        # defocus look "uncertain" and incorrectly handed routing back to the
        # motion-biased image branch.  The maximum available cause reliability
        # is the correct packet-level trust for this sparse expert decision;
        # unsupported modalities have zero reliability by construction.
        support = torch.stack(
            [
                cause_reliability[:, :3].amax(dim=1),
                cause_reliability[:, 3],
                cause_reliability[:, 5],
            ],
            dim=1,
        ).amax(dim=1, keepdim=True)
        physical = evidence / evidence.sum(dim=1, keepdim=True).clamp_min(1e-6)
        cause_evidence = torch.stack([motion, defocus, lowlight], dim=1)
        cause_mix = cause_evidence / cause_evidence.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        acceptance_values = route_code.new_tensor(
            self.cause_route_acceptance
        ).unsqueeze(0)
        route_acceptance = (cause_mix * acceptance_values).sum(
            dim=1, keepdim=True
        )
        # Metadata reliability and routing compatibility answer different
        # questions. q_M says whether a measurement exists and is trustworthy;
        # a_c says how much that physical cause should override the image router.
        #
        #   w = (q_M a_c) w_M(z) + (1 - q_M a_c) w_I.
        effective_support = support * route_acceptance
        routed = effective_support * physical + (1.0 - effective_support) * auto_weights
        residual = 0.35 * torch.tanh(
            self.route_residual(
                torch.cat([image_code, sensor_code, disagreement], dim=1)
            )
        )
        logits = routed.clamp_min(1e-6).log() + residual
        return torch.softmax(logits, dim=1)

    def _sensor_task_expert_weights(
        self,
        sensor_direct: torch.Tensor,
        cause_reliability: torch.Tensor,
        auto_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Select DeMoE's declared task expert from observable measurements.

        The matched DeMoE control receives a declared restoration task and uses
        one of five one-hot experts. TRACE-R does not receive the benchmark
        scenario string. Instead, this route derives the same task decision
        from exposure-synchronized motion, focus, and low-light measurements:

            c = arg priority(defocus, low-light, motion; z_M, q_M).

        The three thresholds are selected on training/validation metadata and
        stored in the checkpoint. If the corresponding modality is unavailable
        or no measured cause exceeds its threshold, DeMoE's image router is
        retained. Historical checkpoints map mixed motion/low-light evidence
        to the low-light expert. New checkpoints may nominate a dedicated
        internal restoration expert through ``sensor_task_mixed_expert``.
        """

        motion_threshold, defocus_threshold, lowlight_threshold = (
            self.sensor_task_thresholds
        )
        motion = sensor_direct[:, :3].amax(dim=1)
        defocus = sensor_direct[:, 3]
        lowlight = sensor_direct[:, 5]
        # Availability and confidence are different quantities. A low
        # autofocus-confidence reading is itself common during defocus and
        # must not erase a directly measured focus-error value. Any non-zero
        # modality reliability therefore makes the measurement eligible for
        # the discrete task nomination; its continuous reliability still
        # controls the posterior and bounded residual correction downstream.
        motion_supported = cause_reliability[:, :3].amax(dim=1) > 1e-6
        defocus_supported = cause_reliability[:, 3] > 1e-6
        lowlight_supported = cause_reliability[:, 5] > 1e-6

        motion_active = motion_supported & (motion >= motion_threshold)
        defocus_active = defocus_supported & (defocus >= defocus_threshold)
        lowlight_active = lowlight_supported & (lowlight >= lowlight_threshold)
        mixed_active = motion_active & lowlight_active

        # DeMoE expert order: defocus, real global motion, local motion,
        # synthetic/global motion, and low light. Priority reproduces the
        # matched baseline's scenario mapping without exposing that label.
        task_index = torch.full_like(motion, -1, dtype=torch.long)
        task_index = torch.where(motion_active, torch.full_like(task_index, 3), task_index)
        task_index = torch.where(lowlight_active, torch.full_like(task_index, 4), task_index)
        if self.sensor_task_mixed_expert is not None:
            task_index = torch.where(
                mixed_active,
                torch.full_like(task_index, self.sensor_task_mixed_expert),
                task_index,
            )
        task_index = torch.where(defocus_active, torch.zeros_like(task_index), task_index)
        available = task_index >= 0
        one_hot = F.one_hot(task_index.clamp_min(0), num_classes=5).to(auto_weights.dtype)
        return torch.where(available[:, None], one_hot, auto_weights)

    def _teacher_to_sensor_task_experts(
        self,
        teacher: torch.Tensor,
        auto_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Map training-only cause targets to the declared DeMoE task route."""

        if teacher.shape[1] != 6:
            raise ValueError("TRACE-R route teacher must have six cause columns")
        motion = teacher[:, 1] > 0.5
        defocus = teacher[:, 2] > 0.5
        lowlight = (teacher[:, 4] > 0.5) | (teacher[:, 5] > 0.5)
        mixed = motion & lowlight
        task_index = torch.full(
            (teacher.shape[0],), -1, device=teacher.device, dtype=torch.long
        )
        task_index = torch.where(motion, torch.full_like(task_index, 3), task_index)
        task_index = torch.where(lowlight, torch.full_like(task_index, 4), task_index)
        if self.sensor_task_mixed_expert is not None:
            task_index = torch.where(
                mixed,
                torch.full_like(task_index, self.sensor_task_mixed_expert),
                task_index,
            )
        task_index = torch.where(defocus, torch.zeros_like(task_index), task_index)
        available = task_index >= 0
        one_hot = F.one_hot(task_index.clamp_min(0), num_classes=5).to(auto_weights.dtype)
        return torch.where(available[:, None], one_hot, auto_weights)

    def _adapt_backbone_feature(
        self,
        stage: str,
        feature: torch.Tensor,
        conditioning: torch.Tensor,
        cause_reliability: torch.Tensor,
    ) -> torch.Tensor:
        """Optional checkpoint-compatible feature adaptation hook.

        The released metadata router leaves DeMoE features unchanged. TRACE-R
        subclasses override this hook to inject the joint image--sensor state
        inside the restoration hierarchy. Keeping the default as an exact
        identity preserves every historical checkpoint and inference result.
        """

        del stage, conditioning, cause_reliability
        return feature

    def _backbone_forward(
        self,
        image: torch.Tensor,
        route_code: torch.Tensor,
        image_code: torch.Tensor,
        sensor_code: torch.Tensor,
        sensor_direct: torch.Tensor,
        cause_reliability: torch.Tensor,
        disagreement: torch.Tensor,
        prompt_teacher_weights: torch.Tensor | None,
        prompt_teacher_mask: torch.Tensor | None,
        active_top_k: int,
        expert_weights_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = image.shape
        padded = self.backbone.check_image_size(image)
        x = self.backbone.intro(padded)
        conditioning = torch.cat(
            [image_code, sensor_code, disagreement, cause_reliability], dim=1
        )
        encodings = []
        bins = []
        for stage_index, (encoder, down) in enumerate(
            zip(self.backbone.encoders, self.backbone.downs)
        ):
            x = encoder(x)
            x = self._adapt_backbone_feature(
                f"encoder_{stage_index}", x, conditioning, cause_reliability
            )
            encodings.append(x)
            x = down(x)

        auto_weights = torch.softmax(self.backbone.mlp_branch(x), dim=1)
        predicted = self._metadata_expert_weights(
            route_code,
            image_code,
            sensor_code,
            cause_reliability,
            disagreement,
            auto_weights,
        )
        sensor_task_weights = self._sensor_task_expert_weights(
            sensor_direct,
            cause_reliability,
            auto_weights,
        )
        # In image-route mode metadata cannot degrade the proven shared
        # restorer. It selects only the residual specialists below. This makes
        # the deployed semantics explicit: image evidence restores generally;
        # measured capture state requests a bounded cause-specific correction.
        if self.backbone_route_mode == "image":
            expert_weights = auto_weights
        elif self.backbone_route_mode == "sensor_task":
            expert_weights = sensor_task_weights
        else:
            expert_weights = predicted
        if prompt_teacher_weights is not None:
            teacher_values = prompt_teacher_weights.to(
                device=image.device, dtype=image.dtype
            )
            teacher = (
                self._teacher_to_sensor_task_experts(teacher_values, auto_weights)
                if self.backbone_route_mode == "sensor_task"
                else self._teacher_to_experts(teacher_values, auto_weights)
            )
            if prompt_teacher_mask is None:
                expert_weights = teacher
            else:
                mask = prompt_teacher_mask.to(
                    device=image.device, dtype=image.dtype
                ).reshape(-1, 1)
                expert_weights = mask * teacher + (1.0 - mask) * predicted
        if expert_weights_override is not None:
            expert_weights = expert_weights_override.to(
                device=image.device, dtype=image.dtype
            )

        self._set_top_k(active_top_k)

        x = self.backbone.middle_blks(x)
        x, expert_bins, _ = self.backbone.experts[0](x, expert_weights)
        x = self._adapt_backbone_feature(
            "bottleneck", x, conditioning, cause_reliability
        )
        bins.append(expert_bins)
        for stage_index, (decoder, up, skip, expert) in enumerate(
            zip(
                self.backbone.decoders,
                self.backbone.ups,
                encodings[::-1],
                self.backbone.experts[1:],
            )
        ):
            x = up(x)
            x = x + skip
            x = decoder(x)
            x, expert_bins, _ = expert(x, expert_weights)
            x = self._adapt_backbone_feature(
                f"decoder_{stage_index}", x, conditioning, cause_reliability
            )
            bins.append(expert_bins)
        restored = self.backbone.ending(x) + padded
        restored = restored[:, :, :height, :width].clamp(0.0, 1.0)
        return restored, expert_weights, predicted, torch.stack(bins, dim=0)

    @staticmethod
    def _compound_mask(
        route_code: torch.Tensor,
        prompt_teacher_weights: torch.Tensor | None,
        prompt_teacher_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return samples supported as joint motion and low-light captures."""

        motion_support = route_code[:, :3].amax(dim=1)
        compound = (motion_support > 0.12) & (route_code[:, 5] > 0.40)
        if prompt_teacher_weights is None:
            return compound
        teacher_compound = prompt_teacher_weights[:, 5] > 0.5
        if prompt_teacher_mask is None:
            return teacher_compound
        return torch.where(
            prompt_teacher_mask.to(device=route_code.device, dtype=torch.bool),
            teacher_compound,
            compound,
        )

    def _apply_refiner(
        self,
        image: torch.Tensor,
        backbone_restored: torch.Tensor,
        code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.refiner is None:
            return (
                backbone_restored,
                image.new_zeros((image.shape[0], 1, image.shape[-2], image.shape[-1])),
                torch.zeros_like(image),
            )
        shared = self.refiner(
            image,
            backbone_restored,
            backbone_restored,
            code,
            code[:, -1],
        )
        if not self.use_cause_refiners or self.cause_refiners is None:
            return shared

        cause_strength = torch.stack(
            [code[:, :3].amax(dim=1), code[:, 3], code[:, 5]], dim=1
        )
        cause_sum = cause_strength.sum(dim=1, keepdim=True)
        cause_weights = cause_strength / cause_sum.clamp_min(1e-6)
        candidates = [
            refiner(
                image,
                backbone_restored,
                backbone_restored,
                code,
                code[:, -1],
            )
            for refiner in self.cause_refiners
        ]
        restored = torch.stack([item[0] for item in candidates], dim=1)
        gate = torch.stack([item[1] for item in candidates], dim=1)
        correction = torch.stack([item[2] for item in candidates], dim=1)
        spatial_weights = cause_weights[:, :, None, None, None]
        mixed_restored = (restored * spatial_weights).sum(dim=1)
        mixed_gate = (gate * spatial_weights).sum(dim=1)
        mixed_correction = (correction * spatial_weights).sum(dim=1)
        has_cause = (cause_sum > 1e-6)[:, :, None, None]
        return (
            torch.where(has_cause, mixed_restored, shared[0]),
            torch.where(has_cause, mixed_gate, shared[1]),
            torch.where(has_cause, mixed_correction, shared[2]),
        )

    def _apply_semantic_adapters(
        self,
        image: torch.Tensor,
        restored: torch.Tensor,
        route_code: torch.Tensor,
        cause_reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply only sensor-supported physical-cause corrections.

        The four weights represent motion, defocus, low light, and the joint
        motion/low-light state.  Reliability masks availability; corruption
        magnitude and availability are intentionally kept distinct. Missing
        metadata therefore gives exactly the shared image restoration.
        """

        zeros = image.new_zeros((image.shape[0], 4))
        zero_map = torch.zeros_like(image)
        if not self.use_semantic_adapters or self.semantic_adapters is None:
            return restored, zeros, zero_map, image.new_zeros(
                (image.shape[0], 1, image.shape[-2], image.shape[-1])
            )

        motion_reliability = cause_reliability[:, :3].amax(dim=1)
        defocus_reliability = cause_reliability[:, 3]
        lowlight_reliability = cause_reliability[:, 5]
        motion = route_code[:, :3].amax(dim=1) * motion_reliability
        defocus = route_code[:, 3] * defocus_reliability
        lowlight = route_code[:, 5] * lowlight_reliability
        compound = torch.minimum(motion, lowlight)
        evidence = torch.stack(
            [
                (motion - compound).clamp_min(0.0),
                defocus,
                (lowlight - compound).clamp_min(0.0),
                compound,
            ],
            dim=1,
        )
        availability = evidence.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        acceptance = image.new_tensor(self.semantic_adapter_acceptance).unsqueeze(0)
        accepted_evidence = evidence * acceptance
        accepted_sum = accepted_evidence.sum(dim=1, keepdim=True)
        weights = accepted_evidence / accepted_sum.clamp_min(1e-6)

        corrections = []
        gates = []
        for adapter in self.semantic_adapters:
            correction, gate = adapter(image, restored, route_code)
            corrections.append(correction)
            gates.append(gate)
        correction_bank = torch.stack(corrections, dim=1)
        gate_bank = torch.stack(gates, dim=1)
        spatial_weights = weights[:, :, None, None, None]
        correction = availability[:, :, None, None] * (
            correction_bank * spatial_weights
        ).sum(dim=1)
        gate = (gate_bank * spatial_weights).sum(dim=1)
        adapted = torch.clamp(restored + correction, 0.0, 1.0)
        return adapted, weights, correction, gate

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
            route_code,
            image_code,
            sensor_code,
            sensor_physical,
            sensor_direct,
            support,
            disagreement,
        ) = self._sensor_state(image, metadata)
        compound = self._compound_mask(
            route_code, prompt_teacher_weights, prompt_teacher_mask
        )
        use_blend = (
            self.use_compound_blend_gate
            and self.compound_blend_head is not None
            and self.top_k == 2
            and self.backbone_route_mode != "sensor_task"
            and bool(compound.any())
        )
        if use_blend:
            top1 = self._backbone_forward(
                image,
                route_code,
                image_code,
                sensor_code,
                sensor_direct,
                support,
                disagreement,
                prompt_teacher_weights,
                prompt_teacher_mask,
                active_top_k=1,
            )
            top2 = self._backbone_forward(
                image,
                route_code,
                image_code,
                sensor_code,
                sensor_direct,
                support,
                disagreement,
                prompt_teacher_weights,
                prompt_teacher_mask,
                active_top_k=2,
            )
            top1_restored, top1_gate, top1_correction = self._apply_refiner(
                image, top1[0], code
            )
            top2_restored, top2_gate, top2_correction = self._apply_refiner(
                image, top2[0], code
            )
            blend = torch.sigmoid(
                self.compound_blend_head(
                    torch.cat([image_code, sensor_code, disagreement, support], dim=1)
                )
            )
            blend = blend * compound.to(blend.dtype).unsqueeze(1)
            spatial_blend = blend[:, :, None, None]
            restored = top1_restored + spatial_blend * (top2_restored - top1_restored)
            backbone_restored = top1[0] + spatial_blend * (top2[0] - top1[0])
            refiner_gate = top1_gate + spatial_blend * (top2_gate - top1_gate)
            refiner_correction = top1_correction + spatial_blend * (
                top2_correction - top1_correction
            )
            expert_weights = top2[1]
            predicted_expert_weights = top2[2]
            expert_bins = top2[3]
        else:
            active_top_k = (
                2
                if self.backbone_route_mode != "sensor_task"
                and self.top_k == 2
                and bool(compound.any())
                else 1
            )
            (
                backbone_restored,
                expert_weights,
                predicted_expert_weights,
                expert_bins,
            ) = self._backbone_forward(
                image,
                route_code,
                image_code,
                sensor_code,
                sensor_direct,
                support,
                disagreement,
                prompt_teacher_weights,
                prompt_teacher_mask,
                active_top_k=active_top_k,
            )
            restored, refiner_gate, refiner_correction = self._apply_refiner(
                image, backbone_restored, code
            )
            blend = image.new_zeros((image.shape[0], 1))
        metadata_candidate_restored = restored
        fallback_restored = restored
        compatibility = image.new_ones((image.shape[0], 1))
        if bool(compound.any()) and self.compound_metadata_acceptance < 1.0:
            # A compound sensor state can be physically correct while the joint
            # correction is locally incompatible with pavement evidence.  The
            # conservative candidate therefore uses only the observable
            # low-light cause.  It is the same frozen DeMoE expert available to
            # the matched baseline and uses no dataset or scenario identifier.
            fallback_weights = image.new_zeros((image.shape[0], 5))
            fallback_weights[:, 4] = 1.0
            fallback_restored = self._backbone_forward(
                image,
                route_code,
                image_code,
                sensor_code,
                sensor_direct,
                support,
                disagreement,
                None,
                None,
                active_top_k=1,
                expert_weights_override=fallback_weights,
            )[0]
            compatibility = image.new_full(
                (image.shape[0], 1), self.compound_metadata_acceptance
            )
            spatial_mask = compound[:, None, None, None]
            compatible = fallback_restored + self.compound_metadata_acceptance * (
                metadata_candidate_restored - fallback_restored
            )
            restored = torch.where(spatial_mask, compatible, restored)
        # Residual specialists refine the already compatibility-checked base.
        # In particular, compound captures start from the robust low-light
        # expert rather than asking the adapter to undo a weak top-2 blend.
        (
            restored,
            semantic_adapter_weights,
            semantic_adapter_correction,
            semantic_adapter_gate,
        ) = self._apply_semantic_adapters(
            image,
            restored,
            route_code,
            support,
        )
        if not return_dict:
            return restored
        return {
            "restored": restored,
            "input": image,
            "metadata_used": image.new_full(
                (image.shape[0],), 1.0 if metadata is not None else 0.0
            ),
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
            "post_prior_correction": refiner_correction,
            "expert_weights": expert_weights,
            "expert_bin_counts": expert_bins,
            "prompt_weights": expert_weights,
            "predicted_prompt_weights": predicted_expert_weights,
            "compound_mask": compound,
            "compound_blend": blend[:, 0],
            "compound_metadata_acceptance": compatibility[:, 0],
            "cause_route_acceptance": image.new_tensor(
                self.cause_route_acceptance
            ).unsqueeze(0).expand(image.shape[0], -1),
            "cause_refiners_enabled": image.new_full(
                (image.shape[0],), float(self.use_cause_refiners)
            ),
            "backbone_route_mode": self.backbone_route_mode,
            "sensor_task_thresholds": image.new_tensor(
                self.sensor_task_thresholds
            ).unsqueeze(0).expand(image.shape[0], -1),
            "sensor_task_mixed_expert": image.new_full(
                (image.shape[0],),
                -1 if self.sensor_task_mixed_expert is None else self.sensor_task_mixed_expert,
            ),
            "semantic_adapters_enabled": image.new_full(
                (image.shape[0],), float(self.use_semantic_adapters)
            ),
            "semantic_adapter_weights": semantic_adapter_weights,
            "semantic_adapter_gate": semantic_adapter_gate,
            "semantic_adapter_correction": semantic_adapter_correction,
            "semantic_adapter_acceptance": image.new_tensor(
                self.semantic_adapter_acceptance
            ).unsqueeze(0).expand(image.shape[0], -1),
            "metadata_candidate_restored": metadata_candidate_restored,
            "compound_fallback_restored": fallback_restored,
            "phi": None,
            "lambda1": None,
            "lambda2": None,
        }
