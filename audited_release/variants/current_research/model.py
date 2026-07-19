# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""
Revised RCAD-Net / RMR-Net architecture file.

Top-level changes from the old version:
1. Kept the public API stable: same main class names, same default RCADNet
   constructor arguments, same forward inputs, and same default restored-image
   output. Existing training/inference code should continue to call RCADNet in
   the same way.
2. Made conditioning safer: FiLM is now bounded and initialized close to an
   identity transform, so old checkpoints/fine-tuning are less likely to start
   with unstable feature scaling.
3. Strengthened task evidence attention: the edge/contrast/darkness/saturation
   evidence path now uses a learnable cue weighting while preserving the same
   forward(features, image) interface.
4. Replaced plain U-Net additive skip fusion with gated skip fusion inside the
   same Up class. The gate is initialized near-open, so behaviour remains close
   to the old additive skip at the start, but the model can suppress degraded
   skip features during training.
5. Upgraded detail preservation: EvidencePreservingDetailSkip now uses
   multi-scale high-pass detail rather than one average-pool high-pass. This is
   better for cracks, pothole rims, patches, and fine pavement texture while
   still being bounded by max_gain.
6. Improved metadata/image-code fusion: CodeBasisFusion keeps the same forward
   signature, but the reliability fusion now sees metadata presence, code
   disagreement, and compact severity/reliability statistics.  The paper treats
   metadata as a calibrated interface; image-only inference is a supported mode
   when metadata is unavailable or unreliable.
7. Kept the old gate_threshold argument for backwards-compatible experiments
   and conservative deployment-policy checks.
8. Kept TDAC/active-contour outputs optional and unchanged. They remain disabled
   by default and should be treated as ablation/measurement support rather than
   a required inference output.
"""

import torch
from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small tensor utilities
# -----------------------------------------------------------------------------


def _normalize_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each image/map independently to [0, 1]-like scale."""
    denom = x.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
    return x / denom


def _gray(image: torch.Tensor) -> torch.Tensor:
    """Fast grayscale proxy that is safe for BCHW tensors."""
    if image.shape[1] == 1:
        return image
    if image.shape[1] >= 3:
        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return image.mean(dim=1, keepdim=True)


def _grad_mag(gray: torch.Tensor) -> torch.Tensor:
    dx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-6)


def _road_prior_like(image: torch.Tensor) -> torch.Tensor:
    """Soft lower-image prior for vehicle-mounted road imagery."""
    y = torch.linspace(
        0.0,
        1.0,
        image.shape[-2],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, -1, 1)
    return 0.35 + 0.65 * y


def evidence_cues(image: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Return label-free road evidence cues: edge, contrast, dark, saturation.

    Shape: B x 4 x H x W. These cues are not labels and should not be reported
    as segmentation masks. They are only detector-oriented saliency proxies.
    """
    image = image.clamp(0.0, 1.0)
    gray = _gray(image)
    edge = _grad_mag(gray)
    blur = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    contrast = torch.abs(gray - blur)
    dark = (1.0 - gray).clamp(0.0, 1.0)
    saturation = image.amax(dim=1, keepdim=True) - image.amin(dim=1, keepdim=True)
    cues = torch.cat([edge, contrast, dark, saturation], dim=1)
    return _normalize_per_image(cues) if normalize else cues


# -----------------------------------------------------------------------------
# Core blocks
# -----------------------------------------------------------------------------


class DepthwiseSeparableConv(nn.Module):
    """Small edge-friendly convolution block used throughout RCAD-Net.

    The final pointwise convolution is zero-initialized so the block starts as a
    stable residual identity. This helps old checkpoints and short fine-tuning
    runs because newly introduced paths do not immediately perturb features.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.res_scale = nn.Parameter(torch.ones(1) * 0.10)
        self._init_last()

    def _init_last(self) -> None:
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale.to(dtype=x.dtype, device=x.device) * self.net(x)


class FiLM(nn.Module):
    """Feature-wise conditioning from degradation or IMU metadata codes.

    Manuscript Eq.:
        FiLM(x, z) = (1 + gamma(z)) * x + beta(z)

    Revised behaviour:
    - gamma and beta are bounded to avoid exploding modulation;
    - final layer is initialized to zero, so the module starts as identity.
    """

    def __init__(self, code_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.gamma_bound = 0.50
        self.beta_bound = 0.25
        self.proj = nn.Sequential(
            nn.Linear(code_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels * 2),
        )
        self._init_last()

    def _init_last(self) -> None:
        last = self.proj[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(code).chunk(2, dim=1)
        gamma = self.gamma_bound * torch.tanh(gamma)
        beta = self.beta_bound * torch.tanh(beta)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class RCADBlock(nn.Module):
    def __init__(self, channels: int, code_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.film = FiLM(code_dim, channels)
        self.conv = DepthwiseSeparableConv(channels)

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return self.conv(self.film(self.norm(x), code))


class EvidenceConditionedBlock(nn.Module):
    """Conditioned residual block with code-aware channel selection.

    This keeps the same external role as the old block but makes the residual
    update safer: the local branch is residual-scaled and the channel gate sees
    both global feature evidence and the degradation/metadata code.
    """

    def __init__(self, channels: int, code_dim: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(1, channels)
        self.film = FiLM(code_dim, channels)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(channels + code_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.ones(1) * 0.10)
        self._init_last()

    def _init_last(self) -> None:
        last = self.local[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        last_gate = self.channel_gate[-2]
        if isinstance(last_gate, nn.Linear):
            nn.init.zeros_(last_gate.weight)
            nn.init.zeros_(last_gate.bias)

    def forward(self, x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        mixed = self.local(self.film(self.norm(x), code))
        pooled = mixed.mean(dim=(2, 3))
        gate = self.channel_gate(torch.cat([pooled, code], dim=1))[:, :, None, None]
        scale = self.res_scale.to(device=x.device, dtype=x.dtype)
        return x + scale * mixed * gate


# -----------------------------------------------------------------------------
# Evidence attention modules
# -----------------------------------------------------------------------------


class DefectAttention(nn.Module):
    """Highlights crack/pothole/lane-marking edges without requiring labels."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.refine = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        edge = _normalize_per_image(_grad_mag(_gray(image.clamp(0.0, 1.0))))
        attention = self.refine(edge)
        return features * (1.0 + attention)


class TaskEvidenceAttention(nn.Module):
    """Label-free visibility gate for detector-relevant road evidence.

    Revised version: cue importance is learnable instead of being only a fixed
    hand-written sum. This keeps the same forward(features, image) signature but
    lets training adapt the attention to different cameras/degradations.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.refine = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        # Starts close to the old fixed prior: edge > contrast > saturation;
        # dark cue remains available through refine but begins lower.
        prior = torch.tensor([0.55, 0.30, 0.05, 0.15], dtype=torch.float32)
        self.cue_logits = nn.Parameter(torch.log(prior / prior.sum()))
        self.focus_threshold = nn.Parameter(torch.tensor(0.10))
        self.focus_slope = nn.Parameter(torch.tensor(8.0))

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        cues = evidence_cues(image)
        weights = torch.softmax(self.cue_logits.to(device=image.device, dtype=image.dtype), dim=0)
        evidence = (cues * weights.view(1, 4, 1, 1)).sum(dim=1, keepdim=True)
        road_prior = _road_prior_like(image)
        slope = self.focus_slope.to(device=image.device, dtype=image.dtype).clamp(2.0, 16.0)
        threshold = self.focus_threshold.to(device=image.device, dtype=image.dtype).clamp(0.02, 0.35)
        focus = torch.sigmoid(slope * (evidence - threshold)) * road_prior
        attention = self.refine(cues)
        attention = attention * (0.20 + 0.80 * focus)
        return features * (1.0 + attention)


class EvidencePreservingDetailSkip(nn.Module):
    """Learned high-frequency skip for detector-relevant road evidence.

    Revised version:
    - uses multi-scale high-pass detail instead of one high-pass filter;
    - keeps output contract identical: returns restored, gate, detail;
    - remains bounded by max_gain, so it cannot become uncontrolled sharpening.
    """

    def __init__(
        self,
        feature_channels: int,
        *,
        code_dim: int = 8,
        max_gain: float = 0.20,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        hidden = max(feature_channels // 2, 8)
        k = max(int(kernel_size), 3)
        if k % 2 == 0:
            k += 1
        self.kernel_size = k
        self.max_gain = float(max_gain)
        self.gate = nn.Sequential(
            nn.Conv2d(feature_channels + 4, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # 3 high-pass scales x 3 RGB channels + 4 evidence cues -> 3 RGB detail.
        self.detail_fuser = nn.Sequential(
            nn.Conv2d(13, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 1),
        )
        self.code_gate = nn.Sequential(
            nn.Linear(code_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self._init_gate()
        self._init_detail_fuser()
        self._init_code_gate()

    def _init_gate(self) -> None:
        last = self.gate[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -2.0)  # sigmoid(-2) ~= 0.12

    def _init_detail_fuser(self) -> None:
        last = self.detail_fuser[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _init_code_gate(self) -> None:
        last = self.code_gate[-2]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _evidence_cues(self, image: torch.Tensor) -> torch.Tensor:
        return evidence_cues(image)

    @staticmethod
    def _highpass_with_kernel(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        blur = F.avg_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return image - blur

    def _highpass(self, image: torch.Tensor) -> torch.Tensor:
        # Public helper kept for compatibility/debugging: returns the middle scale.
        return self._highpass_with_kernel(image, self.kernel_size)

    def _multi_scale_detail(self, image: torch.Tensor, cues: torch.Tensor) -> torch.Tensor:
        k1 = 3
        k2 = self.kernel_size
        k3 = max(self.kernel_size + 4, 9)
        if k3 % 2 == 0:
            k3 += 1
        d1 = self._highpass_with_kernel(image, k1)
        d2 = self._highpass_with_kernel(image, k2)
        d3 = self._highpass_with_kernel(image, k3)
        raw = torch.cat([d1, d2, d3, cues], dim=1)
        learned = self.detail_fuser(raw)
        # Residual detail starts as zero because detail_fuser last conv is zero.
        # Add a conservative middle-scale fallback so the module is useful even
        # early in fine-tuning.
        return 0.50 * d2 + learned

    def _code_scale(
        self,
        code: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if code is None:
            return torch.ones(batch_size, 1, 1, 1, device=device, dtype=dtype)
        code = code.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        if code.ndim == 1:
            code = code[None, :].expand(batch_size, -1)
        learned = self.code_gate(code).view(batch_size, 1, 1, 1)
        zero = code.new_zeros(batch_size, 1)
        motion = torch.maximum(code[:, 0:1], torch.maximum(code[:, 1:2], code[:, 2:3])) if code.shape[1] > 2 else zero
        defocus = code[:, 3:4] if code.shape[1] > 3 else zero
        noise = code[:, 4:5] if code.shape[1] > 4 else zero
        jpeg = code[:, 6:7] if code.shape[1] > 6 else zero
        severity = code[:, -1:] if code.shape[1] > 0 else zero
        copy_risk = (0.30 * motion) + (0.65 * defocus) + (0.50 * noise) + (0.30 * jpeg) + (0.15 * severity)
        safe_scale = (1.0 - copy_risk).clamp(0.10, 1.0).view(batch_size, 1, 1, 1)
        return learned * safe_scale

    def forward(
        self,
        restored_base: torch.Tensor,
        decoder_features: torch.Tensor,
        image: torch.Tensor,
        code: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cues = self._evidence_cues(image)
        gate = torch.sigmoid(self.gate(torch.cat([decoder_features, cues], dim=1)))
        gate = gate * self._code_scale(code, image.shape[0], image.device, image.dtype)
        detail = self._multi_scale_detail(image.clamp(0.0, 1.0), cues)
        restored = torch.clamp(restored_base + self.max_gain * gate * detail, 0.0, 1.0)
        return restored, gate, detail


class CauseAdaptiveResidualHead(nn.Module):
    """Bounded degradation-specific correction routed by the fused code.

    A shared decoder is retained, but motion, defocus, low-light, compound, and
    noise/compression corrections receive separate residual experts:

        r = [m(1-l), d, l(1-m), ml, max(n,j)],  w = r / sum(r),
        I_b = clip(I_d + h_theta + a sum_c w_c tanh(h_c), 0, 1).

    The experts are zero-initialized and bounded. Loading an older checkpoint
    therefore starts from its original output, while training can teach the
    metadata/image-code fusion to select a cause-appropriate correction.
    """

    def __init__(self, channels: int, *, max_residual: float = 0.20) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.max_residual = float(max_residual)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                    nn.Conv2d(channels, hidden, 1),
                    nn.GELU(),
                    nn.Conv2d(hidden, 3, 1),
                )
                for _ in range(5)
            ]
        )
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    @staticmethod
    def routing_weights(code: torch.Tensor) -> torch.Tensor:
        code = code.clamp(0.0, 1.0)
        zero = code.new_zeros(code.shape[0], 1)
        motion = code[:, :3].amax(dim=1, keepdim=True) if code.shape[1] > 2 else zero
        defocus = code[:, 3:4] if code.shape[1] > 3 else zero
        noise = code[:, 4:5] if code.shape[1] > 4 else zero
        lowlight = code[:, 5:6] if code.shape[1] > 5 else zero
        jpeg = code[:, 6:7] if code.shape[1] > 6 else zero
        severity = code[:, -1:] if code.shape[1] else zero
        scores = torch.cat(
            [
                motion * (1.0 - lowlight),
                defocus,
                lowlight * (1.0 - motion),
                motion * lowlight,
                torch.maximum(noise, jpeg),
            ],
            dim=1,
        )
        fallback = (scores.sum(dim=1, keepdim=True) <= 1e-6).to(code.dtype)
        scores[:, 4:5] = scores[:, 4:5] + fallback * severity.clamp_min(0.05)
        scores = scores + 1e-4
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, features: torch.Tensor, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.routing_weights(code)
        residual = features.new_zeros(features.shape[0], 3, features.shape[2], features.shape[3])
        for index, expert in enumerate(self.experts):
            residual = residual + weights[:, index : index + 1, None, None] * torch.tanh(expert(features))
        severity = code[:, -1:].clamp(0.0, 1.0)[:, :, None, None]
        scale = self.max_residual * (0.25 + 0.75 * severity)
        return scale * residual, weights


class IdentityDefectAttention(nn.Module):
    """Ablation module that keeps the backbone identical except defect gating."""

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return features


# -----------------------------------------------------------------------------
# Degradation code estimation and fusion
# -----------------------------------------------------------------------------


class DegradationEncoder(nn.Module):
    """Predict the eight supervised degradation coordinates from the image."""

    def __init__(self, code_dim: int, width: int) -> None:
        super().__init__()
        hidden = max(width, 16)
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, groups=hidden),
            nn.Conv2d(hidden, hidden * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, stride=2, padding=1, groups=hidden * 2),
            nn.Conv2d(hidden * 2, hidden * 2, 1),
            nn.GELU(),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
            nn.Sigmoid(),
        )

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(image))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(image))


class CodeBasisFusion(nn.Module):
    """Reliability-weighted fusion of metadata and image-estimated codes.

    Forward signature is unchanged: forward(metadata_code, estimated_code) -> z.
    The internal reliability weight is stronger than the old version because it receives
    metadata presence, mean disagreement, and compact severity/reliability stats.
    """

    def __init__(
        self,
        code_dim: int,
        *,
        sparsity_strength: float = 1e-4,
        exact_metadata_mode: bool = False,
    ) -> None:
        super().__init__()
        self.code_dim = int(code_dim)
        self.exact_metadata_mode = bool(exact_metadata_mode)
        self.num_basis_groups = 8
        self.basis_dim = self.code_dim * self.num_basis_groups
        self.sparsity_strength = float(sparsity_strength)
        self.basis_gate_logits = nn.Parameter(torch.full((self.basis_dim,), 1.5))
        self.compound_lowlight_logit = nn.Parameter(torch.tensor(-1.1))
        self.embed = nn.Sequential(
            nn.Linear(self.basis_dim, code_dim * 3),
            nn.GELU(),
            nn.Linear(code_dim * 3, code_dim),
            nn.Sigmoid(),
        )
        # meta, est, abs diff, plus 8 reliability/stat channels.
        self.gate = nn.Sequential(
            nn.Linear(code_dim * 3 + 8, code_dim * 2),
            nn.GELU(),
            nn.Linear(code_dim * 2, code_dim),
            nn.Sigmoid(),
        )
        self.last_alpha: torch.Tensor | None = None
        self.last_reliability_stats: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the complete metadata-fusion module for a new training stage."""

        with torch.no_grad():
            self.basis_gate_logits.fill_(1.5)
            self.compound_lowlight_logit.fill_(-1.1)
        for module in (*self.embed.modules(), *self.gate.modules()):
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self._init_metadata_gate()
        self.last_alpha = None
        self.last_reliability_stats = None

    def _init_metadata_gate(self) -> None:
        """Start from a metadata-aware prior, then let training calibrate it.

        When metadata is supplied, the fusion gate starts with a modest
        metadata preference instead of an arbitrary 0.5 split. Counterfactual
        metadata losses in train_rcadnet.py can then teach the same gate to
        reduce trust for deliberately inconsistent metadata. Image-only
        inference remains a valid operating mode through the estimated code.
        """

        last = self.gate[-2]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, 1.2)  # sigmoid(1.2) ~= 0.77 before the missing-metadata mask.

    def _compound_calibrated_code(self, code: torch.Tensor) -> torch.Tensor:
        """Calibrate low-light evidence when motion is simultaneously present.

        Checkpoints trained before the cause-expert extension learned the
        reliability gate under the following compound-cause transform:

            z'_low = z_low * (1 - q_m + q_m * a_ml),
            q_m = 1[max(z_motion) > 0.05],
            a_ml = 0.25 + 0.75 * sigmoid(r_ml).

        Retaining this transform is required for checkpoint-compatible
        inference. Pure low-light frames are unchanged, while the learned
        scalar controls how much low-light conditioning is applied when motion
        is also present.
        """

        if self.exact_metadata_mode or self.code_dim <= 5:
            return code
        motion_present = (code[:, :3].amax(dim=1, keepdim=True) > 0.05).to(code.dtype)
        attenuation = 0.25 + 0.75 * torch.sigmoid(
            self.compound_lowlight_logit.to(device=code.device, dtype=code.dtype)
        )
        calibrated_low = code[:, 5:6] * (1.0 - motion_present + motion_present * attenuation)
        return torch.cat([code[:, :5], calibrated_low, code[:, 6:]], dim=1)

    def _basis(self, code: torch.Tensor) -> torch.Tensor:
        code = code.clamp(0.0, 1.0)
        severity = code[:, -1:].expand_as(code)
        motion = code[:, :3].amax(dim=1, keepdim=True).expand_as(code)
        low_light = code[:, 5:6].expand_as(code) if self.code_dim > 5 else torch.zeros_like(code)
        defocus = code[:, 3:4].expand_as(code) if self.code_dim > 3 else torch.zeros_like(code)
        noise = code[:, 4:5].expand_as(code) if self.code_dim > 4 else torch.zeros_like(code)
        weak_cue = torch.sqrt(code.clamp_min(1e-6))
        severity_interaction = code * severity
        ambiguity = code * (1.0 - code)
        motion_lowlight = code * motion * low_light
        motion_defocus = code * motion * defocus
        lowlight_noise = code * low_light * noise
        # Manuscript basis g(u):
        # [u, u^2, sqrt(u+eps), u*s, u*(1-u),
        #  u*m*l, u*m*d, u*l*n].
        #
        # The last three groups are non-periodic interaction bases. They are
        # what lets the metadata path represent "motion plus low light" as a
        # distinct cause instead of blindly piling independent FiLM signals.
        # It is intentionally non-periodic because degradation severity is not
        # a cyclic variable; basis_sparsity_loss() keeps these gates sparse.
        basis = torch.cat(
            [
                code,
                code.square(),
                weak_cue,
                severity_interaction,
                ambiguity,
                motion_lowlight,
                motion_defocus,
                lowlight_noise,
            ],
            dim=1,
        )
        sparse_gate = torch.sigmoid(self.basis_gate_logits).view(1, -1).to(device=basis.device, dtype=basis.dtype)
        return basis * sparse_gate

    def basis_sparsity_loss(self) -> torch.Tensor:
        gate = torch.sigmoid(self.basis_gate_logits)
        return self.sparsity_strength * gate.mean()

    def forward(self, metadata_code: torch.Tensor, estimated_code: torch.Tensor) -> torch.Tensor:
        metadata_code = self._compound_calibrated_code(metadata_code.clamp(0.0, 1.0))
        estimated_code = self._compound_calibrated_code(estimated_code.clamp(0.0, 1.0))
        meta_has_signal = (metadata_code.abs().sum(dim=1, keepdim=True) > 1e-6).to(metadata_code.dtype)
        meta = self.embed(self._basis(metadata_code))
        est = self.embed(self._basis(estimated_code))
        disagreement = torch.abs(meta - est)
        mean_disagreement = disagreement.mean(dim=1, keepdim=True)
        meta_severity = metadata_code[:, -1:].clamp(0.0, 1.0)
        est_severity = estimated_code[:, -1:].clamp(0.0, 1.0)
        severity_gap = torch.abs(meta_severity - est_severity)
        meta_motion = metadata_code[:, :3].amax(dim=1, keepdim=True)
        est_motion = estimated_code[:, :3].amax(dim=1, keepdim=True)
        meta_low = metadata_code[:, 5:6] if metadata_code.shape[1] > 5 else torch.zeros_like(meta_motion)
        est_low = estimated_code[:, 5:6] if estimated_code.shape[1] > 5 else torch.zeros_like(est_motion)
        meta_compound = meta_motion * meta_low
        est_compound = est_motion * est_low
        compound_gap = torch.abs(meta_compound - est_compound)
        reliability_aux = (
            torch.abs(metadata_code - estimated_code).mean(dim=1, keepdim=True)
            if self.exact_metadata_mode
            else (metadata_code[:, :-1] > 0.05).to(metadata_code.dtype).mean(dim=1, keepdim=True)
        )
        reliability_stats = torch.cat(
            [
                meta_has_signal,
                mean_disagreement,
                meta_severity,
                severity_gap,
                meta_compound,
                est_compound,
                compound_gap,
                reliability_aux,
            ],
            dim=1,
        )
        gate = self.gate(torch.cat([meta, est, disagreement, reliability_stats], dim=1))
        # Manuscript fusion:
        #   z = alpha(z_m, z_hat) g(z_m) + (1-alpha(z_m, z_hat)) g(z_hat).
        # Manuscript availability constraint: alpha = 0 when no metadata are
        # present.  A residual nonzero gate here would mix the learned embedding
        # of an all-zero record into z and would undermine metadata-dropout
        # training.  With the exact mask, missing records use only the supervised
        # image-estimated degradation code.
        gate = gate * meta_has_signal
        self.last_alpha = gate
        self.last_reliability_stats = reliability_stats
        return torch.clamp(gate * meta + (1.0 - gate) * est, 0.0, 1.0)

    def gate_summary(self, *, detach: bool = True) -> dict[str, torch.Tensor] | None:
        """Return the latest metadata-reliability gate for audit logging.

        During training, ``detach=False`` exposes the gate to the metadata
        calibration objective:

            L_gate = ||alpha(z_m, z_hat) - alpha^+||_2^2
                   + ||alpha(z_m^-, z_hat) - alpha^-||_2^2.

        For validation, ablation, and reviewer-facing diagnostics the tensors
        are detached because alpha is an audit signal, not a deployed mask.
        """
        if self.last_alpha is None:
            return None
        alpha = self.last_alpha.detach() if detach else self.last_alpha
        stats = {
            "metadata_alpha": alpha,
            "metadata_alpha_mean": alpha.mean(dim=1),
            "metadata_alpha_min": alpha.amin(dim=1),
            "metadata_alpha_max": alpha.amax(dim=1),
        }
        if self.last_reliability_stats is not None:
            rel = self.last_reliability_stats.detach() if detach else self.last_reliability_stats
            stats.update(
                {
                    "metadata_has_signal": rel[:, 0],
                    "metadata_disagreement": rel[:, 1],
                    "metadata_severity": rel[:, 2],
                    "metadata_severity_gap": rel[:, 3],
                    "metadata_compound_strength": rel[:, 4],
                    "metadata_estimated_compound": rel[:, 5],
                    "metadata_compound_gap": rel[:, 6],
                    (
                        "metadata_raw_disagreement"
                        if self.exact_metadata_mode
                        else "metadata_active_fraction"
                    ): rel[:, 7],
                }
            )
        return stats


class ResidualCodeBasisFusion(CodeBasisFusion):
    """Metadata correction that preserves the image-derived code at startup.

    The legacy basis fuser maps both inputs into a learned latent code before
    interpolation. That is useful when the complete model is trained jointly,
    but it changes the conditioning distribution abruptly when a reliable
    image-only restorer is converted to metadata-conditioned operation.

    This release path instead implements

        z = z_hat + alpha(z_m, z_hat) * (z_m - z_hat),

    where ``z_hat`` is the supervised image-derived degradation code and
    ``z_m`` is the aligned metadata code. The nonlinear, sparsified basis is
    retained only for estimating the per-coordinate reliability ``alpha``.
    A low initial alpha makes transfer from an image-only checkpoint continuous;
    train/validation evidence must then earn a larger metadata correction.
    Missing metadata gives alpha=0 exactly.
    """

    def _init_metadata_gate(self) -> None:
        last = self.gate[-2]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -2.2)  # sigmoid(-2.2) ~= 0.10.

    def forward(self, metadata_code: torch.Tensor, estimated_code: torch.Tensor) -> torch.Tensor:
        metadata_code = self._compound_calibrated_code(metadata_code.clamp(0.0, 1.0))
        estimated_code = self._compound_calibrated_code(estimated_code.clamp(0.0, 1.0))
        meta_has_signal = (metadata_code.abs().sum(dim=1, keepdim=True) > 1e-6).to(metadata_code.dtype)

        meta_features = self.embed(self._basis(metadata_code))
        estimated_features = self.embed(self._basis(estimated_code))
        disagreement = torch.abs(meta_features - estimated_features)
        mean_disagreement = disagreement.mean(dim=1, keepdim=True)

        meta_severity = metadata_code[:, -1:].clamp(0.0, 1.0)
        estimated_severity = estimated_code[:, -1:].clamp(0.0, 1.0)
        severity_gap = torch.abs(meta_severity - estimated_severity)
        meta_motion = metadata_code[:, :3].amax(dim=1, keepdim=True)
        estimated_motion = estimated_code[:, :3].amax(dim=1, keepdim=True)
        meta_low = metadata_code[:, 5:6] if metadata_code.shape[1] > 5 else torch.zeros_like(meta_motion)
        estimated_low = (
            estimated_code[:, 5:6]
            if estimated_code.shape[1] > 5
            else torch.zeros_like(estimated_motion)
        )
        meta_compound = meta_motion * meta_low
        estimated_compound = estimated_motion * estimated_low
        compound_gap = torch.abs(meta_compound - estimated_compound)
        reliability_aux = (
            torch.abs(metadata_code - estimated_code).mean(dim=1, keepdim=True)
            if self.exact_metadata_mode
            else (metadata_code[:, :-1] > 0.05).to(metadata_code.dtype).mean(dim=1, keepdim=True)
        )
        reliability_stats = torch.cat(
            [
                meta_has_signal,
                mean_disagreement,
                meta_severity,
                severity_gap,
                meta_compound,
                estimated_compound,
                compound_gap,
                reliability_aux,
            ],
            dim=1,
        )

        alpha = self.gate(
            torch.cat([meta_features, estimated_features, disagreement, reliability_stats], dim=1)
        )
        alpha = alpha * meta_has_signal
        self.last_alpha = alpha
        self.last_reliability_stats = reliability_stats

        fused = estimated_code + alpha * (metadata_code - estimated_code)
        return fused.clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# U-Net resolution changes
# -----------------------------------------------------------------------------


class Down(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    """Upsampling block with a near-open learned skip gate.

    The class name is unchanged. Existing calls ``up(x, skip)`` still work.
    RCADNet now calls ``up(x, skip, code)`` so the skip can be code-aware.
    """

    def __init__(self, channels: int, code_dim: int | None = None) -> None:
        super().__init__()
        out_channels = channels // 2
        self.proj = nn.Conv2d(channels, out_channels, 1)
        self.skip_gate = nn.Conv2d(out_channels, out_channels, 1)
        self.code_proj = nn.Linear(code_dim, out_channels) if code_dim is not None else None
        self._init_gate()

    def _init_gate(self) -> None:
        nn.init.zeros_(self.skip_gate.weight)
        nn.init.constant_(self.skip_gate.bias, 2.0)  # sigmoid(2) ~= 0.88, near old additive skip.
        if self.code_proj is not None:
            nn.init.zeros_(self.code_proj.weight)
            nn.init.zeros_(self.code_proj.bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, code: torch.Tensor | None = None) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        gate_logits = self.skip_gate(skip)
        if code is not None and self.code_proj is not None:
            code_bias = self.code_proj(code).to(device=skip.device, dtype=skip.dtype)[:, :, None, None]
            gate_logits = gate_logits + code_bias
        gate = torch.sigmoid(gate_logits)
        return x + gate * skip


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------


class RCADNet(nn.Module):
    """Road-Context Adaptive Defect-Preserving restoration network."""

    def __init__(
        self,
        width: int = 32,
        code_dim: int = 8,
        blocks_per_stage: int = 2,
        use_defect_attention: bool = True,
        use_estimated_code: bool = False,
        code_fusion: str = "scenario",
        block_type: str = "simple",
        attention_type: str = "edge",
        conditioning: str = "film",
        use_tdac_head: bool = False,
        detail_preserve: bool = False,
        detail_gain: float = 0.20,
        use_cause_experts: bool = False,
        cause_expert_gain: float = 0.20,
        exact_metadata_mode: bool | None = None,
    ) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.use_estimated_code = use_estimated_code
        self.code_fusion = code_fusion
        self.block_type = block_type
        self.attention_type = attention_type
        self.conditioning = conditioning
        self.use_tdac_head = use_tdac_head
        self.detail_preserve = bool(detail_preserve)
        self.detail_gain = float(detail_gain)
        self.use_cause_experts = bool(use_cause_experts)
        self.cause_expert_gain = float(cause_expert_gain)
        self.exact_metadata_mode = (
            self.use_cause_experts if exact_metadata_mode is None else bool(exact_metadata_mode)
        )

        self.stem = nn.Conv2d(3, width, 3, padding=1)
        if not use_defect_attention or attention_type == "none":
            self.defect_attention = IdentityDefectAttention()
        elif attention_type == "task":
            self.defect_attention = TaskEvidenceAttention(width)
        else:
            self.defect_attention = DefectAttention(width)

        self.code_encoder = DegradationEncoder(code_dim, width) if use_estimated_code else None
        if conditioning == "gated_basis":
            self.code_fuser = CodeBasisFusion(code_dim, exact_metadata_mode=self.exact_metadata_mode)
        elif conditioning == "residual_basis":
            self.code_fuser = ResidualCodeBasisFusion(code_dim, exact_metadata_mode=self.exact_metadata_mode)
        else:
            self.code_fuser = None

        self.enc1 = self._make_blocks(width, code_dim, blocks_per_stage)
        self.down1 = Down(width)
        self.enc2 = self._make_blocks(width * 2, code_dim, blocks_per_stage)
        self.down2 = Down(width * 2)
        self.mid = self._make_blocks(width * 4, code_dim, blocks_per_stage + 1)
        self.up2 = Up(width * 4, code_dim)
        self.dec2 = self._make_blocks(width * 2, code_dim, blocks_per_stage)
        self.up1 = Up(width * 2, code_dim)
        self.dec1 = self._make_blocks(width, code_dim, blocks_per_stage)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        self._init_head()
        self.cause_head = (
            CauseAdaptiveResidualHead(width, max_residual=self.cause_expert_gain)
            if self.use_cause_experts
            else None
        )
        self.last_cause_route: torch.Tensor | None = None
        self.last_cause_residual: torch.Tensor | None = None

        self.detail_skip = (
            EvidencePreservingDetailSkip(width, code_dim=code_dim, max_gain=self.detail_gain)
            if self.detail_preserve
            else None
        )
        self.tdac_head = nn.Conv2d(width, 3, 3, padding=1) if use_tdac_head else None

    def _init_head(self) -> None:
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _make_blocks(self, channels: int, code_dim: int, count: int) -> nn.ModuleList:
        block_cls = EvidenceConditionedBlock if self.block_type == "evidence" else RCADBlock
        return nn.ModuleList([block_cls(channels, code_dim) for _ in range(count)])

    def _run_blocks(self, x: torch.Tensor, code: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        for block in blocks:
            x = block(x, code)
        return x

    def estimate_code(self, image: torch.Tensor) -> torch.Tensor:
        if self.code_encoder is None:
            return torch.zeros(image.shape[0], self.code_dim, device=image.device, dtype=image.dtype)
        return self.code_encoder(image)

    def basis_sparsity_loss(self) -> torch.Tensor:
        if self.code_fuser is None:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        return self.code_fuser.basis_sparsity_loss()

    def reset_code_fuser(self) -> None:
        """Reinitialize metadata fusion while preserving the trained restorer."""

        if self.code_fuser is None:
            raise RuntimeError("Cannot reset metadata fusion because this model has no code fuser.")
        self.code_fuser.reset_parameters()

    def load_pretrained(
        self,
        checkpoint_or_state: dict[str, torch.Tensor] | dict[str, object],
        *,
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """Load old/new RMR-Net checkpoints while skipping incompatible tensors.

        PyTorch's ``load_state_dict(..., strict=False)`` still raises on same-name
        tensors with different shapes. The revised metadata encoder and basis
        fuser intentionally add a few parameters, so this helper implements the
        compatibility policy used in the paper:

            theta <- compatible(theta_old),  theta_new modules stay initialized.

        Returns a small audit dictionary so training/evaluation scripts can log
        exactly which tensors were loaded or skipped.
        """

        state = checkpoint_or_state.get("model", checkpoint_or_state)  # type: ignore[assignment]
        if not isinstance(state, dict):
            raise TypeError("load_pretrained expects a checkpoint dict or state_dict.")

        current = self.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        skipped_shape: list[str] = []
        skipped_missing: list[str] = []
        for name, value in state.items():  # type: ignore[union-attr]
            if name not in current:
                skipped_missing.append(str(name))
                continue
            if tuple(current[name].shape) != tuple(value.shape):
                skipped_shape.append(str(name))
                continue
            compatible[str(name)] = value

        incompatible = self.load_state_dict(compatible, strict=False)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys or skipped_shape or skipped_missing):
            raise RuntimeError(
                "Strict pretrained load failed: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}, "
                f"skipped_shape={skipped_shape}, skipped_missing={skipped_missing}"
            )

        return {
            "loaded": sorted(compatible.keys()),
            "missing_keys": sorted(str(x) for x in incompatible.missing_keys),
            "unexpected_keys": sorted(str(x) for x in incompatible.unexpected_keys),
            "skipped_shape": sorted(skipped_shape),
            "skipped_missing": sorted(skipped_missing),
        }

    def _prepare_code(self, image: torch.Tensor, code: torch.Tensor | None) -> torch.Tensor:
        estimated = self.estimate_code(image) if self.use_estimated_code else None
        # Ablation-only isolation mode. The image degradation encoder can still
        # be supervised through L_code, but its prediction is not injected into
        # the restoration backbone until the subsequent FiLM ablation stage.
        # Normal RMR-Net checkpoints use ``film`` or ``gated_basis``.
        if self.conditioning == "none":
            return torch.zeros(image.shape[0], self.code_dim, device=image.device, dtype=image.dtype)
        if code is None:
            if estimated is not None:
                return estimated
            return torch.zeros(image.shape[0], self.code_dim, device=image.device, dtype=image.dtype)

        if code.ndim == 1:
            code = code[None, :].expand(image.shape[0], -1)
        code = code.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)

        # Preserve old behaviour for scenario/metadata/estimated/average modes.
        # Basis-conditioned modes activate either legacy latent interpolation or
        # the residual metadata correction used by the strict two-stage run.
        if self.code_fuser is not None:
            if estimated is None:
                estimated = torch.zeros_like(code)
            return self.code_fuser(code, estimated)
        if estimated is None or self.code_fusion in {"scenario", "metadata"}:
            return code
        if self.code_fusion == "estimated":
            return estimated
        return torch.clamp(0.5 * code + 0.5 * estimated, 0.0, 1.0)

    def _decode(
        self,
        image: torch.Tensor,
        code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        x1 = self.defect_attention(self.stem(image), image)
        x1 = self._run_blocks(x1, code, self.enc1)
        x2 = self._run_blocks(self.down1(x1), code, self.enc2)
        x3 = self._run_blocks(self.down2(x2), code, self.mid)
        y = self._run_blocks(self.up2(x3, x2, code), code, self.dec2)
        y = self._run_blocks(self.up1(y, x1, code), code, self.dec1)

        # Base residual branch: I_b = clip(I_d + h_theta(I_d, z), 0, 1).
        residual = self.head(y)
        self.last_cause_route = None
        self.last_cause_residual = None
        if self.cause_head is not None:
            cause_residual, cause_route = self.cause_head(y, code)
            residual = residual + cause_residual
            self.last_cause_route = cause_route
            self.last_cause_residual = cause_residual
        restored_base = torch.clamp(image + residual, 0.0, 1.0)

        detail_gate = None
        detail_residual = None
        if self.detail_skip is not None:
            restored, detail_gate, detail_residual = self.detail_skip(restored_base, y, image, code)
        else:
            restored = restored_base

        aux = self.tdac_head(y) if self.tdac_head is not None else None
        return restored, aux, detail_gate, detail_residual

    @staticmethod
    def unpack_tdac_aux(aux: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
        """Convert raw TDAC channels to named differentiable maps."""
        if aux is None:
            return {"phi": None, "lambda1": None, "lambda2": None}
        if aux.shape[1] < 3:
            raise ValueError("TDAC auxiliary head must output phi, lambda1 and lambda2 channels")
        return {
            "phi": torch.tanh(aux[:, 0:1]),
            "lambda1": torch.sigmoid(aux[:, 1:2]) * 4.9 + 0.1,
            "lambda2": torch.sigmoid(aux[:, 2:3]) * 4.9 + 0.1,
        }

    @staticmethod
    def _image_gate_score(image: torch.Tensor) -> torch.Tensor:
        """Low-cost evidence/quality score for deployment gating.

        Higher score means the image likely contains visible road evidence and
        should tolerate/use restoration. Lower score biases toward identity.
        """
        cues = evidence_cues(image)
        edge = cues[:, 0:1]
        contrast = cues[:, 1:2]
        saturation = cues[:, 3:4]
        road_prior = _road_prior_like(image)
        score_map = (0.55 * edge + 0.35 * contrast + 0.10 * saturation) * road_prior
        return score_map.flatten(1).mean(dim=1).clamp(0.0, 1.0)

    def forward(
        self,
        image: torch.Tensor,
        code: torch.Tensor | None = None,
        *,
        return_aux: bool = False,
        return_dict: bool = False,
        return_tuple: bool = False,
        gate_threshold: float | None = None,
        gate_softness: float = 0.03,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor] | dict[str, torch.Tensor | None]:
        metadata_used = code is not None
        code = self._prepare_code(image, code)

        if gate_threshold is not None and gate_softness <= 0 and not return_aux:
            severity = code[:, -1].clamp(0.0, 1.0)
            evidence_score = self._image_gate_score(image)
            combined = torch.maximum(severity, evidence_score)
            if bool((combined < gate_threshold).all().detach().cpu()):
                return image

        restored, aux, detail_gate, detail_residual = self._decode(image, code)
        gate = None
        if gate_threshold is not None:
            severity = code[:, -1].clamp(0.0, 1.0)
            evidence_score = self._image_gate_score(image)
            # Keep severity as the dominant deployment signal, but let image
            # evidence rescue cases where metadata underestimates degradation.
            gate_signal = torch.maximum(severity, 0.75 * severity + 0.25 * evidence_score)
            if gate_softness <= 0:
                gate = (gate_signal >= gate_threshold).to(image.dtype)
            else:
                gate = torch.sigmoid((gate_signal - gate_threshold) / gate_softness).to(image.dtype)
            gate = gate[:, None, None, None]
            restored = image + gate * (restored - image)
            restored = torch.clamp(restored, 0.0, 1.0)

        if not return_aux and not return_dict and not return_tuple:
            return restored

        tdac_maps = self.unpack_tdac_aux(aux)
        if return_tuple:
            return restored, tdac_maps["phi"], tdac_maps["lambda1"], tdac_maps["lambda2"], code[:, -1]

        output = {
            "restored": restored,
            "input": image,
            "metadata_used": image.new_full((image.shape[0],), 1.0 if metadata_used else 0.0),
            "aux": aux,
            "phi": tdac_maps["phi"],
            "lambda1": tdac_maps["lambda1"],
            "lambda2": tdac_maps["lambda2"],
            "code": code,
            "severity": code[:, -1],
            "z_severity": code[:, -1],
            "gate": gate,
            "detail_gate": detail_gate,
            "detail_residual": detail_residual,
            "basis_sparsity": self.basis_sparsity_loss() if self.code_fuser is not None else None,
            "cause_route": self.last_cause_route,
            "cause_residual": self.last_cause_residual,
        }
        if self.code_fuser is not None:
            gate_summary = self.code_fuser.gate_summary(detach=not self.training)
            if gate_summary is not None:
                output.update(gate_summary)
        return output
