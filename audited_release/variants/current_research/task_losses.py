# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""
Revised task-loss module for RMR-Net / RCAD-Net.

Top-level changes from the old version:
1. Kept the public API stable: same function/class names, same default call
   pattern for FrozenDetectorFeatureExtractor, TaskDrivenPerceptualLoss,
   ActiveContourGeometryLoss, DetectorInputAnchorLoss, TaskLossWeights, and
   CompositeTaskLoss. Existing training code should continue to import and call
   these objects in the same way.
2. Made detector-feature losses numerically safer: feature-map alignment is now
   centralized, optional feature normalization prevents one detector layer from
   dominating, and all reductions use stable denominators.
3. Strengthened CQMix while keeping the same function signature: patch masks are
   generated per image, support train/eval behaviour naturally, and preserve the
   restored path as the differentiable branch.
4. Improved label-free road evidence: the visibility mask now combines edge,
   local contrast, high-frequency detail, darkness, saturation, and a lower-road
   prior while remaining detached so it never becomes a learned pseudo-label.
5. Added detector-safe regularizers without breaking old configs: basis sparsity,
   low-evidence identity, and restoration-magnitude penalties are optional fields
   appended to TaskLossWeights with zero defaults.
6. Made active-contour loss safer for ablations: phi/lambda maps are resized and
   bounded robustly, regional terms use stop-gradient image statistics, and an
   area-balance term is available but default-light.
7. Added richer logging in CompositeTaskLoss: raw/weighted terms, evidence
   statistics, optional gate/detail statistics, and final task warmup scale.
8. Preserved the final-paper assumption that active_contour is disabled by
   default. Active contours should remain post-detection measurement unless an
   explicit ablation enables this term.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Basic image utilities
# ---------------------------------------------------------------------


def _safe_odd_kernel(kernel_size: int, minimum: int = 3) -> int:
    k = max(int(kernel_size), int(minimum))
    if k % 2 == 0:
        k += 1
    return k


def _safe_amax_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image max normalization for BCHW tensors."""
    return x / (x.amax(dim=(2, 3), keepdim=True).clamp_min(eps))


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}.")

    if x.shape[1] == 1:
        return x

    if x.shape[1] < 3:
        return x.mean(dim=1, keepdim=True)

    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def image_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]

    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))

    return dx, dy


def gradient_magnitude(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dx, dy = image_gradients(x)
    return torch.sqrt(dx.pow(2) + dy.pow(2) + eps)


def resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[-2:] == target.shape[-2:]:
        return source

    return F.interpolate(
        source,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _align_feature_pair(
    a: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align detector feature tensors by spatial size and channel count."""
    if a.shape[-2:] != b.shape[-2:]:
        b = resize_like(b, a)
    if a.shape[1] != b.shape[1]:
        c = min(a.shape[1], b.shape[1])
        a = a[:, :c]
        b = b[:, :c]
    return a, b


def _normalize_feature(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample, per-layer feature normalization for stable detector losses."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return (x - mean) / std


# ---------------------------------------------------------------------
# Cross-Quality Patch Mix
# ---------------------------------------------------------------------


def cross_quality_patch_mix(
    restored: torch.Tensor,
    clean: torch.Tensor,
    *,
    grid: int = 4,
    prob: float = 0.5,
) -> torch.Tensor:
    """
    Mix restored and clean patches before detector-feature loss.

    Train-time only. It reduces the risk that the restorer learns coherent
    detector-feature artifacts that satisfy the feature loss but hurt held-out
    detection.

    Manuscript Eq. (CQMix):
        I_mix = M * I_r + (1-M) * I_c
    """
    if restored.shape != clean.shape:
        raise ValueError(
            f"restored and clean must have the same shape. "
            f"Got {tuple(restored.shape)} and {tuple(clean.shape)}."
        )

    if prob <= 0.0:
        return restored

    if torch.rand((), device=restored.device).item() > float(prob):
        return restored

    b, _, h, w = restored.shape
    grid = max(int(grid), 1)

    small_mask = torch.randint(
        low=0,
        high=2,
        size=(b, 1, grid, grid),
        device=restored.device,
        dtype=restored.dtype,
    )

    # Avoid degenerate all-clean/all-restored masks for tiny batches/grids.
    if grid > 1:
        flat = small_mask.flatten(1)
        all_zero = flat.sum(dim=1) == 0
        all_one = flat.sum(dim=1) == flat.shape[1]
        if all_zero.any():
            flat[all_zero, 0] = 1.0
        if all_one.any():
            flat[all_one, -1] = 0.0
        small_mask = flat.view(b, 1, grid, grid)

    mask = F.interpolate(small_mask, size=(h, w), mode="nearest")
    return mask * restored + (1.0 - mask) * clean.detach()


# ---------------------------------------------------------------------
# Detector feature extraction
# ---------------------------------------------------------------------


def unwrap_detector(detector: nn.Module) -> nn.Module:
    """
    Best-effort unwrapping for Ultralytics-like detector wrappers.

    Some YOLO objects expose the actual torch model through `.model`.
    We avoid importing Ultralytics here to keep this file dependency-light.
    """
    if hasattr(detector, "model") and isinstance(getattr(detector, "model"), nn.Module):
        return getattr(detector, "model")
    return detector


def _is_reasonable_hook_module(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()

    if isinstance(module, nn.Conv2d):
        return True

    likely = ("conv", "c2f", "c3", "bottleneck", "spp", "sppf", "elan", "block")
    if any(token in name for token in likely):
        if len(list(module.children())) > 0:
            return True

    return False


class FrozenDetectorFeatureExtractor(nn.Module):
    """
    Frozen detector feature extractor using forward hooks.

    The detector is kept in eval mode and its parameters remain frozen. The
    restored/mixed image path remains differentiable back to the restorer.
    """

    def __init__(
        self,
        detector: nn.Module,
        layer_names: Optional[Sequence[str]] = None,
        *,
        max_layers: int = 3,
        input_size: Optional[Tuple[int, int]] = None,
        normalize_imagenet: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.detector = unwrap_detector(detector).eval()
        self.input_size = input_size
        self.normalize_imagenet = bool(normalize_imagenet)
        self.verbose = bool(verbose)

        for param in self.detector.parameters():
            param.requires_grad_(False)

        self.features: Dict[str, torch.Tensor] = {}
        self.handles = []

        named_modules = dict(self.detector.named_modules())

        if layer_names is None:
            candidates = [
                name
                for name, module in named_modules.items()
                if name and _is_reasonable_hook_module(module)
            ]

            if len(candidates) == 0:
                candidates = [
                    name
                    for name, module in named_modules.items()
                    if name and not isinstance(module, (nn.Sequential, nn.ModuleList))
                ]

            if len(candidates) == 0:
                raise RuntimeError("No candidate detector hook layers found.")

            deeper = candidates[len(candidates) // 2 :]
            max_layers = max(int(max_layers), 1)
            if len(deeper) <= max_layers:
                layer_names = deeper
            else:
                # Uniformly sample deeper layers to avoid over-concentrating all
                # hooks in adjacent blocks.
                idx = torch.linspace(0, len(deeper) - 1, steps=max_layers).round().long().tolist()
                layer_names = [deeper[i] for i in idx]

        missing = [name for name in layer_names if name not in named_modules]
        if missing:
            examples = list(named_modules.keys())[:50]
            raise ValueError(
                f"Requested hook layers not found: {missing}. "
                f"Available examples: {examples}"
            )

        self.layer_names = list(layer_names)

        for name in self.layer_names:
            module = named_modules[name]
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

        if self.verbose:
            print(f"[TaskLoss] Hooked detector layers: {self.layer_names}")

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs, output):
            tensor = self._first_tensor(output)
            if tensor is not None:
                self.features[name] = tensor

        return hook

    @staticmethod
    def _first_tensor(output) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (list, tuple)):
            for item in output:
                found = FrozenDetectorFeatureExtractor._first_tensor(item)
                if found is not None:
                    return found

        if isinstance(output, dict):
            for item in output.values():
                found = FrozenDetectorFeatureExtractor._first_tensor(item)
                if found is not None:
                    return found

        return None

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clamp(0.0, 1.0)

        if self.input_size is not None and y.shape[-2:] != self.input_size:
            y = F.interpolate(
                y,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        if self.normalize_imagenet:
            mean = torch.tensor(
                [0.485, 0.456, 0.406],
                device=y.device,
                dtype=y.dtype,
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                [0.229, 0.224, 0.225],
                device=y.device,
                dtype=y.dtype,
            ).view(1, 3, 1, 1)
            y = (y - mean) / std

        return y

    def clear(self) -> None:
        self.features = {}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.clear()
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad_(False)

        prepared = self._prepare_input(x)

        try:
            _ = self.detector(prepared)
        except Exception as exc:
            raise RuntimeError(
                "Frozen detector forward failed inside task loss. "
                "Check detector object, input size, and normalization."
            ) from exc

        if len(self.features) == 0:
            raise RuntimeError(
                "No detector features were captured. "
                "Check hook layer names or detector wrapper."
            )

        # Preserve insertion order of hooks.
        return {name: self.features[name] for name in self.layer_names if name in self.features}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def train(self, mode: bool = True) -> "FrozenDetectorFeatureExtractor":
        """Keep the detector frozen/eval even when parent losses enter train mode."""

        super().train(False)
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad_(False)
        return self


# ---------------------------------------------------------------------
# Task-driven perceptual loss
# ---------------------------------------------------------------------


class TaskDrivenPerceptualLoss(nn.Module):
    """
    Detector-feature perceptual loss.

    Manuscript Eq. (defect-weighted TDP):
        L_TDP = sum_k alpha_k mean((1 + rho R_k(M_c))
                 * (Phi_k(I_r or I_mix) - Phi_k(I_c))^2)

    The clean detector-feature path is computed under no_grad; the restored or
    mixed path remains differentiable back to RMR-Net.
    """

    def __init__(
        self,
        feature_extractor: FrozenDetectorFeatureExtractor,
        *,
        layer_weights: Optional[Dict[str, float]] = None,
        cqmix_grid: int = 4,
        cqmix_prob: float = 0.5,
        defect_mask_weight: float = 0.0,
        defect_mask_power: float = 1.0,
        normalize_features: bool = True,
        robust_charbonnier: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights or {}
        self.cqmix_grid = int(cqmix_grid)
        self.cqmix_prob = float(cqmix_prob)
        self.defect_mask_weight = float(defect_mask_weight)
        self.defect_mask_power = float(defect_mask_power)
        self.normalize_features = bool(normalize_features)
        self.robust_charbonnier = bool(robust_charbonnier)
        self.eps = float(eps)

    def _feature_distance(self, feat_r: torch.Tensor, feat_c: torch.Tensor) -> torch.Tensor:
        if self.normalize_features:
            feat_r = _normalize_feature(feat_r, eps=self.eps)
            feat_c = _normalize_feature(feat_c, eps=self.eps)
        diff = feat_r - feat_c
        if self.robust_charbonnier:
            return torch.sqrt(diff.pow(2) + self.eps * self.eps)
        return diff.pow(2)

    def forward(
        self,
        restored: torch.Tensor,
        clean: torch.Tensor,
        *,
        defect_mask: Optional[torch.Tensor] = None,
        use_cqmix: bool = True,
    ) -> torch.Tensor:
        if use_cqmix:
            detector_input = cross_quality_patch_mix(
                restored,
                clean,
                grid=self.cqmix_grid,
                prob=self.cqmix_prob,
            )
        else:
            detector_input = restored

        restored_features = self.feature_extractor(detector_input)

        with torch.no_grad():
            clean_features = self.feature_extractor(clean)

        total = restored.new_tensor(0.0)
        weight_total = 0.0

        for name, feat_r in restored_features.items():
            if name not in clean_features:
                continue

            feat_c = clean_features[name].detach()
            feat_r, feat_c = _align_feature_pair(feat_r, feat_c)

            weight = float(self.layer_weights.get(name, 1.0))
            if weight <= 0:
                continue

            diff = self._feature_distance(feat_r, feat_c)

            if self.defect_mask_weight > 0:
                if defect_mask is None:
                    defect_mask = road_defect_visibility_mask(
                        clean,
                        power=self.defect_mask_power,
                    )
                mask = resize_like(defect_mask, diff[:, :1]).clamp(0.0, 1.0)
                feature_weight = 1.0 + self.defect_mask_weight * mask
                layer_loss = (diff * feature_weight).mean() / feature_weight.mean().clamp_min(self.eps)
            else:
                layer_loss = diff.mean()

            total = total + weight * layer_loss
            weight_total += weight

        if weight_total <= 0:
            raise RuntimeError("No matching detector features found for TDP loss.")

        return total / float(weight_total)


# ---------------------------------------------------------------------
# Hutchinson Jacobian penalty
# ---------------------------------------------------------------------


def hutchinson_jacobian_penalty(
    feature_extractor: FrozenDetectorFeatureExtractor,
    restored: torch.Tensor,
    *,
    num_probes: int = 1,
) -> torch.Tensor:
    """
    Estimate detector-feature sensitivity to restored image.

    Manuscript Eq. (normalized Hutchinson Jacobian):
        L_J = E_v mean(|grad_{I_r} <Phi(I_r), v / sqrt(|Phi|)>|^2)

    Dividing a feature projection by ``|Phi|`` and then averaging the input
    gradient suppresses the estimate twice; its magnitude approaches zero as
    detector feature maps grow.  The Rademacher vector is therefore normalized
    by ``sqrt(|Phi|)``.  This retains a resolution-independent mean-sensitivity
    penalty while preserving the Hutchinson identity up to the explicitly
    reported per-feature/per-input normalization.

    Keep num_probes small. This term is intentionally expensive.
    """
    if num_probes <= 0:
        return restored.new_tensor(0.0)

    x = restored
    if not x.requires_grad:
        x = restored.detach().requires_grad_(True)
    else:
        x = restored.requires_grad_(True)

    total = restored.new_tensor(0.0)

    for _ in range(int(num_probes)):
        features = feature_extractor(x)

        projection = restored.new_tensor(0.0)
        for feat in features.values():
            v = torch.empty_like(feat).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            projection = projection + (feat * v).sum() / float(max(feat.numel(), 1) ** 0.5)

        grad = torch.autograd.grad(
            projection,
            x,
            create_graph=restored.requires_grad,
            retain_graph=True,
            only_inputs=True,
            allow_unused=False,
        )[0]

        total = total + grad.pow(2).mean()

    return total / float(num_probes)


# ---------------------------------------------------------------------
# Active-contour geometry loss
# ---------------------------------------------------------------------


class ActiveContourGeometryLoss(nn.Module):
    """
    Safeguarded train-time active-contour geometry loss.

    Retained for ablations only. The final paper configuration should leave
    TaskLossWeights.active_contour = 0 and use active contours post-detection
    for measurement/audit instead of as a required train-time objective.
    """

    def __init__(
        self,
        *,
        mu: float = 0.2,
        epsilon: float = 1.0,
        support_floor: float = 1e-4,
        eikonal_weight: float = 0.05,
        region_weight: float = 1.0,
        area_balance_weight: float = 0.01,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.support_floor = float(support_floor)
        self.eikonal_weight = float(eikonal_weight)
        self.region_weight = float(region_weight)
        self.area_balance_weight = float(area_balance_weight)
        self.eps = float(eps)

    def heaviside(self, phi: torch.Tensor) -> torch.Tensor:
        return 0.5 + torch.atan(phi / self.epsilon) / torch.pi

    def dirac(self, phi: torch.Tensor) -> torch.Tensor:
        eps = self.epsilon
        return eps / (torch.pi * (eps * eps + phi * phi))

    def _regional_mean(self, gray: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        numerator = (gray * weight).sum(dim=(-2, -1), keepdim=True)
        denominator = weight.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return numerator / denominator

    def forward(
        self,
        restored: torch.Tensor,
        phi: torch.Tensor,
        lambda1: torch.Tensor,
        lambda2: torch.Tensor,
    ) -> torch.Tensor:
        if phi.shape[-2:] != restored.shape[-2:]:
            phi = F.interpolate(phi, size=restored.shape[-2:], mode="bilinear", align_corners=False)
        if lambda1.shape[-2:] != restored.shape[-2:]:
            lambda1 = F.interpolate(lambda1, size=restored.shape[-2:], mode="bilinear", align_corners=False)
        if lambda2.shape[-2:] != restored.shape[-2:]:
            lambda2 = F.interpolate(lambda2, size=restored.shape[-2:], mode="bilinear", align_corners=False)

        lambda1 = lambda1.clamp(0.1, 5.0)
        lambda2 = lambda2.clamp(0.1, 5.0)
        gray = rgb_to_gray(restored).detach()

        h = self.heaviside(phi).clamp(0.0, 1.0)
        d = self.dirac(phi) + self.support_floor

        mean_inside = self._regional_mean(gray, h)
        mean_outside = self._regional_mean(gray, 1.0 - h)

        grad_phi = gradient_magnitude(phi)
        length_term = self.mu * grad_phi
        region_inside = lambda1 * (gray - mean_inside).pow(2) * h
        region_outside = lambda2 * (gray - mean_outside).pow(2) * (1.0 - h)

        active_contour = (
            d * (length_term + self.region_weight * (region_inside + region_outside))
        ).mean()

        eikonal = (grad_phi - 1.0).pow(2).mean()
        area_balance = (h.mean(dim=(-2, -1)) * (1.0 - h.mean(dim=(-2, -1)))).mean()
        # This is minimized when the support is not collapsed to exactly all-in or
        # all-out. The weight is deliberately tiny by default.
        noncollapse = -area_balance

        return active_contour + self.eikonal_weight * eikonal + self.area_balance_weight * noncollapse


# ---------------------------------------------------------------------
# Detector-safe evidence and anchoring losses
# ---------------------------------------------------------------------


def _lower_road_region(x: torch.Tensor, lower_fraction: float = 0.55) -> torch.Tensor:
    """Return the lower road-heavy crop used by detector-safety evidence terms."""
    if x.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}.")
    frac = min(max(float(lower_fraction), 0.05), 1.0)
    h = x.shape[-2]
    start = int(round(h * (1.0 - frac)))
    return x[..., start:, :]


def local_contrast(x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Differentiable local contrast proxy."""
    gray = rgb_to_gray(x)
    k = _safe_odd_kernel(kernel_size)
    mean = F.avg_pool2d(gray, kernel_size=k, stride=1, padding=k // 2)
    mean_sq = F.avg_pool2d(gray * gray, kernel_size=k, stride=1, padding=k // 2)
    var = (mean_sq - mean * mean).clamp_min(0.0)
    return torch.sqrt(var + 1e-8)


def high_frequency_residual(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Differentiable high-frequency road-texture proxy."""
    gray = rgb_to_gray(x)
    k = _safe_odd_kernel(kernel_size)
    blur = F.avg_pool2d(gray, kernel_size=k, stride=1, padding=k // 2)
    return (gray - blur).abs()


def _road_prior_like(x: torch.Tensor, lower_fraction: float = 0.55) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, x.shape[-2], device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
    base = 0.35 + 0.65 * y
    # Respect lower_fraction by softly increasing emphasis after the likely road
    # start rather than hard-cropping the mask.
    start = 1.0 - min(max(float(lower_fraction), 0.05), 1.0)
    transition = torch.sigmoid(10.0 * (y - start))
    return (0.50 * base + 0.50 * transition).clamp(0.0, 1.0)


def road_defect_visibility_mask(
    x: torch.Tensor,
    *,
    lower_fraction: float = 0.55,
    power: float = 1.0,
    smooth_kernel: int = 7,
) -> torch.Tensor:
    """
    Label-free proxy mask for detector-visible road evidence.

    It is not a segmentation mask. It is used only to weight feature losses and
    detail-copy penalties. The returned tensor is detached by design.
    """
    image = x.clamp(0.0, 1.0)
    gray = rgb_to_gray(image)
    edge = _safe_amax_normalize(gradient_magnitude(gray))
    contrast = _safe_amax_normalize(local_contrast(image))
    highfreq = _safe_amax_normalize(high_frequency_residual(image))
    dark = (1.0 - gray).clamp(0.0, 1.0)
    saturation = image.max(dim=1, keepdim=True).values - image.min(dim=1, keepdim=True).values
    saturation = _safe_amax_normalize(saturation)

    raw = (
        0.40 * edge
        + 0.25 * contrast
        + 0.20 * highfreq
        + 0.10 * saturation
        + 0.05 * dark * (edge + contrast).clamp(0.0, 1.0)
    )
    raw = _safe_amax_normalize(raw)
    mask = raw * _road_prior_like(image, lower_fraction=lower_fraction)

    k = max(int(smooth_kernel), 1)
    if k > 1:
        k = _safe_odd_kernel(k, minimum=1)
        mask = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    if power != 1.0:
        mask = mask.clamp_min(0.0).pow(float(power))
    return mask.clamp(0.0, 1.0).detach()


def road_evidence_vector(x: torch.Tensor, *, lower_fraction: float = 0.55) -> torch.Tensor:
    """
    Per-image road-evidence vector.

    Components: edge, local contrast, high-frequency residual, saturation,
    darkness-weighted edge evidence. The fifth component is appended for richer
    logging/regularization; old callers that only use the first four labels are
    still safe if they treat the vector as opaque.
    """
    road = _lower_road_region(x.clamp(0.0, 1.0), lower_fraction=lower_fraction)
    gray = rgb_to_gray(road)
    edge_map = gradient_magnitude(gray)
    edge = edge_map.flatten(1).mean(dim=1)
    contrast = local_contrast(road).flatten(1).mean(dim=1)
    highfreq = high_frequency_residual(road).flatten(1).mean(dim=1)
    saturation = (
        road.max(dim=1, keepdim=True).values - road.min(dim=1, keepdim=True).values
    ).flatten(1).mean(dim=1)
    dark_edge = ((1.0 - gray).clamp(0.0, 1.0) * edge_map).flatten(1).mean(dim=1)
    return torch.stack([edge, contrast, highfreq, saturation, dark_edge], dim=1)


def road_evidence_nonregression_loss(
    restored: torch.Tensor,
    clean: torch.Tensor,
    degraded: torch.Tensor,
    *,
    lower_fraction: float = 0.55,
    component_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Penalize restored images only when they lose simple road evidence.

    Target: max(clean evidence, degraded evidence). This prevents restoration
    from suppressing detector-useful cues that remain visible in degraded/native
    frames.
    """
    ev_r = road_evidence_vector(restored, lower_fraction=lower_fraction)
    with torch.no_grad():
        ev_c = road_evidence_vector(clean, lower_fraction=lower_fraction)
        ev_d = road_evidence_vector(degraded, lower_fraction=lower_fraction)
        target = torch.maximum(ev_c, ev_d)
    penalty = F.relu(target - ev_r).pow(2)
    if component_weights is not None:
        weights = torch.as_tensor(component_weights, device=penalty.device, dtype=penalty.dtype).view(1, -1)
        if weights.shape[1] != penalty.shape[1]:
            raise ValueError(f"Expected {penalty.shape[1]} evidence weights, got {weights.shape[1]}.")
        penalty = penalty * weights
    return penalty.mean()


class DetectorInputAnchorLoss(nn.Module):
    """
    Weak detector-feature anchor to the degraded/input frame.

    This is deliberately low-weight. It is a safeguard against shifting restored
    frames far from the frozen detector's operating feature distribution.
    """

    def __init__(
        self,
        feature_extractor: FrozenDetectorFeatureExtractor,
        *,
        layer_weights: Optional[Dict[str, float]] = None,
        normalize_features: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights or {}
        self.normalize_features = bool(normalize_features)
        self.eps = float(eps)

    def forward(self, restored: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
        restored_features = self.feature_extractor(restored)
        with torch.no_grad():
            degraded_features = self.feature_extractor(degraded)

        total = restored.new_tensor(0.0)
        weight_total = 0.0
        for name, feat_r in restored_features.items():
            if name not in degraded_features:
                continue
            feat_d = degraded_features[name].detach()
            feat_r, feat_d = _align_feature_pair(feat_r, feat_d)
            if self.normalize_features:
                feat_r = _normalize_feature(feat_r, eps=self.eps)
                feat_d = _normalize_feature(feat_d, eps=self.eps)
            weight = float(self.layer_weights.get(name, 1.0))
            if weight <= 0:
                continue
            total = total + weight * F.mse_loss(feat_r, feat_d)
            weight_total += weight
        if weight_total <= 0:
            raise RuntimeError("No matching detector features found for detector input-anchor loss.")
        return total / float(weight_total)


def restoration_magnitude_loss(restored: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
    """Small optional guard against excessive restoration displacement."""
    return F.smooth_l1_loss(restored, degraded)


def low_evidence_identity_loss(
    restored: torch.Tensor,
    degraded: torch.Tensor,
    clean: torch.Tensor,
    *,
    lower_fraction: float = 0.55,
) -> torch.Tensor:
    """
    Penalize unnecessary changes in low-evidence regions.

    Clean-image evidence is used only as a detached proxy. High-evidence pixels
    are allowed to change; low-evidence background should not drift strongly.
    """
    mask = road_defect_visibility_mask(clean, lower_fraction=lower_fraction)
    background = (1.0 - mask).clamp(0.0, 1.0)
    diff = (restored - degraded).abs().mean(dim=1, keepdim=True)
    return (diff * background).sum() / background.sum().clamp_min(1e-6)


# ---------------------------------------------------------------------
# Composite task loss
# ---------------------------------------------------------------------


@dataclass
class TaskLossWeights:
    tdp: float = 0.02
    jacobian: float = 0.001
    active_contour: float = 0.0
    detector_input_anchor: float = 0.0
    evidence_nonregression: float = 0.0
    detail_copy: float = 0.0
    # Optional new guards. Defaults are zero to preserve old behaviour.
    basis_sparsity: float = 0.0
    restoration_magnitude: float = 0.0
    low_evidence_identity: float = 0.0


class CompositeTaskLoss(nn.Module):
    """
    Task-driven training objective reported in the paper.

    Nonzero weights are treated as required terms. L_base is still computed
    outside this module via RCADLoss; this module returns only the task-driven
    regularizer sum to be added in the same backward pass.

    Manuscript-style objective:
        L_total = L_base + lambda_s||r||_1 + lambda_TDP L_TDP
                + lambda_J L_J + lambda_A L_anchor
                + lambda_NR L_NR + lambda_DC L_DC
                + optional safety guards.
    """

    def __init__(
        self,
        *,
        tdp_loss: Optional[TaskDrivenPerceptualLoss] = None,
        active_contour_loss: Optional[ActiveContourGeometryLoss] = None,
        feature_extractor: Optional[FrozenDetectorFeatureExtractor] = None,
        detector_anchor_loss: Optional[DetectorInputAnchorLoss] = None,
        weights: Optional[TaskLossWeights] = None,
        jacobian_probes: int = 1,
        evidence_lower_fraction: float = 0.55,
    ) -> None:
        super().__init__()

        self.tdp_loss = tdp_loss
        self.active_contour_loss = active_contour_loss
        self.feature_extractor = feature_extractor
        self.detector_anchor_loss = detector_anchor_loss
        self.weights = weights or TaskLossWeights()
        self.jacobian_probes = int(jacobian_probes)
        self.evidence_lower_fraction = float(evidence_lower_fraction)

    @staticmethod
    def _log_scalar(logs: Dict[str, float], name: str, value: torch.Tensor | float) -> None:
        if isinstance(value, torch.Tensor):
            logs[name] = float(value.detach().cpu())
        else:
            logs[name] = float(value)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clean: torch.Tensor,
        *,
        degraded: Optional[torch.Tensor] = None,
        warmup_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if "restored" not in outputs:
            raise KeyError("CompositeTaskLoss expects outputs['restored'].")

        restored = outputs["restored"]
        total = restored.new_tensor(0.0)
        logs: Dict[str, float] = {}

        scale = float(warmup_scale)
        defect_mask = road_defect_visibility_mask(
            clean,
            lower_fraction=self.evidence_lower_fraction,
            power=getattr(self.tdp_loss, "defect_mask_power", 1.0) if self.tdp_loss is not None else 1.0,
        )
        self._log_scalar(logs, "defect_visibility_mask_mean", defect_mask.mean())
        self._log_scalar(logs, "defect_visibility_mask_max", defect_mask.max())

        if self.weights.tdp > 0:
            if self.tdp_loss is None:
                raise RuntimeError("TaskDrivenPerceptualLoss is required because weights.tdp > 0.")
            loss_tdp = self.tdp_loss(restored, clean, defect_mask=defect_mask, use_cqmix=True)
            weighted = scale * self.weights.tdp * loss_tdp
            total = total + weighted
            self._log_scalar(logs, "loss_tdp_raw", loss_tdp)
            self._log_scalar(logs, "loss_tdp_weighted", weighted)

        if self.weights.jacobian > 0 and self.jacobian_probes > 0:
            if self.feature_extractor is None:
                raise RuntimeError("FrozenDetectorFeatureExtractor is required because weights.jacobian > 0.")
            loss_jac = hutchinson_jacobian_penalty(
                self.feature_extractor,
                restored,
                num_probes=self.jacobian_probes,
            )
            weighted = scale * self.weights.jacobian * loss_jac
            total = total + weighted
            self._log_scalar(logs, "loss_jacobian_raw", loss_jac)
            self._log_scalar(logs, "loss_jacobian_weighted", weighted)

        if self.weights.active_contour > 0:
            if self.active_contour_loss is None:
                raise RuntimeError("ActiveContourGeometryLoss is required because weights.active_contour > 0.")
            has_contour = all(k in outputs and outputs[k] is not None for k in ("phi", "lambda1", "lambda2"))
            if not has_contour:
                raise RuntimeError(
                    "RMR-Net must be called with return_aux=True and use_tdac_head=True "
                    "because weights.active_contour > 0."
                )
            loss_ac = self.active_contour_loss(
                restored,
                outputs["phi"],
                outputs["lambda1"],
                outputs["lambda2"],
            )
            weighted = scale * self.weights.active_contour * loss_ac
            total = total + weighted
            self._log_scalar(logs, "loss_active_contour_raw", loss_ac)
            self._log_scalar(logs, "loss_active_contour_weighted", weighted)

        if self.weights.detector_input_anchor > 0:
            if degraded is None:
                raise RuntimeError("degraded input is required because weights.detector_input_anchor > 0.")
            if self.detector_anchor_loss is None:
                raise RuntimeError("DetectorInputAnchorLoss is required because weights.detector_input_anchor > 0.")
            loss_anchor = self.detector_anchor_loss(restored, degraded)
            weighted = scale * self.weights.detector_input_anchor * loss_anchor
            total = total + weighted
            self._log_scalar(logs, "loss_detector_input_anchor_raw", loss_anchor)
            self._log_scalar(logs, "loss_detector_input_anchor_weighted", weighted)

        if self.weights.evidence_nonregression > 0:
            if degraded is None:
                raise RuntimeError("degraded input is required because weights.evidence_nonregression > 0.")
            loss_ev = road_evidence_nonregression_loss(
                restored,
                clean,
                degraded,
                lower_fraction=self.evidence_lower_fraction,
            )
            weighted = scale * self.weights.evidence_nonregression * loss_ev
            total = total + weighted
            self._log_scalar(logs, "loss_evidence_nonregression_raw", loss_ev)
            self._log_scalar(logs, "loss_evidence_nonregression_weighted", weighted)
            with torch.no_grad():
                ev_r = road_evidence_vector(restored, lower_fraction=self.evidence_lower_fraction).mean(dim=0)
                ev_c = road_evidence_vector(clean, lower_fraction=self.evidence_lower_fraction).mean(dim=0)
                ev_d = road_evidence_vector(degraded, lower_fraction=self.evidence_lower_fraction).mean(dim=0)
            labels = ["edge", "contrast", "highfreq", "saturation", "dark_edge"]
            for idx, label in enumerate(labels):
                self._log_scalar(logs, f"evidence_{label}_restored", ev_r[idx])
                self._log_scalar(logs, f"evidence_{label}_clean", ev_c[idx])
                self._log_scalar(logs, f"evidence_{label}_degraded", ev_d[idx])

        if self.weights.detail_copy > 0:
            gate = outputs.get("detail_gate")
            detail = outputs.get("detail_residual")
            if gate is None or detail is None:
                raise RuntimeError("detail_gate and detail_residual are required because weights.detail_copy > 0.")
            copied = (gate * detail).abs().mean(dim=1, keepdim=True)
            background = (1.0 - defect_mask).clamp(0.0, 1.0)
            loss_copy = (copied * background).sum() / background.sum().clamp_min(1e-6)
            weighted = scale * self.weights.detail_copy * loss_copy
            total = total + weighted
            self._log_scalar(logs, "loss_detail_copy_raw", loss_copy)
            self._log_scalar(logs, "loss_detail_copy_weighted", weighted)
            self._log_scalar(logs, "detail_gate_mean", gate.mean())
            self._log_scalar(logs, "detail_gate_max", gate.max())

        if self.weights.basis_sparsity > 0:
            basis_sparsity = outputs.get("basis_sparsity")
            if basis_sparsity is None:
                raise RuntimeError("outputs['basis_sparsity'] is required because weights.basis_sparsity > 0.")
            weighted = scale * self.weights.basis_sparsity * basis_sparsity
            total = total + weighted
            self._log_scalar(logs, "loss_basis_sparsity_raw", basis_sparsity)
            self._log_scalar(logs, "loss_basis_sparsity_weighted", weighted)

        if self.weights.restoration_magnitude > 0:
            if degraded is None:
                raise RuntimeError("degraded input is required because weights.restoration_magnitude > 0.")
            loss_mag = restoration_magnitude_loss(restored, degraded)
            weighted = scale * self.weights.restoration_magnitude * loss_mag
            total = total + weighted
            self._log_scalar(logs, "loss_restoration_magnitude_raw", loss_mag)
            self._log_scalar(logs, "loss_restoration_magnitude_weighted", weighted)

        if self.weights.low_evidence_identity > 0:
            if degraded is None:
                raise RuntimeError("degraded input is required because weights.low_evidence_identity > 0.")
            loss_id = low_evidence_identity_loss(
                restored,
                degraded,
                clean,
                lower_fraction=self.evidence_lower_fraction,
            )
            weighted = scale * self.weights.low_evidence_identity * loss_id
            total = total + weighted
            self._log_scalar(logs, "loss_low_evidence_identity_raw", loss_id)
            self._log_scalar(logs, "loss_low_evidence_identity_weighted", weighted)

        gate = outputs.get("gate")
        if gate is not None:
            self._log_scalar(logs, "deployment_gate_mean", gate.mean())
            self._log_scalar(logs, "deployment_gate_min", gate.min())
            self._log_scalar(logs, "deployment_gate_max", gate.max())

        if "severity" in outputs and outputs["severity"] is not None:
            self._log_scalar(logs, "severity_mean", outputs["severity"].mean())

        self._log_scalar(logs, "loss_task_total", total)
        logs["task_warmup_scale"] = scale

        return total, logs
