from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from models.rmrnet import RMRNet
from rcadnet.dataset import PairedRoadRestorationDataset, list_images
from rcadnet.losses import RCADLoss
from rcadnet.practical_metadata import (
    PRACTICAL_SENSOR_DIM,
    balanced_sensor_dropout,
    counterfactual_sensor_packet,
    perturb_sensor_packet,
    structured_sensor_dropout,
)
from rcadnet.sensor_geometry import practical_psf_geometry_loss
from rcadnet.spatial_physics import physics_reblur_loss
from rcadnet.task_losses import (
    ActiveContourGeometryLoss,
    CompositeTaskLoss,
    DetectorInputAnchorLoss,
    FrozenDetectorFeatureExtractor,
    FrozenDetectorSupervisedLoss,
    TaskDrivenPerceptualLoss,
    TaskLossWeights,
    road_evidence_vector,
)


def apply_primary_protected_gradients(
    primary_loss: torch.Tensor,
    task_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Backpropagate a task objective without opposing the primary objective.

    The restoration/state objective is primary.  If the detector-task gradient
    has a negative global dot product with the primary gradient, remove only
    that opposing component before summing the gradients.  This is a
    deterministic, primary-preserving form of gradient surgery:

        g_t' = g_t - min(0, <g_t,g_p>) g_p / (||g_p||^2 + epsilon)
        g    = g_p + g_t'.

    It avoids the single-backward failure mode in which a large detector
    feature loss erases fidelity or degradation-state learning.  Loss weights
    still control gradient magnitude; this function only resolves conflict.
    """
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable RMR-Net parameters were found")
    if not primary_loss.requires_grad:
        raise RuntimeError("The primary RMR-Net objective is detached")

    has_task = bool(task_loss.requires_grad)
    primary_grads = torch.autograd.grad(
        primary_loss,
        trainable,
        retain_graph=has_task,
        allow_unused=True,
    )
    task_grads = (
        torch.autograd.grad(task_loss, trainable, allow_unused=True)
        if has_task
        else tuple(None for _ in trainable)
    )

    device = primary_loss.device
    dot = torch.zeros((), device=device, dtype=torch.float32)
    primary_sq = torch.zeros_like(dot)
    task_sq = torch.zeros_like(dot)
    for primary_grad, task_grad in zip(primary_grads, task_grads):
        if primary_grad is not None:
            primary_sq = primary_sq + primary_grad.detach().float().square().sum()
        if task_grad is not None:
            task_sq = task_sq + task_grad.detach().float().square().sum()
        if primary_grad is not None and task_grad is not None:
            dot = dot + (
                primary_grad.detach().float() * task_grad.detach().float()
            ).sum()

    conflict = bool((dot < 0).detach().cpu()) and bool(
        (primary_sq > epsilon).detach().cpu()
    )
    coefficient = (
        (dot / (primary_sq + epsilon)).to(dtype=primary_loss.dtype)
        if conflict
        else primary_loss.new_zeros(())
    )
    projected_task_sq = torch.zeros_like(dot)
    for parameter, primary_grad, task_grad in zip(
        trainable,
        primary_grads,
        task_grads,
    ):
        projected_task = task_grad
        if conflict and task_grad is not None and primary_grad is not None:
            projected_task = task_grad - coefficient.to(
                device=task_grad.device,
                dtype=task_grad.dtype,
            ) * primary_grad
        if projected_task is not None:
            projected_task_sq = (
                projected_task_sq
                + projected_task.detach().float().square().sum()
            )
        if primary_grad is None:
            parameter.grad = projected_task
        elif projected_task is None:
            parameter.grad = primary_grad
        else:
            parameter.grad = primary_grad + projected_task

    cosine = dot / (
        torch.sqrt(primary_sq.clamp_min(epsilon))
        * torch.sqrt(task_sq.clamp_min(epsilon))
    )
    return {
        "gradient_conflict": float(conflict),
        "gradient_cosine_before": float(cosine.detach().cpu()) if has_task else 0.0,
        "primary_gradient_norm": float(torch.sqrt(primary_sq).detach().cpu()),
        "task_gradient_norm": float(torch.sqrt(task_sq).detach().cpu()),
        "projected_task_gradient_norm": float(
            torch.sqrt(projected_task_sq).detach().cpu()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RMR/RCAD-Net on paired road restoration folders.")
    parser.add_argument("--data-root", action="append", required=True, help="Dataset root containing scenarios/<scenario>/input and /gt. Repeat to combine datasets.")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="Scenario to train on. Repeat for many.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument(
        "--defect-label-root",
        help=(
            "Training-only YOLO label directory used for defect-centered patch "
            "sampling. Labels are never read by validation, test, or inference."
        ),
    )
    parser.add_argument(
        "--defect-crop-probability",
        type=float,
        default=0.0,
        help=(
            "Probability of centering a training patch on a labelled road "
            "defect; remaining patches are sampled uniformly."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", "--out-dir", dest="out", default="runs/rcadnet")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--code-source",
        choices=["scenario", "zero", "estimated", "fused", "metadata", "metadata_fused", "sensor_fused"],
        default="scenario",
        help=(
            "Conditioning source. metadata_fused combines an eight-value metadata code "
            "with the image estimator; sensor_fused first encodes a practical sensor packet."
        ),
    )
    parser.add_argument("--no-defect-attention", action="store_true", help="Ablation: disable defect attention.")
    parser.add_argument("--aux-code-weight", type=float, default=0.05)
    parser.add_argument("--init-weights", help="Optional RMR/RCAD checkpoint to fine-tune from.")
    parser.add_argument(
        "--resume-checkpoint",
        help="Resume the exact model, AdamW, and AMP state. --epochs is the final global epoch.",
    )
    parser.add_argument("--block-type", choices=["simple", "evidence"], default="simple")
    parser.add_argument("--attention-type", choices=["edge", "task", "none"], default="edge")
    parser.add_argument(
        "--conditioning",
        choices=["none", "film", "gated_basis", "residual_basis"],
        default="gated_basis",
        help=(
            "Conditioning path. 'none' is reserved for the sequential ablation: "
            "it trains the image-code head without injecting that code into the restorer. "
            "'residual_basis' preserves the image-derived code and learns a bounded "
            "metadata correction."
        ),
    )
    parser.add_argument(
        "--reset-code-fuser",
        action="store_true",
        help="Reinitialize only metadata fusion after loading an image-only checkpoint.",
    )
    parser.add_argument(
        "--fusion-only-epochs",
        type=int,
        default=0,
        help="Initially update only code_fuser parameters before unfreezing the full restorer.",
    )
    parser.add_argument(
        "--metadata-branch-only-epochs",
        type=int,
        default=0,
        help=(
            "Initially update only the image-code encoder, metadata fuser, and "
            "cause-specific residual experts. This protects the shared restorer "
            "during a validation-only metadata-branch refinement stage."
        ),
    )
    parser.add_argument(
        "--sensor-state-only-epochs",
        type=int,
        default=0,
        help=(
            "Initially update only the image degradation encoder, practical "
            "sensor encoder, and their joint PSF/cause refiner. This calibrates "
            "partial/noisy telemetry without changing the restoration backbone."
        ),
    )
    parser.add_argument(
        "--sensor-refiner-only-epochs",
        type=int,
        default=0,
        help=(
            "Initially update the image degradation encoder and image-conditioned "
            "PSF/cause refiner while freezing the calibrated sensor encoder and "
            "restoration backbone. This isolates joint posterior calibration."
        ),
    )
    parser.add_argument(
        "--sensor-prior-fusion-only-epochs",
        type=int,
        default=0,
        help=(
            "Initially update only the spatial sensor-prior confidence head. "
            "This learns where a practical motion prior improves the frozen "
            "neural restoration without changing either candidate."
        ),
    )
    parser.add_argument(
        "--post-prior-refiner-only-epochs",
        type=int,
        default=0,
        help=(
            "Initially update only the zero-initialized post-prior evidence "
            "refiner. The validated neural and physical candidates remain "
            "frozen while the bounded detector-aware correction is learned."
        ),
    )
    parser.add_argument(
        "--cause-expert-only-index",
        type=int,
        default=-1,
        help=(
            "When set to 0..4, update only that bounded cause-specific residual "
            "expert for every epoch. Index 3 is the compound motion/low-light "
            "expert. The shared restoration backbone and routing code remain "
            "frozen."
        ),
    )
    parser.add_argument(
        "--physical-state-only",
        action="store_true",
        help=(
            "During training batches, run only the practical image/metadata "
            "physical-state estimator. Validation still runs full restoration."
        ),
    )
    parser.add_argument(
        "--basis-sparsity-weight",
        type=float,
        default=1.0,
        help="Multiplier for the sparse monotone degradation-basis gate when gated_basis conditioning is used.",
    )
    parser.add_argument("--detail-preserve", action="store_true", help="Enable structural evidence-preserving high-frequency skip.")
    parser.add_argument("--detail-gain", type=float, default=0.20, help="Maximum gain for the structural detail skip.")
    parser.add_argument(
        "--cause-experts",
        action="store_true",
        help="Enable bounded motion/defocus/low-light/compound/noise residual experts.",
    )
    parser.add_argument("--cause-expert-gain", type=float, default=0.20)
    parser.add_argument(
        "--cause-compound-boost",
        type=float,
        default=1.0,
        help=(
            "Boost the motion-plus-low-light expert routing score before "
            "normalization. The default preserves historical checkpoints."
        ),
    )
    parser.add_argument("--new-head-lr-mult", type=float, default=10.0, help="LR multiplier for newly added train-time/detail heads.")
    parser.add_argument("--metadata-dropout", type=float, default=0.0)
    parser.add_argument(
        "--metadata-availability-mode",
        choices=["independent", "balanced"],
        default="independent",
        help=(
            "For sensor_fused training, balanced mode samples all seven "
            "partial/missing camera-IMU-vehicle patterns whenever metadata "
            "dropout is active."
        ),
    )
    parser.add_argument("--metadata-noise", type=float, default=0.0)
    parser.add_argument(
        "--metadata-encoding",
        choices=["legacy", "axial_v2"],
        default="legacy",
        help="Versioned metadata-to-code mapping; axial_v2 preserves signed axial blur orientation and removes phantom motion.",
    )
    parser.add_argument("--use-motion-prior", action="store_true")
    parser.add_argument("--motion-prior-k", type=float, default=0.0075)
    parser.add_argument("--motion-prior-blend", type=float, default=0.90)
    parser.add_argument("--motion-prior-nuisance-decay", type=float, default=30.0)
    parser.add_argument("--motion-prior-compound-floor", type=float, default=0.0)
    parser.add_argument("--motion-prior-adaptive-k", type=float, default=0.0)
    parser.add_argument(
        "--motion-prior-compound-source",
        action="store_true",
        help=(
            "For mixed corruption, apply the telemetry Wiener prior to the "
            "neural photometric restoration; pure motion still uses the input."
        ),
    )
    parser.add_argument(
        "--practical-nuisance-deadzone",
        type=float,
        default=0.0,
        help=(
            "Validation-calibrated noise floor for practical sensor nuisance "
            "coordinates before compound-degradation suppression."
        ),
    )
    parser.add_argument(
        "--use-practical-sensor-encoder",
        action="store_true",
        help="Encode the public camera/IMU/vehicle packet inside RMR-Net.",
    )
    parser.add_argument(
        "--sensor-prior-fusion",
        action="store_true",
        help=(
            "Fuse the calibrated sensor-derived deconvolution through a learned "
            "spatial confidence map instead of hard replacement."
        ),
    )
    parser.add_argument(
        "--post-prior-evidence-refiner",
        action="store_true",
        help=(
            "Attach a bounded zero-initialized residual head after the "
            "physics-guided candidate."
        ),
    )
    parser.add_argument(
        "--post-prior-refiner-gain",
        type=float,
        default=0.12,
        help="Maximum absolute post-prior residual before degradation support.",
    )
    parser.add_argument(
        "--post-prior-refiner-support",
        choices=["all", "lowlight"],
        default="all",
        help=(
            "Degradation support for the bounded post-prior adapter. "
            "'lowlight' uses the fused image+sensor low-light coordinate and "
            "therefore leaves pure motion and defocus paths unchanged."
        ),
    )
    parser.add_argument(
        "--practical-prior-source",
        choices=[
            "direct",
            "direct_motion_joint_nuisance",
            "sensor_calibrated",
            "joint",
        ],
        default="joint",
        help=(
            "Physical PSF state for practical packets. 'direct' integrates "
            "public gyro/exposure measurements without a learned calibration "
            "residual; historical checkpoints use 'joint'."
        ),
    )
    parser.add_argument(
        "--sensor-image-psf-refiner",
        action="store_true",
        help=(
            "Refine the sensor-derived image-space PSF with degraded-image "
            "features under per-cause metadata reliability."
        ),
    )
    parser.add_argument("--sensor-dim", type=int, default=PRACTICAL_SENSOR_DIM)
    parser.add_argument(
        "--sensor-gyro-full-scale",
        type=float,
        default=1.0,
        help=(
            "Fixed IMU calibration used to restore physical gyro units from "
            "the normalized practical packet. This is a sensor/protocol "
            "constant, never a per-image renderer parameter."
        ),
    )
    parser.add_argument(
        "--sensor-residual-scale",
        type=float,
        default=0.25,
        help=(
            "Maximum learned calibration residual around the deterministic "
            "sensor-physics code. Use 0 for the direct-physics control; "
            "historical checkpoints use 0.25."
        ),
    )
    parser.add_argument(
        "--spatial-physics",
        action="store_true",
        help=(
            "Enable exposure-synchronised rotational motion fields, spatial "
            "conditioning, and auditable differentiable reblurring."
        ),
    )
    parser.add_argument("--physics-samples", type=int, default=5)
    parser.add_argument("--physics-exposure-min-ms", type=float, default=0.05)
    parser.add_argument("--physics-exposure-max-ms", type=float, default=40.0)
    parser.add_argument(
        "--physics-focal-ratio",
        type=float,
        default=0.75,
        help="Declared camera calibration f_x/image_width used by H_phys.",
    )
    parser.add_argument(
        "--physics-calibration-reliability",
        type=float,
        default=0.50,
        help="Confidence in fixed intrinsics/extrinsics, in [0,1].",
    )
    parser.add_argument(
        "--physics-activation-motion-px",
        type=float,
        default=0.10,
        help="Pixel-motion support below which physics is smoothly rejected.",
    )
    parser.add_argument(
        "--lambda-physics-reblur",
        type=float,
        default=0.0,
        help="Weight of L_phys = rho(H_phys(I_r)-I_d).",
    )
    parser.add_argument(
        "--physics-exclusive-trajectory",
        action="store_true",
        help=(
            "Route raw gyro trajectory only through the deterministic spatial "
            "physics branch. The learned metadata encoder still receives "
            "camera, exposure, vehicle and reliability context."
        ),
    )
    parser.add_argument("--physics-inverse-candidate", action="store_true")
    parser.add_argument("--physics-inverse-iterations", type=int, default=3)
    parser.add_argument("--physics-inverse-blend", type=float, default=1.0)
    parser.add_argument("--physics-decoder-motion-threshold-px", type=float, default=1.5)
    parser.add_argument("--physics-decoder-motion-transition-px", type=float, default=0.35)
    parser.add_argument(
        "--sensor-encoder-weights",
        help=(
            "Optional validation-selected packet-to-cause calibration checkpoint. "
            "Loaded after --init-weights and before restoration fine-tuning."
        ),
    )
    parser.add_argument(
        "--lambda-sensor-cause",
        type=float,
        default=0.0,
        help=(
            "Training-only supervision from hidden synthetic causes to the encoded "
            "sensor code. Hidden causes are targets, never model inputs."
        ),
    )
    parser.add_argument(
        "--lambda-sensor-physical",
        type=float,
        default=0.0,
        help=(
            "Training-only calibration of the sensor-only physical state on "
            "coordinates supported by available telemetry."
        ),
    )
    parser.add_argument(
        "--lambda-image-physical",
        type=float,
        default=0.0,
        help=(
            "Training-only supervision of the degraded-image physical-state "
            "fallback. Private renderer labels are targets, never inputs."
        ),
    )
    parser.add_argument(
        "--lambda-posterior-physical",
        type=float,
        default=0.0,
        help=(
            "Training-only supervision of the joint image-and-metadata "
            "physical posterior over all target coordinates."
        ),
    )
    parser.add_argument(
        "--sensor-motion-coordinate-weight",
        type=float,
        default=2.0,
        help=(
            "Relative weight of the three axial motion coordinates in the "
            "training-only practical PSF calibration target."
        ),
    )
    parser.add_argument(
        "--sensor-psf-geometry-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the geometry-aware blur-length and axial-direction "
            "term on the joint physical posterior."
        ),
    )
    parser.add_argument(
        "--image-psf-geometry-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the same geometry-aware term on the image-only "
            "physical-state fallback."
        ),
    )
    parser.add_argument("--sensor-psf-length-weight", type=float, default=4.0)
    parser.add_argument("--sensor-psf-vector-weight", type=float, default=2.0)
    parser.add_argument("--sensor-psf-direction-weight", type=float, default=0.5)
    parser.add_argument(
        "--lambda-fused-cause",
        type=float,
        default=0.0,
        help=(
            "Supervise the final image-and-metadata posterior degradation code "
            "against the known training corruption state."
        ),
    )
    parser.add_argument(
        "--lambda-sensor-prior-gate",
        type=float,
        default=0.0,
        help=(
            "Train the practical sensor-prior confidence map from paired training "
            "images. The target trusts the physical candidate only where it is "
            "closer to clean than the neural candidate and the motion sensor is "
            "reliable."
        ),
    )
    parser.add_argument(
        "--conditioning-branch-only-epochs",
        type=int,
        default=0,
        help="Initially update metadata branches plus FiLM/channel gates while preserving shared convolutional filters.",
    )
    parser.add_argument(
        "--lambda-metadata-advantage",
        type=float,
        default=0.0,
        help=(
            "Force aligned metadata to be useful: penalize the correct-metadata "
            "restoration loss if it is not lower than image-only and mismatched "
            "metadata controls by a small margin."
        ),
    )
    parser.add_argument(
        "--lambda-metadata-gate",
        type=float,
        default=0.0,
        help=(
            "Supervise the reliability gate: high gate for aligned metadata and "
            "low gate for deliberately mismatched metadata."
        ),
    )
    parser.add_argument("--metadata-advantage-margin", type=float, default=0.002)
    parser.add_argument(
        "--lambda-metadata-tdp-advantage",
        type=float,
        default=0.0,
        help=(
            "Detector-feature metadata advantage. Penalizes aligned metadata "
            "when its TDP loss is not lower than image-only/counterfactual "
            "metadata controls."
        ),
    )
    parser.add_argument("--metadata-tdp-advantage-margin", type=float, default=0.0005)
    parser.add_argument("--metadata-gate-target", type=float, default=0.80)
    parser.add_argument("--metadata-negative-gate-target", type=float, default=0.08)
    parser.add_argument(
        "--metadata-mode",
        choices=["full", "raw_telemetry", "raw_scalar", "zero", "missing", "shuffled"],
        default="full",
        help=(
            "Metadata fields used for conditioning. Use zero/missing for an "
            "image-only metadata-control check, or shuffled to condition each "
            "image on another sample's metadata."
        ),
    )
    parser.add_argument(
        "--loss-profile",
        choices=["simple", "legacy"],
        default="simple",
        help=(
            "simple uses Charbonnier reconstruction plus image-gradient "
            "consistency. legacy restores the older Fourier/visibility pack."
        ),
    )
    parser.add_argument("--charbonnier-epsilon", type=float, default=1e-3)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--freq-weight", type=float, default=0.0)
    parser.add_argument("--defect-weight", type=float, default=0.0)
    parser.add_argument("--visibility-weight", type=float, default=0.0)
    parser.add_argument(
        "--base-weight",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the base restoration loss L_base. Values below 1.0 "
            "enable detector-heavy fine-tuning while keeping a nonzero fidelity anchor."
        ),
    )

    # Composite objective reported in the manuscript.
    parser.add_argument(
        "--use-task-losses",
        action="store_true",
        help=(
            "Enable detector-aware training. The reported profile uses TDP/CQMix "
            "with primary-protected gradients; legacy regularizers require "
            "explicit nonzero coefficients."
        ),
    )
    parser.add_argument("--lambda-tdp", type=float, default=0.02)
    parser.add_argument(
        "--lambda-detector-supervised",
        type=float,
        default=0.0,
        help=(
            "Weight of the frozen detector's supervised box/class/DFL loss on "
            "training crops. Requires --defect-label-root."
        ),
    )
    parser.add_argument(
        "--detector-supervised-cqmix-prob",
        type=float,
        default=0.25,
        help="Probability of clean/restored patch mixing before supervised detector loss.",
    )
    parser.add_argument(
        "--detector-supervised-clean-hinge",
        type=float,
        default=0.25,
        help="Extra penalty when restored detector loss exceeds the clean-reference loss.",
    )
    parser.add_argument(
        "--lambda-jacobian",
        type=float,
        default=0.0,
        help="Legacy cascaded-stability weight. The final simple profile leaves it off.",
    )
    parser.add_argument("--lambda-active-contour", type=float, default=0.0)
    parser.add_argument("--lambda-detector-input-anchor", type=float, default=0.0, help="Weak detector-feature anchor from restored image to degraded/input image.")
    parser.add_argument("--lambda-evidence-nonregression", type=float, default=0.0, help="Hinge penalty when restoration suppresses road evidence below clean/degraded evidence.")
    parser.add_argument("--lambda-detail-copy", type=float, default=0.0, help="Penalize structural detail-skip energy outside detector-visible road-evidence regions.")
    parser.add_argument("--lambda-restoration-magnitude", type=float, default=0.0, help="Optional guard against excessive residual displacement from the degraded/native input.")
    parser.add_argument("--lambda-low-evidence-identity", type=float, default=0.0, help="Optional identity penalty in low-evidence regions to avoid detector-harmful over-restoration.")
    parser.add_argument("--evidence-lower-fraction", type=float, default=0.55, help="Lower image fraction used by road-evidence debug/loss terms.")
    parser.add_argument("--task-loss-warmup-epochs", "--task-warmup-epochs", dest="task_loss_warmup_epochs", type=int, default=5)
    parser.add_argument("--cqmix-grid", type=int, default=4)
    parser.add_argument("--cqmix-prob", type=float, default=0.5)
    parser.add_argument("--tdp-defect-mask-weight", type=float, default=0.0, help="Extra detector-feature weight on clean-target road-defect evidence regions.")
    parser.add_argument("--tdp-defect-mask-power", type=float, default=1.0, help="Power applied to the clean-target defect-evidence proxy used by weighted TDP.")
    parser.add_argument("--jacobian-probes", type=int, default=1)
    parser.add_argument("--detector-hook-layers", type=str, default="")
    parser.add_argument("--detector-max-hook-layers", type=int, default=3)
    parser.add_argument("--detector-input-size", type=int, default=640)
    parser.add_argument("--select-by", type=str, default="val_map50")
    parser.add_argument("--promotion-margin", type=float, default=0.005)
    parser.add_argument("--smoke-test", action="store_true", help="Run one forward/backward batch and exit.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--gradient-strategy",
        choices=["joint", "primary_projected"],
        default="joint",
        help=(
            "joint uses one backward pass. primary_projected removes only the "
            "component of the detector-task gradient that opposes the "
            "restoration/state gradient; AMP must be disabled."
        ),
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--debug-every", type=int, default=0, help="Print JSON debug stats every N training batches. 0 disables periodic batch debug.")
    parser.add_argument("--debug-first-batches", type=int, default=1, help="Print JSON debug stats for the first N batches of each epoch.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help=(
            "Optional transparent cap on batches per epoch for controlled tuning. "
            "0 uses the complete training loader."
        ),
    )

    # Backward-compatible v19-v22 flags.
    parser.add_argument("--task-driven", action="store_true", help="Legacy alias for --use-task-losses.")
    parser.add_argument("--use-tdac-head", action="store_true", help="Enable train-time phi/lambda auxiliary head.")
    parser.add_argument("--tdac-weight", type=float, default=None, help="Legacy alias for --lambda-active-contour.")
    parser.add_argument("--tdac-mu", type=float, default=0.20)
    parser.add_argument("--tdac-epsilon", type=float, default=1.0)
    parser.add_argument("--tdac-window", type=int, default=15, help="Accepted for older logs; current TDAC is global.")
    parser.add_argument("--tdac-eikonal-weight", type=float, default=0.05)
    parser.add_argument("--tdp-yolo-weights", help="Frozen YOLO weights used for TDP/Jacobian.")
    parser.add_argument("--tdp-layers", default="", help="Legacy numeric or named detector hook layers.")
    parser.add_argument("--tdp-layer-weights", default="")
    parser.add_argument("--tdp-weight", type=float, default=None, help="Legacy alias for --lambda-tdp.")
    parser.add_argument("--tdp-no-cqmix", action="store_true")
    parser.add_argument("--tdp-cqmix-patch-size", type=int, default=0, help="Legacy patch-size hint; prefer --cqmix-grid.")
    parser.add_argument("--jacobian-weight", type=float, default=None, help="Legacy alias for --lambda-jacobian.")
    parser.add_argument("--detector-anchor-weight", type=float, default=None, help="Legacy/short alias for --lambda-detector-input-anchor.")
    parser.add_argument("--evidence-nonregression-weight", type=float, default=None, help="Legacy/short alias for --lambda-evidence-nonregression.")
    parser.add_argument("--no-task-warmup", action="store_true")
    parser.add_argument("--gate-threshold", type=float, default=-1.0)
    parser.add_argument("--gate-softness", type=float, default=0.03)
    parser.add_argument("--noise-da-weight", type=float, default=0.0, help="Accepted for compatibility; Noise-DA is not used by the released trainer.")
    parser.add_argument("--noise-da-contrastive-weight", type=float, default=0.05)
    parser.add_argument("--noise-da-real-yolo-data")
    parser.add_argument("--noise-da-real-split", default="train")
    parser.add_argument("--phase2-detector-data")
    parser.add_argument("--phase2-detector-weights")
    parser.add_argument("--phase2-epochs", type=int, default=0)
    parser.add_argument("--phase2-imgsz", type=int, default=640)
    parser.add_argument("--alternate-phase-period", type=int, default=0)
    parser.add_argument("--alternate-detector-data")
    parser.add_argument("--alternate-detector-weights")
    parser.add_argument("--alternate-detector-epochs", type=int, default=0)
    parser.add_argument("--alternate-detector-imgsz", type=int, default=640)

    parser.add_argument("--val-data-root", action="append")
    parser.add_argument("--val-scenario", action="append", dest="val_scenarios")
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--save-every-epoch", action="store_true")
    args = parser.parse_args()

    normalize_task_args(args)
    return args


def normalize_task_args(args: argparse.Namespace) -> None:
    if args.task_driven:
        args.use_task_losses = True
    if args.tdp_weight is not None:
        args.lambda_tdp = float(args.tdp_weight)
    if args.jacobian_weight is not None:
        args.lambda_jacobian = float(args.jacobian_weight)
    if args.detector_anchor_weight is not None:
        args.lambda_detector_input_anchor = float(args.detector_anchor_weight)
    if args.evidence_nonregression_weight is not None:
        args.lambda_evidence_nonregression = float(args.evidence_nonregression_weight)
    if args.tdac_weight is not None:
        args.lambda_active_contour = float(args.tdac_weight)
    if args.use_task_losses and args.lambda_active_contour > 0:
        args.use_tdac_head = True
    if args.use_tdac_head and args.lambda_active_contour <= 0:
        print(
            json.dumps(
                {
                    "warning": "use_tdac_head requested while lambda_active_contour<=0; "
                    "auxiliary contour maps will be emitted but no active-contour loss is optimized."
                }
            ),
            flush=True,
        )
    if args.no_task_warmup:
        args.task_loss_warmup_epochs = 0
    if args.tdp_no_cqmix:
        args.cqmix_prob = 0.0


class YoloImageDataset(Dataset):
    """Loads unpaired native/real images from a YOLO data.yaml split."""

    def __init__(self, data_yaml: str | Path, split: str = "train", patch_size: int = 256) -> None:
        self.data_yaml = Path(data_yaml)
        data = yaml.safe_load(self.data_yaml.read_text(encoding="utf-8"))
        root = Path(data.get("path", self.data_yaml.parent))
        if not root.is_absolute():
            root = (self.data_yaml.parent / root).resolve()
        split_value = Path(data.get(split, f"images/{split}"))
        self.image_dir = split_value if split_value.is_absolute() else root / split_value
        self.patch_size = patch_size
        self.paths = list_images(self.image_dir)
        if not self.paths:
            raise RuntimeError(f"No images found for real split: {self.image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            tensor = TF.to_tensor(image.convert("RGB"))
        _, height, width = tensor.shape
        if min(height, width) < self.patch_size:
            scale = self.patch_size / min(height, width)
            tensor = TF.resize(tensor, (round(height * scale), round(width * scale)), antialias=True)
            _, height, width = tensor.shape
        if height > self.patch_size and width > self.patch_size:
            top = torch.randint(0, height - self.patch_size + 1, ()).item()
            left = torch.randint(0, width - self.patch_size + 1, ()).item()
        else:
            top = max((height - self.patch_size) // 2, 0)
            left = max((width - self.patch_size) // 2, 0)
        return tensor[:, top : top + self.patch_size, left : left + self.patch_size]


def discover_scenarios(data_root: Path) -> list[str]:
    scenarios_dir = data_root / "scenarios"
    return sorted(p.name for p in scenarios_dir.iterdir() if (p / "input").exists() and (p / "gt").exists())


def resolve_amp(args: argparse.Namespace, device: torch.device) -> bool:
    if args.no_amp:
        return False
    if args.amp:
        return device.type == "cuda"
    return device.type == "cuda"


def set_reproducibility_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_warmup_scale(args: argparse.Namespace, epoch: int) -> float:
    warmup_epochs = int(args.task_loss_warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


def parse_csv_floats(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def resolve_hook_layers(detector: Any, raw: str) -> Optional[list[str]]:
    raw = raw.strip()
    if not raw:
        return None
    core = getattr(detector, "model", detector)
    module_names = set(dict(core.named_modules()).keys())
    resolved: list[str] = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        candidates = [item]
        if item.isdigit():
            candidates.append(f"model.{item}")
        hit = next((candidate for candidate in candidates if candidate in module_names), None)
        if hit is None:
            raise ValueError(f"Detector hook layer '{item}' was not found. Tried {candidates}.")
        resolved.append(hit)
    return resolved


def layer_weight_map(layer_names: Optional[list[str]], raw_weights: str) -> dict[str, float]:
    weights = parse_csv_floats(raw_weights)
    if not layer_names or not weights:
        return {}
    return {name: weights[min(index, len(weights) - 1)] for index, name in enumerate(layer_names)}


def build_task_loss(args: argparse.Namespace, device: torch.device) -> Optional[CompositeTaskLoss]:
    if not args.use_task_losses:
        return None
    if not args.tdp_yolo_weights:
        raise ValueError("--tdp-yolo-weights is required when --use-task-losses is enabled.")

    from ultralytics import YOLO

    detector = YOLO(args.tdp_yolo_weights)
    raw_layers = args.detector_hook_layers.strip() or args.tdp_layers.strip()
    hook_layers = resolve_hook_layers(detector, raw_layers)
    detector_size = (args.detector_input_size, args.detector_input_size) if args.detector_input_size > 0 else None
    extractor = FrozenDetectorFeatureExtractor(
        detector=detector,
        layer_names=hook_layers,
        max_layers=args.detector_max_hook_layers,
        input_size=detector_size,
        normalize_imagenet=False,
        verbose=True,
    ).to(device)
    tdp = TaskDrivenPerceptualLoss(
        feature_extractor=extractor,
        layer_weights=layer_weight_map(extractor.layer_names, args.tdp_layer_weights),
        cqmix_grid=args.cqmix_grid,
        cqmix_prob=args.cqmix_prob,
        defect_mask_weight=args.tdp_defect_mask_weight,
        defect_mask_power=args.tdp_defect_mask_power,
    )
    detector_supervised = FrozenDetectorSupervisedLoss(
        detector=detector,
        input_size=detector_size,
        cqmix_grid=args.cqmix_grid,
        cqmix_prob=args.detector_supervised_cqmix_prob,
        clean_hinge_weight=args.detector_supervised_clean_hinge,
    )
    active_contour = None
    if args.lambda_active_contour > 0:
        active_contour = ActiveContourGeometryLoss(
            mu=args.tdac_mu,
            epsilon=args.tdac_epsilon,
            support_floor=1e-4,
            eikonal_weight=args.tdac_eikonal_weight,
            region_weight=1.0,
        )
    anchor = DetectorInputAnchorLoss(
        feature_extractor=extractor,
        layer_weights=layer_weight_map(extractor.layer_names, args.tdp_layer_weights),
    )
    task_loss = CompositeTaskLoss(
        tdp_loss=tdp if args.lambda_tdp > 0 else None,
        detector_supervised_loss=(
            detector_supervised
            if args.lambda_detector_supervised > 0
            else None
        ),
        active_contour_loss=active_contour,
        feature_extractor=extractor if args.lambda_jacobian > 0 else None,
        detector_anchor_loss=anchor if args.lambda_detector_input_anchor > 0 else None,
        weights=TaskLossWeights(
            tdp=args.lambda_tdp,
            detector_supervised=args.lambda_detector_supervised,
            jacobian=args.lambda_jacobian,
            active_contour=args.lambda_active_contour,
            detector_input_anchor=args.lambda_detector_input_anchor,
            evidence_nonregression=args.lambda_evidence_nonregression,
            detail_copy=args.lambda_detail_copy,
            restoration_magnitude=args.lambda_restoration_magnitude,
            low_evidence_identity=args.lambda_low_evidence_identity,
        ),
        jacobian_probes=args.jacobian_probes,
        evidence_lower_fraction=args.evidence_lower_fraction,
    ).to(device)
    return task_loss


def save_checkpoint(
    path: Path,
    model: RMRNet,
    args: argparse.Namespace,
    epoch: int,
    metrics: Optional[dict[str, Any]] = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "arch": {
            "width": args.width,
            "code_dim": model.code_dim,
            "use_defect_attention": not args.no_defect_attention,
            "use_estimated_code": args.code_source in {
                "estimated",
                "fused",
                "metadata_fused",
                "sensor_fused",
            },
            "code_fusion": args.code_source,
            "block_type": args.block_type,
            "attention_type": args.attention_type,
            "conditioning": args.conditioning,
            "use_tdac_head": args.use_tdac_head or (args.use_task_losses and args.lambda_active_contour > 0),
            "detail_preserve": args.detail_preserve,
            "detail_gain": args.detail_gain,
            "use_cause_experts": args.cause_experts,
            "cause_expert_gain": args.cause_expert_gain,
            "cause_compound_boost": args.cause_compound_boost,
            "exact_metadata_mode": args.cause_experts,
            "metadata_encoding": args.metadata_encoding,
            "use_motion_prior": bool(args.use_motion_prior),
            "motion_prior_k": float(args.motion_prior_k),
            "motion_prior_blend": float(args.motion_prior_blend),
            "motion_prior_nuisance_decay": float(args.motion_prior_nuisance_decay),
            "motion_prior_compound_floor": float(args.motion_prior_compound_floor),
            "motion_prior_adaptive_k": float(args.motion_prior_adaptive_k),
            "motion_prior_compound_source": bool(
                args.motion_prior_compound_source
            ),
            "practical_nuisance_deadzone": float(
                args.practical_nuisance_deadzone
            ),
            "use_practical_sensor_encoder": bool(args.use_practical_sensor_encoder),
            "sensor_dim": int(args.sensor_dim),
            "sensor_gyro_full_scale": float(args.sensor_gyro_full_scale),
            "sensor_residual_scale": float(args.sensor_residual_scale),
            "use_sensor_prior_fusion": bool(args.sensor_prior_fusion),
            "use_sensor_image_psf_refiner": bool(
                args.sensor_image_psf_refiner
            ),
            "use_post_prior_evidence_refiner": bool(
                args.post_prior_evidence_refiner
            ),
            "post_prior_refiner_gain": float(
                args.post_prior_refiner_gain
            ),
            "post_prior_refiner_support": (
                args.post_prior_refiner_support
            ),
            "practical_prior_source": args.practical_prior_source,
            "use_spatial_physics": bool(args.spatial_physics),
            "physics_samples": int(args.physics_samples),
            "physics_exposure_min_ms": float(args.physics_exposure_min_ms),
            "physics_exposure_max_ms": float(args.physics_exposure_max_ms),
            "physics_focal_ratio": float(args.physics_focal_ratio),
            "physics_calibration_reliability": float(
                args.physics_calibration_reliability
            ),
            "physics_activation_motion_px": float(
                args.physics_activation_motion_px
            ),
            "physics_exclusive_trajectory": bool(
                args.physics_exclusive_trajectory
            ),
            "use_physics_inverse_candidate": bool(args.physics_inverse_candidate),
            "physics_inverse_iterations": int(args.physics_inverse_iterations),
            "physics_inverse_blend": float(args.physics_inverse_blend),
            "physics_decoder_motion_threshold_px": float(
                args.physics_decoder_motion_threshold_px
            ),
            "physics_decoder_motion_transition_px": float(
                args.physics_decoder_motion_transition_px
            ),
        },
        "epoch": epoch,
        "metrics": metrics or {},
        "args": vars(args),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["amp_scaler"] = scaler.state_dict()
    torch.save(checkpoint, path)


def append_selection_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prepare_codes(args: argparse.Namespace, codes: torch.Tensor, metadata_codes: torch.Tensor, *, training: bool = True) -> torch.Tensor | None:
    if args.code_source == "zero":
        return torch.zeros_like(codes)
    if args.code_source == "estimated":
        return None
    model_codes = (
        metadata_codes
        if args.code_source in {"metadata", "metadata_fused", "sensor_fused"}
        else codes
    )
    if training and args.code_source in {"metadata", "metadata_fused", "sensor_fused"}:
        if args.metadata_dropout > 0:
            if args.code_source == "sensor_fused":
                if args.metadata_availability_mode == "balanced":
                    model_codes = balanced_sensor_dropout(
                        model_codes,
                        args.metadata_dropout,
                    )
                else:
                    model_codes = structured_sensor_dropout(
                        model_codes,
                        args.metadata_dropout,
                    )
            else:
                keep = (
                    torch.rand(
                        model_codes.shape[0],
                        1,
                        device=model_codes.device,
                    )
                    >= args.metadata_dropout
                ).to(model_codes.dtype)
                model_codes = model_codes * keep
        if args.metadata_noise > 0:
            if args.code_source == "sensor_fused":
                model_codes = perturb_sensor_packet(model_codes, args.metadata_noise)
            else:
                model_codes = torch.clamp(
                    model_codes + torch.randn_like(model_codes) * args.metadata_noise,
                    0.0,
                    1.0,
                )
    return model_codes


def prepare_detector_targets(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move crop-aligned training-only YOLO targets to the active device."""

    required = ("detector_classes", "detector_bboxes", "detector_valid")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(
            "Frozen supervised detector loss requires crop-aligned targets; "
            f"missing batch fields: {missing}"
        )
    return {
        "classes": batch["detector_classes"].to(
            device,
            non_blocking=True,
        ),
        "bboxes": batch["detector_bboxes"].to(
            device,
            non_blocking=True,
        ),
        "valid": batch["detector_valid"].to(
            device,
            non_blocking=True,
        ),
    }


def make_mismatched_metadata_codes(metadata_codes: torch.Tensor) -> torch.Tensor:
    """Create a deterministic counterfactual metadata code for training.

    The goal is to give the metadata reliability gate a physically meaningful
    hard negative. For compound degradations, a random vector rotation is too
    weak: the model can still see a dataset-level prior. Instead, the negative
    keeps part of the cause and removes the component that changes the required
    restoration. For mixed motion-plus-low-light samples, z_m^- keeps motion
    but drops low-light, forcing the interaction basis to carry useful signal.

        z_m^- = remove_low_light(z_m)       if motion(z_m) and low_light(z_m)
              = swap_to_motion(z_m)         if low_light-only/defocus
              = swap_to_defocus(z_m)        if motion-only
              = [roll(z_m[0:7]), 1-.5s]     otherwise.

    This supports the paper's metadata-reliability analysis by explicitly
    optimizing aligned metadata against mismatched metadata, instead of merely
    passing both through the same network and hoping the gate learns the
    difference.
    """

    original = metadata_codes.detach().clone()
    if original.shape[1] > 8:
        # Practical telemetry uses a deterministic cause-changing intervention.
        # Batch rolling can pair two records from the same oversampled scenario
        # and provide almost no reliability-gate supervision.
        return counterfactual_sensor_packet(original)
    original = original.clamp(0.0, 1.0)
    if original.shape[1] <= 1:
        return (1.0 - original).clamp(0.0, 1.0)

    motion = original[:, :3].amax(dim=1, keepdim=True)
    low = original[:, 5:6] if original.shape[1] > 5 else torch.zeros_like(motion)
    defocus = original[:, 3:4] if original.shape[1] > 3 else torch.zeros_like(motion)
    noise = original[:, 4:5] if original.shape[1] > 4 else torch.zeros_like(motion)
    severity = original[:, -1:]

    rotated = original.clone()
    if rotated.shape[1] > 2:
        rotated[:, :-1] = torch.roll(rotated[:, :-1], shifts=1, dims=1)
    rotated[:, -1:] = (1.0 - 0.5 * severity).clamp(0.0, 1.0)
    wrong = rotated

    mixed_motion_low = (motion > 0.05) & (low > 0.05)
    motion_only = (motion > 0.05) & (low <= 0.05) & (defocus <= 0.05)
    low_or_defocus = ((low > 0.05) | (defocus > 0.05)) & (motion <= 0.05)

    if original.shape[1] > 5:
        keep_motion_drop_low = original.clone()
        keep_motion_drop_low[:, 5] = 0.0
        keep_motion_drop_low[:, -1:] = torch.maximum(motion, torch.maximum(defocus, noise))
        wrong = torch.where(mixed_motion_low.expand_as(wrong), keep_motion_drop_low, wrong)

    swap_to_defocus = torch.zeros_like(original)
    if swap_to_defocus.shape[1] > 3:
        swap_to_defocus[:, 3:4] = torch.maximum(motion, severity * 0.6)
    swap_to_defocus[:, -1:] = torch.maximum(swap_to_defocus[:, 3:4], 1.0 - 0.5 * severity)
    wrong = torch.where(motion_only.expand_as(wrong), swap_to_defocus, wrong)

    swap_to_motion = torch.zeros_like(original)
    swap_to_motion[:, 2:3] = torch.maximum(low, defocus).clamp(0.0, 1.0)
    swap_to_motion[:, -1:] = torch.maximum(swap_to_motion[:, 2:3], 1.0 - 0.5 * severity)
    wrong = torch.where(low_or_defocus.expand_as(wrong), swap_to_motion, wrong)
    return wrong.clamp(0.0, 1.0)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


def model_forward(
    model: RMRNet,
    inputs: torch.Tensor,
    model_codes: torch.Tensor | None,
    args: argparse.Namespace,
    *,
    need_aux: bool,
    state_only: bool = False,
    physics_codes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    if state_only:
        if model_codes is None:
            raise ValueError("Physical-state-only training requires metadata.")
        return model.physical_state_forward(inputs, model_codes)
    result = model(
        inputs,
        model_codes,
        return_aux=True if need_aux else False,
        return_dict=True if need_aux else False,
        gate_threshold=args.gate_threshold if args.gate_threshold >= 0 else None,
        gate_softness=args.gate_softness,
        physics_code=physics_codes,
    )
    if isinstance(result, dict):
        return result
    return {"restored": result, "phi": None, "lambda1": None, "lambda2": None}


def metadata_objective_enabled(args: argparse.Namespace) -> bool:
    return (
        args.code_source in {"metadata_fused", "sensor_fused"}
        and (
            args.lambda_metadata_advantage > 0
            or args.lambda_metadata_gate > 0
            or args.lambda_metadata_tdp_advantage > 0
            or args.lambda_sensor_prior_gate > 0
        )
    )


def metadata_gate_tensor(
    result: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor | None, str]:
    """Return the deployed metadata-trust tensor for either input interface.

    Compact scenario metadata uses the legacy reliability-gated basis fuser and
    exposes ``metadata_alpha``.  Practical 82-value camera/IMU/vehicle packets
    are reconciled with degraded-image evidence inside
    ``SensorImagePSFRefiner`` and expose ``physical_sensor_weight``.  Both are
    per-cause trust values in [0, 1]; supervising the latter prevents the old
    gate objective from being silently absent on the practical sensor path.
    """

    alpha = result.get("metadata_alpha")
    if isinstance(alpha, torch.Tensor):
        return alpha, "metadata_alpha"
    alpha = result.get("physical_sensor_weight")
    if isinstance(alpha, torch.Tensor):
        return alpha, "physical_sensor_weight"
    return None, "unavailable"


def physical_state_objective_enabled(args: argparse.Namespace) -> bool:
    """Return whether train-only physical-state supervision needs aux outputs."""

    return (
        args.code_source == "sensor_fused"
        and (
            args.lambda_sensor_physical > 0
            or args.lambda_image_physical > 0
            or args.lambda_posterior_physical > 0
            or args.sensor_psf_geometry_weight > 0
            or args.image_psf_geometry_weight > 0
        )
    )


def masked_sensor_state_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    *,
    coordinate_weight: torch.Tensor | None = None,
    sample_available: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise only degradation coordinates observable by present sensors.

    Partial metadata is a normal operating condition, not a binary ablation.
    Camera, IMU, and vehicle fields support different degradation causes.  The
    sensor-only branches must therefore not be penalized for coordinates whose
    required modality is absent:

        L_sensor = sum(b_i q_ij w_j ell(z_ij, z*_ij))
                   / sum(b_i q_ij w_j),

    where ``q_ij`` is the detached cause-wise sensor support and ``b_i`` marks
    availability of a private train/validation calibration target.  The fused
    image-plus-sensor posterior is supervised separately over every coordinate,
    so unsupported causes fall back to image evidence instead of being ignored.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"Sensor state shapes differ: {prediction.shape} vs {target.shape}"
        )
    if support.shape != prediction.shape:
        raise ValueError(
            f"Sensor support shape {support.shape} does not match "
            f"prediction {prediction.shape}"
        )
    weight = support.detach().to(
        device=prediction.device,
        dtype=prediction.dtype,
    ).clamp(0.0, 1.0)
    if coordinate_weight is not None:
        weight = weight * coordinate_weight.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
    if sample_available is not None:
        available = sample_available.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).reshape(prediction.shape[0], 1)
        weight = weight * available
    element_loss = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    )
    active_weight = weight.sum()
    loss = (element_loss * weight).sum() / active_weight.clamp_min(1.0)
    active_fraction = (weight > 0).to(prediction.dtype).mean()
    return loss, active_fraction


def available_physical_state_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_available: torch.Tensor,
    *,
    coordinate_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise a joint/image physical state wherever a private target exists.

    Unlike ``masked_sensor_state_loss``, this objective is not masked by packet
    availability. Missing telemetry is precisely when the image branch must
    learn to recover the state. The private renderer state is a train/validation
    label and is never passed to ``model.forward``.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"Physical state shapes differ: {prediction.shape} vs {target.shape}"
        )
    available = sample_available.to(
        device=prediction.device,
        dtype=prediction.dtype,
    ).reshape(prediction.shape[0], 1)
    weight = torch.ones_like(prediction) * available
    if coordinate_weight is not None:
        weight = weight * coordinate_weight.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
    pointwise = F.smooth_l1_loss(prediction, target, reduction="none")
    return (pointwise * weight).sum() / weight.sum().clamp_min(1.0)


def add_metadata_central_losses(
    *,
    model: RMRNet,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    metadata_codes: torch.Tensor,
    result: dict[str, torch.Tensor | None],
    base_loss: torch.Tensor,
    criterion: RCADLoss,
    task_loss: Optional[CompositeTaskLoss],
    warmup_scale: float,
    args: argparse.Namespace,
    need_aux: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Counterfactual losses that calibrate the metadata reliability gate.

    Let z_m be the aligned metadata code, z_hat the image-estimated code, and
    z_m^- a deliberately mismatched metadata code.  For metadata-reliability
    runs, the fused model is optimized with two counterfactual terms:

        L_meta = max(0, L(I_r(z_m), I_c)
                        - min[L(I_r(z_hat), I_c), L(I_r(z_m^-), I_c)]
                        + margin)

        L_meta-tdp = max(0, TDP(I_r(z_m), I_c)
                            - min[TDP(I_r(z_hat), I_c), TDP(I_r(z_m^-), I_c)]
                            + margin)

        L_gate = ||alpha(z_m, z_hat)-alpha^+||_2^2
               + ||alpha(z_m^-, z_hat)-alpha^-||_2^2

    The image-only and mismatched restoration losses are detached in L_meta, so
    the model cannot improve the objective by making control branches worse.
    The mismatched gate path remains differentiable for L_gate, forcing the same
    reliability module to trust aligned metadata and reject counterfactual
    metadata.
    """

    if not metadata_objective_enabled(args):
        return {}, {}

    terms: dict[str, torch.Tensor] = {}
    logs: dict[str, float] = {}

    wrong_codes = make_mismatched_metadata_codes(metadata_codes)
    image_only_result: dict[str, torch.Tensor | None] | None = None
    wrong_result: dict[str, torch.Tensor | None] | None = None
    has_signal = (metadata_codes.detach().abs().sum(dim=1, keepdim=True) > 1e-6).to(metadata_codes.dtype)
    metadata_presence = has_signal.mean()

    needs_control_restores = (
        args.lambda_metadata_advantage > 0
        or args.lambda_metadata_tdp_advantage > 0
        or args.lambda_sensor_prior_gate > 0
    )
    if needs_control_restores:
        # L_meta uses the image-only branch only as a fixed counterfactual
        # comparator.  Building its restorer graph cannot improve that branch
        # because the comparator is stop-gradient in Eqs. (metadata advantage),
        # and needlessly increases GPU memory.  The aligned branch remains
        # fully differentiable; the mismatched branch remains differentiable
        # because its reliability coefficient is optimized by L_gate below.
        with torch.no_grad():
            image_only_result = model_forward(model, inputs, None, args, need_aux=False)
        if args.lambda_metadata_gate > 0 or args.lambda_sensor_prior_gate > 0:
            wrong_result = model_forward(
                model,
                inputs,
                wrong_codes,
                args,
                need_aux=True,
            )
        else:
            # The wrong-metadata branch is only a fixed comparator when gate
            # supervision is disabled. Keeping it out of the autograd graph
            # reduces memory without changing the aligned metadata gradient.
            with torch.no_grad():
                wrong_result = model_forward(
                    model,
                    inputs,
                    wrong_codes,
                    args,
                    need_aux=False,
                )

    if args.lambda_metadata_advantage > 0:
        if image_only_result is None or wrong_result is None:
            raise RuntimeError("Metadata restoration advantage requested but control branches were not computed.")
        image_loss = criterion(image_only_result["restored"], targets, inputs, None).detach()
        wrong_loss = criterion(wrong_result["restored"], targets, inputs, wrong_codes).detach()
        control_loss = torch.minimum(image_loss, wrong_loss)
        advantage = F.relu(base_loss - control_loss + float(args.metadata_advantage_margin)) * metadata_presence
        weighted_advantage = float(args.lambda_metadata_advantage) * advantage
        terms["metadata_advantage"] = weighted_advantage
        logs.update(
            {
                "metadata_correct_loss": float(base_loss.detach().cpu()),
                "metadata_image_control_loss": float(image_loss.detach().cpu()),
                "metadata_wrong_control_loss": float(wrong_loss.detach().cpu()),
                "metadata_best_control_loss": float(control_loss.detach().cpu()),
                "metadata_presence": float(metadata_presence.detach().cpu()),
                "metadata_advantage_unweighted": float(advantage.detach().cpu()),
                "metadata_advantage_loss": float(weighted_advantage.detach().cpu()),
            }
        )

    if args.lambda_metadata_tdp_advantage > 0:
        if task_loss is None or task_loss.tdp_loss is None:
            raise RuntimeError("--lambda-metadata-tdp-advantage requires --use-task-losses with TDP enabled.")
        if image_only_result is None or wrong_result is None:
            raise RuntimeError("Metadata TDP advantage requested but control branches were not computed.")
        correct_tdp = task_loss.tdp_loss(result["restored"], targets, defect_mask=None, use_cqmix=False)
        with torch.no_grad():
            image_tdp = task_loss.tdp_loss(
                image_only_result["restored"].detach(),
                targets,
                defect_mask=None,
                use_cqmix=False,
            )
            wrong_tdp = task_loss.tdp_loss(
                wrong_result["restored"].detach(),
                targets,
                defect_mask=None,
                use_cqmix=False,
            )
            control_tdp = torch.minimum(image_tdp, wrong_tdp)
        tdp_advantage = F.relu(
            correct_tdp - control_tdp + float(args.metadata_tdp_advantage_margin)
        ) * metadata_presence
        weighted_tdp_advantage = float(args.lambda_metadata_tdp_advantage) * float(warmup_scale) * tdp_advantage
        terms["metadata_tdp_advantage"] = weighted_tdp_advantage
        logs.update(
            {
                "metadata_tdp_correct": float(correct_tdp.detach().cpu()),
                "metadata_tdp_image_control": float(image_tdp.detach().cpu()),
                "metadata_tdp_wrong_control": float(wrong_tdp.detach().cpu()),
                "metadata_tdp_best_control": float(control_tdp.detach().cpu()),
                "metadata_tdp_advantage_unweighted": float(tdp_advantage.detach().cpu()),
                "metadata_tdp_advantage_loss": float(weighted_tdp_advantage.detach().cpu()),
            }
        )

    if args.lambda_metadata_gate > 0:
        if wrong_result is None:
            wrong_result = model_forward(model, inputs, wrong_codes, args, need_aux=True)
        alpha, alpha_source = metadata_gate_tensor(result)
        wrong_alpha, wrong_alpha_source = metadata_gate_tensor(wrong_result)
        if isinstance(alpha, torch.Tensor) and isinstance(wrong_alpha, torch.Tensor):
            target_high = torch.full_like(alpha, float(args.metadata_gate_target))
            target_low = torch.full_like(alpha, float(args.metadata_negative_gate_target))
            signal = has_signal.to(device=alpha.device, dtype=alpha.dtype)
            cause_reliability = result.get("sensor_cause_reliability")
            if isinstance(cause_reliability, torch.Tensor):
                # The deployed gate is alpha = reliability * sigmoid(logits).
                # An unobservable cause therefore cannot have alpha=0.85.
                # Match the positive target to stop-gradient observability.
                positive_support = cause_reliability.detach().to(
                    device=alpha.device,
                    dtype=alpha.dtype,
                ).clamp(0.0, 1.0)
                target_pos = signal * target_high * positive_support
            else:
                positive_support = torch.ones_like(alpha)
                target_pos = signal * target_high + (1.0 - signal) * target_low
            target_neg = torch.full_like(wrong_alpha, float(args.metadata_negative_gate_target))
            gate_unweighted = F.mse_loss(alpha, target_pos) + F.mse_loss(wrong_alpha, target_neg)
            weighted_gate = float(args.lambda_metadata_gate) * gate_unweighted
            terms["metadata_gate"] = weighted_gate
            logs.update(
                {
                    "metadata_gate_unweighted": float(gate_unweighted.detach().cpu()),
                    "metadata_gate_loss": float(weighted_gate.detach().cpu()),
                    "metadata_alpha_correct_mean": float(alpha.detach().mean().cpu()),
                    "metadata_alpha_wrong_mean": float(wrong_alpha.detach().mean().cpu()),
                    "metadata_gate_is_practical": float(
                        alpha_source == "physical_sensor_weight"
                    ),
                    "metadata_wrong_gate_is_practical": float(
                        wrong_alpha_source == "physical_sensor_weight"
                    ),
                    "metadata_positive_gate_target_mean": float(
                        target_pos.detach().mean().cpu()
                    ),
                    "metadata_cause_reliability_mean": float(
                        positive_support.detach().mean().cpu()
                    ),
                }
            )
        else:
            raise RuntimeError(
                "Metadata gate loss was requested, but model output did not include "
                "metadata_alpha or physical_sensor_weight. Use metadata_fused "
                "or sensor_fused with return_dict=True/return_aux=True."
            )

    if args.lambda_sensor_prior_gate > 0:
        if wrong_result is None:
            wrong_result = model_forward(
                model,
                inputs,
                wrong_codes,
                args,
                need_aux=True,
            )

        def prior_gate_term(
            branch: dict[str, torch.Tensor | None],
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            alpha = branch.get("sensor_prior_alpha")
            neural = branch.get("neural_restored")
            physical = branch.get("motion_prior")
            cause_gate = branch.get("motion_prior_gate")
            reliability = branch.get("sensor_cause_reliability")
            tensors = (alpha, neural, physical, cause_gate, reliability)
            if not all(isinstance(value, torch.Tensor) for value in tensors):
                raise RuntimeError(
                    "--lambda-sensor-prior-gate requires practical sensor-prior "
                    "fusion outputs from return_dict=True."
                )

            # Train-time oracle preference for the confidence map:
            #
            #   a* = g_motion g_sensor sigmoid((e_n - e_p) / temperature),
            #
            # where e_n and e_p are local errors of the neural and physical
            # candidates. Only a* is detached; no clean target is available at
            # inference. IMU reliability can support motion coordinates only.
            neural_error = (neural - targets).abs().mean(dim=1, keepdim=True)
            physical_error = (physical - targets).abs().mean(dim=1, keepdim=True)
            preference = torch.sigmoid(
                (neural_error - physical_error) / 0.02
            )
            motion_support = cause_gate[:, None, None, None].to(alpha.dtype)
            sensor_support = reliability[:, :3].mean(
                dim=1, keepdim=True
            )[:, :, None, None].to(alpha.dtype)
            target_alpha = (
                preference * motion_support * sensor_support
            ).detach()

            candidate_gap = (physical - neural).abs().mean(
                dim=1, keepdim=True
            ).detach()
            gap_scale = candidate_gap.flatten(1).amax(
                dim=1
            )[:, None, None, None].clamp_min(1e-4)
            importance = 0.25 + 0.75 * (candidate_gap / gap_scale).clamp(
                0.0, 1.0
            )
            loss = (importance * (alpha - target_alpha).square()).mean()
            return loss, alpha.detach().mean(), target_alpha.mean()

        correct_prior_loss, correct_prior_alpha, correct_prior_target = (
            prior_gate_term(result)
        )
        wrong_prior_loss, wrong_prior_alpha, wrong_prior_target = (
            prior_gate_term(wrong_result)
        )
        prior_gate_loss = 0.5 * (
            correct_prior_loss + wrong_prior_loss
        )
        weighted_prior_gate = (
            float(args.lambda_sensor_prior_gate) * prior_gate_loss
        )
        terms["sensor_prior_gate"] = weighted_prior_gate
        logs.update(
            {
                "sensor_prior_gate_unweighted": float(
                    prior_gate_loss.detach().cpu()
                ),
                "sensor_prior_gate_loss": float(
                    weighted_prior_gate.detach().cpu()
                ),
                "sensor_prior_alpha_correct_mean": float(
                    correct_prior_alpha.cpu()
                ),
                "sensor_prior_target_correct_mean": float(
                    correct_prior_target.detach().cpu()
                ),
                "sensor_prior_alpha_wrong_mean": float(
                    wrong_prior_alpha.cpu()
                ),
                "sensor_prior_target_wrong_mean": float(
                    wrong_prior_target.detach().cpu()
                ),
            }
        )

    # Alpha is an audit signal even when no explicit gate target is optimized.
    # Recording it prevents a nominal metadata model with alpha ~= 0 from being
    # misreported as evidence that external metadata affected restoration.
    alpha, alpha_source = metadata_gate_tensor(result)
    if isinstance(alpha, torch.Tensor):
        logs["metadata_alpha_correct_mean"] = float(alpha.detach().mean().cpu())
        logs["metadata_gate_is_practical"] = float(
            alpha_source == "physical_sensor_weight"
        )
    if wrong_result is not None:
        wrong_alpha, wrong_alpha_source = metadata_gate_tensor(wrong_result)
        if isinstance(wrong_alpha, torch.Tensor):
            logs["metadata_alpha_wrong_mean"] = float(wrong_alpha.detach().mean().cpu())
            logs["metadata_wrong_gate_is_practical"] = float(
                wrong_alpha_source == "physical_sensor_weight"
            )

    return terms, logs


@torch.no_grad()
def validate(model: RMRNet, loader: DataLoader, criterion: RCADLoss, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["gt"].to(device, non_blocking=True)
        codes = batch["code"].to(device, non_blocking=True)
        metadata_codes = batch["metadata_code"].to(device, non_blocking=True)
        model_codes = prepare_codes(args, codes, metadata_codes, training=False)
        result = model_forward(
            model,
            inputs,
            model_codes,
            args,
            need_aux=False,
            physics_codes=metadata_codes if args.spatial_physics else None,
        )
        outputs = result["restored"]
        loss = criterion(outputs, targets, inputs, model_codes)
        batch_count = inputs.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_psnr += float(psnr(outputs, targets).sum().detach().cpu())
        count += batch_count
    return {"val_loss": total_loss / max(count, 1), "val_psnr": total_psnr / max(count, 1)}




def _tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, float]:
    t = tensor.detach()
    return {
        f"{name}_mean": float(t.mean().cpu()),
        f"{name}_std": float(t.std(unbiased=False).cpu()),
        f"{name}_min": float(t.min().cpu()),
        f"{name}_max": float(t.max().cpu()),
    }


def debug_training_stats(
    *,
    epoch: int,
    batch_index: int,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    model_codes: torch.Tensor | None,
    metadata_codes: torch.Tensor,
    result: dict[str, torch.Tensor | None],
    loss_value: torch.Tensor,
    base_loss: torch.Tensor,
    task_logs: dict[str, float] | None = None,
    grad_norm: float | None = None,
    tag: str = "train_debug",
) -> dict[str, Any]:
    with torch.no_grad():
        residual = outputs - inputs
        batch_psnr = psnr(outputs, targets).mean()
        ev_in = road_evidence_vector(inputs).mean(dim=0)
        ev_out = road_evidence_vector(outputs).mean(dim=0)
        ev_gt = road_evidence_vector(targets).mean(dim=0)
        payload: dict[str, Any] = {
            "tag": tag,
            "epoch": epoch,
            "batch_index": batch_index,
            "loss_total": float(loss_value.detach().cpu()),
            "loss_base": float(base_loss.detach().cpu()),
            "psnr_batch": float(batch_psnr.detach().cpu()),
            "residual_abs_mean": float(residual.abs().mean().detach().cpu()),
            "residual_abs_p95": float(torch.quantile(residual.abs().flatten(), 0.95).detach().cpu()),
            "residual_abs_max": float(residual.abs().max().detach().cpu()),
            "restored_changed_fraction_gt_0p01": float((residual.abs() > 0.01).float().mean().detach().cpu()),
        }
        payload.update(_tensor_stats("input", inputs))
        payload.update(_tensor_stats("restored", outputs))
        payload.update(_tensor_stats("target", targets))
        payload.update(_tensor_stats("metadata_code", metadata_codes))
        if model_codes is not None:
            payload.update(_tensor_stats("model_code", model_codes))
        for idx, label in enumerate(["edge", "contrast", "highfreq", "saturation"]):
            payload[f"evidence_{label}_input"] = float(ev_in[idx].detach().cpu())
            payload[f"evidence_{label}_restored"] = float(ev_out[idx].detach().cpu())
            payload[f"evidence_{label}_target"] = float(ev_gt[idx].detach().cpu())
        for aux_name in (
            "phi",
            "lambda1",
            "lambda2",
            "severity",
            "gate",
            "detail_gate",
            "detail_residual",
            "metadata_alpha",
            "metadata_alpha_mean",
            "metadata_alpha_min",
            "metadata_alpha_max",
            "metadata_has_signal",
            "metadata_disagreement",
            "metadata_severity",
            "metadata_severity_gap",
        ):
            value = result.get(aux_name)
            if isinstance(value, torch.Tensor):
                payload.update(_tensor_stats(aux_name, value))
        if task_logs:
            for key, value in task_logs.items():
                if isinstance(value, (int, float)):
                    payload[key] = float(value)
        if grad_norm is not None:
            payload["grad_norm"] = float(grad_norm)
    return payload

def smoke_test(
    model: RMRNet,
    loader: DataLoader,
    criterion: RCADLoss,
    task_loss: Optional[CompositeTaskLoss],
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> None:
    model.train()
    batch = next(iter(loader))
    inputs = batch["input"].to(device, non_blocking=True)
    targets = batch["gt"].to(device, non_blocking=True)
    codes = batch["code"].to(device, non_blocking=True)
    metadata_codes = batch["metadata_code"].to(device, non_blocking=True)
    cause_targets = batch["cause_target"].to(device, non_blocking=True)
    physical_targets = batch["physical_target"].to(device, non_blocking=True)
    physical_target_available = batch["physical_target_available"].to(
        device,
        non_blocking=True,
    )
    detector_targets = (
        prepare_detector_targets(batch, device)
        if args.lambda_detector_supervised > 0
        else None
    )
    model_codes = prepare_codes(args, codes, metadata_codes, training=True)
    result = model_forward(
        model,
        inputs,
        model_codes,
        args,
        need_aux=task_loss is not None
        or args.use_tdac_head
        or (args.conditioning in {"gated_basis", "residual_basis"} and args.basis_sparsity_weight > 0)
        or args.lambda_sensor_cause > 0
        or physical_state_objective_enabled(args)
        or args.lambda_fused_cause > 0
        or metadata_objective_enabled(args)
        or args.lambda_physics_reblur > 0,
        state_only=args.physical_state_only,
        physics_codes=metadata_codes if args.spatial_physics else None,
    )
    outputs = result["restored"]
    if args.base_weight > 0:
        base = criterion(outputs, targets, inputs, model_codes)
        weighted_base = float(args.base_weight) * base
    else:
        # A zero coefficient must remove the branch from autograd entirely.
        # Keeping ``0 * L_base`` in the graph can still propagate NaN through
        # FFT derivatives during isolated physical-state calibration.
        with torch.no_grad():
            base = criterion(outputs, targets, inputs, model_codes)
        weighted_base = outputs.new_zeros(())
    total = weighted_base
    logs: dict[str, Any] = {
        "input_shape": list(inputs.shape),
        "restored_shape": list(outputs.shape),
        "base_loss": float(base.detach().cpu()),
        "base_weight": float(args.base_weight),
        "weighted_base_loss": float(weighted_base.detach().cpu()),
        "keys": sorted(k for k, v in result.items() if v is not None),
    }
    if args.lambda_physics_reblur > 0:
        reblurred = result.get("physics_reblurred")
        reliability = result.get("physics_reliability")
        if not isinstance(reblurred, torch.Tensor) or not isinstance(
            reliability, torch.Tensor
        ):
            raise RuntimeError(
                "--lambda-physics-reblur requires --spatial-physics and an "
                "82-value practical sensor packet"
            )
        physics_raw = physics_reblur_loss(
            reblurred,
            inputs,
            reliability,
        )
        physics_weighted = float(args.lambda_physics_reblur) * physics_raw
        total = total + physics_weighted
        logs["physics_reblur_raw"] = float(physics_raw.detach().cpu())
        logs["physics_reblur_weighted"] = float(
            physics_weighted.detach().cpu()
        )
        motion_px = result.get("physics_motion_px")
        if isinstance(motion_px, torch.Tensor):
            logs["physics_motion_px_mean"] = float(
                motion_px.detach().mean().cpu()
            )
    if (
        args.code_source
        in {"estimated", "fused", "metadata_fused", "sensor_fused"}
        and args.aux_code_weight > 0
    ):
        target_codes = (
            cause_targets
            if args.code_source == "sensor_fused"
            else metadata_codes
            if args.code_source == "metadata_fused"
            else codes
        )
        aux_code = float(args.aux_code_weight) * F.smooth_l1_loss(
            model.estimate_code(inputs),
            target_codes,
        )
        total = total + aux_code
        logs["aux_code_loss"] = float(aux_code.detach().cpu())
    if args.code_source == "sensor_fused" and args.lambda_sensor_cause > 0:
        sensor_code = result.get("sensor_code")
        sensor_support = result.get("sensor_cause_reliability")
        if not isinstance(sensor_code, torch.Tensor):
            raise RuntimeError("Smoke test did not receive sensor_code.")
        if not isinstance(sensor_support, torch.Tensor):
            raise RuntimeError(
                "Smoke test did not receive sensor_cause_reliability."
            )
        sensor_cause_unweighted, sensor_active_fraction = (
            masked_sensor_state_loss(
                sensor_code,
                cause_targets,
                sensor_support,
            )
        )
        sensor_cause = float(args.lambda_sensor_cause) * (
            sensor_cause_unweighted
        )
        total = total + sensor_cause
        logs["sensor_cause_loss"] = float(sensor_cause.detach().cpu())
        logs["sensor_cause_active_fraction"] = float(
            sensor_active_fraction.detach().cpu()
        )
    if physical_state_objective_enabled(args):
        sensor_physical = result.get("sensor_only_physical_code")
        image_physical = result.get("image_physical_code")
        posterior_physical = result.get("sensor_calibrated_physical_code")
        sensor_support = result.get("sensor_cause_reliability")
        if not isinstance(sensor_physical, torch.Tensor):
            raise RuntimeError(
                "Smoke test did not receive sensor_only_physical_code."
            )
        if not isinstance(image_physical, torch.Tensor):
            raise RuntimeError(
                "Smoke test did not receive image_physical_code."
            )
        if not isinstance(posterior_physical, torch.Tensor):
            raise RuntimeError(
                "Smoke test did not receive sensor_calibrated_physical_code."
            )
        if not isinstance(sensor_support, torch.Tensor):
            raise RuntimeError(
                "Smoke test did not receive sensor_cause_reliability."
            )
        motion_weight = float(args.sensor_motion_coordinate_weight)
        coordinate_weight = posterior_physical.new_tensor(
            [
                motion_weight,
                motion_weight,
                motion_weight,
                1.0,
                1.0,
                1.0,
                0.5,
                1.0,
            ]
        )[None, :]
        sensor_unweighted, sensor_active_fraction = (
            masked_sensor_state_loss(
                sensor_physical,
                physical_targets,
                sensor_support,
                coordinate_weight=coordinate_weight,
                sample_available=physical_target_available,
            )
        )
        sensor_loss = float(args.lambda_sensor_physical) * sensor_unweighted
        total = total + sensor_loss
        logs["sensor_physical_loss"] = float(sensor_loss.detach().cpu())
        logs["sensor_physical_active_fraction"] = float(
            sensor_active_fraction.detach().cpu()
        )
        image_unweighted = available_physical_state_loss(
            image_physical,
            physical_targets,
            physical_target_available,
            coordinate_weight=coordinate_weight,
        )
        image_loss = float(args.lambda_image_physical) * image_unweighted
        total = total + image_loss
        logs["image_physical_loss"] = float(image_loss.detach().cpu())
        posterior_unweighted = available_physical_state_loss(
            posterior_physical,
            physical_targets,
            physical_target_available,
            coordinate_weight=coordinate_weight,
        )
        posterior_loss = (
            float(args.lambda_posterior_physical) * posterior_unweighted
        )
        total = total + posterior_loss
        logs["posterior_physical_loss"] = float(
            posterior_loss.detach().cpu()
        )
        full_support = torch.ones_like(sensor_support)
        image_geometry, image_metrics = practical_psf_geometry_loss(
            image_physical,
            physical_targets,
            full_support,
            sample_available=physical_target_available,
            length_weight=args.sensor_psf_length_weight,
            vector_weight=args.sensor_psf_vector_weight,
            direction_weight=args.sensor_psf_direction_weight,
        )
        weighted_image_geometry = (
            float(args.image_psf_geometry_weight) * image_geometry
        )
        total = total + weighted_image_geometry
        logs["image_psf_geometry_loss"] = float(
            weighted_image_geometry.detach().cpu()
        )
        posterior_geometry, posterior_metrics = practical_psf_geometry_loss(
            posterior_physical,
            physical_targets,
            full_support,
            sample_available=physical_target_available,
            length_weight=args.sensor_psf_length_weight,
            vector_weight=args.sensor_psf_vector_weight,
            direction_weight=args.sensor_psf_direction_weight,
        )
        weighted_posterior_geometry = (
            float(args.sensor_psf_geometry_weight) * posterior_geometry
        )
        total = total + weighted_posterior_geometry
        logs["posterior_psf_geometry_loss"] = float(
            weighted_posterior_geometry.detach().cpu()
        )
        for name, value in image_metrics.items():
            logs[f"image_psf_{name}"] = float(value.detach().cpu())
        for name, value in posterior_metrics.items():
            logs[f"posterior_psf_{name}"] = float(value.detach().cpu())
    if args.code_source == "sensor_fused" and args.lambda_fused_cause > 0:
        posterior_code = result.get("code")
        if not isinstance(posterior_code, torch.Tensor):
            raise RuntimeError("Smoke test did not receive posterior code.")
        posterior_loss = float(args.lambda_fused_cause) * F.smooth_l1_loss(
            posterior_code,
            cause_targets,
        )
        total = total + posterior_loss
        logs["fused_cause_loss"] = float(
            posterior_loss.detach().cpu()
        )
    if task_loss is not None:
        task_value, task_logs = task_loss(
            result,
            targets,
            degraded=inputs,
            detector_targets=detector_targets,
            warmup_scale=1.0,
        )
        total = total + task_value
        logs["task_loss"] = float(task_value.detach().cpu())
        logs.update(task_logs)
    basis_sparsity = result.get("basis_sparsity")
    if (
        args.conditioning in {"gated_basis", "residual_basis"}
        and args.basis_sparsity_weight > 0
        and isinstance(basis_sparsity, torch.Tensor)
    ):
        sparse_loss = float(args.basis_sparsity_weight) * basis_sparsity
        total = total + sparse_loss
        logs["basis_sparsity_loss"] = float(sparse_loss.detach().cpu())
    metadata_terms, metadata_logs = add_metadata_central_losses(
        model=model,
        inputs=inputs,
        targets=targets,
        metadata_codes=model_codes if model_codes is not None else metadata_codes,
        result=result,
        base_loss=base,
        criterion=criterion,
        task_loss=task_loss,
        warmup_scale=1.0,
        args=args,
        need_aux=True,
    )
    for name, value in metadata_terms.items():
        total = total + value
        logs[f"{name}_loss"] = float(value.detach().cpu())
    logs.update(metadata_logs)
    total.backward()
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().norm().cpu())
    logs["total_loss"] = float(total.detach().cpu())
    logs["grad_norm_sum"] = grad_norm
    logs["finite"] = bool(torch.isfinite(total).detach().cpu())
    logs["debug_stats"] = debug_training_stats(
        epoch=0,
        batch_index=0,
        inputs=inputs,
        outputs=outputs,
        targets=targets,
        model_codes=model_codes,
        metadata_codes=metadata_codes,
        result=result,
        loss_value=total,
        base_loss=base,
        task_logs=logs,
        grad_norm=grad_norm,
        tag="smoke_debug",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_test.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    print(json.dumps(logs, indent=2), flush=True)


def run_detector_adaptation(args: argparse.Namespace, out_dir: Path, epoch: int) -> dict[str, object]:
    data = args.alternate_detector_data or args.phase2_detector_data
    weights = args.alternate_detector_weights or args.phase2_detector_weights
    if not data or not weights or args.alternate_detector_epochs <= 0:
        return {"epoch": epoch, "phase": "detector_adaptation_skipped", "reason": "missing_detector_args"}
    from ultralytics import YOLO

    detector = YOLO(weights)
    result = detector.train(
        data=data,
        epochs=args.alternate_detector_epochs,
        imgsz=args.alternate_detector_imgsz,
        project=str(out_dir / "alternate_detector"),
        name=f"epoch_{epoch:03d}",
    )
    return {"epoch": epoch, "phase": "detector_adaptation", "result": str(result)}


def is_detector_adaptation_epoch(args: argparse.Namespace, epoch: int) -> bool:
    return args.alternate_phase_period > 0 and epoch % args.alternate_phase_period == 0


def main() -> None:
    print(
        "[RMR-Net] train_rcadnet.py is kept for backward compatibility; "
        "use train_rmrnet.py for reported RMR-Net runs."
    )
    args = parse_args()
    if args.gradient_strategy == "primary_projected" and args.amp and not args.no_amp:
        raise ValueError(
            "--gradient-strategy primary_projected requires --no-amp so the "
            "two objective gradients are compared in the same scale"
        )
    if args.lambda_detector_supervised > 0:
        if not args.use_task_losses:
            raise ValueError(
                "--lambda-detector-supervised requires --use-task-losses"
            )
        if not args.defect_label_root:
            raise ValueError(
                "--lambda-detector-supervised requires --defect-label-root"
            )
    if args.code_source == "sensor_fused":
        args.use_practical_sensor_encoder = True
        if args.sensor_dim != PRACTICAL_SENSOR_DIM:
            raise ValueError(
                f"sensor_fused requires --sensor-dim {PRACTICAL_SENSOR_DIM}; "
                f"received {args.sensor_dim}"
            )
        if args.sensor_gyro_full_scale <= 0.0:
            raise ValueError("--sensor-gyro-full-scale must be greater than zero")
    if not 0.0 <= args.sensor_residual_scale <= 0.25:
        raise ValueError("--sensor-residual-scale must be in [0, 0.25]")
    if args.sensor_prior_fusion and (
        not args.use_motion_prior or not args.use_practical_sensor_encoder
    ):
        raise ValueError(
            "--sensor-prior-fusion requires --use-motion-prior and "
            "--use-practical-sensor-encoder (or --code-source sensor_fused)"
        )
    if args.sensor_image_psf_refiner and not args.use_practical_sensor_encoder:
        raise ValueError(
            "--sensor-image-psf-refiner requires "
            "--use-practical-sensor-encoder or --code-source sensor_fused"
        )
    if args.cause_expert_only_index not in {-1, 0, 1, 2, 3, 4}:
        raise ValueError("--cause-expert-only-index must be -1 or an index from 0 to 4")
    if args.cause_compound_boost < 1.0:
        raise ValueError("--cause-compound-boost must be at least 1.0")
    if args.cause_expert_only_index >= 0 and not args.cause_experts:
        raise ValueError("--cause-expert-only-index requires --cause-experts")
    if args.physical_state_only and (
        not args.sensor_image_psf_refiner
        or args.code_source != "sensor_fused"
        or args.base_weight != 0
        or args.use_task_losses
        or metadata_objective_enabled(args)
    ):
        raise ValueError(
            "--physical-state-only requires sensor_fused, "
            "--sensor-image-psf-refiner, --base-weight 0, and no detector or "
            "metadata-advantage losses. It is an isolated calibration stage."
        )
    set_reproducibility_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_roots = [Path(root) for root in args.data_root]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        print(json.dumps({"warning": "CUDA requested but unavailable; using CPU."}), flush=True)
    if args.noise_da_weight > 0:
        print(json.dumps({"warning": "Noise-DA flags are accepted for compatibility but are not active in the released composite-loss trainer."}), flush=True)

    scenarios = args.scenarios or discover_scenarios(data_roots[0])
    datasets = [
        PairedRoadRestorationDataset(
            root,
            scenarios,
            patch_size=args.patch_size,
            train=True,
            metadata_mode=args.metadata_mode,
            metadata_encoding=args.metadata_encoding,
            # Directional metadata remains in the camera coordinate frame.
            # Do not mirror the image without also transforming its telemetry.
            horizontal_flip_probability=(
                0.0
                if (
                    args.spatial_physics
                    or args.code_source
                    in {"metadata", "metadata_fused", "sensor_fused"}
                )
                else 0.5
            ),
            defect_label_root=args.defect_label_root,
            defect_crop_probability=args.defect_crop_probability,
        )
        for root in data_roots
    ]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    val_loader = None
    if args.val_data_root:
        val_roots = [Path(root) for root in args.val_data_root]
        val_scenarios = args.val_scenarios or scenarios
        val_sets = [
            PairedRoadRestorationDataset(
                root,
                val_scenarios,
                patch_size=args.patch_size,
                train=False,
                metadata_mode=args.metadata_mode,
                metadata_encoding=args.metadata_encoding,
            )
            for root in val_roots
        ]
        val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = RMRNet(
        width=args.width,
        use_defect_attention=not args.no_defect_attention,
        use_estimated_code=args.code_source in {"estimated", "fused", "metadata_fused", "sensor_fused"},
        code_fusion=args.code_source,
        block_type=args.block_type,
        attention_type=args.attention_type,
        conditioning=args.conditioning,
        use_tdac_head=args.use_tdac_head or (args.use_task_losses and args.lambda_active_contour > 0),
        detail_preserve=args.detail_preserve,
        detail_gain=args.detail_gain,
        use_cause_experts=args.cause_experts,
        cause_expert_gain=args.cause_expert_gain,
        cause_compound_boost=args.cause_compound_boost,
        exact_metadata_mode=args.cause_experts,
        use_motion_prior=args.use_motion_prior,
        motion_prior_k=args.motion_prior_k,
        motion_prior_blend=args.motion_prior_blend,
        motion_prior_nuisance_decay=args.motion_prior_nuisance_decay,
        motion_prior_compound_floor=args.motion_prior_compound_floor,
        motion_prior_adaptive_k=args.motion_prior_adaptive_k,
        motion_prior_compound_source=args.motion_prior_compound_source,
        practical_nuisance_deadzone=args.practical_nuisance_deadzone,
        use_practical_sensor_encoder=args.use_practical_sensor_encoder,
        sensor_dim=args.sensor_dim,
        sensor_gyro_full_scale=args.sensor_gyro_full_scale,
        sensor_residual_scale=args.sensor_residual_scale,
        use_sensor_prior_fusion=args.sensor_prior_fusion,
        use_sensor_image_psf_refiner=args.sensor_image_psf_refiner,
        use_post_prior_evidence_refiner=args.post_prior_evidence_refiner,
        post_prior_refiner_gain=args.post_prior_refiner_gain,
        post_prior_refiner_support=args.post_prior_refiner_support,
        practical_prior_source=args.practical_prior_source,
        use_spatial_physics=args.spatial_physics,
        physics_samples=args.physics_samples,
        physics_exposure_min_ms=args.physics_exposure_min_ms,
        physics_exposure_max_ms=args.physics_exposure_max_ms,
        physics_focal_ratio=args.physics_focal_ratio,
        physics_calibration_reliability=(
            args.physics_calibration_reliability
        ),
        physics_activation_motion_px=args.physics_activation_motion_px,
        physics_exclusive_trajectory=args.physics_exclusive_trajectory,
        use_physics_inverse_candidate=args.physics_inverse_candidate,
        physics_inverse_iterations=args.physics_inverse_iterations,
        physics_inverse_blend=args.physics_inverse_blend,
        physics_decoder_motion_threshold_px=args.physics_decoder_motion_threshold_px,
        physics_decoder_motion_transition_px=args.physics_decoder_motion_transition_px,
        enable_aux_contour=False,
    ).to(device)
    if args.init_weights and args.resume_checkpoint:
        raise ValueError("--init-weights and --resume-checkpoint are mutually exclusive")
    resume_payload: dict[str, Any] | None = None
    if args.resume_checkpoint:
        resume_payload = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(resume_payload["model"], strict=True)
        print(
            json.dumps(
                {
                    "resume_checkpoint": args.resume_checkpoint,
                    "resume_epoch": int(resume_payload.get("epoch", 0)),
                    "resume_policy": "strict model plus exact AdamW and AMP state",
                }
            ),
            flush=True,
        )
    elif args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location=device)
        if hasattr(model, "load_pretrained"):
            load_report = model.load_pretrained(checkpoint, strict=False)
            print(
                json.dumps(
                    {
                        "checkpoint_load_policy": "shape-compatible transfer",
                        "checkpoint_loaded_tensors": len(load_report["loaded"]),
                        "checkpoint_missing_keys": load_report["missing_keys"],
                        "checkpoint_unexpected_keys": load_report["unexpected_keys"],
                        "checkpoint_skipped_shape": load_report["skipped_shape"],
                        "checkpoint_skipped_missing": load_report["skipped_missing"],
                    }
                ),
                flush=True,
            )
        else:
            incompatible = model.load_state_dict(checkpoint["model"], strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(
                    json.dumps(
                        {
                            "checkpoint_load_missing": incompatible.missing_keys,
                            "checkpoint_load_unexpected": incompatible.unexpected_keys,
                        }
                    ),
                    flush=True,
                )
        print(json.dumps({"loaded_init_weights": args.init_weights}), flush=True)
    if args.sensor_encoder_weights:
        if model.sensor_encoder is None:
            raise ValueError(
                "--sensor-encoder-weights requires --use-practical-sensor-encoder "
                "or --code-source sensor_fused"
            )
        sensor_payload = torch.load(args.sensor_encoder_weights, map_location=device)
        sensor_state = sensor_payload.get("sensor_encoder", sensor_payload)
        incompatible = model.sensor_encoder.load_state_dict(
            sensor_state,
            strict=False,
        )
        allowed_missing_prefixes = (
            "availability_conditioning_experts.",
            "availability_physical_experts.",
        )
        disallowed_missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Compatible practical sensor calibration load failed: "
                f"missing={disallowed_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        print(
            json.dumps(
                {
                    "loaded_sensor_encoder_weights": args.sensor_encoder_weights,
                    "sensor_calibration_epoch": sensor_payload.get("epoch"),
                    "sensor_calibration_metrics": sensor_payload.get("metrics"),
                    "sensor_calibration_test_used": sensor_payload.get("test_split_used"),
                    "initialized_availability_expert_tensors": list(
                        incompatible.missing_keys
                    ),
                }
            ),
            flush=True,
        )
    if args.reset_code_fuser:
        if resume_payload is not None:
            raise ValueError("--reset-code-fuser cannot be combined with --resume-checkpoint")
        model.reset_code_fuser()
        print(json.dumps({"reset_code_fuser": True, "conditioning": args.conditioning}), flush=True)

    criterion = RCADLoss(
        edge_weight=args.edge_weight,
        freq_weight=args.freq_weight,
        defect_weight=args.defect_weight,
        visibility_weight=args.visibility_weight,
        profile=args.loss_profile,
        charbonnier_epsilon=args.charbonnier_epsilon,
    )
    task_loss = build_task_loss(args, device)
    boosted_keywords = (
        "tdac_head",
        "detail_skip",
        "code_encoder",
        "code_fuser",
        "cause_head",
        "sensor_encoder",
        "sensor_prior_fusion",
        "sensor_image_psf_refiner",
        "post_prior_evidence_refiner",
        "physics_feature_encoder",
    )
    boosted_params = []
    base_params = []
    boosted_names = []
    for name, param in model.named_parameters():
        if any(token in name for token in boosted_keywords):
            boosted_params.append(param)
            boosted_names.append(name)
        else:
            base_params.append(param)
    param_groups = [{"params": base_params, "lr": args.lr}]
    if boosted_params:
        param_groups.append({"params": boosted_params, "lr": args.lr * max(float(args.new_head_lr_mult), 1.0)})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=resolve_amp(args, device))
    if resume_payload is not None:
        if "optimizer" not in resume_payload or "amp_scaler" not in resume_payload:
            raise RuntimeError(
                "Exact resume requested, but the checkpoint lacks optimizer/AMP state. "
                "Use --init-weights only for an explicitly disclosed fine-tuning transfer."
            )
        optimizer.load_state_dict(resume_payload["optimizer"])
        scaler.load_state_dict(resume_payload["amp_scaler"])

    audit_config = {
        "training_profile": "detector_safe_composite_release",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seed": int(args.seed),
        "data_roots": [str(p) for p in data_roots],
        "scenarios": scenarios,
        "train_size": len(dataset),
        "val_size": len(val_loader.dataset) if val_loader is not None else 0,
        "task_losses_enabled": bool(args.use_task_losses),
        "lambda_tdp": args.lambda_tdp,
        "lambda_detector_supervised": args.lambda_detector_supervised,
        "lambda_jacobian": args.lambda_jacobian,
        "lambda_active_contour": args.lambda_active_contour,
        "lambda_detector_input_anchor": args.lambda_detector_input_anchor,
        "lambda_evidence_nonregression": args.lambda_evidence_nonregression,
        "lambda_detail_copy": args.lambda_detail_copy,
        "lambda_restoration_magnitude": args.lambda_restoration_magnitude,
        "lambda_low_evidence_identity": args.lambda_low_evidence_identity,
        "gradient_strategy": args.gradient_strategy,
        "lambda_metadata_advantage": args.lambda_metadata_advantage,
        "lambda_metadata_gate": args.lambda_metadata_gate,
        "lambda_metadata_tdp_advantage": args.lambda_metadata_tdp_advantage,
        "lambda_sensor_cause": args.lambda_sensor_cause,
        "lambda_sensor_physical": args.lambda_sensor_physical,
        "lambda_image_physical": args.lambda_image_physical,
        "lambda_posterior_physical": args.lambda_posterior_physical,
        "sensor_motion_coordinate_weight": (
            args.sensor_motion_coordinate_weight
        ),
        "sensor_psf_geometry_weight": args.sensor_psf_geometry_weight,
        "image_psf_geometry_weight": args.image_psf_geometry_weight,
        "sensor_psf_length_weight": args.sensor_psf_length_weight,
        "sensor_psf_vector_weight": args.sensor_psf_vector_weight,
        "sensor_psf_direction_weight": args.sensor_psf_direction_weight,
        "lambda_fused_cause": args.lambda_fused_cause,
        "lambda_sensor_prior_gate": args.lambda_sensor_prior_gate,
        "metadata_advantage_margin": args.metadata_advantage_margin,
        "metadata_tdp_advantage_margin": args.metadata_tdp_advantage_margin,
        "metadata_gate_target": args.metadata_gate_target,
        "metadata_negative_gate_target": args.metadata_negative_gate_target,
        "tdp_defect_mask_weight": args.tdp_defect_mask_weight,
        "tdp_defect_mask_power": args.tdp_defect_mask_power,
        "basis_sparsity_weight": args.basis_sparsity_weight,
        "base_weight": args.base_weight,
        "detail_preserve": args.detail_preserve,
        "detail_gain": args.detail_gain,
        "use_practical_sensor_encoder": bool(args.use_practical_sensor_encoder),
        "sensor_dim": int(args.sensor_dim),
        "sensor_gyro_full_scale": float(args.sensor_gyro_full_scale),
        "sensor_residual_scale": float(args.sensor_residual_scale),
        "use_sensor_prior_fusion": bool(args.sensor_prior_fusion),
        "use_post_prior_evidence_refiner": bool(
            args.post_prior_evidence_refiner
        ),
        "post_prior_refiner_gain": float(args.post_prior_refiner_gain),
        "post_prior_refiner_support": args.post_prior_refiner_support,
            "practical_prior_source": args.practical_prior_source,
            "use_spatial_physics": bool(args.spatial_physics),
            "physics_samples": int(args.physics_samples),
            "physics_exposure_min_ms": float(args.physics_exposure_min_ms),
            "physics_exposure_max_ms": float(args.physics_exposure_max_ms),
            "physics_focal_ratio": float(args.physics_focal_ratio),
            "physics_calibration_reliability": float(
                args.physics_calibration_reliability
            ),
            "physics_activation_motion_px": float(
                args.physics_activation_motion_px
            ),
            "physics_exclusive_trajectory": bool(
                args.physics_exclusive_trajectory
            ),
            "use_physics_inverse_candidate": bool(args.physics_inverse_candidate),
            "physics_inverse_iterations": int(args.physics_inverse_iterations),
            "physics_inverse_blend": float(args.physics_inverse_blend),
            "physics_decoder_motion_threshold_px": float(
                args.physics_decoder_motion_threshold_px
            ),
            "physics_decoder_motion_transition_px": float(
                args.physics_decoder_motion_transition_px
            ),
        "use_sensor_image_psf_refiner": bool(
            args.sensor_image_psf_refiner
        ),
        "sensor_encoder_weights": args.sensor_encoder_weights,
        "new_head_lr_mult": args.new_head_lr_mult,
        "reset_code_fuser": bool(args.reset_code_fuser),
        "metadata_availability_mode": args.metadata_availability_mode,
        "metadata_dropout": float(args.metadata_dropout),
        "metadata_noise": float(args.metadata_noise),
        "horizontal_flip_probability": (
            0.0
            if args.code_source in {"metadata", "metadata_fused", "sensor_fused"}
            else 0.5
        ),
        "fusion_only_epochs": int(args.fusion_only_epochs),
        "metadata_branch_only_epochs": int(args.metadata_branch_only_epochs),
        "sensor_state_only_epochs": int(args.sensor_state_only_epochs),
        "sensor_refiner_only_epochs": int(args.sensor_refiner_only_epochs),
        "sensor_prior_fusion_only_epochs": int(
            args.sensor_prior_fusion_only_epochs
        ),
        "post_prior_refiner_only_epochs": int(
            args.post_prior_refiner_only_epochs
        ),
        "cause_expert_only_index": int(args.cause_expert_only_index),
        "cause_compound_boost": float(args.cause_compound_boost),
        "physical_state_only": bool(args.physical_state_only),
        "conditioning_branch_only_epochs": int(
            args.conditioning_branch_only_epochs
        ),
        "boosted_parameter_names": boosted_names,
        "selection_policy": "Training saves PSNR/loss checkpoints. Detector-mAP promotion must be performed externally on validation restored YOLO splits.",
        "args": vars(args),
    }
    (out_dir / "audit_config.json").write_text(json.dumps(audit_config, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit_config.items() if k != "args"}), flush=True)

    if args.smoke_test:
        smoke_test(model, loader, criterion, task_loss, args, device, out_dir)
        return

    prior_history: list[dict[str, Any]] = []
    history_csv = out_dir / "selection_history.csv"
    if resume_payload is not None and history_csv.exists():
        with history_csv.open(newline="", encoding="utf-8") as handle:
            prior_history = [dict(row) for row in csv.DictReader(handle)]
    history: list[dict[str, Any]] = list(prior_history)
    best_val_psnr = max((float(row["val_psnr"]) for row in prior_history if row.get("val_psnr")), default=float("-inf"))
    best_val_loss = min((float(row["val_loss"]) for row in prior_history if row.get("val_loss")), default=float("inf"))
    start_epoch = int(resume_payload.get("epoch", 0)) + 1 if resume_payload is not None else 1
    if start_epoch > args.epochs:
        raise ValueError(f"Resume checkpoint is already at epoch {start_epoch - 1}, beyond requested final epoch {args.epochs}")
    staged_scopes = sum(
        int(value > 0)
        for value in (
            args.fusion_only_epochs,
            args.metadata_branch_only_epochs,
            args.sensor_state_only_epochs,
            args.sensor_refiner_only_epochs,
            args.sensor_prior_fusion_only_epochs,
            args.post_prior_refiner_only_epochs,
            args.conditioning_branch_only_epochs,
            1 if args.cause_expert_only_index >= 0 else 0,
        )
    )
    if staged_scopes > 1:
        raise ValueError(
            "--fusion-only-epochs, --metadata-branch-only-epochs, "
            "--sensor-state-only-epochs, --sensor-refiner-only-epochs, "
            "--sensor-prior-fusion-only-epochs, and "
            "--post-prior-refiner-only-epochs, and "
            "--conditioning-branch-only-epochs/--cause-expert-only-index "
            "are mutually exclusive"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if is_detector_adaptation_epoch(args, epoch):
            for param in model.parameters():
                param.requires_grad_(False)
            row = run_detector_adaptation(args, out_dir, epoch)
            history.append(row)
            print(json.dumps(row), flush=True)
            for param in model.parameters():
                param.requires_grad_(True)
            continue

        fusion_only = int(args.fusion_only_epochs) > 0 and epoch <= int(args.fusion_only_epochs)
        metadata_branch_only = (
            int(args.metadata_branch_only_epochs) > 0
            and epoch <= int(args.metadata_branch_only_epochs)
        )
        sensor_state_only = (
            int(args.sensor_state_only_epochs) > 0
            and epoch <= int(args.sensor_state_only_epochs)
        )
        sensor_refiner_only = (
            int(args.sensor_refiner_only_epochs) > 0
            and epoch <= int(args.sensor_refiner_only_epochs)
        )
        sensor_prior_fusion_only = (
            int(args.sensor_prior_fusion_only_epochs) > 0
            and epoch <= int(args.sensor_prior_fusion_only_epochs)
        )
        post_prior_refiner_only = (
            int(args.post_prior_refiner_only_epochs) > 0
            and epoch <= int(args.post_prior_refiner_only_epochs)
        )
        conditioning_branch_only = (
            int(args.conditioning_branch_only_epochs) > 0
            and epoch <= int(args.conditioning_branch_only_epochs)
        )
        cause_expert_only_prefix = (
            f"cause_head.experts.{int(args.cause_expert_only_index)}."
            if args.cause_expert_only_index >= 0
            else None
        )
        metadata_branch_prefixes = (
            "code_encoder.",
            "code_fuser.",
            "cause_head.",
            "sensor_encoder.",
            "sensor_prior_fusion.",
            "sensor_image_psf_refiner.",
        )
        sensor_state_prefixes = (
            "code_encoder.",
            "sensor_encoder.",
            "sensor_image_psf_refiner.",
        )
        conditioning_tokens = (".film.", ".channel_gate.", "detail_skip.code_gate.")
        for name, parameter in model.named_parameters():
            if fusion_only:
                trainable = name.startswith("code_fuser.")
            elif metadata_branch_only:
                trainable = name.startswith(metadata_branch_prefixes)
            elif sensor_state_only:
                trainable = name.startswith(sensor_state_prefixes)
            elif sensor_refiner_only:
                trainable = name.startswith(
                    ("code_encoder.", "sensor_image_psf_refiner.")
                )
            elif sensor_prior_fusion_only:
                trainable = name.startswith("sensor_prior_fusion.")
            elif post_prior_refiner_only:
                trainable = name.startswith("post_prior_evidence_refiner.")
            elif cause_expert_only_prefix is not None:
                trainable = name.startswith(cause_expert_only_prefix)
            elif conditioning_branch_only:
                trainable = name.startswith(metadata_branch_prefixes) or any(
                    token in name for token in conditioning_tokens
                )
            else:
                trainable = True
            parameter.requires_grad_(trainable)
        model.train()
        running = 0.0
        component_sums: dict[str, float] = {}
        scale = task_warmup_scale(args, epoch)
        use_amp = resolve_amp(args, device)
        trained_batches = 0
        for batch_index_for_cap, batch in enumerate(loader):
            if args.max_train_batches > 0 and batch_index_for_cap >= args.max_train_batches:
                break
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["gt"].to(device, non_blocking=True)
            codes = batch["code"].to(device, non_blocking=True)
            metadata_codes = batch["metadata_code"].to(device, non_blocking=True)
            cause_targets = batch["cause_target"].to(device, non_blocking=True)
            physical_targets = batch["physical_target"].to(device, non_blocking=True)
            physical_target_available = batch["physical_target_available"].to(
                device,
                non_blocking=True,
            )
            detector_targets = (
                prepare_detector_targets(batch, device)
                if args.lambda_detector_supervised > 0
                else None
            )
            model_codes = prepare_codes(args, codes, metadata_codes, training=True)
            optimizer.zero_grad(set_to_none=True)
            loss_terms: dict[str, torch.Tensor] = {}
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                result = model_forward(
                    model,
                    inputs,
                    model_codes,
                    args,
                    need_aux=(
                        task_loss is not None
                        or args.use_tdac_head
                        or (args.conditioning in {"gated_basis", "residual_basis"} and args.basis_sparsity_weight > 0)
                        or args.lambda_sensor_cause > 0
                        or physical_state_objective_enabled(args)
                        or args.lambda_fused_cause > 0
                        or metadata_objective_enabled(args)
                        or args.lambda_physics_reblur > 0
                    ),
                    state_only=args.physical_state_only,
                    physics_codes=(
                        metadata_codes if args.spatial_physics else None
                    ),
                )
                outputs = result["restored"]
                if args.base_weight > 0:
                    base_loss = criterion(
                        outputs,
                        targets,
                        inputs,
                        model_codes,
                    )
                    weighted_base_loss = (
                        float(args.base_weight) * base_loss
                    )
                else:
                    # Fully detach a disabled loss branch. This is important
                    # for refiner-only calibration because a literal zero
                    # multiplier does not protect against NaN FFT gradients.
                    with torch.no_grad():
                        base_loss = criterion(
                            outputs,
                            targets,
                            inputs,
                            model_codes,
                        )
                    weighted_base_loss = outputs.new_zeros(())
                loss = weighted_base_loss
                loss_terms["restoration"] = base_loss
                loss_terms["restoration_weighted"] = weighted_base_loss
                if args.lambda_physics_reblur > 0:
                    reblurred = result.get("physics_reblurred")
                    reliability = result.get("physics_reliability")
                    if not isinstance(reblurred, torch.Tensor) or not isinstance(
                        reliability, torch.Tensor
                    ):
                        raise RuntimeError(
                            "Physics loss requested without spatial-physics outputs"
                        )
                    physics_raw = physics_reblur_loss(
                        reblurred,
                        inputs,
                        reliability,
                    )
                    physics_weighted = (
                        float(args.lambda_physics_reblur) * physics_raw
                    )
                    loss = loss + physics_weighted
                    loss_terms["physics_reblur"] = physics_raw
                    loss_terms["physics_reblur_weighted"] = physics_weighted
                if args.code_source in {"estimated", "fused", "metadata_fused", "sensor_fused"} and args.aux_code_weight > 0:
                    target_codes = (
                        cause_targets
                        if args.code_source == "sensor_fused"
                        else metadata_codes
                        if args.code_source == "metadata_fused"
                        else codes
                    )
                    aux_code = args.aux_code_weight * F.smooth_l1_loss(model.estimate_code(inputs), target_codes)
                    loss = loss + aux_code
                    loss_terms["aux_code"] = aux_code
                if args.code_source == "sensor_fused" and args.lambda_sensor_cause > 0:
                    sensor_code = result.get("sensor_code")
                    sensor_support = result.get("sensor_cause_reliability")
                    if not isinstance(sensor_code, torch.Tensor):
                        raise RuntimeError(
                            "sensor_fused training requires result['sensor_code']; "
                            "enable --use-practical-sensor-encoder"
                        )
                    if not isinstance(sensor_support, torch.Tensor):
                        raise RuntimeError(
                            "sensor_fused training requires "
                            "result['sensor_cause_reliability']"
                        )
                    sensor_cause_unweighted, sensor_cause_active = (
                        masked_sensor_state_loss(
                            sensor_code,
                            cause_targets,
                            sensor_support,
                        )
                    )
                    sensor_cause = (
                        float(args.lambda_sensor_cause)
                        * sensor_cause_unweighted
                    )
                    loss = loss + sensor_cause
                    loss_terms["sensor_cause"] = sensor_cause
                    component_sums["sensor_cause_unweighted"] = (
                        component_sums.get("sensor_cause_unweighted", 0.0)
                        + float(sensor_cause_unweighted.detach().cpu())
                    )
                    component_sums["sensor_cause_active_fraction"] = (
                        component_sums.get(
                            "sensor_cause_active_fraction",
                            0.0,
                        )
                        + float(sensor_cause_active.detach().cpu())
                    )
                if physical_state_objective_enabled(args):
                    sensor_physical = result.get(
                        "sensor_only_physical_code"
                    )
                    image_physical = result.get("image_physical_code")
                    posterior_physical = result.get(
                        "sensor_calibrated_physical_code"
                    )
                    sensor_support = result.get(
                        "sensor_cause_reliability"
                    )
                    if not isinstance(sensor_physical, torch.Tensor):
                        raise RuntimeError(
                            "Physical-state training requires "
                            "result['sensor_only_physical_code']."
                        )
                    if not isinstance(image_physical, torch.Tensor):
                        raise RuntimeError(
                            "Physical-state training requires "
                            "result['image_physical_code']."
                        )
                    if not isinstance(posterior_physical, torch.Tensor):
                        raise RuntimeError(
                            "Physical-state training requires "
                            "result['sensor_calibrated_physical_code']."
                        )
                    if not isinstance(sensor_support, torch.Tensor):
                        raise RuntimeError(
                            "Physical sensor calibration requires "
                            "result['sensor_cause_reliability']."
                        )
                    # Training-only physical-state calibration:
                    #
                    # L_sensor = sum_ij b_i q_ij w_j ell(z_S-z*)
                    #            / sum_ij b_i q_ij w_j,
                    # L_image  = sum_ij b_i w_j ell(z_I-z*) / sum_ij b_i w_j,
                    # L_post   = sum_ij b_i w_j ell(z-z*) / sum_ij b_i w_j.
                    #
                    # where b_i marks private calibration labels and q_ij is
                    # detached sensor support. Missing telemetry therefore
                    # creates no sensor-only gradient, while the image fallback
                    # and joint posterior remain supervised. z* is never passed
                    # into model.forward().
                    motion_weight = float(
                        args.sensor_motion_coordinate_weight
                    )
                    coordinate_weight = posterior_physical.new_tensor(
                        [
                            motion_weight,
                            motion_weight,
                            motion_weight,
                            1.0,
                            1.0,
                            1.0,
                            0.5,
                            1.0,
                        ]
                    )[None, :]
                    sensor_unweighted, sensor_active = (
                        masked_sensor_state_loss(
                            sensor_physical,
                            physical_targets,
                            sensor_support,
                            coordinate_weight=coordinate_weight,
                            sample_available=physical_target_available,
                        )
                    )
                    sensor_loss = (
                        float(args.lambda_sensor_physical)
                        * sensor_unweighted
                    )
                    loss = loss + sensor_loss
                    loss_terms["sensor_physical"] = sensor_loss
                    component_sums["sensor_physical_unweighted"] = (
                        component_sums.get(
                            "sensor_physical_unweighted",
                            0.0,
                        )
                        + float(sensor_unweighted.detach().cpu())
                    )
                    component_sums[
                        "sensor_physical_active_fraction"
                    ] = (
                        component_sums.get(
                            "sensor_physical_active_fraction",
                            0.0,
                        )
                        + float(sensor_active.detach().cpu())
                    )
                    image_unweighted = available_physical_state_loss(
                        image_physical,
                        physical_targets,
                        physical_target_available,
                        coordinate_weight=coordinate_weight,
                    )
                    image_loss = (
                        float(args.lambda_image_physical)
                        * image_unweighted
                    )
                    loss = loss + image_loss
                    loss_terms["image_physical"] = image_loss
                    component_sums["image_physical_unweighted"] = (
                        component_sums.get(
                            "image_physical_unweighted",
                            0.0,
                        )
                        + float(image_unweighted.detach().cpu())
                    )
                    posterior_unweighted = available_physical_state_loss(
                        posterior_physical,
                        physical_targets,
                        physical_target_available,
                        coordinate_weight=coordinate_weight,
                    )
                    posterior_loss = (
                        float(args.lambda_posterior_physical)
                        * posterior_unweighted
                    )
                    loss = loss + posterior_loss
                    loss_terms["posterior_physical"] = posterior_loss
                    component_sums["posterior_physical_unweighted"] = (
                        component_sums.get(
                            "posterior_physical_unweighted",
                            0.0,
                        )
                        + float(posterior_unweighted.detach().cpu())
                    )
                    full_support = torch.ones_like(sensor_support)
                    image_geometry, image_metrics = (
                        practical_psf_geometry_loss(
                            image_physical,
                            physical_targets,
                            full_support,
                            sample_available=physical_target_available,
                            length_weight=args.sensor_psf_length_weight,
                            vector_weight=args.sensor_psf_vector_weight,
                            direction_weight=args.sensor_psf_direction_weight,
                        )
                    )
                    weighted_image_geometry = (
                        float(args.image_psf_geometry_weight)
                        * image_geometry
                    )
                    loss = loss + weighted_image_geometry
                    loss_terms["image_psf_geometry"] = (
                        weighted_image_geometry
                    )
                    component_sums["image_psf_geometry_unweighted"] = (
                        component_sums.get(
                            "image_psf_geometry_unweighted",
                            0.0,
                        )
                        + float(image_geometry.detach().cpu())
                    )
                    posterior_geometry, posterior_metrics = (
                        practical_psf_geometry_loss(
                            posterior_physical,
                            physical_targets,
                            full_support,
                            sample_available=physical_target_available,
                            length_weight=args.sensor_psf_length_weight,
                            vector_weight=args.sensor_psf_vector_weight,
                            direction_weight=args.sensor_psf_direction_weight,
                        )
                    )
                    weighted_posterior_geometry = (
                        float(args.sensor_psf_geometry_weight)
                        * posterior_geometry
                    )
                    loss = loss + weighted_posterior_geometry
                    loss_terms["posterior_psf_geometry"] = (
                        weighted_posterior_geometry
                    )
                    component_sums[
                        "posterior_psf_geometry_unweighted"
                    ] = (
                        component_sums.get(
                            "posterior_psf_geometry_unweighted",
                            0.0,
                        )
                        + float(posterior_geometry.detach().cpu())
                    )
                    for name, value in image_metrics.items():
                        key = f"image_psf_{name}"
                        component_sums[key] = (
                            component_sums.get(key, 0.0)
                            + float(value.detach().cpu())
                        )
                    for name, value in posterior_metrics.items():
                        key = f"posterior_psf_{name}"
                        component_sums[key] = (
                            component_sums.get(key, 0.0)
                            + float(value.detach().cpu())
                        )
                if (
                    args.code_source == "sensor_fused"
                    and args.lambda_fused_cause > 0
                ):
                    posterior_code = result.get("code")
                    if not isinstance(posterior_code, torch.Tensor):
                        raise RuntimeError(
                            "Posterior supervision requires result['code']."
                        )
                    # Joint image/metadata degradation posterior:
                    # z = z_I + A(I_d,m,q) * (z_M-z_I).
                    posterior_unweighted = F.smooth_l1_loss(
                        posterior_code,
                        cause_targets,
                    )
                    posterior_loss = (
                        float(args.lambda_fused_cause)
                        * posterior_unweighted
                    )
                    loss = loss + posterior_loss
                    loss_terms["fused_cause"] = posterior_loss
                    component_sums["fused_cause_unweighted"] = (
                        component_sums.get(
                            "fused_cause_unweighted",
                            0.0,
                        )
                        + float(posterior_unweighted.detach().cpu())
                    )
                basis_sparsity = result.get("basis_sparsity")
                if (
                    args.conditioning in {"gated_basis", "residual_basis"}
                    and args.basis_sparsity_weight > 0
                    and isinstance(basis_sparsity, torch.Tensor)
                ):
                    sparse_loss = float(args.basis_sparsity_weight) * basis_sparsity
                    loss = loss + sparse_loss
                    loss_terms["basis_sparsity"] = sparse_loss
                task_logs: dict[str, float] = {}
                if task_loss is not None:
                    task_value, task_logs = task_loss(
                        result,
                        targets,
                        degraded=inputs,
                        detector_targets=detector_targets,
                        warmup_scale=scale,
                    )
                    loss = loss + task_value
                    loss_terms["task_total"] = task_value
                    for name, value in task_logs.items():
                        component_sums[name] = component_sums.get(name, 0.0) + float(value)
                metadata_terms, metadata_logs = add_metadata_central_losses(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    metadata_codes=model_codes if model_codes is not None else metadata_codes,
                    result=result,
                    base_loss=base_loss,
                    criterion=criterion,
                    task_loss=task_loss,
                    warmup_scale=scale,
                    args=args,
                    need_aux=True,
                )
                task_objective = task_value if task_loss is not None else outputs.new_zeros(())
                for name, value in metadata_terms.items():
                    loss = loss + value
                    loss_terms[name] = value
                    if name == "metadata_tdp_advantage":
                        task_objective = task_objective + value
                for name, value in metadata_logs.items():
                    component_sums[name] = component_sums.get(name, 0.0) + float(value)
                primary_objective = loss - task_objective
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise FloatingPointError(
                    f"Non-finite training objective at epoch {epoch}; "
                    "the optimizer step was not executed."
                )
            if args.gradient_strategy == "primary_projected":
                gradient_logs = apply_primary_protected_gradients(
                    primary_objective,
                    task_objective,
                    list(model.parameters()),
                )
                for name, value in gradient_logs.items():
                    component_sums[name] = component_sums.get(name, 0.0) + value
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            grad_norm_value = float(grad_norm_tensor.detach().cpu()) if isinstance(grad_norm_tensor, torch.Tensor) else float(grad_norm_tensor)
            if not math.isfinite(grad_norm_value):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {epoch}; "
                    "the optimizer step was not executed."
                )
            if args.debug_first_batches > 0 or args.debug_every > 0:
                batch_index = int(component_sums.get("_batch_index", 0))
                should_debug = batch_index < args.debug_first_batches or (args.debug_every > 0 and batch_index % args.debug_every == 0)
                if should_debug:
                    print(json.dumps(debug_training_stats(
                        epoch=epoch,
                        batch_index=batch_index,
                        inputs=inputs,
                        outputs=outputs,
                        targets=targets,
                        model_codes=model_codes,
                        metadata_codes=metadata_codes,
                        result=result,
                        loss_value=loss,
                        base_loss=base_loss,
                        task_logs=task_logs,
                        grad_norm=grad_norm_value,
                    )), flush=True)
                component_sums["_batch_index"] = batch_index + 1
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
            trained_batches += 1
            for name, value in loss_terms.items():
                component_sums[f"loss_{name}"] = component_sums.get(f"loss_{name}", 0.0) + float(value.detach().cpu())

        epoch_batches = max(trained_batches, 1)
        row: dict[str, Any] = {
            "epoch": epoch,
            "phase": "rmr_update",
            "optimization_scope": (
                "metadata_fusion_only"
                if fusion_only
                else "metadata_branch_only"
                if metadata_branch_only
                else "sensor_state_only"
                if sensor_state_only
                else "sensor_refiner_only"
                if sensor_refiner_only
                else "sensor_prior_fusion_only"
                if sensor_prior_fusion_only
                else "post_prior_refiner_only"
                if post_prior_refiner_only
                else f"cause_expert_{int(args.cause_expert_only_index)}_only"
                if cause_expert_only_prefix is not None
                else "conditioning_branch_only"
                if conditioning_branch_only
                else "full_model"
            ),
            "loss": running / epoch_batches,
            "train_batches": trained_batches,
            "train_batches_total": len(loader),
            "max_train_batches": int(args.max_train_batches),
            "task_warmup_scale": scale,
            "effective_tdp_weight": (args.lambda_tdp * scale) if task_loss is not None else 0.0,
            "effective_detector_supervised_weight": (
                args.lambda_detector_supervised * scale
                if task_loss is not None
                else 0.0
            ),
            "effective_jacobian_weight": (args.lambda_jacobian * scale) if task_loss is not None else 0.0,
            "effective_active_contour_weight": (args.lambda_active_contour * scale) if task_loss is not None else 0.0,
            "effective_detector_input_anchor_weight": (args.lambda_detector_input_anchor * scale) if task_loss is not None else 0.0,
            "effective_evidence_nonregression_weight": (args.lambda_evidence_nonregression * scale) if task_loss is not None else 0.0,
            "effective_detail_copy_weight": (args.lambda_detail_copy * scale) if task_loss is not None else 0.0,
            "effective_restoration_magnitude_weight": (args.lambda_restoration_magnitude * scale) if task_loss is not None else 0.0,
            "effective_low_evidence_identity_weight": (args.lambda_low_evidence_identity * scale) if task_loss is not None else 0.0,
            "effective_metadata_advantage_weight": args.lambda_metadata_advantage,
            "effective_metadata_gate_weight": args.lambda_metadata_gate,
            "selection_note": "Detector mAP checkpoint selection is performed externally on validation splits only; this trainer saves explicit PSNR/loss checkpoints for audit.",
        }
        for name, value in sorted(component_sums.items()):
            if name.startswith("_"):
                continue
            row[name] = value / epoch_batches
        if val_loader is not None and epoch % max(args.val_every, 1) == 0:
            row.update(validate(model, val_loader, criterion, args, device))
        history.append(row)
        print(json.dumps(row), flush=True)

        save_checkpoint(out_dir / "rcadnet_last.pth", model, args, epoch, row, optimizer=optimizer, scaler=scaler)
        if args.save_every_epoch:
            save_checkpoint(
                out_dir / f"rcadnet_epoch_{epoch:03d}.pth",
                model,
                args,
                epoch,
                row,
                optimizer=optimizer,
                scaler=scaler,
            )
        if "val_psnr" in row and row["val_psnr"] > best_val_psnr:
            best_val_psnr = float(row["val_psnr"])
            save_checkpoint(out_dir / "rcadnet_best_psnr.pth", model, args, epoch, row, optimizer=optimizer, scaler=scaler)
            save_checkpoint(out_dir / "rcadnet_best.pth", model, args, epoch, row, optimizer=optimizer, scaler=scaler)  # backward-compatible alias
            (out_dir / "best_by_val_psnr.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        if "val_loss" in row and row["val_loss"] < best_val_loss:
            best_val_loss = float(row["val_loss"])
            save_checkpoint(out_dir / "rcadnet_best_loss.pth", model, args, epoch, row, optimizer=optimizer, scaler=scaler)
            (out_dir / "best_by_val_loss.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        append_selection_history(out_dir / "selection_history.csv", {k: v for k, v in row.items() if not isinstance(v, dict)})

    if args.phase2_epochs > 0:
        if not args.phase2_detector_data or not args.phase2_detector_weights:
            raise ValueError("--phase2-detector-data and --phase2-detector-weights are required for detector phase 2")
        from ultralytics import YOLO

        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
        detector = YOLO(args.phase2_detector_weights)
        phase2 = detector.train(
            data=args.phase2_detector_data,
            epochs=args.phase2_epochs,
            imgsz=args.phase2_imgsz,
            project=str(out_dir / "phase2_detector"),
            name="restored_patch_finetune",
        )
        (out_dir / "phase2_detector_result.json").write_text(json.dumps({"result": str(phase2)}, indent=2), encoding="utf-8")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
