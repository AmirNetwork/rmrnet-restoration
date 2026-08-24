#!/usr/bin/env python3
"""Matched target-domain adaptation for the RMR-P comparison.

Every method sees the same balanced stream of paired crops and is optimized
for the same number of effective batches with the same AdamW schedule.  The
common objective is used by every restorer:

    L_common = L_charbonnier + lambda_g L_gradient
             + lambda_TDP L_feature + lambda_det L_box/class.

The final term is the clean-normalized supervised objective of the frozen
dataset detector. It is optional for backward compatibility, but when enabled
its coefficient is identical for every method in the matched suite.

RMR-P additionally learns the quantity that its competing image-only models
do not receive: a reliability-aware corruption state from the public capture
packet. The private cause and physical labels supervise that state only on
training data; they are never passed to the model and are absent from test
sidecars. In the manuscript notation:

    z_I = g_I(I_d),
    z_M = clip[h(m) + r(m) delta tanh(Delta_psi(m)), 0, 1],
    z = clip[q*z_M + (1-q)*z_I + delta_p tanh p_psi(...), 0, 1],
    L_RMR-P = L_common + lambda_z L_cause + lambda_p L_phys.

This script deliberately does not evaluate a test split.  It writes epoch
checkpoints for a separate validation-only selection stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from baselines.demoe_adapter import DeMoEAdapter, demoe_task_from_scenario
from baselines.dfpir_adapter import DFPIRAdapter
from baselines.instructir_adapter import InstructIRAdapter
from baselines.nafnet_metadata import MetadataNAFNetRoad
from baselines.nafnet_road import NAFNetRoad
from models.rmrnet import RMRP
from models.rmrp_metadata_demoe import RMRPMetadataDeMoE
from models.rmrp_prompted_dfpir import RMRPPromptedDFPIR
from rcadnet.dataset import PairedRoadRestorationDataset
from rcadnet.losses import RCADLoss
from rcadnet.model import RCADNet
from rcadnet.practical_metadata import balanced_sensor_dropout, perturb_sensor_packet
from rcadnet.task_losses import (
    FrozenDetectorFeatureExtractor,
    FrozenDetectorSupervisedLoss,
    TaskDrivenPerceptualLoss,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENARIOS = (
    "clean",
    "motion_horizontal_medium",
    "defocus_medium",
    "lowlight_medium",
    "mixed_motion_lowlight",
)


def parse_tagged(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected TAG=PATH")
    tag, raw = value.split("=", 1)
    tag = tag.strip().lower()
    if not tag:
        raise argparse.ArgumentTypeError("Dataset tag cannot be empty")
    return tag, Path(raw).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("rmrp", "nafnet", "nafnet_meta", "demoe", "dfpir", "instructir"),
        required=True,
    )
    parser.add_argument("--data-root", type=parse_tagged, action="append", required=True)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--detector", type=parse_tagged, action="append", required=True)
    parser.add_argument("--init-weights", action="append", required=True)
    parser.add_argument("--lm-head-weights")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--effective-batch-size", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--new-module-lr-multiplier",
        type=float,
        default=1.0,
        help=(
            "RMR-P-only learning-rate multiplier for identity-initialized "
            "sensor-prior fusion and post-prior refinement modules."
        ),
    )
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--tdp-weight", type=float, default=0.08)
    parser.add_argument(
        "--detector-supervised-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the clean-normalized frozen-detector box/class loss. "
            "Use the same value for every method in a matched comparison."
        ),
    )
    parser.add_argument("--tdp-input-size", type=int, default=320)
    parser.add_argument("--tdp-warmup-epochs", type=int, default=3)
    parser.add_argument(
        "--detector-cqmix-probability",
        type=float,
        default=0.25,
        help="Probability of clean/restored patch mixing inside detector supervision.",
    )
    parser.add_argument(
        "--detector-clean-hinge-weight",
        type=float,
        default=0.25,
        help="Weight of the clean-relative detector non-regression hinge.",
    )
    parser.add_argument("--state-weight", type=float, default=0.50)
    parser.add_argument("--physical-weight", type=float, default=0.20)
    parser.add_argument("--metadata-dropout", type=float, default=0.35)
    parser.add_argument("--metadata-noise", type=float, default=0.015)
    parser.add_argument(
        "--metadata-mismatch-probability",
        type=float,
        default=0.10,
        help=(
            "Probability that a training sample receives another sample's sensor "
            "record. Sensor-only targets follow the donor record, while image and "
            "joint targets remain tied to the current image."
        ),
    )
    parser.add_argument(
        "--metadata-curriculum-epochs",
        type=int,
        default=0,
        help=(
            "Number of initial epochs with aligned, complete sensor records. "
            "This lets the restorer learn the physical task before robustness "
            "augmentation is introduced."
        ),
    )
    parser.add_argument(
        "--metadata-curriculum-ramp-epochs",
        type=int,
        default=0,
        help=(
            "Number of epochs used to linearly ramp metadata dropout, noise, "
            "and mismatch from zero to their configured target values."
        ),
    )
    parser.add_argument("--route-teacher-epochs", type=int, default=0)
    parser.add_argument("--route-teacher-ramp-epochs", type=int, default=0)
    parser.add_argument(
        "--rmrp-sensor-route-mode",
        choices=("posterior", "physical_fused"),
        default="posterior",
        help="RMR-P state used for task routing and continuous feature conditioning.",
    )
    parser.add_argument(
        "--rmrp-bounded-refiner",
        action="store_true",
        help="Enable RMR-P's identity-initialized state-conditioned output refiner.",
    )
    parser.add_argument("--rmrp-refiner-gain", type=float, default=0.12)
    parser.add_argument(
        "--rmrp-cause-refiners",
        action="store_true",
        help=(
            "Enable bounded motion, defocus, and low-light correction experts "
            "selected from the fused physical degradation state."
        ),
    )
    parser.add_argument("--rmrp-cause-refiner-gain", type=float, default=0.08)
    parser.add_argument(
        "--rmrp-backbone-route-mode",
        choices=("metadata", "image", "sensor_task"),
        default="metadata",
        help=(
            "Select whether observable metadata also routes the shared DeMoE "
            "backbone or only the cause-specific residual adapters."
        ),
    )
    parser.add_argument(
        "--rmrp-semantic-adapters",
        action="store_true",
        help=(
            "Enable identity-initialized motion, defocus, low-light, and "
            "compound residual specialists selected by sensor evidence."
        ),
    )
    parser.add_argument("--rmrp-semantic-adapter-gain", type=float, default=0.25)
    parser.add_argument(
        "--rmrp-freeze-semantic-adapters-only",
        action="store_true",
        help=(
            "Freeze the shared restorer and sensor encoder and optimize only "
            "the four newly enabled metadata residual specialists."
        ),
    )
    parser.add_argument(
        "--rmrp-freeze-cause-refiners-only",
        action="store_true",
        help=(
            "Freeze the validated router and restoration trunk; optimize only "
            "the three newly enabled cause-specific correction experts."
        ),
    )
    parser.add_argument(
        "--rmrp-freeze-routed-experts-only",
        action="store_true",
        help=(
            "Freeze the shared DeMoE trunk and RMR-P controller; optimize only "
            "the DeMoE expert blocks listed by --rmrp-routed-expert-indices."
        ),
    )
    parser.add_argument(
        "--rmrp-routed-expert-indices",
        type=int,
        nargs="+",
        default=(0, 3, 4),
        help=(
            "DeMoE expert indices calibrated when "
            "--rmrp-freeze-routed-experts-only is active. DeMoE orders its "
            "experts as defocus=0, real-motion=1, local-motion=2, "
            "synthetic/global-motion=3, and low-light=4."
        ),
    )
    parser.add_argument(
        "--rmrp-freeze-backbone",
        action="store_true",
        help=(
            "Freeze the matched restoration backbone and optimize only RMR-P's "
            "sensor-state, routing, identity-initialized FiLM, and bounded "
            "correction modules."
        ),
    )
    parser.add_argument(
        "--rmrp-compound-blend-gate",
        action="store_true",
        help=(
            "Enable the image-and-sensor reliability gate that blends sparse "
            "top-1 and top-2 DeMoE outputs only for compound corruption."
        ),
    )
    parser.add_argument("--rmrp-compound-blend-init", type=float, default=0.65)
    parser.add_argument(
        "--rmrp-compound-metadata-acceptance",
        type=float,
        default=None,
        help=(
            "Validation-selected acceptance of the joint metadata candidate "
            "relative to the conservative low-light candidate for compound states."
        ),
    )
    parser.add_argument(
        "--rmrp-freeze-compound-gate-only",
        action="store_true",
        help=(
            "Freeze all existing RMR-P parameters and calibrate only the newly "
            "enabled compound blend gate."
        ),
    )
    parser.add_argument(
        "--rmrp-compound-motion-blend",
        type=float,
        default=0.0,
        help=(
            "For metadata-confirmed motion-plus-low-light samples, blend this "
            "fraction of the motion restoration with the low-light restoration. "
            "The value must be selected on validation data only."
        ),
    )
    parser.add_argument(
        "--rmrp-compound-refiner",
        action="store_true",
        help=(
            "Enable RMR-P's identity-initialized residual expert for samples "
            "whose observable state supports joint motion and low light."
        ),
    )
    parser.add_argument("--rmrp-compound-refiner-gain", type=float, default=0.18)
    parser.add_argument("--defect-label-root", type=parse_tagged, action="append")
    parser.add_argument("--defect-crop-probability", type=float, default=0.60)
    parser.add_argument(
        "--dataset-gradient-surgery",
        action="store_true",
        help=(
            "Apply equal-domain PCGrad to the restoration gradients. This is "
            "intended for joint IVCNZ/PCM calibration where the two frozen "
            "detectors can otherwise issue opposing updates."
        ),
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume model, optimizer, scheduler, epoch, and step from a verified checkpoint.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TaggedDataset(Dataset):
    def __init__(self, tag: str, dataset: PairedRoadRestorationDataset) -> None:
        self.tag = tag
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.dataset[index])
        row["dataset_tag"] = self.tag
        return row


def build_dataset(
    roots: Sequence[tuple[str, Path]],
    scenarios: Sequence[str],
    patch_size: int,
    label_roots: dict[str, Path],
    defect_crop_probability: float,
) -> tuple[ConcatDataset, torch.Tensor, dict[str, int]]:
    tagged: list[TaggedDataset] = []
    group_counts: Counter[tuple[str, str]] = Counter()
    for tag, root in roots:
        dataset = PairedRoadRestorationDataset(
            root,
            scenarios,
            patch_size=patch_size,
            train=True,
            metadata_mode="full",
            defect_label_root=label_roots.get(tag),
            defect_crop_probability=defect_crop_probability,
        )
        tagged_dataset = TaggedDataset(tag, dataset)
        tagged.append(tagged_dataset)
        group_counts.update((tag, scenario) for _, _, scenario in dataset.samples)

    weights: list[float] = []
    for tagged_dataset in tagged:
        for _, _, scenario in tagged_dataset.dataset.samples:
            weights.append(1.0 / float(group_counts[(tagged_dataset.tag, scenario)]))
    summary = {f"{tag}:{scenario}": count for (tag, scenario), count in sorted(group_counts.items())}
    return ConcatDataset(tagged), torch.tensor(weights, dtype=torch.double), summary


def average_state_dicts(paths: Sequence[Path]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    states = [payload.get("model", payload) for payload in payloads]
    if not all(isinstance(state, dict) for state in states):
        raise TypeError("Every initialization must contain a state dictionary")
    shared = set(states[0])
    for state in states[1:]:
        shared &= set(state)
    averaged: dict[str, torch.Tensor] = {}
    for name in sorted(shared):
        values = [state[name] for state in states]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue
        if not all(value.shape == values[0].shape for value in values):
            continue
        if torch.is_floating_point(values[0]):
            averaged[name] = torch.stack([value.float() for value in values]).mean(0).to(values[0].dtype)
        else:
            averaged[name] = values[0]
    return averaged, payloads[0]


def rmrp_architecture(payload: dict[str, Any]) -> dict[str, Any]:
    arch = dict(payload.get("arch", {}))
    accepted = set(inspect.signature(RCADNet.__init__).parameters)
    kwargs = {key: value for key, value in arch.items() if key in accepted}
    # Both added blocks are checkpoint-compatible. SensorPriorFusion starts
    # near the historical physical candidate but can fall back spatially to
    # the neural restoration when mixed noise/low light makes Wiener inversion
    # unsafe. The bounded refiner starts as an exact identity.
    kwargs.update(
        {
            "use_sensor_prior_fusion": True,
            "use_post_prior_evidence_refiner": True,
            "post_prior_refiner_gain": 0.20,
            "post_prior_refiner_support": "all",
            "use_tdac_head": False,
        }
    )
    return kwargs


class TrainableRestorer(nn.Module):
    def __init__(
        self,
        kind: str,
        init_paths: Sequence[Path],
        device: torch.device,
        lm_head_weights: Path | None,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.device = device
        self.arch: dict[str, Any] = {"model": kind}
        self.load_report: dict[str, Any] = {
            "initializations": [str(path) for path in init_paths],
            "sha256": [sha256(path) for path in init_paths],
        }

        if kind == "rmrp":
            if len(init_paths) != 1:
                raise ValueError("RMR-P uses one auditable backbone initialization")
            payload = torch.load(init_paths[0], map_location="cpu", weights_only=False)
            source_arch = dict(payload.get("arch", {})) if isinstance(payload, dict) else {}
            if source_arch.get("backbone") == "demoe_sensor_router":
                adapter = DeMoEAdapter(None, device=device, smoke=True)
                model = RMRPMetadataDeMoE(
                    adapter.model,
                    sensor_gyro_full_scale=float(
                        source_arch.get("sensor_gyro_full_scale", 4.0)
                    ),
                    top_k=int(source_arch.get("top_k", 2)),
                    refiner_gain=float(source_arch.get("refiner_gain", 0.12)),
                    use_refiner=bool(source_arch.get("use_refiner", True)),
                    use_compound_blend_gate=bool(
                        source_arch.get("use_compound_blend_gate", False)
                    ),
                    compound_blend_init=float(
                        source_arch.get("compound_blend_init", 0.65)
                    ),
                    compound_metadata_acceptance=float(
                        source_arch.get("compound_metadata_acceptance", 1.0)
                    ),
                    cause_route_acceptance=tuple(
                        source_arch.get("cause_route_acceptance", (1.0, 1.0, 1.0))
                    ),
                    use_cause_refiners=bool(
                        source_arch.get("use_cause_refiners", False)
                    ),
                    cause_refiner_gain=float(
                        source_arch.get("cause_refiner_gain", 0.08)
                    ),
                    backbone_route_mode=str(
                        source_arch.get("backbone_route_mode", "metadata")
                    ),
                    sensor_task_thresholds=tuple(
                        source_arch.get(
                            "sensor_task_thresholds", (0.18, 0.10, 0.385)
                        )
                    ),
                    use_semantic_adapters=bool(
                        source_arch.get("use_semantic_adapters", False)
                    ),
                    semantic_adapter_gain=float(
                        source_arch.get("semantic_adapter_gain", 0.25)
                    ),
                    semantic_adapter_acceptance=tuple(
                        source_arch.get(
                            "semantic_adapter_acceptance",
                            (1.0, 1.0, 1.0, 1.0),
                        )
                    ),
                ).to(device)
                incompatible = model.load_state_dict(
                    payload.get("model", payload), strict=True
                )
                missing_keys = list(incompatible.missing_keys)
                unexpected_keys = list(incompatible.unexpected_keys)
                initialization_kind = "RMR-P metadata-routed DeMoE checkpoint"
            elif source_arch.get("model") == "demoe":
                # Start from the already matched, target-adapted DeMoE control.
                # Its expert functions are loaded exactly; RMR-P adds only the
                # observable-sensor controller and identity-initialized bounded
                # correction. This avoids erasing the strongest fair baseline
                # while the metadata path learns its routing semantics.
                adapter = DeMoEAdapter(None, device=device, smoke=True)
                incompatible_backbone = adapter.model.load_state_dict(
                    payload.get("model", payload), strict=True
                )
                model = RMRPMetadataDeMoE(
                    adapter.model,
                    sensor_gyro_full_scale=4.0,
                    top_k=2,
                    use_refiner=False,
                ).to(device)
                missing_keys = list(incompatible_backbone.missing_keys)
                unexpected_keys = list(incompatible_backbone.unexpected_keys)
                initialization_kind = (
                    "matched DeMoE backbone plus identity RMR-P metadata modules"
                )
            elif init_paths[0].name.lower() == "demoe.pt":
                adapter = DeMoEAdapter(init_paths[0], device=device, task="scenario")
                model = RMRPMetadataDeMoE(
                    adapter.model,
                    sensor_gyro_full_scale=4.0,
                    top_k=2,
                    use_refiner=False,
                ).to(device)
                missing_keys = []
                unexpected_keys = []
                initialization_kind = (
                    "official DeMoE backbone plus identity RMR-P metadata modules"
                )
            elif source_arch.get("backbone") == "dfpir_sensor_prompt":
                adapter = DFPIRAdapter(None, device=str(device), use_clip=False)
                source_state = payload.get("model", payload)
                basis_count = int(
                    source_arch.get(
                        "prompt_basis_count",
                        source_state.get("prompt_embeddings", torch.zeros(3, 512)).shape[0],
                    )
                )
                model = RMRPPromptedDFPIR(
                    adapter.model,
                    torch.zeros(basis_count, 512, device=device),
                    sensor_gyro_full_scale=float(source_arch.get("sensor_gyro_full_scale", 4.0)),
                    prompt_residual_scale=float(source_arch.get("prompt_residual_scale", 0.10)),
                    refiner_gain=float(source_arch.get("refiner_gain", 0.12)),
                    prompt_router=str(source_arch.get("prompt_router", "hard")),
                    sensor_route_mode=str(source_arch.get("sensor_route_mode", "posterior")),
                    use_refiner=bool(source_arch.get("use_refiner", True)),
                    compound_motion_blend=float(
                        source_arch.get("compound_motion_blend", 0.0)
                    ),
                    use_compound_refiner=bool(
                        source_arch.get("use_compound_refiner", False)
                    ),
                    compound_refiner_gain=float(
                        source_arch.get("compound_refiner_gain", 0.18)
                    ),
                    use_cause_refiners=bool(
                        source_arch.get("use_cause_refiners", False)
                    ),
                    cause_refiner_gain=float(
                        source_arch.get("cause_refiner_gain", 0.08)
                    ),
                ).to(device)
                incompatible = model.load_state_dict(payload.get("model", payload), strict=False)
                missing_keys = list(incompatible.missing_keys)
                unexpected_keys = list(incompatible.unexpected_keys)
                initialization_kind = "RMR-P prompted-DFPIR checkpoint"
            elif "DFPIR" in init_paths[0].name.upper():
                adapter = DFPIRAdapter(init_paths[0], device=str(device), use_clip=True)
                prompt_embeddings = torch.cat(
                    [
                        adapter._text_code("clean"),
                        adapter._text_code("motion_horizontal_medium"),
                        adapter._text_code("defocus_medium"),
                        adapter._text_code("gaussian_sigma3"),
                        adapter._text_code("lowlight_medium"),
                        adapter._text_code("mixed_motion_lowlight"),
                    ],
                    dim=0,
                )
                model = RMRPPromptedDFPIR(
                    adapter.model,
                    prompt_embeddings,
                    prompt_router="hard",
                    sensor_route_mode="posterior",
                    use_refiner=False,
                ).to(device)
                missing_keys = []
                unexpected_keys = []
                initialization_kind = "official DFPIR backbone plus identity RMR-P metadata modules"
            else:
                averaged, legacy_payload = average_state_dicts(init_paths)
                kwargs = rmrp_architecture(legacy_payload)
                legacy_model = RMRP(enable_aux_contour=False, **kwargs).to(device)
                incompatible = legacy_model.load_state_dict(averaged, strict=False)
                legacy_model.metadata_encoding = legacy_payload.get("arch", {}).get(
                    "metadata_encoding", "legacy"
                )
                self.model = legacy_model
                self.arch = dict(legacy_payload.get("arch", {}))
                self.arch.update(kwargs)
                self.arch["method_name"] = "RMR-P"
                self.load_report.update(
                    {
                        "missing_keys": list(incompatible.missing_keys),
                        "unexpected_keys": list(incompatible.unexpected_keys),
                        "model_soup_members": len(init_paths),
                        "initialization_kind": "legacy compact RMR-P",
                    }
                )
                model = None
            if model is not None:
                self.model = model
                if isinstance(model, RMRPMetadataDeMoE):
                    self.arch = {
                        "model": "rmrp",
                        "method_name": "RMR-P",
                        "backbone": "demoe_sensor_router",
                        "sensor_dim": 82,
                        "sensor_gyro_full_scale": float(
                            model.sensor_gyro_full_scale
                        ),
                        "top_k": int(model.top_k),
                        "refiner_gain": float(model.refiner_gain),
                        "use_refiner": bool(model.use_refiner),
                        "use_compound_blend_gate": bool(
                            model.use_compound_blend_gate
                        ),
                        "compound_blend_init": float(model.compound_blend_init),
                        "compound_metadata_acceptance": float(
                            model.compound_metadata_acceptance
                        ),
                        "cause_route_acceptance": list(
                            model.cause_route_acceptance
                        ),
                        "use_cause_refiners": bool(model.use_cause_refiners),
                        "cause_refiner_gain": float(model.cause_refiner_gain),
                        "backbone_route_mode": str(model.backbone_route_mode),
                        "sensor_task_thresholds": list(
                            model.sensor_task_thresholds
                        ),
                        "use_semantic_adapters": bool(
                            model.use_semantic_adapters
                        ),
                        "semantic_adapter_gain": float(
                            model.semantic_adapter_gain
                        ),
                        "scenario_labels_at_inference": False,
                    }
                else:
                    self.arch = {
                        "model": "rmrp",
                        "method_name": "RMR-P",
                        "backbone": "dfpir_sensor_prompt",
                        "sensor_dim": 82,
                        "sensor_gyro_full_scale": float(model.sensor_gyro_full_scale),
                        "prompt_residual_scale": float(model.prompt_residual_scale),
                        "continuous_state_film": True,
                        "prompt_basis_count": int(model.prompt_basis_count),
                        "refiner_gain": float(model.refiner_gain),
                        "prompt_router": str(model.prompt_router),
                        "sensor_route_mode": str(model.sensor_route_mode),
                        "use_refiner": bool(model.use_refiner),
                        "compound_motion_blend": float(model.compound_motion_blend),
                        "use_compound_refiner": bool(model.use_compound_refiner),
                        "compound_refiner_gain": float(model.compound_refiner_gain),
                        "use_cause_refiners": bool(model.use_cause_refiners),
                        "cause_refiner_gain": float(model.cause_refiner_gain),
                    }
                self.load_report.update(
                    {
                        "missing_keys": missing_keys,
                        "unexpected_keys": unexpected_keys,
                        "model_soup_members": 1,
                        "initialization_kind": initialization_kind,
                    }
                )
        elif kind == "nafnet":
            averaged, payload = average_state_dicts(init_paths)
            width = int(payload.get("arch", {}).get("width", 32))
            self.model = NAFNetRoad(width=width).to(device)
            incompatible = self.model.load_state_dict(averaged, strict=False)
            self.arch = {"model": kind, "width": width}
            self.load_report.update(
                {
                    "missing_keys": list(incompatible.missing_keys),
                    "unexpected_keys": list(incompatible.unexpected_keys),
                    "model_soup_members": len(init_paths),
                }
            )
        elif kind == "nafnet_meta":
            # A sensor-conditioned NAFNet control starts from exactly the same
            # road-trained image function as NAFNet. CodeFiLM is initialized to
            # identity, so any later gain must be learned from the same public
            # 82-value packet and matched target-adaptation stream as RMR-P.
            averaged, payload = average_state_dicts(init_paths)
            width = int(payload.get("arch", {}).get("width", 32))
            self.model = MetadataNAFNetRoad(width=width, code_dim=82).to(device)
            incompatible = self.model.load_state_dict(averaged, strict=False)
            self.arch = {"model": kind, "width": width, "code_dim": 82}
            self.load_report.update(
                {
                    "missing_keys": list(incompatible.missing_keys),
                    "unexpected_keys": list(incompatible.unexpected_keys),
                    "model_soup_members": len(init_paths),
                    "identity_initialized_sensor_conditioning": True,
                }
            )
        elif kind == "demoe":
            adapter = DeMoEAdapter(init_paths[0], device=device, task="scenario")
            self.model = adapter.model
            self.arch = {"model": kind, "task": "scenario"}
        elif kind == "dfpir":
            adapter = DFPIRAdapter(init_paths[0], device=str(device), use_clip=True)
            self.model = adapter.model
            self.text_code = adapter._text_code
            self.arch = {"model": kind, "clip_prompt": True}
        elif kind == "instructir":
            if lm_head_weights is None:
                raise ValueError("--lm-head-weights is required for InstructIR")
            adapter = InstructIRAdapter(init_paths[0], lm_head_weights, device=device)
            self.model = adapter.model
            self.encode_prompt = adapter.encode_prompt
            self.arch = {"model": kind, "language_head": str(lm_head_weights)}
        else:  # pragma: no cover
            raise ValueError(kind)

        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        if kind == "rmrp" and isinstance(self.model, RMRPPromptedDFPIR):
            # DFPIR's task prompt controls a discrete top-k permutation. The
            # backward-compatible prompt-delta tensors are intentionally not
            # optimized; the eight-coordinate state is learned through the
            # differentiable multi-scale FiLM adapters instead.
            for parameter in self.model.prompt_delta.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def instruction(scenario: str) -> str:
        name = scenario.lower()
        if "clean" in name or "identity" in name:
            corruption = "no material corruption; preserve the native image"
        elif "mixed" in name:
            corruption = "combined camera motion and low illumination"
        elif "lowlight" in name:
            corruption = "low illumination and sensor noise"
        elif "defocus" in name:
            corruption = "defocus blur"
        else:
            corruption = "camera-motion blur"
        return (
            f"Correct {corruption} in this road inspection image while preserving "
            "thin cracks, pothole rims, lane markings, and pavement texture."
        )

    def _grouped_forward(self, image: torch.Tensor, scenarios: Sequence[str]) -> torch.Tensor:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, scenario in enumerate(scenarios):
            groups[str(scenario)].append(index)
        outputs: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for scenario, group in groups.items():
            idx = torch.tensor(group, device=image.device, dtype=torch.long)
            subset = image.index_select(0, idx)
            if self.kind == "demoe":
                task = demoe_task_from_scenario(scenario, "scenario")
                prediction = self.model(subset, task=task)["output"]
            elif self.kind == "dfpir":
                code = self.text_code(scenario).expand(subset.shape[0], -1)
                prediction = self.model(subset, code)
            elif self.kind == "instructir":
                embedding = self.encode_prompt(self.instruction(scenario)).expand(subset.shape[0], -1)
                prediction = self.model(subset, embedding)
            else:  # pragma: no cover
                raise ValueError(self.kind)
            outputs.append(prediction)
            indices.append(idx)
        merged = torch.cat(outputs, dim=0)
        order = torch.argsort(torch.cat(indices, dim=0))
        return merged.index_select(0, order).clamp(0.0, 1.0)

    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor,
        scenarios: Sequence[str],
        prompt_teacher_weights: torch.Tensor | None = None,
        prompt_teacher_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        if self.kind == "rmrp":
            result = self.model(
                image,
                metadata,
                return_dict=True,
                return_aux=False,
                prompt_teacher_weights=prompt_teacher_weights,
                prompt_teacher_mask=prompt_teacher_mask,
            )
            return result["restored"], result
        if self.kind == "nafnet":
            restored = self.model(image).clamp(0.0, 1.0)
            return restored, {"restored": restored}
        if self.kind == "nafnet_meta":
            restored = self.model(image, metadata).clamp(0.0, 1.0)
            return restored, {"restored": restored}
        restored = self._grouped_forward(image, scenarios)
        return restored, {"restored": restored}

    def train(self, mode: bool = True) -> "TrainableRestorer":
        super().train(mode)
        self.model.train(mode)
        # DeMoE's router contains BatchNorm1d. Keep only the router in eval
        # mode so batch-one accumulation remains valid and deterministic.
        if self.kind == "demoe" and hasattr(self.model, "mlp_branch"):
            self.model.mlp_branch.eval()
        if self.kind == "rmrp" and isinstance(self.model, RMRPMetadataDeMoE):
            # Batch-one accumulation is used on the 6-GB development GPU.
            # The released DeMoE router's BatchNorm statistics remain frozen,
            # while gradients still flow through its image evidence.
            self.model.backbone.mlp_branch.eval()
        return self

    def state_for_inference(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()


def available_state_loss(
    prediction: torch.Tensor | None,
    target: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    if prediction is None:
        return target.new_zeros(())
    per_sample = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=1)
    mask = available.reshape(-1).to(per_sample.dtype)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def causewise_available_state_loss(
    prediction: torch.Tensor | None,
    target: torch.Tensor,
    availability: torch.Tensor | None,
    sample_available: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise only state coordinates supported by the available sensors.

    The practical packet can be partially observed. Requiring a sensor-only
    branch to predict an unsupported coordinate would teach a dataset prior
    rather than a measurement-conditioned state. The model's cause-wise
    reliability is therefore also the supervision mask. Image and joint
    branches remain supervised on the complete training target.
    """

    if prediction is None:
        return target.new_zeros(())
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    if availability is None:
        mask = torch.ones_like(error)
    else:
        # Stop-gradient prevents the state estimator from reducing its own
        # supervision by collapsing the learned reliability/support values.
        mask = availability.detach().to(device=error.device, dtype=error.dtype)
        if mask.shape != error.shape:
            raise ValueError(
                f"Cause-wise availability shape {tuple(mask.shape)} does not "
                f"match state shape {tuple(error.shape)}"
            )
    if sample_available is not None:
        mask = mask * sample_available.reshape(-1, 1).to(mask.dtype)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


class SensorRecordReservoir:
    """Training-only donor bank for record/image mismatch augmentation.

    DFPIR requires micro-batch one on a 6 GB GPU, so a within-batch shuffle has
    no donor. This bounded reservoir restores the intended augmentation without
    changing the effective optimizer batch. For each current sample, the donor
    with the largest cause-state distance is used when a mismatch is sampled.
    """

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError("sensor mismatch reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.metadata: torch.Tensor | None = None
        self.cause: torch.Tensor | None = None
        self.physical: torch.Tensor | None = None
        self.physical_available: torch.Tensor | None = None

    def donors(
        self,
        cause_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.cause is None:
            return None
        distance = torch.cdist(cause_target.float(), self.cause.float(), p=1)
        indices = distance.argmax(dim=1)
        assert self.metadata is not None
        assert self.physical is not None
        assert self.physical_available is not None
        return (
            self.metadata.index_select(0, indices),
            self.cause.index_select(0, indices),
            self.physical.index_select(0, indices),
            self.physical_available.index_select(0, indices),
        )

    def update(
        self,
        metadata: torch.Tensor,
        cause_target: torch.Tensor,
        physical_target: torch.Tensor,
        physical_available: torch.Tensor,
    ) -> None:
        records = (
            metadata.detach().clone(),
            cause_target.detach().clone(),
            physical_target.detach().clone(),
            physical_available.detach().clone(),
        )
        names = ("metadata", "cause", "physical", "physical_available")
        for name, value in zip(names, records):
            previous = getattr(self, name)
            combined = value if previous is None else torch.cat([previous, value], dim=0)
            setattr(self, name, combined[-self.capacity :])


def mismatch_sensor_records(
    metadata: torch.Tensor,
    cause_target: torch.Tensor,
    physical_target: torch.Tensor,
    physical_available: torch.Tensor,
    probability: float,
    reservoir: SensorRecordReservoir | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inject record/image mismatches without corrupting sensor supervision.

    For a mismatched sample, the sensor-only target follows the donor record.
    The image and joint posterior targets remain attached to the current image,
    so the restoration objective rewards the compatibility gate for rejecting
    a contradictory record instead of forcing the sensor encoder to lie.
    """

    batch = metadata.shape[0]
    sensor_cause = cause_target
    sensor_physical = physical_target
    sensor_physical_available = physical_available
    mismatched = torch.zeros(batch, device=metadata.device, dtype=torch.bool)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("metadata mismatch probability must be in [0, 1]")
    donors = reservoir.donors(cause_target) if reservoir is not None else None
    if probability <= 0.0 or (batch < 2 and donors is None):
        return (
            metadata,
            sensor_cause,
            sensor_physical,
            sensor_physical_available,
            mismatched,
        )
    if donors is None:
        donor = torch.roll(torch.arange(batch, device=metadata.device), shifts=1)
        donor_metadata = metadata[donor]
        donor_cause = cause_target[donor]
        donor_physical = physical_target[donor]
        donor_available = physical_available[donor]
    else:
        donor_metadata, donor_cause, donor_physical, donor_available = donors
    mismatched = torch.rand(batch, device=metadata.device) < float(probability)
    metadata = torch.where(mismatched[:, None], donor_metadata, metadata)
    sensor_cause = torch.where(mismatched[:, None], donor_cause, cause_target)
    sensor_physical = torch.where(
        mismatched[:, None], donor_physical, physical_target
    )
    sensor_physical_available = torch.where(
        mismatched, donor_available, physical_available
    )
    return (
        metadata,
        sensor_cause,
        sensor_physical,
        sensor_physical_available,
        mismatched,
    )


def metadata_robustness_scale(
    epoch: int,
    aligned_epochs: int,
    ramp_epochs: int,
) -> float:
    """Return the curriculum multiplier for sensor robustness augmentation.

    The model first observes image/telemetry pairs without artificial missingness
    or mismatch.  Robustness is then introduced gradually.  Epochs are one-based.
    """

    if epoch < 1:
        raise ValueError("epoch must be one-based and positive")
    if aligned_epochs < 0 or ramp_epochs < 0:
        raise ValueError("metadata curriculum lengths must be non-negative")
    if epoch <= aligned_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    return min(1.0, (epoch - aligned_epochs) / float(ramp_epochs))


def route_teacher_probability(epoch: int, full_epochs: int, ramp_epochs: int) -> float:
    """Return the training-only scheduled-sampling probability."""

    if epoch < 1 or full_epochs < 0 or ramp_epochs < 0:
        raise ValueError("invalid route-teacher schedule")
    if epoch <= full_epochs:
        return 1.0
    if ramp_epochs == 0:
        return 0.0
    return max(0.0, 1.0 - (epoch - full_epochs) / float(ramp_epochs))


def cause_targets_to_prompt_weights(
    cause_target: torch.Tensor,
    prompt_basis_count: int,
) -> torch.Tensor | None:
    """Map private training causes to the six public prompt operators.

    The order is preserve, motion, defocus, noise, low light, and compound
    motion--low-light. These targets are never constructed at validation or
    inference time.
    """

    if prompt_basis_count != 6:
        return None
    motion = cause_target[:, :3].amax(dim=1)
    defocus = cause_target[:, 3]
    noise = torch.maximum(cause_target[:, 4], cause_target[:, 6])
    lowlight = cause_target[:, 5]
    route = torch.stack([motion, defocus, noise, lowlight], dim=1).argmax(dim=1) + 1
    clean = cause_target[:, :7].amax(dim=1) <= 0.05
    compound = (motion > 0.05) & (lowlight > 0.05)
    route = torch.where(clean, torch.zeros_like(route), route)
    route = torch.where(compound, torch.full_like(route, 5), route)
    return F.one_hot(route, num_classes=6).to(cause_target.dtype)


def metadata_state_loss(
    outputs: dict[str, torch.Tensor | None],
    cause_target: torch.Tensor,
    physical_target: torch.Tensor,
    physical_available: torch.Tensor,
    sensor_cause_target: torch.Tensor | None = None,
    sensor_physical_target: torch.Tensor | None = None,
    sensor_physical_available: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    image_code = outputs.get("image_degradation_code")
    sensor_code = outputs.get("sensor_code")
    joint_code = outputs.get("code")
    terms: list[torch.Tensor] = []
    logs: dict[str, float] = {}
    sensor_cause_target = (
        cause_target if sensor_cause_target is None else sensor_cause_target
    )
    sensor_physical_target = (
        physical_target if sensor_physical_target is None else sensor_physical_target
    )
    sensor_physical_available = (
        physical_available
        if sensor_physical_available is None
        else sensor_physical_available
    )
    sensor_support = outputs.get("sensor_cause_reliability")
    for name, value, target, weight in (
        ("image_cause", image_code, cause_target, 0.5),
        ("sensor_cause", sensor_code, sensor_cause_target, 0.5),
        ("joint_cause", joint_code, cause_target, 1.0),
    ):
        if isinstance(value, torch.Tensor):
            if name == "sensor_cause":
                loss = causewise_available_state_loss(
                    value,
                    target,
                    sensor_support if isinstance(sensor_support, torch.Tensor) else None,
                )
            else:
                loss = F.smooth_l1_loss(value, target)
            terms.append(weight * loss)
            logs[name] = float(loss.detach().cpu())

    physical_terms: list[torch.Tensor] = []
    for name, value, target, available, weight in (
        (
            "image_physical",
            outputs.get("image_physical_code"),
            physical_target,
            physical_available,
            0.25,
        ),
        (
            "sensor_physical",
            outputs.get("sensor_only_physical_code"),
            sensor_physical_target,
            sensor_physical_available,
            0.25,
        ),
        (
            "joint_physical",
            outputs.get("sensor_calibrated_physical_code"),
            physical_target,
            physical_available,
            0.50,
        ),
    ):
        if isinstance(value, torch.Tensor):
            if name == "sensor_physical":
                loss = causewise_available_state_loss(
                    value,
                    target,
                    sensor_support if isinstance(sensor_support, torch.Tensor) else None,
                    available,
                )
            else:
                loss = available_state_loss(value, target, available)
            physical_terms.append(weight * loss)
            logs[name] = float(loss.detach().cpu())
    cause = sum(terms, cause_target.new_zeros(()))
    physical = sum(physical_terms, cause_target.new_zeros(()))
    logs["cause_state"] = float(cause.detach().cpu())
    logs["physical_state"] = float(physical.detach().cpu())
    return cause, {**logs, "physical_tensor": physical}  # type: ignore[dict-item]


def build_detector_losses(
    detectors: Sequence[tuple[str, Path]],
    device: torch.device,
    input_size: int,
    cqmix_probability: float,
    clean_hinge_weight: float,
) -> tuple[
    dict[str, TaskDrivenPerceptualLoss],
    dict[str, FrozenDetectorSupervisedLoss],
]:
    from ultralytics import YOLO

    tdp_result: dict[str, TaskDrivenPerceptualLoss] = {}
    supervised_result: dict[str, FrozenDetectorSupervisedLoss] = {}
    for tag, weights in detectors:
        detector = YOLO(str(weights)).model.to(device).eval()
        # Keep the executable Ultralytics DetectionModel intact. Backbone
        # hooks are addressed through its internal module-list namespace.
        extractor = FrozenDetectorFeatureExtractor(
            detector,
            layer_names=("model.2", "model.4"),
            input_size=(input_size, input_size),
            verbose=True,
        )
        tdp_result[tag] = TaskDrivenPerceptualLoss(
            extractor,
            layer_weights={"model.2": 0.5, "model.4": 1.0},
            cqmix_prob=0.0,
            defect_mask_weight=1.0,
            normalize_features=True,
        )
        supervised_result[tag] = FrozenDetectorSupervisedLoss(
            detector,
            input_size=(input_size, input_size),
            cqmix_prob=cqmix_probability,
            clean_hinge_weight=clean_hinge_weight,
        )
    return tdp_result, supervised_result


def grouped_tdp(
    losses: dict[str, TaskDrivenPerceptualLoss],
    restored: torch.Tensor,
    target: torch.Tensor,
    tags: Sequence[str],
) -> torch.Tensor:
    total = restored.new_zeros(())
    count = 0
    for tag in sorted(set(tags)):
        indices = [index for index, value in enumerate(tags) if value == tag]
        idx = torch.tensor(indices, device=restored.device, dtype=torch.long)
        total = total + len(indices) * losses[tag](
            restored.index_select(0, idx),
            target.index_select(0, idx),
            use_cqmix=False,
        )
        count += len(indices)
    return total / max(count, 1)


def grouped_detector_supervised(
    losses: dict[str, FrozenDetectorSupervisedLoss],
    restored: torch.Tensor,
    target: torch.Tensor,
    classes: torch.Tensor,
    bboxes: torch.Tensor,
    valid: torch.Tensor,
    tags: Sequence[str],
) -> torch.Tensor:
    """Apply each dataset's frozen detector to its own labelled crops.

    The detector is frozen. Gradients flow only through the restored image,
    and boxes have already been transformed by the paired crop/flip loader.
    Empty-target crops remain valid negatives for the detector objective.
    """

    total = restored.new_zeros(())
    count = 0
    for tag in sorted(set(tags)):
        indices = [index for index, value in enumerate(tags) if value == tag]
        idx = torch.tensor(indices, device=restored.device, dtype=torch.long)
        loss = losses[tag](
            restored.index_select(0, idx),
            target.index_select(0, idx),
            classes=classes.index_select(0, idx),
            bboxes=bboxes.index_select(0, idx),
            valid=valid.index_select(0, idx),
        )
        total = total + len(indices) * loss
        count += len(indices)
    return total / max(count, 1)


def save_checkpoint(
    path: Path,
    restorer: TrainableRestorer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    row: dict[str, Any],
) -> None:
    payload = {
        "model": restorer.state_for_inference(),
        "arch": restorer.arch,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "metrics": row,
        "method_name": "RMR-P" if restorer.kind == "rmrp" else restorer.kind,
    }
    torch.save(payload, path)
    if restorer.kind == "instructir":
        torch.save(restorer.state_for_inference(), path.with_name(path.stem + "_image_state.pt"))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def accumulate_domain_gradients(
    parameters: Sequence[torch.nn.Parameter],
    buffer: list[torch.Tensor] | None,
) -> list[torch.Tensor]:
    """Accumulate one unscaled micro-batch gradient for a PCGrad domain."""

    if buffer is None:
        buffer = [torch.zeros_like(parameter) for parameter in parameters]
    for target, parameter in zip(buffer, parameters):
        if parameter.grad is not None:
            target.add_(parameter.grad.detach())
    return buffer


def apply_equal_domain_pcgrad(
    parameters: Sequence[torch.nn.Parameter],
    buffers: dict[str, list[torch.Tensor]],
    counts: Counter[str],
) -> tuple[bool, float]:
    """Write an equal-domain PCGrad update into ``parameter.grad``.

    For the two detector domains, with mean gradients ``g_a`` and ``g_b``,
    conflicting components are removed symmetrically:

        g'_a = g_a - min(<g_a,g_b>, 0) g_b / (||g_b||^2 + eps),
        g'_b = g_b - min(<g_a,g_b>, 0) g_a / (||g_a||^2 + eps).

    The optimizer receives ``(g'_a + g'_b)/2``. A single present domain is
    passed through unchanged, which is used only for a final partial window.
    """

    tags = sorted(buffers)
    if not tags:
        raise RuntimeError("PCGrad received no domain gradients")
    means = {
        tag: [gradient / max(int(counts[tag]), 1) for gradient in buffers[tag]]
        for tag in tags
    }
    conflict = False
    cosine = 1.0
    if len(tags) == 2:
        left, right = tags
        dot = sum(
            (a.float() * b.float()).sum() for a, b in zip(means[left], means[right])
        )
        left_norm = sum((a.float() ** 2).sum() for a in means[left])
        right_norm = sum((b.float() ** 2).sum() for b in means[right])
        cosine = float(
            (dot / (left_norm.sqrt() * right_norm.sqrt()).clamp_min(1e-12))
            .detach()
            .cpu()
        )
        if float(dot.detach().cpu()) < 0.0:
            conflict = True
            left_scale = dot / right_norm.clamp_min(1e-12)
            right_scale = dot / left_norm.clamp_min(1e-12)
            projected_left = [
                a - left_scale.to(dtype=a.dtype) * b
                for a, b in zip(means[left], means[right])
            ]
            projected_right = [
                b - right_scale.to(dtype=b.dtype) * a
                for a, b in zip(means[left], means[right])
            ]
            means[left] = projected_left
            means[right] = projected_right
    elif len(tags) > 2:
        raise RuntimeError("This audited PCGrad implementation supports two domains")

    for index, parameter in enumerate(parameters):
        merged = sum(means[tag][index] for tag in tags) / float(len(tags))
        parameter.grad = merged
    return conflict, cosine


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    resume_path = args.resume_from.resolve() if args.resume_from else None
    if resume_path is None:
        args.out.mkdir(parents=True, exist_ok=False)
    else:
        if not args.out.exists():
            raise FileNotFoundError(f"Resume output directory does not exist: {args.out}")
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
    scenarios = tuple(args.scenarios or DEFAULT_SCENARIOS)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    init_paths = (
        [resume_path]
        if resume_path is not None
        else [path.resolve() for path in map(Path, args.init_weights)]
    )
    detector_paths = [(tag, path.resolve()) for tag, path in args.detector]
    label_roots = {tag: path.resolve() for tag, path in (args.defect_label_root or [])}
    roots = [(tag, path.resolve()) for tag, path in args.data_root]
    for path in [*init_paths, *(path for _, path in detector_paths), *(path for _, path in roots)]:
        if not path.exists():
            raise FileNotFoundError(path)

    micro_batch = args.micro_batch_size
    if micro_batch <= 0:
        micro_batch = 1 if args.model == "dfpir" else 2
    if args.effective_batch_size % micro_batch != 0:
        raise ValueError("effective batch size must be divisible by micro batch size")
    for name in ("metadata_dropout", "metadata_mismatch_probability"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.metadata_noise < 0.0:
        raise ValueError("metadata_noise must be non-negative")
    if not 0.0 <= args.detector_cqmix_probability <= 1.0:
        raise ValueError("detector_cqmix_probability must be in [0, 1]")
    if args.detector_clean_hinge_weight < 0.0:
        raise ValueError("detector_clean_hinge_weight must be non-negative")
    if args.metadata_curriculum_epochs < 0:
        raise ValueError("metadata_curriculum_epochs must be non-negative")
    if args.metadata_curriculum_ramp_epochs < 0:
        raise ValueError("metadata_curriculum_ramp_epochs must be non-negative")
    if args.route_teacher_epochs < 0 or args.route_teacher_ramp_epochs < 0:
        raise ValueError("route teacher schedule lengths must be non-negative")
    if not 0.0 <= args.rmrp_compound_motion_blend <= 1.0:
        raise ValueError("rmrp_compound_motion_blend must be in [0, 1]")
    if not 0.0 < args.rmrp_compound_blend_init < 1.0:
        raise ValueError("rmrp_compound_blend_init must be strictly between 0 and 1")
    if args.rmrp_freeze_compound_gate_only and not args.rmrp_compound_blend_gate:
        raise ValueError(
            "--rmrp-freeze-compound-gate-only requires --rmrp-compound-blend-gate"
        )
    if args.rmrp_freeze_cause_refiners_only and not args.rmrp_cause_refiners:
        raise ValueError(
            "--rmrp-freeze-cause-refiners-only requires --rmrp-cause-refiners"
        )
    routed_expert_indices = tuple(dict.fromkeys(args.rmrp_routed_expert_indices))
    if not routed_expert_indices:
        raise ValueError("--rmrp-routed-expert-indices cannot be empty")
    if any(index < 0 or index > 4 for index in routed_expert_indices):
        raise ValueError("RMR-P routed expert indices must be in [0, 4]")
    args.rmrp_routed_expert_indices = list(routed_expert_indices)
    if args.rmrp_freeze_semantic_adapters_only and not args.rmrp_semantic_adapters:
        raise ValueError(
            "--rmrp-freeze-semantic-adapters-only requires --rmrp-semantic-adapters"
        )
    exclusive_calibrations = sum(
        bool(value)
        for value in (
            args.rmrp_freeze_cause_refiners_only,
            args.rmrp_freeze_routed_experts_only,
            args.rmrp_freeze_compound_gate_only,
            args.rmrp_freeze_semantic_adapters_only,
        )
    )
    if exclusive_calibrations > 1:
        raise ValueError("RMR-P calibration-only modes are mutually exclusive")
    if (
        args.rmrp_compound_metadata_acceptance is not None
        and not 0.0 <= args.rmrp_compound_metadata_acceptance <= 1.0
    ):
        raise ValueError("rmrp_compound_metadata_acceptance must be in [0, 1]")
    accumulation = args.effective_batch_size // micro_batch
    if args.dataset_gradient_surgery:
        if micro_batch != 1:
            raise ValueError("--dataset-gradient-surgery requires micro batch size 1")
        if len({tag for tag, _ in args.data_root}) != 2:
            raise ValueError("--dataset-gradient-surgery requires exactly two data roots")

    dataset, sampler_weights, group_counts = build_dataset(
        roots,
        scenarios,
        args.patch_size,
        label_roots,
        args.defect_crop_probability,
    )
    restorer = TrainableRestorer(
        args.model,
        init_paths,
        device,
        Path(args.lm_head_weights).resolve() if args.lm_head_weights else None,
    ).to(device)
    if args.model == "rmrp" and isinstance(restorer.model, RMRPPromptedDFPIR):
        restorer.model.sensor_route_mode = args.rmrp_sensor_route_mode
        if args.rmrp_bounded_refiner:
            restorer.model.enable_bounded_refiner(args.rmrp_refiner_gain)
        if args.rmrp_compound_refiner:
            restorer.model.enable_compound_refiner(args.rmrp_compound_refiner_gain)
        if args.rmrp_cause_refiners:
            restorer.model.enable_cause_refiners(args.rmrp_cause_refiner_gain)
        restorer.model.compound_motion_blend = float(args.rmrp_compound_motion_blend)
        if args.rmrp_freeze_backbone:
            # Keep the matched DFPIR image prior fixed while learning only the
            # zero-initialized FiLM layers inserted by RMR-P.  These adapters
            # live inside ``backbone`` and therefore must be re-enabled after
            # freezing the original DFPIR parameters.
            for name, parameter in restorer.model.backbone.named_parameters():
                parameter.requires_grad_("condition_affine" in name)
        if args.rmrp_freeze_cause_refiners_only:
            for parameter in restorer.model.parameters():
                parameter.requires_grad_(False)
            if restorer.model.cause_refiners is None:
                raise RuntimeError("RMR-P cause refiners were not enabled")
            for parameter in restorer.model.cause_refiners.parameters():
                parameter.requires_grad_(True)
        restorer.arch.update(
            {
                "sensor_route_mode": restorer.model.sensor_route_mode,
                "use_refiner": bool(restorer.model.use_refiner),
                "refiner_gain": float(restorer.model.refiner_gain),
                "compound_motion_blend": float(restorer.model.compound_motion_blend),
                "use_compound_refiner": bool(restorer.model.use_compound_refiner),
                "compound_refiner_gain": float(restorer.model.compound_refiner_gain),
                "use_cause_refiners": bool(restorer.model.use_cause_refiners),
                "cause_refiner_gain": float(restorer.model.cause_refiner_gain),
                "cause_refiners_only_calibration": bool(
                    args.rmrp_freeze_cause_refiners_only
                ),
                "backbone_frozen_during_sensor_adaptation": bool(
                    args.rmrp_freeze_backbone
                ),
                "continuous_film_trainable_during_sensor_adaptation": bool(
                    args.rmrp_freeze_backbone
                ),
            }
        )
    if args.model == "rmrp" and isinstance(restorer.model, RMRPMetadataDeMoE):
        restorer.model.backbone_route_mode = str(args.rmrp_backbone_route_mode)
        if args.rmrp_bounded_refiner:
            restorer.model.enable_bounded_refiner(args.rmrp_refiner_gain)
        if args.rmrp_cause_refiners:
            restorer.model.enable_cause_refiners(args.rmrp_cause_refiner_gain)
        if args.rmrp_semantic_adapters:
            restorer.model.enable_semantic_adapters(
                args.rmrp_semantic_adapter_gain
            )
        if args.rmrp_compound_blend_gate and not restorer.model.use_compound_blend_gate:
            restorer.model.enable_compound_blend_gate(args.rmrp_compound_blend_init)
        if args.rmrp_compound_metadata_acceptance is not None:
            restorer.model.compound_metadata_acceptance = float(
                args.rmrp_compound_metadata_acceptance
            )
        if args.rmrp_freeze_backbone:
            for parameter in restorer.model.backbone.parameters():
                parameter.requires_grad_(False)
        if args.rmrp_freeze_compound_gate_only:
            for parameter in restorer.model.parameters():
                parameter.requires_grad_(False)
            if restorer.model.compound_blend_head is None:
                raise RuntimeError("RMR-P compound blend head was not enabled")
            for parameter in restorer.model.compound_blend_head.parameters():
                parameter.requires_grad_(True)
        if args.rmrp_freeze_cause_refiners_only:
            for parameter in restorer.model.parameters():
                parameter.requires_grad_(False)
            if restorer.model.cause_refiners is None:
                raise RuntimeError("RMR-P cause refiners were not enabled")
            for parameter in restorer.model.cause_refiners.parameters():
                parameter.requires_grad_(True)
        if args.rmrp_freeze_semantic_adapters_only:
            for parameter in restorer.model.parameters():
                parameter.requires_grad_(False)
            if restorer.model.semantic_adapters is None:
                raise RuntimeError("RMR-P semantic adapters were not enabled")
            for parameter in restorer.model.semantic_adapters.parameters():
                parameter.requires_grad_(True)
        if args.rmrp_freeze_routed_experts_only:
            for parameter in restorer.model.parameters():
                parameter.requires_grad_(False)
            for block in restorer.model.backbone.experts:
                for expert_index in routed_expert_indices:
                    for parameter in block.experts[expert_index].parameters():
                        parameter.requires_grad_(True)
        restorer.arch.update(
            {
                "top_k": int(restorer.model.top_k),
                "use_refiner": bool(restorer.model.use_refiner),
                "refiner_gain": float(restorer.model.refiner_gain),
                "use_compound_blend_gate": bool(
                    restorer.model.use_compound_blend_gate
                ),
                "compound_blend_init": float(restorer.model.compound_blend_init),
                "compound_metadata_acceptance": float(
                    restorer.model.compound_metadata_acceptance
                ),
                "cause_route_acceptance": list(
                    restorer.model.cause_route_acceptance
                ),
                "use_cause_refiners": bool(restorer.model.use_cause_refiners),
                "cause_refiner_gain": float(restorer.model.cause_refiner_gain),
                "backbone_route_mode": str(restorer.model.backbone_route_mode),
                "sensor_task_thresholds": list(
                    restorer.model.sensor_task_thresholds
                ),
                "use_semantic_adapters": bool(
                    restorer.model.use_semantic_adapters
                ),
                "semantic_adapter_gain": float(
                    restorer.model.semantic_adapter_gain
                ),
                "semantic_adapter_acceptance": list(
                    restorer.model.semantic_adapter_acceptance
                ),
                "semantic_adapters_only_calibration": bool(
                    args.rmrp_freeze_semantic_adapters_only
                ),
                "cause_refiners_only_calibration": bool(
                    args.rmrp_freeze_cause_refiners_only
                ),
                "routed_experts_only_calibration": bool(
                    args.rmrp_freeze_routed_experts_only
                ),
                "routed_expert_indices": list(routed_expert_indices),
                "compound_gate_only_calibration": bool(
                    args.rmrp_freeze_compound_gate_only
                ),
                "scenario_labels_at_inference": False,
                "backbone_frozen_during_sensor_adaptation": bool(
                    args.rmrp_freeze_backbone
                    or args.rmrp_freeze_compound_gate_only
                    or args.rmrp_freeze_semantic_adapters_only
                ),
                "shared_backbone_frozen_during_routed_expert_calibration": bool(
                    args.rmrp_freeze_routed_experts_only
                ),
            }
        )
    common_loss = RCADLoss(edge_weight=args.edge_weight, profile="simple").to(device)
    tdp_losses, detector_supervised_losses = build_detector_losses(
        detector_paths,
        device,
        args.tdp_input_size,
        args.detector_cqmix_probability,
        args.detector_clean_hinge_weight,
    )

    trainable_named = [
        (name, parameter)
        for name, parameter in restorer.named_parameters()
        if parameter.requires_grad
    ]
    new_module_tokens = (
        "model.sensor_prior_fusion.",
        "model.post_prior_evidence_refiner.",
        "model.sensor_encoder.",
        "model.image_state.",
        "model.posterior_refine.",
        "model.route_residual.",
        ".condition_affine.",
        "model.refiner.",
        "model.compound_refiner.",
        "model.compound_blend_head.",
        "model.cause_refiners.",
        "model.semantic_adapters.",
    )
    new_module_parameters = [
        parameter
        for name, parameter in trainable_named
        if any(token in name for token in new_module_tokens)
    ]
    core_parameters = [
        parameter
        for name, parameter in trainable_named
        if not any(token in name for token in new_module_tokens)
    ]
    optimizer_groups: list[dict[str, Any]] = []
    if core_parameters:
        optimizer_groups.append(
            {"params": core_parameters, "lr": args.lr, "group_name": "core"}
        )
    if new_module_parameters:
        optimizer_groups.append(
            {
                "params": new_module_parameters,
                "lr": args.lr * args.new_module_lr_multiplier,
                "group_name": "identity_initialized_modules",
            }
        )
    if not optimizer_groups:
        raise RuntimeError("No trainable parameters remain after applying freeze policy")
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(args.samples_per_epoch / args.effective_batch_size)
    total_steps = max(1, args.epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * args.min_lr_ratio,
    )
    uses_dfpir_backbone = args.model == "dfpir" or restorer.arch.get("backbone") == "dfpir_sensor_prompt"
    # PCGrad stores and projects unscaled per-domain gradients. Running that
    # small calibration path in FP32 avoids mixing GradScaler state across the
    # two independently accumulated detector objectives.
    use_amp = (
        device.type == "cuda"
        and not uses_dfpir_backbone
        and not args.dataset_gradient_surgery
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    global_step = 0
    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        saved_args = resume_payload.get("args", {})
        critical = [
            "model",
            "epochs",
            "samples_per_epoch",
            "effective_batch_size",
            "micro_batch_size",
            "lr",
            "min_lr_ratio",
            "weight_decay",
        ]
        if args.model == "rmrp":
            critical.extend(
                [
                    "metadata_dropout",
                    "metadata_noise",
                    "metadata_mismatch_probability",
                    "metadata_curriculum_epochs",
                    "metadata_curriculum_ramp_epochs",
                    "route_teacher_epochs",
                    "route_teacher_ramp_epochs",
                    "rmrp_sensor_route_mode",
                    "rmrp_bounded_refiner",
                    "rmrp_refiner_gain",
                    "rmrp_cause_refiners",
                    "rmrp_cause_refiner_gain",
                    "rmrp_freeze_cause_refiners_only",
                    "rmrp_freeze_routed_experts_only",
                    "rmrp_routed_expert_indices",
                    "rmrp_backbone_route_mode",
                    "rmrp_semantic_adapters",
                    "rmrp_semantic_adapter_gain",
                    "rmrp_freeze_semantic_adapters_only",
                    "rmrp_freeze_backbone",
                    "rmrp_compound_blend_gate",
                    "rmrp_compound_blend_init",
                    "rmrp_compound_metadata_acceptance",
                    "rmrp_freeze_compound_gate_only",
                    "rmrp_compound_motion_blend",
                    "rmrp_compound_refiner",
                    "rmrp_compound_refiner_gain",
                    "dataset_gradient_surgery",
                ]
            )
        resume_defaults = {
            "rmrp_compound_motion_blend": 0.0,
            "rmrp_compound_blend_gate": False,
            "rmrp_compound_blend_init": 0.65,
            "rmrp_compound_metadata_acceptance": None,
            "rmrp_freeze_compound_gate_only": False,
            "rmrp_cause_refiners": False,
            "rmrp_cause_refiner_gain": 0.08,
            "rmrp_freeze_cause_refiners_only": False,
            "rmrp_freeze_routed_experts_only": False,
            "rmrp_routed_expert_indices": [0, 3, 4],
            "rmrp_backbone_route_mode": "metadata",
            "rmrp_semantic_adapters": False,
            "rmrp_semantic_adapter_gain": 0.25,
            "rmrp_freeze_semantic_adapters_only": False,
            "rmrp_freeze_backbone": False,
            "rmrp_compound_refiner": False,
            "rmrp_compound_refiner_gain": 0.18,
            "dataset_gradient_surgery": False,
        }
        disagreements = {
            name: (saved_args.get(name, resume_defaults.get(name)), getattr(args, name))
            for name in critical
            if saved_args.get(name, resume_defaults.get(name)) != getattr(args, name)
        }
        if disagreements:
            raise RuntimeError(f"Resume protocol mismatch: {disagreements}")
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        saved_epoch = int(resume_payload["epoch"])
        start_epoch = saved_epoch + 1
        global_step = int(
            resume_payload.get("metrics", {}).get(
                "optimizer_step", saved_epoch * steps_per_epoch
            )
        )
        if start_epoch > args.epochs:
            raise RuntimeError(
                f"Checkpoint epoch {saved_epoch} already reaches requested epoch {args.epochs}"
            )

    audit = {
        "protocol": "matched target-domain adaptation v1",
        "test_split_used": False,
        "selection": "external validation-only detector mAP",
        "model": args.model,
        "method_display": "RMR-P" if args.model == "rmrp" else args.model,
        "training_roots": {tag: str(path) for tag, path in roots},
        "scenarios": list(scenarios),
        "group_counts": group_counts,
        "samples_per_epoch": args.samples_per_epoch,
        "effective_batch_size": args.effective_batch_size,
        "micro_batch_size": micro_batch,
        "gradient_accumulation": accumulation,
        "optimizer_steps": total_steps,
        "optimizer_parameter_groups": {
            str(group.get("group_name", index)): {
                "parameters": sum(parameter.numel() for parameter in group["params"]),
                "initial_lr": float(group["lr"]),
            }
            for index, group in enumerate(optimizer_groups)
        },
        "common_objective": {
            "charbonnier": 1.0,
            "gradient": args.edge_weight,
            "detector_feature": args.tdp_weight,
            "detector_supervised": args.detector_supervised_weight,
            "detector_cqmix_probability": args.detector_cqmix_probability,
            "detector_clean_hinge_weight": args.detector_clean_hinge_weight,
            "detector_feature_layers": ["model.2", "model.4"],
            "detector_feature_layer_weights": [0.5, 1.0],
            "dataset_gradient_surgery": bool(args.dataset_gradient_surgery),
            "dataset_gradient_merge": (
                "equal-domain symmetric PCGrad"
                if args.dataset_gradient_surgery
                else "standard effective-batch mean"
            ),
        },
        "rmrp_state_objective": {
            "cause": args.state_weight if args.model == "rmrp" else 0.0,
            "physical": args.physical_weight if args.model == "rmrp" else 0.0,
            "metadata_dropout": args.metadata_dropout if args.model == "rmrp" else 0.0,
            "metadata_noise": args.metadata_noise if args.model == "rmrp" else 0.0,
            "metadata_mismatch_probability": (
                args.metadata_mismatch_probability if args.model == "rmrp" else 0.0
            ),
            "metadata_curriculum_aligned_epochs": (
                args.metadata_curriculum_epochs if args.model == "rmrp" else 0
            ),
            "metadata_curriculum_ramp_epochs": (
                args.metadata_curriculum_ramp_epochs if args.model == "rmrp" else 0
            ),
            "route_teacher_epochs": (
                args.route_teacher_epochs if args.model == "rmrp" else 0
            ),
            "route_teacher_ramp_epochs": (
                args.route_teacher_ramp_epochs if args.model == "rmrp" else 0
            ),
            "sensor_state_masking": args.model == "rmrp",
        },
        "load_report": restorer.load_report,
        "args": json_ready(vars(args)),
    }
    if resume_path is None:
        (args.out / "audit_config.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
    else:
        audit_path = args.out / "audit_config.json"
        if not audit_path.exists():
            raise FileNotFoundError(
                f"Cannot resume an unaudited training directory: {audit_path}"
            )
        resume_history_path = args.out / "resume_history.json"
        resume_history = (
            json.loads(resume_history_path.read_text(encoding="utf-8"))
            if resume_history_path.exists()
            else []
        )
        resume_history.append(
            {
                "checkpoint": str(resume_path),
                "checkpoint_sha256": sha256(resume_path),
                "saved_epoch": int(resume_payload["epoch"]),
                "optimizer_step": global_step,
                "resumed_unix": time.time(),
                "test_split_used": False,
            }
        )
        resume_history_path.write_text(
            json.dumps(resume_history, indent=2), encoding="utf-8"
        )

    # Epoch zero is a required validation reference. It distinguishes gains
    # produced by matched adaptation from gains already present in public or
    # road-domain initialization weights.
    if resume_path is None:
        save_checkpoint(
            args.out / f"{args.model}_epoch_000.pth",
            restorer,
            optimizer,
            scheduler,
            0,
            args,
            {"epoch": 0, "optimizer_step": 0, "selection_eligible": True},
        )

    optimizer.zero_grad(set_to_none=True)
    stop_after = args.smoke_steps if args.smoke_steps > 0 else None
    for epoch in range(start_epoch, args.epochs + 1):
        set_seed(args.seed + epoch)
        generator = torch.Generator().manual_seed(args.seed + 1009 * epoch)
        sampler = WeightedRandomSampler(
            sampler_weights,
            num_samples=args.samples_per_epoch,
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(
            dataset,
            batch_size=micro_batch,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        restorer.train(True)
        mismatch_reservoir = SensorRecordReservoir(capacity=32)
        epoch_sums: defaultdict[str, float] = defaultdict(float)
        epoch_start = time.perf_counter()
        optimizer_steps = 0
        micro_steps = 0
        robustness_scale = (
            metadata_robustness_scale(
                epoch,
                args.metadata_curriculum_epochs,
                args.metadata_curriculum_ramp_epochs,
            )
            if args.model == "rmrp"
            else 0.0
        )
        effective_metadata_dropout = args.metadata_dropout * robustness_scale
        effective_metadata_noise = args.metadata_noise * robustness_scale
        effective_metadata_mismatch = (
            args.metadata_mismatch_probability * robustness_scale
        )
        teacher_probability = (
            route_teacher_probability(
                epoch,
                args.route_teacher_epochs,
                args.route_teacher_ramp_epochs,
            )
            if args.model == "rmrp"
            else 0.0
        )
        pcgrad_parameters = [
            parameter for parameter in restorer.parameters() if parameter.requires_grad
        ]
        domain_gradient_buffers: dict[str, list[torch.Tensor]] = {}
        domain_gradient_counts: Counter[str] = Counter()
        micro_steps_since_update = 0

        def step_pcgrad_window() -> None:
            nonlocal optimizer_steps, global_step, micro_steps_since_update
            conflict, cosine = apply_equal_domain_pcgrad(
                pcgrad_parameters,
                domain_gradient_buffers,
                domain_gradient_counts,
            )
            torch.nn.utils.clip_grad_norm_(pcgrad_parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optimizer_steps += 1
            global_step += 1
            epoch_sums["pcgrad_updates"] += 1.0
            epoch_sums["pcgrad_conflicts"] += float(conflict)
            epoch_sums["pcgrad_cosine"] += cosine
            domain_gradient_buffers.clear()
            domain_gradient_counts.clear()
            micro_steps_since_update = 0

        for batch_index, batch in enumerate(loader, start=1):
            degraded = batch["input"].to(device, non_blocking=True)
            target = batch["gt"].to(device, non_blocking=True)
            metadata = batch["metadata_code"].to(device, non_blocking=True)
            cause_target = batch["cause_target"].to(device, non_blocking=True)
            physical_target = batch["physical_target"].to(device, non_blocking=True)
            physical_available = batch["physical_target_available"].to(device, non_blocking=True)
            detector_classes = batch["detector_classes"].to(device, non_blocking=True)
            detector_bboxes = batch["detector_bboxes"].to(device, non_blocking=True)
            detector_valid = batch["detector_valid"].to(device, non_blocking=True)
            scenarios_batch = list(batch["scenario"])
            tags_batch = list(batch["dataset_tag"])

            if args.model == "rmrp":
                original_sensor_record = (
                    metadata,
                    cause_target,
                    physical_target,
                    physical_available,
                )
                (
                    metadata,
                    sensor_cause_target,
                    sensor_physical_target,
                    sensor_physical_available,
                    metadata_mismatched,
                ) = mismatch_sensor_records(
                    metadata,
                    cause_target,
                    physical_target,
                    physical_available,
                    effective_metadata_mismatch,
                    mismatch_reservoir,
                )
                mismatch_reservoir.update(*original_sensor_record)
                metadata = balanced_sensor_dropout(
                    metadata, effective_metadata_dropout
                )
                metadata = perturb_sensor_packet(metadata, effective_metadata_noise)
            else:
                sensor_cause_target = cause_target
                sensor_physical_target = physical_target
                sensor_physical_available = physical_available
                metadata_mismatched = torch.zeros_like(
                    physical_available, dtype=torch.bool
                )

            warmup = min(1.0, epoch / max(args.tdp_warmup_epochs, 1))
            prompt_teacher_weights = None
            prompt_teacher_mask = None
            if args.model == "rmrp" and teacher_probability > 0.0:
                prompt_basis_count = int(
                    getattr(restorer.model, "prompt_basis_count", 0)
                )
                prompt_teacher_weights = cause_targets_to_prompt_weights(
                    cause_target,
                    prompt_basis_count,
                )
                if prompt_teacher_weights is not None:
                    prompt_teacher_mask = (
                        torch.rand(cause_target.shape[0], device=device)
                        < teacher_probability
                    )
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                restored, outputs = restorer(
                    degraded,
                    metadata,
                    scenarios_batch,
                    prompt_teacher_weights=prompt_teacher_weights,
                    prompt_teacher_mask=prompt_teacher_mask,
                )
                base = common_loss(restored, target)
                tdp = grouped_tdp(tdp_losses, restored, target, tags_batch)
                detector_supervised = base.new_zeros(())
                if args.detector_supervised_weight > 0.0:
                    detector_supervised = grouped_detector_supervised(
                        detector_supervised_losses,
                        restored,
                        target,
                        detector_classes,
                        detector_bboxes,
                        detector_valid,
                        tags_batch,
                    )
                state = base.new_zeros(())
                physical = base.new_zeros(())
                state_logs: dict[str, Any] = {}
                if args.model == "rmrp":
                    state, state_logs = metadata_state_loss(
                        outputs,
                        cause_target,
                        physical_target,
                        physical_available,
                        sensor_cause_target,
                        sensor_physical_target,
                        sensor_physical_available,
                    )
                    physical = state_logs.pop("physical_tensor")
                # Manuscript Eq. (8): the detector and state terms enter the
                # same backward pass; validation, never test data, selects the
                # checkpoint and the later compound compatibility coefficient.
                total = (
                    base
                    + warmup * args.tdp_weight * tdp
                    + warmup
                    * args.detector_supervised_weight
                    * detector_supervised
                    + args.state_weight * state
                    + args.physical_weight * physical
                )
                scaled = total / accumulation

            if args.dataset_gradient_surgery:
                if len(set(tags_batch)) != 1:
                    raise RuntimeError("PCGrad micro-batch mixed multiple domains")
                total.backward()
                tag = str(tags_batch[0])
                domain_gradient_buffers[tag] = accumulate_domain_gradients(
                    pcgrad_parameters,
                    domain_gradient_buffers.get(tag),
                )
                domain_gradient_counts[tag] += 1
                optimizer.zero_grad(set_to_none=True)
                micro_steps_since_update += 1
            else:
                scaler.scale(scaled).backward()
            micro_steps += 1
            if (
                args.dataset_gradient_surgery
                and micro_steps_since_update >= accumulation
                and len(domain_gradient_buffers) == 2
            ):
                step_pcgrad_window()
            elif not args.dataset_gradient_surgery and micro_steps % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(restorer.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
                global_step += 1

            epoch_sums["loss"] += float(total.detach().cpu())
            epoch_sums["base"] += float(base.detach().cpu())
            epoch_sums["tdp"] += float(tdp.detach().cpu())
            epoch_sums["detector_supervised"] += float(
                detector_supervised.detach().cpu()
            )
            epoch_sums["state"] += float(state.detach().cpu())
            epoch_sums["physical"] += float(physical.detach().cpu())
            epoch_sums["metadata_mismatch_fraction"] += float(
                metadata_mismatched.float().mean().detach().cpu()
            )
            if prompt_teacher_mask is not None:
                epoch_sums["route_teacher_fraction"] += float(
                    prompt_teacher_mask.float().mean().detach().cpu()
                )
            if stop_after is not None and global_step >= stop_after:
                break

        if args.dataset_gradient_surgery and domain_gradient_buffers:
            step_pcgrad_window()

        count = max(micro_steps, 1)
        row = {
            "epoch": epoch,
            "optimizer_step": global_step,
            "loss": epoch_sums["loss"] / count,
            "base": epoch_sums["base"] / count,
            "tdp": epoch_sums["tdp"] / count,
            "detector_supervised": (
                epoch_sums["detector_supervised"] / count
            ),
            "state": epoch_sums["state"] / count,
            "physical": epoch_sums["physical"] / count,
            "metadata_mismatch_fraction": (
                epoch_sums["metadata_mismatch_fraction"] / count
            ),
            "metadata_robustness_scale": robustness_scale,
            "metadata_dropout_probability": effective_metadata_dropout,
            "metadata_noise_sigma": effective_metadata_noise,
            "metadata_mismatch_probability": effective_metadata_mismatch,
            "route_teacher_probability": teacher_probability,
            "route_teacher_fraction": epoch_sums["route_teacher_fraction"] / count,
            "pcgrad_conflict_fraction": (
                epoch_sums["pcgrad_conflicts"]
                / max(epoch_sums["pcgrad_updates"], 1.0)
            ),
            "pcgrad_mean_domain_cosine": (
                epoch_sums["pcgrad_cosine"]
                / max(epoch_sums["pcgrad_updates"], 1.0)
            ),
            "lr": optimizer.param_groups[0]["lr"],
            "new_module_lr": (
                optimizer.param_groups[1]["lr"]
                if len(optimizer.param_groups) > 1
                else optimizer.param_groups[0]["lr"]
            ),
            "seconds": time.perf_counter() - epoch_start,
            "peak_cuda_MiB": (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if device.type == "cuda"
                else 0.0
            ),
        }
        append_csv(args.out / "history.csv", row)
        print(json.dumps(row), flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs or stop_after is not None:
            save_checkpoint(
                args.out / f"{args.model}_epoch_{epoch:03d}.pth",
                restorer,
                optimizer,
                scheduler,
                epoch,
                args,
                row,
            )
        if stop_after is not None and global_step >= stop_after:
            (args.out / "SMOKE_ONLY").write_text(
                "This directory is a wiring test and must not be used as evidence.\n",
                encoding="utf-8",
            )
            break

    if stop_after is None:
        last_checkpoint = args.out / f"{args.model}_epoch_{args.epochs:03d}.pth"
        completion = {
            "status": "complete",
            "model": args.model,
            "epochs": args.epochs,
            "optimizer_steps": global_step,
            "last_checkpoint": str(last_checkpoint.resolve()),
            "last_checkpoint_sha256": sha256(last_checkpoint),
            "test_split_used": False,
        }
        (args.out / "TRAINING_COMPLETE.json").write_text(
            json.dumps(completion, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
