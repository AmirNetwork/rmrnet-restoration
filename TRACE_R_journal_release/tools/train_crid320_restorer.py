#!/usr/bin/env python3
"""Fine-tune TRACE-R and restoration baselines from labelled field images.

CRID has native camera images and box labels but no paired latent sharp target.
Consequently this trainer does not pretend that a second clean image exists.
It minimizes a frozen-detector supervised objective on labelled crops while
regularizing the returned image toward the native observation:

    L_field = lambda_det L_det(I_r, Y)
             + lambda_id rho(I_r - I_d)
             + lambda_edge ||grad I_r - grad I_d||_1
             + lambda_tv ||grad(I_r - I_d)||_1.

Candidate model families receive the same crop stream, detector,
optimizer-step budget, and objective. Architecture-specific trainable scopes
are recorded in ``adaptation_audit.json``. The final CRID TRACE-R policy is a
separate validation-only composition: it inherits the selected DFPIR candidate,
retains sensor feature adapters at identity, and uses the measured 82-value
camera/IMU/vehicle packet to gate a bounded output correction. See
``build_crid320_staged_trace_init.py`` and
``freeze_crid320_trace_field_policy.py``. No test annotation is used here.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcadnet.practical_metadata import sensor_packet_from_mapping
from rcadnet.task_losses import FrozenDetectorSupervisedLoss
from models.rmrp_prompted_dfpir import RMRPPromptedDFPIR
from train_matched_restorer import TrainableRestorer


DEFAULT_EXPORT = (
    ROOT
    / "datasets"
    / "crid320_annotation_20260829"
    / "exports"
    / "latest"
)
DEFAULT_METADATA = (
    ROOT
    / "experiments"
    / "geotagged_cam1_complete_sbg_ins_metadata_v4_20260811"
    / "metadata"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=("rmrp", "demoe", "nafnet", "dfpir", "instructir"),
    )
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--lm-head-weights", type=Path)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--cached-candidate-root",
        type=Path,
        help=(
            "Optional native-resolution frozen DFPIR candidates. Valid only "
            "with --trace-refiner-only; avoids recomputing the frozen backbone."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=360)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument(
        "--full-frame-long-side",
        type=int,
        default=0,
        help=(
            "If positive, replace random square crops with aspect-preserving "
            "full frames whose long side has this many pixels. This keeps long "
            "cracks and road context intact for full-resolution field calibration."
        ),
    )
    parser.add_argument("--defect-crop-probability", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=2.0e-6)
    parser.add_argument("--trace-adapter-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--trace-adapters-only",
        action="store_true",
        help=(
            "Freeze the restoration backbone and update only TRACE-R's sensor, "
            "state-reconciliation, FiLM, and native-gate modules."
        ),
    )
    parser.add_argument(
        "--trace-gate-only",
        action="store_true",
        help=(
            "Freeze the restoration candidate and train only TRACE-R's image/sensor "
            "state and automatic native gate."
        ),
    )
    parser.add_argument(
        "--trace-refiner-only",
        action="store_true",
        help=(
            "Freeze the matched restoration candidate and native strength; "
            "update only the image/sensor state and bounded post-prior refiner."
        ),
    )
    parser.add_argument("--gate-utility-weight", type=float, default=0.0)
    parser.add_argument("--gate-utility-interval", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--detector-weight", type=float, default=1.0)
    parser.add_argument("--identity-weight", type=float, default=0.50)
    parser.add_argument("--edge-weight", type=float, default=0.10)
    parser.add_argument("--residual-tv-weight", type=float, default=0.03)
    parser.add_argument("--detector-input-size", type=int, default=640)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA FP16 autocast with gradient scaling for memory-heavy baselines.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--smoke-steps", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_to_label(path: Path) -> Path:
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an images directory: {path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_labels(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return boxes
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Malformed YOLO label at {path}:{line_number}")
        cls = int(float(values[0]))
        cx, cy, bw, bh = (float(value) for value in values[1:])
        x1 = (cx - 0.5 * bw) * width
        y1 = (cy - 0.5 * bh) * height
        x2 = (cx + 0.5 * bw) * width
        y2 = (cy + 0.5 * bh) * height
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


class CRIDNativeCropDataset(Dataset):
    """Deterministic epoch-aware CRID training crop stream."""

    def __init__(
        self,
        manifest: Path,
        metadata_root: Path,
        *,
        patch_size: int,
        samples_per_epoch: int,
        defect_crop_probability: float,
        seed: int,
        candidate_root: Path | None = None,
        full_frame_long_side: int = 0,
    ) -> None:
        self.paths = [
            Path(line.strip()).resolve()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(self.paths) != 180:
            raise RuntimeError("Field fine-tuning requires the locked 180-frame train block")
        self.metadata_root = metadata_root
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.defect_crop_probability = float(defect_crop_probability)
        self.seed = int(seed)
        self.candidate_root = candidate_root
        self.full_frame_long_side = int(full_frame_long_side)
        self.epoch = 0
        self._sizes: dict[Path, tuple[int, int]] = {}
        self._labels: dict[Path, list[tuple[int, float, float, float, float]]] = {}
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                width, height = image.size
            self._sizes[path] = (width, height)
            self._labels[path] = read_yolo_labels(image_to_label(path), width, height)
            metadata_path = metadata_root / f"{path.stem}.json"
            if not metadata_path.exists():
                raise FileNotFoundError(metadata_path)
            if candidate_root is not None:
                candidate_path = candidate_root / f"{path.stem}.jpg"
                if not candidate_path.exists():
                    raise FileNotFoundError(candidate_path)
                with Image.open(candidate_path) as candidate_image:
                    if candidate_image.size != (width, height):
                        raise RuntimeError(
                            f"Cached candidate geometry mismatch for {path.name}"
                        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _crop_origin(
        self,
        path: Path,
        rng: random.Random,
    ) -> tuple[int, int]:
        width, height = self._sizes[path]
        patch = self.patch_size
        labels = self._labels[path]
        if labels and rng.random() < self.defect_crop_probability:
            _cls, x1, y1, x2, y2 = rng.choice(labels)
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            jitter_x = rng.uniform(-0.20, 0.20) * patch
            jitter_y = rng.uniform(-0.20, 0.20) * patch
            left = int(round(center_x + jitter_x - 0.5 * patch))
            top = int(round(center_y + jitter_y - 0.5 * patch))
        else:
            left = rng.randint(0, max(width - patch, 0))
            top = rng.randint(0, max(height - patch, 0))
        return min(max(left, 0), max(width - patch, 0)), min(
            max(top, 0), max(height - patch, 0)
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch + index)
        path = self.paths[(index * 104729 + self.epoch * 15485863) % len(self.paths)]
        width, height = self._sizes[path]
        if self.full_frame_long_side > 0:
            return self._full_frame_item(path, rng, width, height)
        left, top = self._crop_origin(path, rng)
        patch = self.patch_size
        with Image.open(path) as source:
            image = source.convert("RGB")
            crop = TF.crop(
                image,
                top,
                left,
                min(patch, height - top),
                min(patch, width - left),
            )
        candidate_crop: Image.Image | None = None
        if self.candidate_root is not None:
            with Image.open(self.candidate_root / f"{path.stem}.jpg") as source:
                candidate_crop = TF.crop(
                    source.convert("RGB"),
                    top,
                    left,
                    min(patch, height - top),
                    min(patch, width - left),
                )
        if crop.size != (patch, patch):
            padded = Image.new("RGB", (patch, patch), (114, 114, 114))
            padded.paste(crop, (0, 0))
            crop = padded
            if candidate_crop is not None:
                candidate_padded = Image.new("RGB", (patch, patch), (114, 114, 114))
                candidate_padded.paste(candidate_crop, (0, 0))
                candidate_crop = candidate_padded

        transformed: list[tuple[int, float, float, float, float]] = []
        for cls, x1, y1, x2, y2 in self._labels[path]:
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            if not (left <= center_x < left + patch and top <= center_y < top + patch):
                continue
            xx1 = min(max(x1 - left, 0.0), float(patch))
            yy1 = min(max(y1 - top, 0.0), float(patch))
            xx2 = min(max(x2 - left, 0.0), float(patch))
            yy2 = min(max(y2 - top, 0.0), float(patch))
            if xx2 - xx1 >= 2.0 and yy2 - yy1 >= 2.0:
                transformed.append((cls, xx1, yy1, xx2, yy2))

        flip = rng.random() < 0.5
        if flip:
            crop = TF.hflip(crop)
            if candidate_crop is not None:
                candidate_crop = TF.hflip(candidate_crop)
            transformed = [
                (cls, patch - x2, y1, patch - x1, y2)
                for cls, x1, y1, x2, y2 in transformed
            ]
        tensor = TF.to_tensor(crop)
        classes = torch.tensor([item[0] for item in transformed], dtype=torch.float32)
        boxes = []
        for _cls, x1, y1, x2, y2 in transformed:
            boxes.append(
                (
                    0.5 * (x1 + x2) / patch,
                    0.5 * (y1 + y2) / patch,
                    (x2 - x1) / patch,
                    (y2 - y1) / patch,
                )
            )
        bboxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        metadata = json.loads(
            (self.metadata_root / f"{path.stem}.json").read_text(encoding="utf-8")
        )
        packet = sensor_packet_from_mapping(metadata, device="cpu")
        result = {
            "image": tensor,
            "metadata": packet,
            "classes": classes,
            "bboxes": bboxes,
            "name": path.name,
        }
        if candidate_crop is not None:
            result["candidate"] = TF.to_tensor(candidate_crop)
        return result

    def _full_frame_item(
        self,
        path: Path,
        rng: random.Random,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        """Return one aspect-preserving frame with every native annotation.

        CRID images are sharp native captures, so this calibration stream must
        preserve global road context rather than manufacture paired corruption.
        The image and frozen restoration candidate undergo identical geometry.
        """

        scale = self.full_frame_long_side / float(max(width, height))
        output_width = max(1, int(round(width * scale)))
        output_height = max(1, int(round(height * scale)))
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (output_width, output_height), Image.Resampling.LANCZOS
            )
        candidate: Image.Image | None = None
        if self.candidate_root is not None:
            with Image.open(self.candidate_root / f"{path.stem}.jpg") as source:
                candidate = source.convert("RGB").resize(
                    (output_width, output_height), Image.Resampling.LANCZOS
                )

        flip = rng.random() < 0.5
        if flip:
            image = TF.hflip(image)
            if candidate is not None:
                candidate = TF.hflip(candidate)

        classes = torch.tensor(
            [item[0] for item in self._labels[path]], dtype=torch.float32
        )
        boxes = []
        for _cls, x1, y1, x2, y2 in self._labels[path]:
            cx = 0.5 * (x1 + x2) / width
            cy = 0.5 * (y1 + y2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            boxes.append((1.0 - cx if flip else cx, cy, bw, bh))
        bboxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        metadata = json.loads(
            (self.metadata_root / f"{path.stem}.json").read_text(encoding="utf-8")
        )
        result = {
            "image": TF.to_tensor(image),
            "metadata": sensor_packet_from_mapping(metadata, device="cpu"),
            "classes": classes,
            "bboxes": bboxes,
            "name": path.name,
        }
        if candidate is not None:
            result["candidate"] = TF.to_tensor(candidate)
        return result


def collate_native(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_targets = max((int(row["classes"].numel()) for row in batch), default=0)
    max_targets = max(max_targets, 1)
    classes = torch.zeros((len(batch), max_targets), dtype=torch.float32)
    bboxes = torch.zeros((len(batch), max_targets, 4), dtype=torch.float32)
    valid = torch.zeros((len(batch), max_targets), dtype=torch.bool)
    for index, row in enumerate(batch):
        count = int(row["classes"].numel())
        if count:
            classes[index, :count] = row["classes"]
            bboxes[index, :count] = row["bboxes"]
            valid[index, :count] = True
    result = {
        "image": torch.stack([row["image"] for row in batch]),
        "metadata": torch.stack([row["metadata"] for row in batch]),
        "classes": classes,
        "bboxes": bboxes,
        "valid": valid,
        "names": [row["name"] for row in batch],
    }
    if all("candidate" in row for row in batch):
        result["candidate"] = torch.stack([row["candidate"] for row in batch])
    return result


def image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return image[..., 1:, :] - image[..., :-1, :], image[..., :, 1:] - image[..., :, :-1]


def charbonnier(value: torch.Tensor, eps: float = 1.0e-3) -> torch.Tensor:
    return torch.sqrt(value.square() + eps * eps).mean()


def field_scenarios(model_name: str, batch_size: int) -> list[str]:
    """Return the benchmark's predeclared field-survey control input.

    DeMoE is evaluated through its released automatic router. DFPIR and
    InstructIR require a text/task condition, for which ``native`` denotes the
    same camera-motion preservation instruction used by the existing field
    protocol. TRACE-R and NAFNet ignore this string.
    """

    scenario = "auto" if model_name == "demoe" else "native"
    return [scenario] * int(batch_size)


def configure_parameters(
    restorer: TrainableRestorer,
    model_name: str,
    base_lr: float,
    trace_adapter_lr: float,
    trace_adapters_only: bool = False,
    trace_gate_only: bool = False,
    trace_refiner_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if model_name not in {"rmrp", "dfpir", "instructir"}:
        parameters = [parameter for parameter in restorer.parameters() if parameter.requires_grad]
        return [{"params": parameters, "lr": base_lr}], {
            "scope": "full restoration model",
            "trainable": sum(parameter.numel() for parameter in parameters),
        }

    for parameter in restorer.parameters():
        parameter.requires_grad_(False)

    if model_name == "dfpir":
        # DFPIR's 31 M-parameter transformer requires roughly 90 minutes for a
        # single 360-crop full-backbone epoch on the audited 6 GB GPU. Field
        # adaptation therefore calibrates its deployment-facing decoder while
        # preserving the released encoder and CLIP-conditioned feature prior.
        # This is an architecture-specific parameter-efficient update; the
        # data stream, detector objective, and 7,200 optimizer steps are not
        # reduced.
        named_modules = (
            ("decoder_level1", restorer.model.decoder_level1),
            ("refinement", restorer.model.refinement),
            ("output", restorer.model.output),
        )
        parameters: list[nn.Parameter] = []
        for _name, module in named_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                parameters.append(parameter)
        return [{"params": parameters, "lr": base_lr}], {
            "scope": "decoder_level1 + refinement + output",
            "trainable": sum(parameter.numel() for parameter in parameters),
            "frozen": sum(
                parameter.numel()
                for parameter in restorer.parameters()
                if not parameter.requires_grad
            ),
        }

    if model_name == "instructir":
        # InstructIR is calibrated through its published instruction-
        # conditioning blocks and output projection. The image prior remains
        # fixed, while the native-road instruction response is trainable.
        named_modules = (
            ("enc_cond", restorer.model.enc_cond),
            ("dec_cond", restorer.model.dec_cond),
            ("ending", restorer.model.ending),
        )
        parameters = []
        for _name, module in named_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                parameters.append(parameter)
        return [{"params": parameters, "lr": base_lr}], {
            "scope": "enc_cond + dec_cond + ending",
            "trainable": sum(parameter.numel() for parameter in parameters),
            "frozen": sum(
                parameter.numel()
                for parameter in restorer.parameters()
                if not parameter.requires_grad
            ),
        }

    if isinstance(restorer.model, RMRPPromptedDFPIR):
        # TRACE-R/DFPIR receives exactly the decoder scope used by the matched
        # DFPIR control, plus the observable-sensor state, zero-initialized
        # continuous FiLM layers, and automatic native pass-through gate.
        base_modules = (
            restorer.model.backbone.decoder_level1,
            restorer.model.backbone.refinement,
            restorer.model.backbone.output,
        )
        # The CRID detail mode deliberately calibrates only the low-dimensional
        # gain head. Its physical sensor state remains fixed, preventing a
        # small native field set from redefining telemetry semantics merely to
        # fit detector labels.
        detail_refiner_only = bool(
            trace_refiner_only and restorer.model.refiner_mode == "detail"
        )
        trace_modules: list[nn.Module] = []
        if not detail_refiner_only:
            trace_modules.extend(
                [restorer.model.sensor_encoder, restorer.model.image_state]
            )
        if not trace_gate_only and not trace_refiner_only:
            trace_modules.append(restorer.model.posterior_refine)
        if restorer.model.refiner is not None:
            trace_modules.append(restorer.model.refiner)
        if restorer.model.native_gate_head is not None and not trace_refiner_only:
            trace_modules.append(restorer.model.native_gate_head)
        if not trace_gate_only and not trace_refiner_only:
            trace_modules.extend(
                module
                for name, module in restorer.model.backbone.named_modules()
                if name.endswith("condition_affine")
            )
        base_parameters: list[nn.Parameter] = []
        trace_parameters: list[nn.Parameter] = []
        if not trace_adapters_only and not trace_gate_only and not trace_refiner_only:
            for module in base_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
                    base_parameters.append(parameter)
        for module in trace_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                trace_parameters.append(parameter)
        parameter_groups = [{"params": trace_parameters, "lr": trace_adapter_lr}]
        if base_parameters:
            parameter_groups.insert(0, {"params": base_parameters, "lr": base_lr})
        return parameter_groups, {
            "scope": (
                "image/sensor state + native pass-through gate"
                if trace_gate_only
                else (
                    "image/sensor state + bounded post-prior refiner"
                    if trace_refiner_only and not detail_refiner_only
                    else (
                        "fixed physical state + sensor-conditioned detail gain"
                        if detail_refiner_only
                        else (
                            "sensor state + continuous FiLM + native pass-through gate"
                            if trace_adapters_only
                            else "matched DFPIR decoder + sensor state + continuous FiLM + "
                            "native pass-through gate"
                        )
                    )
                )
            ),
            "staged_trace_adapters_only": bool(trace_adapters_only),
            "staged_trace_gate_only": bool(trace_gate_only),
            "staged_trace_refiner_only": bool(trace_refiner_only),
            "base_trainable": sum(parameter.numel() for parameter in base_parameters),
            "trace_trainable": sum(parameter.numel() for parameter in trace_parameters),
            "trainable": sum(
                parameter.numel() for parameter in (*base_parameters, *trace_parameters)
            ),
            "frozen": sum(
                parameter.numel()
                for parameter in restorer.parameters()
                if not parameter.requires_grad
            ),
        }

    trace_modules = (
        restorer.model.sensor_encoder,
        restorer.model.image_state,
        restorer.model.posterior_refine,
        restorer.model.route_residual,
        restorer.model.feature_adapters,
    )
    adapter_parameters: list[nn.Parameter] = []
    for module in trace_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            adapter_parameters.append(parameter)
    # The detector-facing field update is intentionally confined to TRACE-R's
    # sensor controller and low-rank adapters. The shared DeMoE image prior is
    # preserved, which prevents a 180-frame field set from erasing controlled
    # restoration knowledge.
    return [{"params": adapter_parameters, "lr": trace_adapter_lr}], {
        "scope": "sensor controller + image state + posterior + low-rank adapters",
        "trainable": sum(parameter.numel() for parameter in adapter_parameters),
        "frozen": sum(
            parameter.numel()
            for parameter in restorer.parameters()
            if not parameter.requires_grad
        ),
    }


def save_checkpoint(
    path: Path,
    restorer: TrainableRestorer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    stage1_epochs = int(restorer.arch.get("stage1_epochs", 0))
    torch.save(
        {
            "model": restorer.state_for_inference(),
            "arch": restorer.arch,
            "epoch": int(epoch),
            "stage2_epoch": int(epoch) if stage1_epochs else None,
            "total_matched_epoch": int(stage1_epochs + epoch),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "metrics": metrics,
            "method_name": "TRACE-R" if args.model == "rmrp" else args.model,
            "field_adaptation": True,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    args.init_weights = args.init_weights.resolve()
    args.detector = args.detector.resolve()
    args.export_root = args.export_root.resolve()
    args.metadata_root = args.metadata_root.resolve()
    args.cached_candidate_root = (
        args.cached_candidate_root.resolve()
        if args.cached_candidate_root is not None
        else None
    )
    args.out = args.out.resolve()
    args.lm_head_weights = args.lm_head_weights.resolve() if args.lm_head_weights else None
    for path in (args.init_weights, args.detector, args.export_root, args.metadata_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.model == "instructir" and args.lm_head_weights is None:
        raise ValueError("InstructIR requires --lm-head-weights")
    if args.trace_adapters_only and args.model != "rmrp":
        raise ValueError("--trace-adapters-only is valid only for --model rmrp")
    if args.trace_gate_only and args.model != "rmrp":
        raise ValueError("--trace-gate-only is valid only for --model rmrp")
    if args.trace_refiner_only and args.model != "rmrp":
        raise ValueError("--trace-refiner-only is valid only for --model rmrp")
    trace_scopes = sum(
        int(value)
        for value in (
            args.trace_gate_only,
            args.trace_adapters_only,
            args.trace_refiner_only,
        )
    )
    if trace_scopes > 1:
        raise ValueError(
            "Choose only one TRACE-specific staged adaptation scope"
        )
    if args.gate_utility_weight < 0.0:
        raise ValueError("--gate-utility-weight must be non-negative")
    if args.gate_utility_interval <= 0:
        raise ValueError("--gate-utility-interval must be positive")
    if args.gate_utility_weight and not args.trace_gate_only:
        raise ValueError("Gate utility supervision requires --trace-gate-only")
    if args.cached_candidate_root is not None:
        if not args.trace_refiner_only:
            raise ValueError(
                "--cached-candidate-root requires --trace-refiner-only"
            )
        if not args.cached_candidate_root.exists():
            raise FileNotFoundError(args.cached_candidate_root)
    if args.patch_size % 16:
        raise ValueError("--patch-size must be divisible by 16")
    if args.full_frame_long_side < 0:
        raise ValueError("--full-frame-long-side must be non-negative")
    if args.full_frame_long_side and args.samples_per_epoch != 180:
        raise ValueError(
            "Full-frame calibration requires exactly one pass over the locked "
            "180-frame training block per epoch"
        )

    manifest = args.export_root / "train.txt"
    dataset = CRIDNativeCropDataset(
        manifest,
        args.metadata_root,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        defect_crop_probability=args.defect_crop_probability,
        seed=args.seed,
        candidate_root=args.cached_candidate_root,
        full_frame_long_side=args.full_frame_long_side,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_native,
        pin_memory=True,
    )
    restorer = TrainableRestorer(
        args.model,
        [args.init_weights],
        device,
        args.lm_head_weights,
    ).to(device)
    parameter_groups, parameter_audit = configure_parameters(
        restorer,
        args.model,
        args.lr,
        args.trace_adapter_lr,
        trace_adapters_only=args.trace_adapters_only,
        trace_gate_only=args.trace_gate_only,
        trace_refiner_only=args.trace_refiner_only,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=min(args.lr, args.trace_adapter_lr) * 0.05,
    )
    detector = YOLO(str(args.detector)).model.to(device).eval()
    detector_loss = FrozenDetectorSupervisedLoss(
        detector,
        input_size=(args.detector_input_size, args.detector_input_size),
        letterbox=bool(args.full_frame_long_side),
        cqmix_prob=0.0,
        clean_hinge_weight=0.50,
        normalization_floor=1.0,
    ).to(device)
    use_amp = bool(args.amp and device.type == "cuda")
    uses_dfpir_backbone = (
        args.model == "dfpir"
        or restorer.arch.get("backbone") == "dfpir_sensor_prompt"
    )
    amp_dtype = torch.bfloat16 if uses_dfpir_backbone else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(use_amp and amp_dtype == torch.float16)
    )

    args.out.mkdir(parents=True, exist_ok=True)
    audit = {
        "protocol": "matched CRID-320 native-field restorer adaptation",
        "model": args.model,
        "test_split_opened": False,
        "train_manifest": str(manifest),
        "train_manifest_sha256": sha256(manifest),
        "train_images": 180,
        "metadata_root": str(args.metadata_root) if args.model == "rmrp" else None,
        "initial_checkpoint": str(args.init_weights),
        "initial_checkpoint_sha256": sha256(args.init_weights),
        "detector": str(args.detector),
        "detector_sha256": sha256(args.detector),
        "objective": {
            "frozen_detector_supervised": args.detector_weight,
            "native_identity_charbonnier": args.identity_weight,
            "native_edge": args.edge_weight,
            "restoration_residual_tv": args.residual_tv_weight,
            "training_only_gate_utility": args.gate_utility_weight,
            "gate_utility_interval": args.gate_utility_interval,
            "gate_utility_strengths": [0.0, 0.5, 1.0],
        },
        "baseline_conditioning": {
            "demoe": "released automatic router",
            "dfpir": "native camera-motion task code",
            "instructir": "native camera-motion instruction",
            "nafnet": "image only",
            "rmrp": "measured 82-value camera/IMU/vehicle packet",
        },
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "optimizer_steps": args.epochs * args.samples_per_epoch,
        "staged_training": bool(restorer.arch.get("staged_training", False)),
        "stage1_epochs": int(restorer.arch.get("stage1_epochs", 0)),
        "stage1_optimizer_steps": int(restorer.arch.get("stage1_epochs", 0))
        * args.samples_per_epoch,
        "total_matched_optimizer_steps": (
            args.epochs + int(restorer.arch.get("stage1_epochs", 0))
        )
        * args.samples_per_epoch,
        "patch_size": args.patch_size,
        "full_frame_long_side": args.full_frame_long_side,
        "detector_letterbox": bool(args.full_frame_long_side),
        "automatic_mixed_precision": use_amp,
        "automatic_mixed_precision_dtype": (
            str(amp_dtype).replace("torch.", "") if use_amp else None
        ),
        "defect_crop_probability": args.defect_crop_probability,
        "cached_frozen_candidate_root": (
            str(args.cached_candidate_root)
            if args.cached_candidate_root is not None
            else None
        ),
        "parameter_audit": parameter_audit,
        "seed": args.seed,
        "started_unix": time.time(),
    }
    (args.out / "adaptation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    start_epoch = 1
    if args.resume_from is not None:
        payload = torch.load(args.resume_from.resolve(), map_location=device, weights_only=False)
        restorer.model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1

    history: list[dict[str, float]] = []
    steps = 0
    for epoch in range(start_epoch, args.epochs + 1):
        dataset.set_epoch(epoch)
        restorer.train(True)
        sums = {
            "loss": 0.0,
            "detector": 0.0,
            "identity": 0.0,
            "edge": 0.0,
            "residual_tv": 0.0,
            "gate_utility": 0.0,
            "gate_target": 0.0,
            "native_gate": 0.0,
            "gate_supervised": 0.0,
            "gate_target_eta0": 0.0,
            "gate_target_eta05": 0.0,
            "gate_target_eta1": 0.0,
        }
        count = 0
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            classes = batch["classes"].to(device, non_blocking=True)
            bboxes = batch["bboxes"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Only the restoration forward uses FP16. The frozen detector loss
            # stays in FP32 because its box/classification heads can overflow
            # under half precision on sparse full-resolution field crops.
            if "candidate" in batch:
                candidate = batch["candidate"].to(device, non_blocking=True)
                model = restorer.model
                if not isinstance(model, RMRPPromptedDFPIR) or model.refiner is None:
                    raise RuntimeError(
                        "Cached candidates require prompted DFPIR with a refiner"
                    )
                # Deployment-aligned cached forward.  The frozen DFPIR output
                # was produced once at native resolution.  Only the observable
                # image/sensor state and bounded correction remain in the graph.
                with torch.amp.autocast("cuda", enabled=False):
                    (
                        code,
                        image_code,
                        sensor_code,
                        _sensor_physical,
                        support,
                        _sensor_direct,
                    ) = model._sensor_state(image.float(), metadata.float())
                    refiner_support = model.refiner_support_floor + (
                        1.0 - model.refiner_support_floor
                    ) * code[:, -1]
                    refined, refiner_gate, correction = model.refiner(
                        image.float(),
                        candidate.float(),
                        candidate.float(),
                        code,
                        refiner_support,
                    )
                    disagreement = (sensor_code - image_code).abs()
                    gate_features = torch.cat(
                        [image_code, sensor_code, disagreement, support], dim=1
                    )
                    native_gate = torch.sigmoid(
                        model.native_gate_head.float()(gate_features)
                    )
                    restored = torch.clamp(
                        image.float()
                        + native_gate[:, :, None, None]
                        * (refined - image.float()),
                        0.0,
                        1.0,
                    )
                state = {
                    "restored": restored,
                    "neural_restored": candidate,
                    "native_gate": native_gate[:, 0],
                    "post_prior_gate": refiner_gate,
                    "post_prior_correction": correction,
                    "posterior_degradation_code": code,
                }
            else:
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    restored, state = restorer(
                        image,
                        metadata,
                        field_scenarios(args.model, image.shape[0]),
                    )
            restored = restored.float()
            identity = charbonnier(restored - image)
            restored_dy, restored_dx = image_gradients(restored)
            image_dy, image_dx = image_gradients(image)
            edge = 0.5 * (
                F.l1_loss(restored_dy, image_dy) + F.l1_loss(restored_dx, image_dx)
            )
            residual_dy, residual_dx = image_gradients(restored - image)
            residual_tv = 0.5 * (residual_dy.abs().mean() + residual_dx.abs().mean())
            task = detector_loss(
                restored,
                image,
                classes=classes,
                bboxes=bboxes,
                valid=valid,
            )
            gate_utility = restored.new_zeros(())
            gate_target = restored.new_zeros(())
            native_gate = state.get("native_gate")
            supervise_gate = bool(
                args.gate_utility_weight
                and native_gate is not None
                and steps % args.gate_utility_interval == 0
            )
            if supervise_gate:
                # Training-only utility target:
                # eta* = argmin_{eta in {0,.5,1}} L_det(I + eta*(I_b-I), Y).
                # The candidate I_b is frozen in --trace-gate-only mode; only
                # the observable image/sensor controller learns eta*.
                candidate = state["neural_restored"].detach().float()
                strengths = restored.new_tensor((0.0, 0.5, 1.0))
                candidates = [
                    image.float() + strength * (candidate - image.float())
                    for strength in strengths
                ]
                candidate_losses = detector_loss.detached_candidate_losses(
                    candidates,
                    classes=classes,
                    bboxes=bboxes,
                    valid=valid,
                )
                gate_target = strengths[candidate_losses.argmin()]
                gate_utility = F.mse_loss(
                    native_gate.float().mean(), gate_target.float()
                )
            total = (
                args.detector_weight * task
                + args.identity_weight * identity
                + args.edge_weight * edge
                + args.residual_tv_weight * residual_tv
                + args.gate_utility_weight * gate_utility
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite field loss at epoch {epoch}, step {steps}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in parameter_groups for parameter in group["params"]],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            values = {
                "loss": total,
                "detector": task,
                "identity": identity,
                "edge": edge,
                "residual_tv": residual_tv,
                "gate_utility": gate_utility,
                "gate_target": gate_target,
                "native_gate": (
                    native_gate.float().mean()
                    if native_gate is not None
                    else restored.new_zeros(())
                ),
                "gate_supervised": restored.new_tensor(float(supervise_gate)),
                "gate_target_eta0": restored.new_tensor(
                    float(supervise_gate and float(gate_target) == 0.0)
                ),
                "gate_target_eta05": restored.new_tensor(
                    float(supervise_gate and float(gate_target) == 0.5)
                ),
                "gate_target_eta1": restored.new_tensor(
                    float(supervise_gate and float(gate_target) == 1.0)
                ),
            }
            for key, value in values.items():
                sums[key] += float(value.detach().cpu())
            count += 1
            steps += 1
            if args.smoke_steps and steps >= args.smoke_steps:
                break
        scheduler.step()
        row = {key: value / max(count, 1) for key, value in sums.items()}
        supervised = max(sums["gate_supervised"], 1.0)
        row["gate_utility_supervised"] = sums["gate_utility"] / supervised
        row["gate_target_supervised_mean"] = sums["gate_target"] / supervised
        row["gate_target_eta0_fraction"] = sums["gate_target_eta0"] / supervised
        row["gate_target_eta05_fraction"] = sums["gate_target_eta05"] / supervised
        row["gate_target_eta1_fraction"] = sums["gate_target_eta1"] / supervised
        row["epoch"] = float(epoch)
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.out / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if epoch % args.save_every == 0 or epoch == args.epochs or args.smoke_steps:
            save_checkpoint(
                args.out / f"{args.model}_field_epoch_{epoch:03d}.pth",
                restorer,
                optimizer,
                scheduler,
                epoch,
                args,
                row,
            )
        if args.smoke_steps and steps >= args.smoke_steps:
            break

    (args.out / "adaptation_complete.json").write_text(
        json.dumps(
            {
                **audit,
                "status": "complete" if not args.smoke_steps else "smoke_only",
                "completed_unix": time.time(),
                "steps_completed": steps,
                "test_split_opened": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
