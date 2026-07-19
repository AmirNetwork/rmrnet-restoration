# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import csv
import json
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

from rcadnet import RCADNet
from rcadnet.dataset import PairedRoadRestorationDataset, list_images
from rcadnet.losses import RCADLoss
from rcadnet.task_losses import (
    ActiveContourGeometryLoss,
    CompositeTaskLoss,
    DetectorInputAnchorLoss,
    FrozenDetectorFeatureExtractor,
    TaskDrivenPerceptualLoss,
    TaskLossWeights,
    road_evidence_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RMR/RCAD-Net on paired road restoration folders.")
    parser.add_argument("--data-root", action="append", required=True, help="Dataset root containing scenarios/<scenario>/input and /gt. Repeat to combine datasets.")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="Scenario to train on. Repeat for many.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", "--out-dir", dest="out", default="runs/rcadnet")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--code-source",
        choices=["scenario", "zero", "estimated", "fused", "metadata", "metadata_fused"],
        default="scenario",
        help="Conditioning source. metadata_fused combines metadata with the learned image estimator.",
    )
    parser.add_argument("--no-defect-attention", action="store_true", help="Ablation: disable defect attention.")
    parser.add_argument("--aux-code-weight", type=float, default=0.05)
    parser.add_argument("--init-weights", help="Optional RMR/RCAD checkpoint to fine-tune from.")
    parser.add_argument("--block-type", choices=["simple", "evidence"], default="simple")
    parser.add_argument("--attention-type", choices=["edge", "task", "none"], default="edge")
    parser.add_argument("--conditioning", choices=["film", "gated_basis"], default="gated_basis")
    parser.add_argument(
        "--basis-sparsity-weight",
        type=float,
        default=1.0,
        help="Multiplier for the sparse monotone degradation-basis gate when gated_basis conditioning is used.",
    )
    parser.add_argument("--detail-preserve", action="store_true", help="Enable structural evidence-preserving high-frequency skip.")
    parser.add_argument("--detail-gain", type=float, default=0.20, help="Maximum gain for the structural detail skip.")
    parser.add_argument("--new-head-lr-mult", type=float, default=10.0, help="LR multiplier for newly added train-time/detail heads.")
    parser.add_argument("--metadata-dropout", type=float, default=0.0)
    parser.add_argument("--metadata-noise", type=float, default=0.0)
    parser.add_argument(
        "--metadata-mode",
        choices=["full", "raw_telemetry", "raw_scalar"],
        default="full",
        help="Metadata fields used for conditioning.",
    )
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--freq-weight", type=float, default=0.05)
    parser.add_argument("--defect-weight", type=float, default=0.10)
    parser.add_argument("--visibility-weight", type=float, default=0.0)

    # New paper-facing composite objective.
    parser.add_argument(
        "--use-task-losses",
        action="store_true",
        help=(
            "Enable the detector/task regularizer pack. The final paper path uses "
            "TDP/CQMix, Jacobian, detector-anchor, and evidence non-regression; "
            "the active-contour loss stays disabled unless --lambda-active-contour > 0."
        ),
    )
    parser.add_argument("--lambda-tdp", type=float, default=0.02)
    parser.add_argument("--lambda-jacobian", type=float, default=0.001)
    parser.add_argument("--lambda-active-contour", type=float, default=0.0)
    parser.add_argument("--lambda-detector-input-anchor", type=float, default=0.0, help="v24: weak detector-feature anchor from restored image to degraded/input image.")
    parser.add_argument("--lambda-evidence-nonregression", type=float, default=0.0, help="v24: hinge penalty when restoration suppresses road evidence below clean/degraded evidence.")
    parser.add_argument("--lambda-detail-copy", type=float, default=0.0, help="Penalize structural detail-skip energy outside detector-visible road-evidence regions.")
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
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--debug-every", type=int, default=0, help="Print JSON debug stats every N training batches. 0 disables periodic batch debug.")
    parser.add_argument("--debug-first-batches", type=int, default=1, help="Print JSON debug stats for the first N batches of each epoch.")

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
    parser.add_argument("--noise-da-weight", type=float, default=0.0, help="Accepted for old commands; Noise-DA is not used by this v23 trainer.")
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
    if args.use_task_losses:
        if args.visibility_weight == 0.0:
            args.visibility_weight = 0.08
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
        active_contour_loss=active_contour,
        feature_extractor=extractor if args.lambda_jacobian > 0 else None,
        detector_anchor_loss=anchor if args.lambda_detector_input_anchor > 0 else None,
        weights=TaskLossWeights(
            tdp=args.lambda_tdp,
            jacobian=args.lambda_jacobian,
            active_contour=args.lambda_active_contour,
            detector_input_anchor=args.lambda_detector_input_anchor,
            evidence_nonregression=args.lambda_evidence_nonregression,
            detail_copy=args.lambda_detail_copy,
        ),
        jacobian_probes=args.jacobian_probes,
        evidence_lower_fraction=args.evidence_lower_fraction,
    ).to(device)
    return task_loss


def save_checkpoint(path: Path, model: RCADNet, args: argparse.Namespace, epoch: int, metrics: Optional[dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "arch": {
            "width": args.width,
            "code_dim": model.code_dim,
            "use_defect_attention": not args.no_defect_attention,
            "use_estimated_code": args.code_source in {"estimated", "fused", "metadata_fused"},
            "code_fusion": args.code_source,
            "block_type": args.block_type,
            "attention_type": args.attention_type,
            "conditioning": args.conditioning,
            "use_tdac_head": args.use_tdac_head or (args.use_task_losses and args.lambda_active_contour > 0),
            "detail_preserve": args.detail_preserve,
            "detail_gain": args.detail_gain,
        },
        "epoch": epoch,
        "metrics": metrics or {},
        "args": vars(args),
    }
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
    model_codes = metadata_codes if args.code_source in {"metadata", "metadata_fused"} else codes
    if training and args.code_source in {"metadata", "metadata_fused"}:
        if args.metadata_dropout > 0:
            keep = (torch.rand(model_codes.shape[0], 1, device=model_codes.device) >= args.metadata_dropout).to(model_codes.dtype)
            model_codes = model_codes * keep
        if args.metadata_noise > 0:
            model_codes = torch.clamp(model_codes + torch.randn_like(model_codes) * args.metadata_noise, 0.0, 1.0)
    return model_codes


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1).clamp_min(1e-10)
    return -10.0 * torch.log10(mse)


def model_forward(
    model: RCADNet,
    inputs: torch.Tensor,
    model_codes: torch.Tensor | None,
    args: argparse.Namespace,
    *,
    need_aux: bool,
) -> dict[str, torch.Tensor | None]:
    result = model(
        inputs,
        model_codes,
        return_aux=True if need_aux else False,
        return_dict=True if need_aux else False,
        gate_threshold=args.gate_threshold if args.gate_threshold >= 0 else None,
        gate_softness=args.gate_softness,
    )
    if isinstance(result, dict):
        return result
    return {"restored": result, "phi": None, "lambda1": None, "lambda2": None}


@torch.no_grad()
def validate(model: RCADNet, loader: DataLoader, criterion: RCADLoss, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
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
        result = model_forward(model, inputs, model_codes, args, need_aux=False)
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
        for aux_name in ("phi", "lambda1", "lambda2", "severity", "gate", "detail_gate", "detail_residual"):
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
    model: RCADNet,
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
    model_codes = prepare_codes(args, codes, metadata_codes, training=True)
    result = model_forward(
        model,
        inputs,
        model_codes,
        args,
        need_aux=task_loss is not None
        or args.use_tdac_head
        or (args.conditioning == "gated_basis" and args.basis_sparsity_weight > 0),
    )
    outputs = result["restored"]
    base = criterion(outputs, targets, inputs, model_codes)
    total = base
    logs: dict[str, Any] = {
        "input_shape": list(inputs.shape),
        "restored_shape": list(outputs.shape),
        "base_loss": float(base.detach().cpu()),
        "keys": sorted(k for k, v in result.items() if v is not None),
    }
    if task_loss is not None:
        task_value, task_logs = task_loss(result, targets, degraded=inputs, warmup_scale=1.0)
        total = total + task_value
        logs["task_loss"] = float(task_value.detach().cpu())
        logs.update(task_logs)
    basis_sparsity = result.get("basis_sparsity")
    if (
        args.conditioning == "gated_basis"
        and args.basis_sparsity_weight > 0
        and isinstance(basis_sparsity, torch.Tensor)
    ):
        sparse_loss = float(args.basis_sparsity_weight) * basis_sparsity
        total = total + sparse_loss
        logs["basis_sparsity_loss"] = float(sparse_loss.detach().cpu())
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
        "use train_rmrnet.py for paper-facing runs."
    )
    args = parse_args()
    set_reproducibility_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_roots = [Path(root) for root in args.data_root]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        print(json.dumps({"warning": "CUDA requested but unavailable; using CPU."}), flush=True)
    if args.noise_da_weight > 0:
        print(json.dumps({"warning": "Noise-DA flags are accepted but not active in v23 composite-loss trainer."}), flush=True)

    scenarios = args.scenarios or discover_scenarios(data_roots[0])
    datasets = [
        PairedRoadRestorationDataset(root, scenarios, patch_size=args.patch_size, train=True, metadata_mode=args.metadata_mode)
        for root in data_roots
    ]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    val_loader = None
    if args.val_data_root:
        val_roots = [Path(root) for root in args.val_data_root]
        val_scenarios = args.val_scenarios or scenarios
        val_sets = [
            PairedRoadRestorationDataset(root, val_scenarios, patch_size=args.patch_size, train=False, metadata_mode=args.metadata_mode)
            for root in val_roots
        ]
        val_dataset = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = RCADNet(
        width=args.width,
        use_defect_attention=not args.no_defect_attention,
        use_estimated_code=args.code_source in {"estimated", "fused", "metadata_fused"},
        code_fusion=args.code_source,
        block_type=args.block_type,
        attention_type=args.attention_type,
        conditioning=args.conditioning,
        use_tdac_head=args.use_tdac_head or (args.use_task_losses and args.lambda_active_contour > 0),
        detail_preserve=args.detail_preserve,
        detail_gain=args.detail_gain,
    ).to(device)
    if args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location=device)
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

    criterion = RCADLoss(
        edge_weight=args.edge_weight,
        freq_weight=args.freq_weight,
        defect_weight=args.defect_weight,
        visibility_weight=args.visibility_weight,
    )
    task_loss = build_task_loss(args, device)
    boosted_keywords = ("tdac_head", "detail_skip", "code_encoder", "code_fuser")
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

    audit_config = {
        "version": "v24_detector_safe_debug",
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
        "lambda_jacobian": args.lambda_jacobian,
        "lambda_active_contour": args.lambda_active_contour,
        "lambda_detector_input_anchor": args.lambda_detector_input_anchor,
        "lambda_evidence_nonregression": args.lambda_evidence_nonregression,
        "lambda_detail_copy": args.lambda_detail_copy,
        "tdp_defect_mask_weight": args.tdp_defect_mask_weight,
        "tdp_defect_mask_power": args.tdp_defect_mask_power,
        "basis_sparsity_weight": args.basis_sparsity_weight,
        "detail_preserve": args.detail_preserve,
        "detail_gain": args.detail_gain,
        "new_head_lr_mult": args.new_head_lr_mult,
        "boosted_parameter_names": boosted_names,
        "selection_policy": "Training saves PSNR/loss checkpoints. Detector-mAP promotion must be performed externally on validation restored YOLO splits.",
        "args": vars(args),
    }
    (out_dir / "audit_config.json").write_text(json.dumps(audit_config, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit_config.items() if k != "args"}), flush=True)

    if args.smoke_test:
        smoke_test(model, loader, criterion, task_loss, args, device, out_dir)
        return

    history: list[dict[str, Any]] = []
    best_val_psnr = float("-inf")
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        if is_detector_adaptation_epoch(args, epoch):
            for param in model.parameters():
                param.requires_grad_(False)
            row = run_detector_adaptation(args, out_dir, epoch)
            history.append(row)
            print(json.dumps(row), flush=True)
            for param in model.parameters():
                param.requires_grad_(True)
            continue

        model.train()
        running = 0.0
        component_sums: dict[str, float] = {}
        scale = task_warmup_scale(args, epoch)
        use_amp = resolve_amp(args, device)
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["gt"].to(device, non_blocking=True)
            codes = batch["code"].to(device, non_blocking=True)
            metadata_codes = batch["metadata_code"].to(device, non_blocking=True)
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
                        or (args.conditioning == "gated_basis" and args.basis_sparsity_weight > 0)
                    ),
                )
                outputs = result["restored"]
                base_loss = criterion(outputs, targets, inputs, model_codes)
                loss = base_loss
                loss_terms["restoration"] = base_loss
                if args.code_source in {"estimated", "fused", "metadata_fused"} and args.aux_code_weight > 0:
                    target_codes = metadata_codes if args.code_source == "metadata_fused" else codes
                    aux_code = args.aux_code_weight * F.smooth_l1_loss(model.estimate_code(inputs), target_codes)
                    loss = loss + aux_code
                    loss_terms["aux_code"] = aux_code
                basis_sparsity = result.get("basis_sparsity")
                if (
                    args.conditioning == "gated_basis"
                    and args.basis_sparsity_weight > 0
                    and isinstance(basis_sparsity, torch.Tensor)
                ):
                    sparse_loss = float(args.basis_sparsity_weight) * basis_sparsity
                    loss = loss + sparse_loss
                    loss_terms["basis_sparsity"] = sparse_loss
                task_logs: dict[str, float] = {}
                if task_loss is not None:
                    task_value, task_logs = task_loss(result, targets, degraded=inputs, warmup_scale=scale)
                    loss = loss + task_value
                    loss_terms["task_total"] = task_value
                    for name, value in task_logs.items():
                        component_sums[name] = component_sums.get(name, 0.0) + float(value)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            grad_norm_value = float(grad_norm_tensor.detach().cpu()) if isinstance(grad_norm_tensor, torch.Tensor) else float(grad_norm_tensor)
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
            for name, value in loss_terms.items():
                component_sums[f"loss_{name}"] = component_sums.get(f"loss_{name}", 0.0) + float(value.detach().cpu())

        row: dict[str, Any] = {
            "epoch": epoch,
            "phase": "rmr_update",
            "loss": running / max(len(loader), 1),
            "task_warmup_scale": scale,
            "effective_tdp_weight": (args.lambda_tdp * scale) if task_loss is not None else 0.0,
            "effective_jacobian_weight": (args.lambda_jacobian * scale) if task_loss is not None else 0.0,
            "effective_active_contour_weight": (args.lambda_active_contour * scale) if task_loss is not None else 0.0,
            "effective_detector_input_anchor_weight": (args.lambda_detector_input_anchor * scale) if task_loss is not None else 0.0,
            "effective_evidence_nonregression_weight": (args.lambda_evidence_nonregression * scale) if task_loss is not None else 0.0,
            "effective_detail_copy_weight": (args.lambda_detail_copy * scale) if task_loss is not None else 0.0,
            "selection_note": "Detector mAP checkpoint selection is performed externally on validation splits only; this trainer saves explicit PSNR/loss checkpoints for audit.",
        }
        for name, value in sorted(component_sums.items()):
            if name.startswith("_"):
                continue
            row[name] = value / max(len(loader), 1)
        if val_loader is not None and epoch % max(args.val_every, 1) == 0:
            row.update(validate(model, val_loader, criterion, args, device))
        history.append(row)
        print(json.dumps(row), flush=True)

        save_checkpoint(out_dir / "rcadnet_last.pth", model, args, epoch, row)
        if args.save_every_epoch:
            save_checkpoint(out_dir / f"rcadnet_epoch_{epoch:03d}.pth", model, args, epoch, row)
        if "val_psnr" in row and row["val_psnr"] > best_val_psnr:
            best_val_psnr = float(row["val_psnr"])
            save_checkpoint(out_dir / "rcadnet_best_psnr.pth", model, args, epoch, row)
            save_checkpoint(out_dir / "rcadnet_best.pth", model, args, epoch, row)  # backward-compatible alias
            (out_dir / "best_by_val_psnr.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        if "val_loss" in row and row["val_loss"] < best_val_loss:
            best_val_loss = float(row["val_loss"])
            save_checkpoint(out_dir / "rcadnet_best_loss.pth", model, args, epoch, row)
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
