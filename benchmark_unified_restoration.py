from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms import functional as TF

from baselines.demoe_adapter import DeMoEAdapter
from baselines.dfpir_adapter import DFPIRAdapter
from baselines.nafnet_road import NAFNetRoad
from rcadnet import RCADNet, code_from_metadata, code_from_scenario
from rcadnet.dataset import IMAGE_EXTS, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified restoration benchmark for RCAD-Net and DFPIR-style baselines.")
    parser.add_argument("--data-root", required=True, help="Root containing scenarios/<scenario>/input and /gt.")
    parser.add_argument("--scenario", action="append", dest="scenarios", help="Scenario name. Repeat or omit for all.")
    parser.add_argument("--model", action="append", choices=["rcadnet", "dfpir", "nafnet", "demoe"], required=True)
    parser.add_argument("--rcadnet-weights")
    parser.add_argument("--rcadnet-code-source", choices=["scenario", "metadata", "blind"], default="scenario")
    parser.add_argument("--rcadnet-gate-threshold", type=float, default=-1.0, help="If >=0, enable RMR severity-gated pass-through.")
    parser.add_argument("--rcadnet-gate-softness", type=float, default=0.03, help="Set to 0 for hard clean-frame bypass.")
    parser.add_argument("--nafnet-weights")
    parser.add_argument("--dfpir-weights")
    parser.add_argument("--dfpir-smoke", action="store_true", help="Use tiny random DFPIR for pipeline testing only.")
    parser.add_argument("--dfpir-clip", action="store_true", help="Use CLIP text prompts for DFPIR conditioning.")
    parser.add_argument("--demoe-weights")
    parser.add_argument("--demoe-task", choices=["auto", "scenario", "defocus", "global_motion", "local_motion", "synth_global_motion", "low_light"], default="auto")
    parser.add_argument("--demoe-smoke", action="store_true", help="Use random DeMoE weights for pipeline tests only.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=0, help="Resize long side for fast tests. 0 keeps original size.")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed warmup passes per model/scenario.")
    parser.add_argument("--out", default="runs/unified_benchmark")
    return parser.parse_args()


def discover_scenarios(data_root: Path) -> list[str]:
    return sorted(p.name for p in (data_root / "scenarios").iterdir() if (p / "input").exists() and (p / "gt").exists())


def load_image(path: Path, max_side: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_side and max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
        return TF.to_tensor(image).unsqueeze(0)


def pad_to_multiple(image: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image, (height, width)


def unpad(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    return image[..., :height, :width]


def tensor_to_numpy(image: torch.Tensor):
    return image.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()


def psnr_ssim(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred_np = tensor_to_numpy(pred).clip(0, 1)
    target_np = tensor_to_numpy(target).clip(0, 1)
    psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
    ssim = structural_similarity(target_np, pred_np, channel_axis=2, data_range=1.0)
    return float(psnr), float(ssim)


def load_rcadnet(weights: str, device: torch.device) -> RCADNet:
    checkpoint = torch.load(weights, map_location=device)
    arch = checkpoint.get("arch", {})
    model = RCADNet(
        width=arch.get("width", 32),
        code_dim=arch.get("code_dim", 8),
        use_defect_attention=arch.get("use_defect_attention", True),
        use_estimated_code=arch.get("use_estimated_code", False),
        code_fusion=arch.get("code_fusion", "scenario"),
        block_type=arch.get("block_type", "simple"),
        attention_type=arch.get("attention_type", "edge"),
        conditioning=arch.get("conditioning", "film"),
        use_tdac_head=arch.get("use_tdac_head", False),
        detail_preserve=arch.get("detail_preserve", False),
        detail_gain=arch.get("detail_gain", 0.20),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def load_nafnet(weights: str, device: torch.device) -> NAFNetRoad:
    checkpoint = torch.load(weights, map_location=device)
    arch = checkpoint.get("arch", {})
    model = NAFNetRoad(width=arch.get("width", 32)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


@torch.inference_mode()
def run_rcadnet(
    model: RCADNet,
    image: torch.Tensor,
    scenario: str,
    device: torch.device,
    code_source: str,
    input_path: Path | None = None,
    gate_threshold: float | None = None,
    gate_softness: float = 0.03,
) -> torch.Tensor:
    if code_source == "blind":
        code = None
    elif code_source == "metadata" and input_path is not None:
        metadata_path = input_path.parent.parent / "metadata" / f"{input_path.stem}.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            code = code_from_metadata(metadata, device=device)
        else:
            code = code_from_scenario(scenario, device=device)
    else:
        code = code_from_scenario(scenario, device=device)
    return model(image.to(device), code, gate_threshold=gate_threshold, gate_softness=gate_softness)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    scenarios = args.scenarios or discover_scenarios(data_root)

    models = {}
    if "rcadnet" in args.model:
        if not args.rcadnet_weights:
            raise ValueError("--rcadnet-weights is required for model rcadnet")
        models["RMR-Net"] = {
            "runner": load_rcadnet(args.rcadnet_weights, device),
            "backend": "GPU-confirmed" if device.type == "cuda" else "CPU-forced",
            "kind": "rcadnet",
        }
    if "nafnet" in args.model:
        if not args.nafnet_weights:
            raise ValueError("--nafnet-weights is required for model nafnet")
        models["NAFNet-road"] = {
            "runner": load_nafnet(args.nafnet_weights, device),
            "backend": "GPU-confirmed" if device.type == "cuda" else "CPU-forced",
            "kind": "nafnet",
        }
    if "dfpir" in args.model:
        if not args.dfpir_weights and not args.dfpir_smoke:
            raise ValueError("--dfpir-weights is required for real DFPIR; use --dfpir-smoke for pipeline tests")
        models["DFPIR-CVPR2025"] = {
            "runner": DFPIRAdapter(args.dfpir_weights, device=str(device), smoke=args.dfpir_smoke, use_clip=args.dfpir_clip),
            "backend": "GPU-confirmed" if device.type == "cuda" else "CPU-forced",
            "kind": "dfpir",
        }
    if "demoe" in args.model:
        if not args.demoe_weights and not args.demoe_smoke:
            raise ValueError("--demoe-weights is required for real DeMoE; use --demoe-smoke only for pipeline tests")
        runner = DeMoEAdapter(
            args.demoe_weights,
            device=device,
            task=args.demoe_task,
            smoke=args.demoe_smoke,
            strict=not args.demoe_smoke,
        )
        models["DeMoE-CVPRW2025"] = {
            "runner": runner,
            "backend": runner.backend_status,
            "kind": "demoe",
        }

    rows = []
    for scenario in scenarios:
        input_dir = data_root / "scenarios" / scenario / "input"
        gt_dir = data_root / "scenarios" / scenario / "gt"
        image_paths = list_images(input_dir)
        if args.limit:
            image_paths = image_paths[: args.limit]
        for model_name, model_info in models.items():
            kind = model_info["kind"]
            timings = []
            psnrs = []
            ssims = []
            if image_paths and args.warmup:
                warmup_image = load_image(image_paths[0], args.max_side)
                warmup_image, _ = pad_to_multiple(warmup_image)
                for _ in range(args.warmup):
                    if kind == "rcadnet":
                        _ = run_rcadnet(
                            model_info["runner"],
                            warmup_image,
                            scenario,
                            device,
                            args.rcadnet_code_source,
                            image_paths[0],
                            args.rcadnet_gate_threshold if args.rcadnet_gate_threshold >= 0 else None,
                            args.rcadnet_gate_softness,
                        )
                    elif kind == "nafnet":
                        _ = model_info["runner"](warmup_image.to(device))
                    elif kind == "demoe":
                        _ = model_info["runner"](warmup_image.to(device), scenario=scenario, task=args.demoe_task)
                    else:
                        _ = model_info["runner"](warmup_image, scenario)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            for input_path in image_paths:
                gt_path = gt_dir / input_path.name
                if not gt_path.exists():
                    continue
                image = load_image(input_path, args.max_side)
                target = load_image(gt_path, args.max_side)
                image, original_size = pad_to_multiple(image)
                target, _ = pad_to_multiple(target)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                if kind == "rcadnet":
                    restored = run_rcadnet(
                        model_info["runner"],
                        image,
                        scenario,
                        device,
                        args.rcadnet_code_source,
                        input_path,
                        args.rcadnet_gate_threshold if args.rcadnet_gate_threshold >= 0 else None,
                        args.rcadnet_gate_softness,
                    )
                elif kind == "nafnet":
                    restored = model_info["runner"](image.to(device))
                elif kind == "demoe":
                    restored = model_info["runner"](image.to(device), scenario=scenario, task=args.demoe_task)
                else:
                    restored = model_info["runner"](image, scenario)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)

                restored = unpad(restored.cpu(), original_size)
                target = unpad(target.cpu(), original_size)
                psnr, ssim = psnr_ssim(restored, target)
                psnrs.append(psnr)
                ssims.append(ssim)

            row = {
                "model": model_name,
                "scenario": scenario,
                "images": len(psnrs),
                "psnr": statistics.mean(psnrs) if psnrs else None,
                "ssim": statistics.mean(ssims) if ssims else None,
                "mean_runtime_ms": statistics.mean(timings) if timings else None,
                "median_runtime_ms": statistics.median(timings) if timings else None,
                "runtime_backend": model_info["backend"],
                "include_in_gpu_speed_ranking": "yes" if device.type == "cuda" else "no",
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
