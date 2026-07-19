# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

"""Profile deployed RMR-Net complexity from a checkpoint.

Reports parameters, approximate convolution/linear FLOPs, CUDA runtime, and
peak allocated memory for a single-image inference path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcadnet import RCADNet, code_from_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("experiments/v30_submission_readiness/rmrnet_complexity.json"))
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--scenario", default="motion_horizontal_medium")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_model(checkpoint: dict[str, Any], device: torch.device) -> RCADNet:
    arch = checkpoint.get("arch", {})
    model = RCADNet(
        width=arch.get("width", 32),
        code_dim=arch.get("code_dim", 8),
        blocks_per_stage=arch.get("blocks_per_stage", 2),
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
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def count_flops(model: nn.Module, image: torch.Tensor, code: torch.Tensor) -> int:
    flops = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        out = output
        batch, out_channels, out_h, out_w = out.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        flops += int(batch * out_channels * out_h * out_w * kernel_ops * 2)
        if module.bias is not None:
            flops += int(batch * out_channels * out_h * out_w)

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        in_features = module.in_features
        out_features = module.out_features
        batch = int(inputs[0].numel() / max(in_features, 1))
        flops += int(batch * in_features * out_features * 2)
        if module.bias is not None:
            flops += int(batch * out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        _ = model(image, code)
    for hook in hooks:
        hook.remove()
    return flops


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location=device)
    model = build_model(checkpoint, device)
    image = torch.rand(1, 3, args.height, args.width, device=device)
    code = code_from_scenario(args.scenario, device=device)

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops = count_flops(model, image, code)

    timings = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for idx in range(args.warmup + args.runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(image, code)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            if idx >= args.warmup:
                timings.append(elapsed)

    result = {
        "weights": str(args.weights.resolve()),
        "height": args.height,
        "width": args.width,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "backend": "CUDA" if device.type == "cuda" else "CPU",
        "parameters": params,
        "trainable_parameters": trainable_params,
        "gflops_approx": flops / 1e9,
        "mean_runtime_ms": statistics.mean(timings),
        "median_runtime_ms": statistics.median(timings),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        "runs": args.runs,
        "warmup": args.warmup,
        "flop_note": "Conv2d and Linear only; multiply-add counted as two FLOPs.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
