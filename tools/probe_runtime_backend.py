from __future__ import annotations

import argparse
import json
import sys
import statistics
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcadnet import RCADNet, code_from_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe RCAD-Net runtime/backend for fair benchmark tables.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--scenario", default="motion_random_medium")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.weights, map_location=device)
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
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    image = torch.rand(1, 3, args.height, args.width, device=device)
    code = code_from_scenario(args.scenario, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    timings = []
    with torch.inference_mode():
        for index in range(args.warmup + args.runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(image, code)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if index >= args.warmup:
                timings.append(elapsed_ms)

    row = {
        "model": "RCAD-Net",
        "runtime_backend": "GPU-confirmed" if device.type == "cuda" else "CPU-forced",
        "include_in_gpu_speed_ranking": "yes" if device.type == "cuda" else "no",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "height": args.height,
        "width": args.width,
        "mean_runtime_ms": statistics.mean(timings),
        "median_runtime_ms": statistics.median(timings),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
    }
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
