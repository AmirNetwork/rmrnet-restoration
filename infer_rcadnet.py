from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from rcadnet import RCADNet, code_from_scenario
from rcadnet.dataset import IMAGE_EXTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RCAD-Net inference on an image folder.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-threshold", type=float, default=-1.0, help="If >=0, pass through clean/low-severity inputs.")
    parser.add_argument("--gate-softness", type=float, default=0.03, help="Set to 0 for hard clean-frame bypass.")
    return parser.parse_args()


def load_model(weights: str, device: torch.device) -> RCADNet:
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.weights, device)
    code = code_from_scenario(args.scenario, device=device)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
            restored = model(
                tensor,
                code,
                gate_threshold=args.gate_threshold if args.gate_threshold >= 0 else None,
                gate_softness=args.gate_softness,
            )[0].detach().cpu()
            TF.to_pil_image(restored).save(output_dir / path.name)
            print(output_dir / path.name, flush=True)


if __name__ == "__main__":
    main()
