# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.demoe_adapter import DeMoEAdapter
from baselines.dfpir_adapter import DFPIRAdapter
from baselines.nafnet_road import NAFNetRoad
from rcadnet import RCADNet, code_from_metadata, code_from_scenario


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a YOLO image split and copy labels for detection evaluation.")
    parser.add_argument("--data", required=True, help="Degraded YOLO data.yaml.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--model", choices=["rcadnet", "dfpir", "nafnet", "demoe"], required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rcadnet-weights")
    parser.add_argument("--dfpir-weights")
    parser.add_argument("--nafnet-weights")
    parser.add_argument("--demoe-weights")
    parser.add_argument("--demoe-task", choices=["auto", "scenario", "defocus", "global_motion", "local_motion", "synth_global_motion", "low_light"], default="auto")
    parser.add_argument("--demoe-smoke", action="store_true", help="Use random DeMoE weights for wiring tests only; never report these as benchmark results.")
    parser.add_argument("--dfpir-clip", action="store_true")
    parser.add_argument("--gate-threshold", type=float, default=-1.0, help="RMR clean-frame pass-through threshold; disabled when negative.")
    parser.add_argument("--gate-softness", type=float, default=0.03, help="Set to 0 for hard bypass of clean/low-severity images.")
    parser.add_argument("--residual-strength", type=float, default=1.0, help="Blend restored output as input + eta * (restored - input). Use eta<1 for residual perception policy.")
    parser.add_argument("--debug-every", type=int, default=0, help="Print JSON restoration stats every N images. 0 disables per-image debug.")
    parser.add_argument(
        "--rcadnet-code-source",
        choices=["scenario", "metadata", "blind"],
        default="scenario",
        help="RMR/RCAD conditioning: scenario label, metadata JSON, or blind image-estimated code only.",
    )
    return parser.parse_args()


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


def pad_to_multiple(image: torch.Tensor, multiple: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image, (height, width)




def image_stats(name: str, tensor: torch.Tensor) -> dict[str, float]:
    t = tensor.detach()
    return {
        f"{name}_mean": float(t.mean().cpu()),
        f"{name}_std": float(t.std(unbiased=False).cpu()),
        f"{name}_min": float(t.min().cpu()),
        f"{name}_max": float(t.max().cpu()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    source_root = Path(config["path"])
    image_dir = source_root / config[args.split]
    label_dir = source_root / config[args.split].replace("images", "labels")
    metadata_dir = source_root / "metadata" / args.split
    out = Path(args.out)
    out_image_dir = out / "images" / args.split
    out_label_dir = out / "labels" / args.split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "rcadnet":
        if not args.rcadnet_weights:
            raise ValueError("--rcadnet-weights is required")
        model = load_rcadnet(args.rcadnet_weights, device)
        scenario_code = code_from_scenario(args.scenario, device=device)
    elif args.model == "nafnet":
        if not args.nafnet_weights:
            raise ValueError("--nafnet-weights is required")
        model = load_nafnet(args.nafnet_weights, device)
        scenario_code = None
    elif args.model == "demoe":
        if not args.demoe_weights and not args.demoe_smoke:
            raise ValueError("--demoe-weights is required for real DeMoE; use --demoe-smoke only for pipeline tests")
        model = DeMoEAdapter(
            args.demoe_weights,
            device=device,
            task=args.demoe_task,
            smoke=args.demoe_smoke,
            strict=not args.demoe_smoke,
        )
        scenario_code = None
    else:
        if not args.dfpir_weights:
            raise ValueError("--dfpir-weights is required")
        model = DFPIRAdapter(args.dfpir_weights, device=str(device), use_clip=args.dfpir_clip)
        scenario_code = None

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    with torch.inference_mode():
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
            tensor, original_size = pad_to_multiple(tensor)
            if args.model == "rcadnet":
                if args.rcadnet_code_source == "blind":
                    code = None
                elif args.rcadnet_code_source == "metadata":
                    metadata_path = metadata_dir / f"{path.stem}.json"
                    if metadata_path.exists():
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        code = code_from_metadata(metadata, device=device)
                    else:
                        code = scenario_code
                else:
                    code = scenario_code
                restored = model(
                    tensor,
                    code,
                    gate_threshold=args.gate_threshold if args.gate_threshold >= 0 else None,
                    gate_softness=args.gate_softness,
                )
            elif args.model == "nafnet":
                restored = model(tensor)
            elif args.model == "demoe":
                restored = model(tensor, scenario=args.scenario, task=args.demoe_task)
            else:
                restored = model(tensor, args.scenario)
            if args.residual_strength != 1.0:
                eta = max(0.0, min(float(args.residual_strength), 1.0))
                restored = (tensor + eta * (restored - tensor)).clamp(0.0, 1.0)
            restored = restored[..., : original_size[0], : original_size[1]]
            if args.debug_every > 0 and index % args.debug_every == 0:
                residual = restored - tensor[..., : original_size[0], : original_size[1]]
                debug = {
                    "tag": "restore_yolo_split_debug",
                    "index": index,
                    "image": path.name,
                    "model": args.model,
                    "scenario": args.scenario,
                    "residual_strength": float(args.residual_strength),
                    "residual_abs_mean": float(residual.abs().mean().cpu()),
                    "residual_abs_max": float(residual.abs().max().cpu()),
                }
                debug.update(image_stats("input", tensor[..., : original_size[0], : original_size[1]]))
                debug.update(image_stats("restored", restored))
                print(json.dumps(debug), flush=True)
            TF.to_pil_image(restored[0].detach().cpu()).save(out_image_dir / path.name)
            label_path = label_dir / path.with_suffix(".txt").name
            if label_path.exists():
                shutil.copy2(label_path, out_label_dir / label_path.name)

    data_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": config.get("train", "images/train"),
        "val": f"images/{args.split}",
        "test": f"images/{args.split}",
        "names": config["names"],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print({"model": args.model, "scenario": args.scenario, "images": len(paths), "out": str(out)})


if __name__ == "__main__":
    main()
