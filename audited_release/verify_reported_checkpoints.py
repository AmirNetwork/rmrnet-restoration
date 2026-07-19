# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

"""Strictly verify the source/checkpoint pair reported in the paper."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reported_controlled_main"
sys.path.insert(0, str(SOURCE))

from rcadnet.model import RCADNet  # noqa: E402


EXPECTED_HASHES = {
    "ivcnz_epoch_028.pth": "3c6a1a8e582639fade7c5cae9cbb301e3f2987d600c06584adacb14c1ab538dd",
    "pcm_epoch_028.pth": "e580d2bf0bb8cc3319afbcca3b3d1cb96ef340667497f445b6685c2ebbe7eec7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(config: dict) -> RCADNet:
    args = config["args"]
    return RCADNet(
        width=int(args["width"]),
        use_defect_attention=not bool(args["no_defect_attention"]),
        use_estimated_code=args["code_source"] in {"estimated", "fused", "metadata_fused"},
        code_fusion=args["code_source"],
        block_type=args["block_type"],
        attention_type=args["attention_type"],
        conditioning=args["conditioning"],
        use_tdac_head=bool(args["use_tdac_head"]),
        detail_preserve=bool(args["detail_preserve"]),
        detail_gain=float(args["detail_gain"]),
    )


def verify(dataset: str) -> dict:
    checkpoint_path = ROOT / "reported_checkpoints" / f"{dataset}_epoch_028.pth"
    config_path = ROOT / "reported_checkpoints" / f"{dataset}_audit_config.json"
    actual_hash = sha256(checkpoint_path)
    if actual_hash != EXPECTED_HASHES[checkpoint_path.name]:
        raise RuntimeError(f"SHA-256 mismatch for {checkpoint_path.name}: {actual_hash}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.inference_mode():
        image = torch.rand(1, 3, 64, 64)
        code = torch.rand(1, 8)
        output = model(image, code)
    if output.shape != image.shape or not torch.isfinite(output).all():
        raise RuntimeError(f"Invalid inference output for {dataset}: {tuple(output.shape)}")

    return {
        "dataset": dataset,
        "checkpoint": checkpoint_path.name,
        "sha256": actual_hash,
        "state_tensors": len(state),
        "strict_load": "PASS",
        "inference_shape": list(output.shape),
    }


def main() -> None:
    report = {"source": "reported_controlled_main", "checks": [verify("ivcnz"), verify("pcm")]}
    output = ROOT / "checkpoint_compatibility_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
