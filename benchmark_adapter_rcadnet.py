from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from rcadnet import RCADNet, code_from_metadata, code_from_scenario


class RCADNetAdapter:
    """Benchmark adapter template for scripts/benchmark_all_models.py.

    Keep this class small so it is easy to wrap in the existing benchmark's
    adapter pattern. It accepts PIL images or HWC uint8 numpy arrays and returns
    an HWC uint8 RGB numpy array, matching common restoration benchmark code.

    Paper equation trace:
        I_r = clip(I_b + eta_d * G_d * D(I_d), 0, 1)

    The detail-preserving path is controlled by checkpoint fields
    ``detail_preserve`` and ``detail_gain``. They must be forwarded here so the
    benchmark adapter evaluates the same deployed RMR-Net architecture as the
    manuscript, not a legacy RCADNet-only fallback.
    """

    name = "RMR-Net"

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cuda",
        scenario: str | None = None,
        width: int | None = None,
        gate_threshold: float | None = None,
        gate_softness: float = 0.03,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.scenario = scenario or "motion_random_medium"
        self.gate_threshold = gate_threshold
        self.gate_softness = gate_softness
        self.model = self._load_model(width)
        self.backend_status = "GPU-confirmed" if self.device.type == "cuda" else "CPU-forced"

    def _load_model(self, width: int | None) -> RCADNet:
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        arch = checkpoint.get("arch", {})
        model = RCADNet(
            width=width or arch.get("width", 32),
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
        ).to(self.device)
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError:
            # Older checkpoints may not carry the final detail-skip or aux-head
            # keys. Falling back to strict=False preserves reproducibility while
            # making the compatibility boundary explicit.
            model.load_state_dict(checkpoint["model"], strict=False)
        model.eval()
        return model

    def set_scenario(self, scenario: str) -> None:
        self.scenario = scenario

    def restore_image(self, image: Image.Image | np.ndarray, metadata: dict[str, Any] | None = None) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[..., :3].astype(np.uint8))
        tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(self.device)

        if metadata and "degradation_code" in metadata:
            code = torch.as_tensor(metadata["degradation_code"], dtype=tensor.dtype, device=self.device)
        elif metadata:
            code = code_from_metadata(metadata, device=self.device)
        else:
            code = code_from_scenario(metadata.get("scenario", self.scenario) if metadata else self.scenario, device=self.device)

        with torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            output = self.model(
                tensor,
                code,
                gate_threshold=self.gate_threshold,
                gate_softness=self.gate_softness,
            )[0].detach().cpu()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        return np.asarray(TF.to_pil_image(output), dtype=np.uint8)

    def runtime_backend_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "runtime_backend": self.backend_status,
            "include_in_gpu_speed_ranking": "yes" if self.device.type == "cuda" else "no",
            "weights": str(self.weights_path),
        }
