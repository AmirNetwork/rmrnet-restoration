from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


DFPIR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "DFPIR-main"
if str(DFPIR_ROOT) not in sys.path:
    sys.path.insert(0, str(DFPIR_ROOT))

from net.model import ChannelShuffle_skip_textguaid  # noqa: E402


DFPIR_PROMPTS = {
    "denoise": "Gaussian noise with a standard deviation of 25",
    "derain": "Rain degradation with rain lines",
    "dehaze": "Hazy degradation with normal haze",
    "deblur": "Blur degradation with motion blur",
    "lowlight": "Lowlight degradation",
}


def task_from_scenario(scenario: str) -> str:
    name = scenario.lower()
    if "rain" in name:
        return "derain"
    if "haze" in name:
        return "dehaze"
    if "lowlight" in name or "low_light" in name:
        return "lowlight"
    if "noise" in name or "gaussian" in name:
        return "denoise"
    return "deblur"


class DFPIRAdapter(nn.Module):
    """Windows-friendly wrapper around the official CVPR 2025 DFPIR model.

    If `weights` is omitted, use `smoke=True` to instantiate a tiny random model
    for pipeline testing only. Official comparisons require official DFPIR
    weights and the default architecture.
    """

    name = "DFPIR-CVPR2025"

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "cuda",
        smoke: bool = False,
        use_clip: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.smoke = smoke
        self.use_clip = use_clip
        self.clip_model = None
        self._text_cache: dict[str, torch.Tensor] = {}

        if smoke and weights is None:
            self.model = ChannelShuffle_skip_textguaid(
                dim=8,
                num_blocks=[1, 1, 1, 1],
                num_refinement_blocks=1,
                heads=[1, 1, 1, 1],
                device=str(self.device),
            )
        else:
            self.model = ChannelShuffle_skip_textguaid(device=str(self.device))
        self.model.to(self.device).eval()

        if weights:
            checkpoint = torch.load(weights, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            self.model.load_state_dict(state_dict, strict=False)

        if use_clip:
            import clip

            self.clip = clip
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def _text_code(self, scenario: str) -> torch.Tensor:
        task = task_from_scenario(scenario)
        if task in self._text_cache:
            return self._text_cache[task]
        if self.use_clip and self.clip_model is not None:
            tokens = self.clip.tokenize([DFPIR_PROMPTS[task]]).to(self.device)
            with torch.no_grad():
                code = self.clip_model.encode_text(tokens).float()
            self._text_cache[task] = code
            return code
        # Zero text code is only for smoke tests where no official CLIP/weights
        # path is being used. It keeps the benchmark harness deterministic.
        code = torch.zeros(1, 512, device=self.device)
        self._text_cache[task] = code
        return code

    @torch.inference_mode()
    def forward(self, image: torch.Tensor, scenario: str) -> torch.Tensor:
        image = image.to(self.device)
        code = self._text_code(scenario)
        output = self.model(image, code)
        return torch.clamp(output, 0.0, 1.0)

    @property
    def backend_status(self) -> str:
        if self.device.type == "cuda":
            return "GPU-confirmed" if not self.smoke else "GPU-confirmed-smoke"
        return "CPU-forced"
