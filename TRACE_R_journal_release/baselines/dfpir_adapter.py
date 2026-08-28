from __future__ import annotations

# TRACE-R integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>.
# The DFPIR implementation and weights retain their upstream authorship/license.

import sys
from pathlib import Path

import torch
from torch import nn


DFPIR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "DFPIR-main"


def _load_dfpir_model_class() -> type[nn.Module]:
    """Load the official DFPIR class only when that baseline is instantiated.

    TRACE-R and the other baselines can therefore be imported and unit-tested
    before the separately licensed DFPIR repository has been downloaded.
    """

    if not DFPIR_ROOT.exists():
        raise FileNotFoundError(
            f"DFPIR repository was not found at {DFPIR_ROOT}. "
            "Clone the official implementation into third_party/DFPIR-main."
        )
    root_text = str(DFPIR_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from net.model import ChannelShuffle_skip_textguaid

    return ChannelShuffle_skip_textguaid


DFPIR_PROMPTS = {
    "denoise": "Gaussian noise with a standard deviation of 25",
    "derain": "Rain degradation with rain lines",
    "dehaze": "Hazy degradation with normal haze",
    "deblur": "Blur degradation with motion blur",
    "lowlight": "Lowlight degradation",
}


_LEGACY_CHECKPOINT_ATTENTION_SUFFIXES = (".attn1", ".attn2", ".attn3")
_LEGACY_CHECKPOINT_ATTENTION_MODULES = (
    "encoder_shuffle_channel1.select_attn",
    "encoder_shuffle_channel2.select_attn",
    "encoder_shuffle_channel3.select_attn",
    "latent_shuffle_channel.select_attn",
)


def load_official_dfpir_state(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, object]:
    """Apply the released DFPIR checkpoint with source-native key filtering.

    The official DFPIR evaluation script filters a released checkpoint against
    the current model state before loading. The published five-degradation
    checkpoint covers every tensor used by the current source, but also retains
    twelve obsolete ``attn1``--``attn3`` scalars from an earlier attention
    variant; the current source uses ``attn4``. We reproduce that policy while
    refusing any missing current tensor or any other checkpoint-only key.
    """

    model_state = model.state_dict()
    model_keys = set(model_state)
    checkpoint_keys = set(state_dict)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    allowed_unexpected = {
        f"{module}{suffix}"
        for module in _LEGACY_CHECKPOINT_ATTENTION_MODULES
        for suffix in _LEGACY_CHECKPOINT_ATTENTION_SUFFIXES
    }
    unsupported = sorted(set(unexpected) - allowed_unexpected)
    if missing or unsupported:
        raise RuntimeError(
            "DFPIR checkpoint is incompatible with the official source model: "
            f"missing={missing}, unsupported_checkpoint_keys={unsupported}"
        )

    compatible = dict(model_state)
    compatible.update({key: state_dict[key] for key in model_state})
    model.load_state_dict(compatible, strict=True)
    return {
        "model_tensor_count": len(model_state),
        "checkpoint_tensor_count": len(state_dict),
        "ignored_legacy_keys": unexpected,
        "coverage": 1.0,
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
        self.clip_device = torch.device("cpu")
        self._text_cache: dict[str, torch.Tensor] = {}
        model_class = _load_dfpir_model_class()

        if weights is not None and not smoke and not use_clip:
            raise ValueError(
                "Official DFPIR inference requires its CLIP degradation prompt. "
                "Set use_clip=True; a zero text vector is permitted only for smoke tests."
            )

        if smoke and weights is None:
            self.model = model_class(
                dim=8,
                num_blocks=[1, 1, 1, 1],
                num_refinement_blocks=1,
                heads=[1, 1, 1, 1],
                device=str(self.device),
            )
        else:
            self.model = model_class(device=str(self.device))
        self.model.to(self.device).eval()

        if weights:
            checkpoint = torch.load(weights, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            self.checkpoint_load_report = load_official_dfpir_state(
                self.model, state_dict
            )

        if use_clip:
            import clip

            self.clip = clip
            # CLIP is used once per degradation prompt. Keeping it on CPU
            # preserves the official embedding while leaving GPU memory for
            # native-resolution DFPIR tiles.
            self.clip_model, _ = clip.load("ViT-B/32", device=self.clip_device)
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def _text_code(self, scenario: str) -> torch.Tensor:
        task = task_from_scenario(scenario)
        if task in self._text_cache:
            return self._text_cache[task]
        if self.use_clip and self.clip_model is not None:
            tokens = self.clip.tokenize([DFPIR_PROMPTS[task]]).to(self.clip_device)
            with torch.no_grad():
                code = self.clip_model.encode_text(tokens).float().to(self.device)
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
