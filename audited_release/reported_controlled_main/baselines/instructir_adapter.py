# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


INSTRUCTIR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "InstructIR-main"


def _add_instructir_to_path() -> None:
    if not INSTRUCTIR_ROOT.exists():
        raise FileNotFoundError(
            f"InstructIR repository was not found at {INSTRUCTIR_ROOT}. "
            "Download https://github.com/mv-lab/InstructIR into third_party/InstructIR-main."
        )
    root_text = str(INSTRUCTIR_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


class InstructIRAdapter(nn.Module):
    """Adapter for the official ECCV 2024 InstructIR baseline.

    InstructIR uses natural-language restoration instructions as metadata. The
    benchmark uses two modes:
      * generic: a fixed prompt with no camera/geotag information;
      * metadata: a prompt generated from EXIF and pose metadata.

    The model is not trained in this project. It loads the released image model
    and language-head weights and keeps them frozen for fair inference.
    """

    name = "InstructIR-ECCV2024"

    def __init__(
        self,
        image_weights: str | Path,
        lm_head_weights: str | Path,
        *,
        device: str | torch.device = "cuda",
        text_model: str = "TaylorAI/bge-micro-v2",
    ) -> None:
        super().__init__()
        _add_instructir_to_path()

        from models import instructir  # type: ignore
        from text.models import LMHead, LanguageModel  # type: ignore

        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.text_model_name = text_model
        self.prompt_cache: dict[str, torch.Tensor] = {}

        self.model = instructir.create_model(
            input_channels=3,
            width=32,
            enc_blks=[2, 2, 4, 8],
            middle_blk_num=4,
            dec_blks=[2, 2, 2, 2],
            txtdim=256,
        ).to(self.device)
        self.model.load_state_dict(torch.load(image_weights, map_location=self.device), strict=True)
        self.model.eval()

        # The released language model is frozen and runs on CPU by default in
        # the official demo. We keep that behavior to avoid GPU memory spikes.
        self.language_model = LanguageModel(model=text_model)
        self.language_model.eval()
        self.lm_head = LMHead(embedding_dim=384, hidden_dim=256, num_classes=7)
        self.lm_head.load_state_dict(torch.load(lm_head_weights, map_location="cpu"), strict=True)
        self.lm_head.eval()

        for module in (self.model, self.language_model, self.lm_head):
            for param in module.parameters():
                param.requires_grad = False

    @torch.inference_mode()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        if prompt not in self.prompt_cache:
            lm_embedding = self.language_model(prompt)
            text_embedding, _deg_pred = self.lm_head(lm_embedding)
            self.prompt_cache[prompt] = text_embedding.to(self.device)
        return self.prompt_cache[prompt]

    @torch.inference_mode()
    def forward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        image = image.to(self.device)
        text_embedding = self.encode_prompt(prompt)
        output = self.model(image, text_embedding)
        return output.clamp(0.0, 1.0)

    @property
    def backend_status(self) -> str:
        return "GPU-confirmed" if self.device.type == "cuda" else "CPU-forced"


def generic_road_prompt() -> str:
    return (
        "Restore this road inspection image while preserving pothole rims, "
        "cracks, lane markings, and pavement texture."
    )


def metadata_prompt(record: dict[str, Any]) -> str:
    """Create a compact natural-language prompt from geotagged camera metadata."""

    exif = record.get("exif", {}) or {}
    csv = record.get("csv", {}) or {}
    yaw = csv.get("yaw", "")
    pitch = csv.get("pitch", "")
    roll = csv.get("roll", "")
    altitude = csv.get("ht", csv.get("altitude", ""))
    exposure = exif.get("ExposureTime", "")
    brightness = exif.get("BrightnessValue", "")
    zoom = exif.get("DigitalZoomRatio", "")
    model = exif.get("Model", "cam1")
    return (
        "Restore a high-resolution road inspection image from camera "
        f"{model}. Preserve small cracks, pothole rims, patches, and road "
        f"texture. Use the capture metadata as a restoration prior: yaw {yaw}, "
        f"pitch {pitch}, roll {roll}, altitude {altitude}, exposure {exposure}, "
        f"brightness {brightness}, zoom {zoom}. Avoid hallucinating new defects."
    )
