from __future__ import annotations

import importlib
import sys
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "InstructIR-main"
HF_CACHE = ROOT / ".hf_cache"
TRANSFORMERS_CACHE = HF_CACHE / "transformers"


def _add_instructir_to_path() -> None:
    if not INSTRUCTIR_ROOT.exists():
        raise FileNotFoundError(
            f"InstructIR repository was not found at {INSTRUCTIR_ROOT}. "
            "Download https://github.com/mv-lab/InstructIR into third_party/InstructIR-main."
        )
    root_text = str(INSTRUCTIR_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _import_official_modules() -> tuple[Any, Any, Any]:
    """Import InstructIR despite this project also having a ``models`` package.

    The official repository uses absolute imports such as
    ``models.nafnet_utils``.  ``restore_yolo_split.py`` has already imported
    this project's ``models.rmrnet`` by the time the adapter is constructed,
    so a normal ``from models import instructir`` resolves to the wrong
    package.  Temporarily expose the vendored repository as ``models`` during
    import, then restore the host package.  The returned class/module objects
    remain valid after the namespace is restored.
    """

    host_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "models" or name.startswith("models.")
    }
    for name in host_modules:
        sys.modules.pop(name, None)
    root_text = str(INSTRUCTIR_ROOT)
    already_present = root_text in sys.path
    if already_present:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    try:
        instructir = importlib.import_module("models.instructir")
        text_models = importlib.import_module("text.models")
        return instructir, text_models.LMHead, text_models.LanguageModel
    finally:
        for name in tuple(sys.modules):
            if name == "models" or name.startswith("models."):
                sys.modules.pop(name, None)
        sys.modules.update(host_modules)
        if root_text in sys.path:
            sys.path.remove(root_text)
        if already_present:
            sys.path.append(root_text)


def _configure_offline_hf_cache() -> None:
    """Force InstructIR text embeddings to use the project-local HF cache.

    The paper benchmark is expected to run offline/reproducibly. InstructIR's
    released language branch needs TaylorAI/bge-micro-v2; the cached snapshot is
    stored under .hf_cache/transformers. Setting these variables before
    constructing the language model prevents slow network retries during GT49
    native-field evaluation.
    """

    if TRANSFORMERS_CACHE.exists():
        os.environ.setdefault("HF_HOME", str(HF_CACHE))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE))
        os.environ.setdefault("HF_HUB_CACHE", str(TRANSFORMERS_CACHE))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class InstructIRAdapter(nn.Module):
    """Adapter for the official ECCV 2024 InstructIR baseline.

    InstructIR uses natural-language restoration instructions as metadata. The
    benchmark uses two modes:
      * generic: a fixed prompt with no camera/geotag information;
      * metadata: a prompt generated from EXIF and pose metadata.

    The adapter accepts either the released raw image-model state dict or an
    audited target-adaptation checkpoint whose state dict is stored under the
    ``model`` key.  It keeps the modules frozen for ordinary benchmark
    inference; the matched-training wrapper may explicitly unfreeze the image
    model after construction.
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
        _configure_offline_hf_cache()
        _add_instructir_to_path()

        instructir, LMHead, LanguageModel = _import_official_modules()

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
        image_payload = torch.load(
            image_weights,
            map_location=self.device,
            weights_only=False,
        )
        image_state = (
            image_payload["model"]
            if isinstance(image_payload, dict) and "model" in image_payload
            else image_payload
        )
        if not isinstance(image_state, dict) or not image_state:
            raise TypeError(
                "InstructIR image checkpoint must contain a non-empty state dict"
            )
        non_tensor_keys = [
            key for key, value in image_state.items() if not torch.is_tensor(value)
        ]
        if non_tensor_keys:
            raise TypeError(
                "InstructIR image state dict contains non-tensor entries: "
                + ", ".join(non_tensor_keys[:5])
            )
        self.model.load_state_dict(image_state, strict=True)
        self.model.eval()

        # The released language model is frozen and runs on CPU by default in
        # the official demo. We keep that behavior to avoid GPU memory spikes.
        self.language_model = LanguageModel(model=text_model)
        self.language_model.eval()
        self.lm_head = LMHead(embedding_dim=384, hidden_dim=256, num_classes=7)
        self.lm_head.load_state_dict(
            torch.load(lm_head_weights, map_location="cpu", weights_only=False), strict=True
        )
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
