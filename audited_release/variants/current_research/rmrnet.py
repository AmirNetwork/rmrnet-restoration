# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from rcadnet.model import RCADNet


TensorOutput = Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[Any, ...]]


class AuxiliaryContourHead(nn.Module):
    """
    Lightweight auxiliary geometry head for legacy train-time contour studies.

    The outputs are not segmentation predictions and should not be reported as
    pixel-level masks. The final paper configuration disables active-contour
    loss and uses active contours after detection for measurement only. This
    head remains for checkpoint compatibility and ablation work.

    Outputs
    -------
    phi:
        Bounded level-set-like map.
    lambda1, lambda2:
        Positive bounded region weights in [0.1, 5.0].
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        phi_scale: float = 3.0,
        lambda_min: float = 0.1,
        lambda_max: float = 5.0,
    ) -> None:
        super().__init__()

        if lambda_max <= lambda_min:
            raise ValueError("lambda_max must be greater than lambda_min.")

        self.phi_scale = float(phi_scale)
        self.lambda_min = float(lambda_min)
        self.lambda_span = float(lambda_max - lambda_min)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )

        self._init_last_layer()

    def _init_last_layer(self) -> None:
        last = self.head[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, restored: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.head(restored)

        phi = self.phi_scale * torch.tanh(raw[:, 0:1])
        lambda1 = self.lambda_min + self.lambda_span * torch.sigmoid(raw[:, 1:2])
        lambda2 = self.lambda_min + self.lambda_span * torch.sigmoid(raw[:, 2:3])

        return {
            "phi": phi,
            "lambda1": lambda1,
            "lambda2": lambda2,
        }


class RMRNet(RCADNet):
    """
    Road Metadata-aware Restoration Network.

    This class keeps historical RCADNet compatibility while exposing the
    reported RMR-Net interface. Metadata-conditioned inference is supported
    when calibrated metadata is available, and image-only inference is a valid
    operating mode through the learned image-derived degradation code.

    Key properties:
    - old RCADNet checkpoints can be transferred with strict=False for
      subsequent retraining, while reported inference requires strict=True;
    - normal inference still returns a restored tensor by default;
    - return_dict=True exposes stable named outputs for training;
    - return_aux=True adds train-time auxiliary contour maps;
    - aligned metadata and image-only inference both use the same restoration
      backbone; the reliability gate determines how strongly metadata is used.
    """

    def __init__(
        self,
        *args: Any,
        enable_aux_contour: bool = True,
        clamp_output: bool = True,
        aux_hidden_channels: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.enable_aux_contour = bool(enable_aux_contour)
        self.clamp_output = bool(clamp_output)

        self.aux_contour_head: Optional[AuxiliaryContourHead]
        if self.enable_aux_contour:
            self.aux_contour_head = AuxiliaryContourHead(
                in_channels=3,
                hidden_channels=aux_hidden_channels,
            )
        else:
            self.aux_contour_head = None

    # ------------------------------------------------------------------
    # Compatibility utilities
    # ------------------------------------------------------------------

    def _call_base_forward(
        self,
        image: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        *,
        return_tuple: bool = False,
        return_dict: bool = False,
        return_aux: bool = False,
        **kwargs: Any,
    ) -> TensorOutput:
        """
        Safely call RCADNet.forward even if its signature differs between
        versions of the project.

        The fallback order is intentionally conservative:
        1. image + metadata + return flags + kwargs
        2. image + metadata + kwargs
        3. image + metadata
        4. image only
        """
        base_forward = super().forward

        candidate_calls = []

        candidate_calls.append(
            lambda: base_forward(
                image,
                metadata,
                return_tuple=return_tuple,
                return_dict=return_dict,
                return_aux=return_aux,
                **kwargs,
            )
        )
        candidate_calls.append(lambda: base_forward(image, metadata, **kwargs))
        candidate_calls.append(lambda: base_forward(image, metadata))
        candidate_calls.append(lambda: base_forward(image))

        last_error: Optional[Exception] = None

        for call in candidate_calls:
            try:
                return call()
            except TypeError as exc:
                last_error = exc
                continue

        raise TypeError(
            "Could not call RCADNet.forward using any compatible signature. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _extract_restored(output: TensorOutput) -> torch.Tensor:
        """
        Convert common RCADNet output formats into a restored image tensor.
        """
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, dict):
            for key in ("restored", "image", "out", "output", "pred", "prediction"):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value

            tensor_values = [v for v in output.values() if isinstance(v, torch.Tensor)]
            if len(tensor_values) == 1:
                return tensor_values[0]

            raise KeyError(
                "RCADNet returned a dict, but no restored image key was found. "
                "Expected one of: restored, image, out, output, pred, prediction."
            )

        if isinstance(output, tuple):
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    return item
            raise ValueError("RCADNet returned a tuple without a 4D image tensor.")

        raise TypeError(f"Unsupported RCADNet output type: {type(output)!r}")

    def load_pretrained_compat(
        self,
        checkpoint_path: str,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = False,
        key_candidates: Tuple[str, ...] = ("model", "state_dict", "model_state_dict"),
    ) -> Dict[str, Any]:
        """
        Load old RCADNet/RMRNet checkpoints safely.

        ``strict=False`` is provided only for initializing a new training run
        from an older checkpoint. Evaluation and paper-result generation must
        use ``strict=True`` so no deployed parameter remains untrained.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if isinstance(checkpoint, dict):
            state_dict = None
            for key in key_candidates:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint
        else:
            raise TypeError("Checkpoint must be a state-dict-like object.")

        cleaned_state = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            cleaned_state[new_key] = value

        result = self.load_state_dict(cleaned_state, strict=strict)

        return {
            "checkpoint_path": checkpoint_path,
            "strict": strict,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        *,
        return_tuple: bool = False,
        return_dict: bool = False,
        return_aux: bool = False,
        **kwargs: Any,
    ) -> TensorOutput:
        base_output = self._call_base_forward(
            image,
            metadata,
            return_tuple=return_tuple,
            return_dict=return_dict,
            return_aux=return_aux,
            **kwargs,
        )

        restored = self._extract_restored(base_output)

        if self.clamp_output:
            restored = restored.clamp(0.0, 1.0)

        aux: Dict[str, torch.Tensor] = {}
        if return_aux and self.aux_contour_head is not None:
            aux = self.aux_contour_head(restored)

        if return_dict:
            output: Dict[str, torch.Tensor] = {}
            if isinstance(base_output, dict):
                output.update({k: v for k, v in base_output.items() if isinstance(v, torch.Tensor) or v is None})
            output.update(
                {
                    "restored": restored,
                    "input": image,
                }
            )

            if metadata is not None:
                output["metadata"] = metadata
                output["metadata_used"] = torch.ones(
                    image.shape[0],
                    1,
                    device=image.device,
                    dtype=image.dtype,
                )
            else:
                output["metadata_used"] = torch.zeros(
                    image.shape[0],
                    1,
                    device=image.device,
                    dtype=image.dtype,
                )

            output.update(aux)
            return output

        if return_tuple:
            if aux:
                return restored, aux["phi"], aux["lambda1"], aux["lambda2"]
            return (restored,)

        return restored
