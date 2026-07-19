# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEMOE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "DeMoE-main"

DEMOE_TASKS = {
    "auto",
    "scenario",
    "defocus",
    "global_motion",
    "local_motion",
    "synth_global_motion",
    "low_light",
}


def demoe_task_from_scenario(scenario: str, requested_task: str = "auto") -> str:
    """Map our benchmark scenario names to DeMoE's task labels.

    DeMoE can route automatically, but its paper also reports manually selected
    experts. Use `requested_task="auto"` to keep DeMoE's own router. Use
    `requested_task="scenario"` to choose the manual expert from our scenario
    name, which separates restoration capacity from router errors.
    """

    if requested_task == "auto":
        return "auto"

    if requested_task != "scenario":
        if requested_task not in DEMOE_TASKS:
            raise ValueError(f"Unknown DeMoE task '{requested_task}'. Valid tasks: {sorted(DEMOE_TASKS)}")
        return requested_task

    name = scenario.lower()
    if "defocus" in name:
        return "defocus"
    if "lowlight" in name or "low_light" in name or "dark" in name:
        return "low_light"
    if "local" in name or "rolling" in name:
        return "local_motion"
    if "native" in name or "real" in name or "kitti" in name:
        return "global_motion"
    if "motion" in name or "vibration" in name or "shake" in name:
        return "synth_global_motion"
    return "auto"


def _add_demoe_to_path(repo_root: Path) -> None:
    if not repo_root.exists():
        raise FileNotFoundError(
            f"DeMoE repository was not found at {repo_root}. "
            "Clone or unzip https://github.com/cidautai/DeMoE into third_party/DeMoE-main."
        )
    for candidate in (repo_root / "archs", repo_root):
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        cleaned[key] = value
    return cleaned


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _clean_state_dict(value)
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return _clean_state_dict(checkpoint)
    raise TypeError("Could not find a valid DeMoE state_dict in the checkpoint.")


class DeMoEAdapter(nn.Module):
    """Windows/CUDA-friendly adapter for the official DeMoE all-in-one baseline.

    The official repository runs inference through a Linux-oriented torchrun
    script. This wrapper imports the released architecture directly so our
    benchmark can call it like the other restoration baselines.
    """

    name = "DeMoE-MoE-Decoder"

    def __init__(
        self,
        weights: str | Path | None,
        device: str | torch.device = "cuda",
        task: str = "auto",
        repo_root: str | Path | None = None,
        k_used: int = 1,
        smoke: bool = False,
        strict: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.default_task = task
        self.smoke = smoke
        self.strict = strict
        self.repo_root = Path(repo_root) if repo_root else DEMOE_ROOT
        self.load_report: dict[str, Any] = {
            "weights": str(weights) if weights else None,
            "smoke": smoke,
            "strict": strict,
            "k_used": k_used,
        }

        if task not in DEMOE_TASKS:
            raise ValueError(f"Unknown DeMoE task '{task}'. Valid tasks: {sorted(DEMOE_TASKS)}")
        if not weights and not smoke:
            raise FileNotFoundError(
                "Official DeMoE weights are required for benchmark use. "
                "Download DeMoE.pt from the official repository's OneDrive link "
                "and pass --demoe-weights weights/demoe/DeMoE.pt. "
                "Use --demoe-smoke only for wiring tests."
            )

        _add_demoe_to_path(self.repo_root)
        try:
            # Import the file module directly. The third-party `archs` package
            # initializer imports ptflops for complexity reporting, which is not
            # needed for inference and can break otherwise valid installations.
            from DeMoE import DeMoE  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on third-party imports.
            raise ImportError(
                "Could not import DeMoE. Install the extra dependencies in "
                "requirements-demoe-extra.txt and keep the repo at third_party/DeMoE-main."
            ) from exc

        self.model = DeMoE(
            img_channel=3,
            width=32,
            middle_blk_num=2,
            enc_blk_nums=[2, 2, 2, 2],
            dec_blk_nums=[2, 2, 2, 2],
            num_exp=5,
            k_used=k_used,
        ).to(self.device)

        if weights:
            checkpoint = torch.load(weights, map_location=self.device)
            state_dict = _extract_state_dict(checkpoint)
            incompatible = self.model.load_state_dict(state_dict, strict=strict)
            self.load_report["missing_keys"] = list(getattr(incompatible, "missing_keys", []))
            self.load_report["unexpected_keys"] = list(getattr(incompatible, "unexpected_keys", []))

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        scenario: str = "",
        task: str | None = None,
        return_info: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        image = image.to(self.device)
        selected_task = demoe_task_from_scenario(scenario, task or self.default_task)
        outputs = self.model(image, task=selected_task)
        restored = torch.clamp(outputs["output"], 0.0, 1.0)
        if not return_info:
            return restored
        return {
            "output": restored,
            "task": selected_task,
            "router_weights": outputs.get("pred_labels"),
            "expert_bin_counts": outputs.get("bin_counts"),
            "load_report": self.load_report,
        }

    @property
    def backend_status(self) -> str:
        if self.device.type == "cuda":
            return "GPU-confirmed-smoke" if self.smoke else "GPU-confirmed"
        return "CPU-forced"
