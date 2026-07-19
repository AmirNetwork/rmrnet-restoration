# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .scenario_codes import code_from_metadata, code_from_scenario


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def metadata_for_mode(metadata: dict, mode: str) -> dict:
    """Return metadata with privileged fields removed for stricter audits."""

    if mode == "full":
        return metadata
    filtered = dict(metadata)
    if mode in {"raw_telemetry", "raw_scalar"}:
        for key in ("blur_length_px", "blur_angle_deg", "telemetry_strength", "blur_scale"):
            filtered.pop(key, None)
    if mode == "raw_scalar":
        for key in ("raw_oxts_yaw_rate_radps", "raw_oxts_lateral_accel_mps2", "raw_oxts_forward_accel_mps2"):
            filtered.pop(key, None)
    return filtered


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


class PairedRoadRestorationDataset(Dataset):
    """Loads benchmark folders: scenarios/<scenario>/input and /gt."""

    def __init__(
        self,
        data_root: str | Path,
        scenarios: Sequence[str],
        patch_size: int = 256,
        train: bool = True,
        metadata_mode: str = "full",
    ) -> None:
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.train = train
        self.metadata_mode = metadata_mode
        self.samples: list[tuple[Path, Path, str]] = []

        for scenario in scenarios:
            input_dir = self.data_root / "scenarios" / scenario / "input"
            gt_dir = self.data_root / "scenarios" / scenario / "gt"
            if not input_dir.exists() or not gt_dir.exists():
                raise FileNotFoundError(f"Missing scenario folders for {scenario}: {input_dir} / {gt_dir}")
            for input_path in list_images(input_dir):
                gt_path = gt_dir / input_path.name
                if gt_path.exists():
                    self.samples.append((input_path, gt_path, scenario))

        if not self.samples:
            raise RuntimeError(f"No paired images found under {self.data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            return TF.to_tensor(image.convert("RGB"))

    def _crop_pair(self, input_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, height, width = input_tensor.shape
        if height < self.patch_size or width < self.patch_size:
            scale = self.patch_size / min(height, width)
            new_size = (round(height * scale), round(width * scale))
            input_tensor = TF.resize(input_tensor, new_size, antialias=True)
            gt_tensor = TF.resize(gt_tensor, new_size, antialias=True)
            _, height, width = input_tensor.shape

        size = self.patch_size
        if self.train and height > size and width > size:
            top = torch.randint(0, height - size + 1, ()).item()
            left = torch.randint(0, width - size + 1, ()).item()
        else:
            top = max((height - size) // 2, 0)
            left = max((width - size) // 2, 0)
        return (
            input_tensor[:, top : top + size, left : left + size],
            gt_tensor[:, top : top + size, left : left + size],
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        input_path, gt_path, scenario = self.samples[index]
        input_tensor = self._load_rgb(input_path)
        gt_tensor = self._load_rgb(gt_path)
        input_tensor, gt_tensor = self._crop_pair(input_tensor, gt_tensor)

        if self.train and torch.rand(()) < 0.5:
            input_tensor = torch.flip(input_tensor, dims=(2,))
            gt_tensor = torch.flip(gt_tensor, dims=(2,))

        scenario_code = code_from_scenario(scenario)
        metadata_path = input_path.parent.parent / "metadata" / f"{input_path.stem}.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = metadata_for_mode(metadata, self.metadata_mode)
            metadata_code = code_from_metadata(metadata)
        else:
            metadata_code = scenario_code

        return {
            "input": input_tensor,
            "gt": gt_tensor,
            "code": scenario_code,
            "metadata_code": metadata_code,
            "scenario": scenario,
            "name": input_path.name,
        }
