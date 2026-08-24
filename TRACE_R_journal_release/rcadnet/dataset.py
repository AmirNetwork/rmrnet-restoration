# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .practical_metadata import PRACTICAL_SENSOR_DIM, sensor_packet_from_mapping
from .scenario_codes import code_from_metadata, code_from_scenario


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def metadata_for_mode(metadata: dict, mode: str) -> dict:
    """Return metadata with privileged fields removed for stricter audits."""

    if mode == "full":
        return metadata
    if mode in {"zero", "missing"}:
        return {}
    filtered = dict(metadata)
    if mode in {"raw_telemetry", "raw_scalar"}:
        # KITTI sidecars can carry two explicitly audited 82-value packets:
        # a raw OXTS packet and a generator-aligned upper-bound packet. Raw
        # telemetry must replace, not merely accompany, the upper-bound packet.
        raw_packet = filtered.get("raw_practical_sensor_packet")
        if raw_packet is not None:
            filtered["practical_sensor_packet"] = raw_packet
            filtered["sensor_packet_variant"] = "raw_oxts"
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
        metadata_encoding: str = "legacy",
        horizontal_flip_probability: float = 0.5,
        defect_label_root: str | Path | None = None,
        defect_crop_probability: float = 0.0,
        max_detector_boxes: int = 64,
    ) -> None:
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.train = train
        self.metadata_mode = metadata_mode
        self.metadata_encoding = metadata_encoding
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.defect_label_root = (
            Path(defect_label_root) if defect_label_root is not None else None
        )
        self.defect_crop_probability = float(defect_crop_probability)
        self.max_detector_boxes = int(max_detector_boxes)
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if not 0.0 <= self.defect_crop_probability <= 1.0:
            raise ValueError("defect_crop_probability must be in [0, 1]")
        if self.max_detector_boxes <= 0:
            raise ValueError("max_detector_boxes must be greater than zero")
        if (
            self.defect_crop_probability > 0.0
            and (
                self.defect_label_root is None
                or not self.defect_label_root.exists()
            )
        ):
            raise FileNotFoundError(
                "Defect-aware crop sampling requires an existing YOLO label "
                f"directory, got {self.defect_label_root}"
            )
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

    def _load_detector_boxes(
        self,
        input_path: Path,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Read training-only YOLO labels as native-pixel ``class, xyxy`` boxes."""

        if self.defect_label_root is None:
            return torch.zeros((0, 5), dtype=torch.float32)
        label_path = self.defect_label_root / f"{input_path.stem}.txt"
        if not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)
        boxes: list[list[float]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = float(parts[0])
                center_x = float(parts[1]) * width
                center_y = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height
            except ValueError:
                continue
            boxes.append(
                [
                    class_id,
                    center_x - box_width / 2.0,
                    center_y - box_height / 2.0,
                    center_x + box_width / 2.0,
                    center_y + box_height / 2.0,
                ]
            )
        if not boxes:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    @staticmethod
    def _box_centers(boxes: torch.Tensor) -> list[tuple[float, float]]:
        if boxes.numel() == 0:
            return []
        center_x = 0.5 * (boxes[:, 1] + boxes[:, 3])
        center_y = 0.5 * (boxes[:, 2] + boxes[:, 4])
        return list(zip(center_y.tolist(), center_x.tolist()))

    def _crop_pair(
        self,
        input_tensor: torch.Tensor,
        gt_tensor: torch.Tensor,
        input_path: Path,
        detector_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, height, width = input_tensor.shape
        if height < self.patch_size or width < self.patch_size:
            scale = self.patch_size / min(height, width)
            new_size = (round(height * scale), round(width * scale))
            input_tensor = TF.resize(input_tensor, new_size, antialias=True)
            gt_tensor = TF.resize(gt_tensor, new_size, antialias=True)
            _, new_height, new_width = input_tensor.shape
            if detector_boxes.numel() > 0:
                detector_boxes = detector_boxes.clone()
                detector_boxes[:, [1, 3]] *= float(new_width) / float(width)
                detector_boxes[:, [2, 4]] *= float(new_height) / float(height)
            height, width = new_height, new_width

        size = self.patch_size
        if self.train and height > size and width > size:
            defect_centers = self._box_centers(detector_boxes)
            use_defect = (
                bool(defect_centers)
                and torch.rand(()) < self.defect_crop_probability
            )
            if use_defect:
                center_y, center_x = defect_centers[
                    torch.randint(0, len(defect_centers), ()).item()
                ]
                # Modest jitter prevents the detector-guided crops from always
                # centering the same object while keeping it inside the patch.
                jitter = 0.15 * size
                center_y += float((torch.rand(()) * 2.0 - 1.0) * jitter)
                center_x += float((torch.rand(()) * 2.0 - 1.0) * jitter)
                top = round(center_y - size / 2)
                left = round(center_x - size / 2)
                top = min(max(top, 0), height - size)
                left = min(max(left, 0), width - size)
            else:
                top = torch.randint(0, height - size + 1, ()).item()
                left = torch.randint(0, width - size + 1, ()).item()
        else:
            top = max((height - size) // 2, 0)
            left = max((width - size) // 2, 0)

        cropped_boxes = torch.zeros((0, 5), dtype=torch.float32)
        if detector_boxes.numel() > 0:
            boxes = detector_boxes.clone()
            original_width = (boxes[:, 3] - boxes[:, 1]).clamp_min(1e-6)
            original_height = (boxes[:, 4] - boxes[:, 2]).clamp_min(1e-6)
            original_area = original_width * original_height
            center_x = 0.5 * (boxes[:, 1] + boxes[:, 3])
            center_y = 0.5 * (boxes[:, 2] + boxes[:, 4])

            boxes[:, [1, 3]] -= float(left)
            boxes[:, [2, 4]] -= float(top)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0.0, float(size))
            boxes[:, [2, 4]] = boxes[:, [2, 4]].clamp(0.0, float(size))

            clipped_width = boxes[:, 3] - boxes[:, 1]
            clipped_height = boxes[:, 4] - boxes[:, 2]
            retained_area = clipped_width.clamp_min(0.0) * clipped_height.clamp_min(0.0)
            center_inside = (
                (center_x >= left)
                & (center_x <= left + size)
                & (center_y >= top)
                & (center_y <= top + size)
            )
            keep = (
                center_inside
                & (clipped_width >= 2.0)
                & (clipped_height >= 2.0)
                & (retained_area / original_area >= 0.20)
            )
            cropped_boxes = boxes[keep]

        return (
            input_tensor[:, top : top + size, left : left + size],
            gt_tensor[:, top : top + size, left : left + size],
            cropped_boxes,
        )

    def _pad_detector_targets(
        self,
        boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert cropped ``class, xyxy`` boxes to padded normalized YOLO targets."""

        classes = torch.full(
            (self.max_detector_boxes,),
            -1,
            dtype=torch.long,
        )
        normalized = torch.zeros(
            (self.max_detector_boxes, 4),
            dtype=torch.float32,
        )
        valid = torch.zeros(
            (self.max_detector_boxes,),
            dtype=torch.bool,
        )
        if boxes.numel() == 0:
            return classes, normalized, valid

        size = float(self.patch_size)
        box_width = (boxes[:, 3] - boxes[:, 1]).clamp_min(0.0)
        box_height = (boxes[:, 4] - boxes[:, 2]).clamp_min(0.0)
        area = box_width * box_height
        order = torch.argsort(area, descending=True)[: self.max_detector_boxes]
        boxes = boxes[order]
        count = boxes.shape[0]

        classes[:count] = boxes[:, 0].to(torch.long)
        normalized[:count, 0] = 0.5 * (boxes[:, 1] + boxes[:, 3]) / size
        normalized[:count, 1] = 0.5 * (boxes[:, 2] + boxes[:, 4]) / size
        normalized[:count, 2] = (boxes[:, 3] - boxes[:, 1]) / size
        normalized[:count, 3] = (boxes[:, 4] - boxes[:, 2]) / size
        normalized[:count] = normalized[:count].clamp(0.0, 1.0)
        valid[:count] = True
        return classes, normalized, valid

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        input_path, gt_path, scenario = self.samples[index]
        input_tensor = self._load_rgb(input_path)
        gt_tensor = self._load_rgb(gt_path)
        detector_boxes = self._load_detector_boxes(
            input_path,
            input_tensor.shape[1],
            input_tensor.shape[2],
        )
        input_tensor, gt_tensor, detector_boxes = self._crop_pair(
            input_tensor,
            gt_tensor,
            input_path,
            detector_boxes,
        )

        # A horizontal image flip changes a directional blur kernel. Unless the
        # camera/IMU coordinate frame and every directional supervision target
        # are transformed consistently, applying the flip would pair one
        # telemetry packet with two contradictory PSFs. Metadata-conditioned
        # training therefore passes probability=0 from train_rcadnet.py.
        if (
            self.train
            and self.horizontal_flip_probability > 0.0
            and torch.rand(()) < self.horizontal_flip_probability
        ):
            input_tensor = torch.flip(input_tensor, dims=(2,))
            gt_tensor = torch.flip(gt_tensor, dims=(2,))
            if detector_boxes.numel() > 0:
                detector_boxes = detector_boxes.clone()
                old_x1 = detector_boxes[:, 1].clone()
                old_x2 = detector_boxes[:, 3].clone()
                detector_boxes[:, 1] = float(self.patch_size) - old_x2
                detector_boxes[:, 3] = float(self.patch_size) - old_x1

        detector_classes, detector_bboxes, detector_valid = (
            self._pad_detector_targets(detector_boxes)
        )

        scenario_code = code_from_scenario(scenario)
        cause_target = scenario_code
        physical_target = torch.zeros_like(scenario_code)
        physical_target_available = torch.tensor(0.0, dtype=torch.float32)
        metadata_source_path = input_path
        if self.metadata_mode == "shuffled" and len(self.samples) > 1:
            # Deterministic metadata shuffling for reviewer audits:
            # condition the image on another sample's metadata without changing
            # the image/target pair. This tests whether metadata alignment
            # matters rather than merely providing a dataset-level prior.
            shuffled_index = (index + max(1, len(self.samples) // 2)) % len(self.samples)
            metadata_source_path = self.samples[shuffled_index][0]
        metadata_path = metadata_source_path.parent.parent / "metadata" / f"{metadata_source_path.stem}.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = metadata_for_mode(metadata, self.metadata_mode)
            if metadata:
                if "practical_sensor_packet" in metadata or "sensor_packet" in metadata:
                    metadata_code = sensor_packet_from_mapping(metadata)
                    # Calibration labels are intentionally stored outside the
                    # public inference sidecar. They supervise the sensor
                    # encoder during training but are never passed to RMR-Net.
                    private_target_path = (
                        input_path.parent.parent
                        / "private_calibration"
                        / f"{input_path.stem}.json"
                    )
                    target = None
                    physical = None
                    if private_target_path.exists():
                        private_target = json.loads(
                            private_target_path.read_text(encoding="utf-8")
                        )
                        target = private_target.get("training_cause_target_code")
                        physical = private_target.get("training_physical_target_code")
                    if target is not None:
                        cause_target = torch.tensor(target, dtype=torch.float32)
                    if physical is not None:
                        physical_target = torch.tensor(physical, dtype=torch.float32)
                        physical_target_available = torch.tensor(1.0, dtype=torch.float32)
                else:
                    metadata_code = code_from_metadata(metadata, encoding=self.metadata_encoding)
                    cause_target = metadata_code
            else:
                # Practical-metadata datasets keep the packet dimension even
                # for unavailable/zero controls so DataLoader collation and the
                # in-network sensor encoder remain well defined.
                public_path = input_path.parent.parent / "metadata" / f"{input_path.stem}.json"
                if public_path.exists():
                    public = json.loads(public_path.read_text(encoding="utf-8"))
                    practical = "practical_sensor_packet" in public or "sensor_packet" in public
                else:
                    practical = False
                metadata_code = (
                    torch.zeros(PRACTICAL_SENSOR_DIM, dtype=torch.float32)
                    if practical
                    else torch.zeros_like(scenario_code)
                )
        else:
            metadata_code = scenario_code

        return {
            "input": input_tensor,
            "gt": gt_tensor,
            "code": scenario_code,
            "metadata_code": metadata_code,
            "cause_target": cause_target,
            # Private renderer parameters are calibration labels only. They are
            # consumed by the training loss and are never passed to RMR-Net.
            "physical_target": physical_target,
            "physical_target_available": physical_target_available,
            # Training-only detector targets. They are transformed with the
            # exact resize/crop/flip applied above and never enter inference.
            "detector_classes": detector_classes,
            "detector_bboxes": detector_bboxes,
            "detector_valid": detector_valid,
            "scenario": scenario,
            "name": input_path.name,
        }
