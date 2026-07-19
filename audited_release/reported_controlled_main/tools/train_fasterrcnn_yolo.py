# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

"""Train/evaluate a Faster R-CNN detector on a YOLO-format road dataset.

This script is the non-YOLO detector-family protocol requested by review. It is
not used to create manuscript numbers unless it is actually run and its outputs
are added to the provenance table. The implementation keeps the same train/val
/test split references as the YOLO data.yaml files used elsewhere.

Example:
python tools/train_fasterrcnn_yolo.py ^
  --data datasets/road_damage_pcm_yolo/data.yaml ^
  --out runs/fasterrcnn_pcm_clean ^
  --epochs 20 --batch-size 2 --device cuda --pretrained ^
  --eval-data datasets/pcm_yolo_defocus_test/data.yaml ^
  --eval-data datasets/pcm_yolo_defocus_test_rmrnet_v27_ep002/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_split(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_data_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    cfg["_root"] = root
    cfg["_path"] = str(path)
    return cfg


def find_images(path: Path) -> list[Path]:
    if path.is_file():
        return [Path(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


class YoloDetectionDataset(Dataset):
    def __init__(self, image_dir: Path, nc: int):
        self.images = find_images(image_dir)
        self.nc = nc
        if not self.images:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, image_path: Path) -> Path:
        parts = list(image_path.parts)
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes: list[list[float]] = []
        labels: list[int] = []
        label_path = self._label_path(image_path)
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.strip().split()
                if len(fields) < 5:
                    continue
                cls = int(float(fields[0]))
                cx, cy, bw, bh = map(float, fields[1:5])
                x1 = max(0.0, (cx - bw / 2.0) * width)
                y1 = max(0.0, (cy - bh / 2.0) * height)
                x2 = min(float(width), (cx + bw / 2.0) * width)
                y2 = min(float(height), (cy + bh / 2.0) * height)
                if x2 > x1 and y2 > y1 and 0 <= cls < self.nc:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls + 1)  # Faster R-CNN reserves 0 for background.
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return F.to_tensor(image), target, str(image_path)


def collate(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    lt = torch.maximum(a[:, None, :2], b[:, :2])
    rb = torch.minimum(a[:, None, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a[:, None] + area_b - inter + 1e-7)


def average_precision(tp: list[int], fp: list[int], n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if not tp:
        return 0.0
    tp_t = torch.tensor(tp, dtype=torch.float32).cumsum(0)
    fp_t = torch.tensor(fp, dtype=torch.float32).cumsum(0)
    recall = tp_t / max(1, n_gt)
    precision = tp_t / torch.clamp(tp_t + fp_t, min=1.0)
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    precision = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])
    for i in range(precision.numel() - 2, -1, -1):
        precision[i] = torch.maximum(precision[i], precision[i + 1])
    idx = torch.where(recall[1:] != recall[:-1])[0]
    return float(((recall[idx + 1] - recall[idx]) * precision[idx + 1]).sum())


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, nc: int, score_thr: float) -> dict[str, Any]:
    model.eval()
    detections = {c: [] for c in range(1, nc + 1)}
    gt_count = {c: 0 for c in range(1, nc + 1)}
    gt_seen: dict[tuple[int, int], set[int]] = {}
    gt_boxes_by_image_class: dict[tuple[int, int], torch.Tensor] = {}
    for images, targets, _paths in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for batch_idx, (out, target) in enumerate(zip(outputs, targets)):
            image_id = int(target["image_id"][0])
            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            for c in range(1, nc + 1):
                class_boxes = gt_boxes[gt_labels == c]
                gt_count[c] += int(class_boxes.shape[0])
                gt_boxes_by_image_class[(image_id, c)] = class_boxes
                gt_seen[(image_id, c)] = set()
            keep = out["scores"] >= score_thr
            boxes = out["boxes"][keep]
            labels = out["labels"][keep]
            scores = out["scores"][keep]
            for box, label, score in zip(boxes, labels, scores):
                c = int(label)
                if 1 <= c <= nc:
                    detections[c].append((float(score), image_id, box.detach()))

    ap = {}
    for c in range(1, nc + 1):
        tp, fp = [], []
        for _score, image_id, box in sorted(detections[c], key=lambda x: x[0], reverse=True):
            gt_boxes = gt_boxes_by_image_class.get((image_id, c), torch.empty((0, 4), device=device))
            if gt_boxes.numel() == 0:
                fp.append(1)
                tp.append(0)
                continue
            ious = box_iou(box[None, :], gt_boxes).squeeze(0)
            best_iou, best_idx = torch.max(ious, dim=0)
            best_int = int(best_idx)
            if float(best_iou) >= 0.5 and best_int not in gt_seen[(image_id, c)]:
                tp.append(1)
                fp.append(0)
                gt_seen[(image_id, c)].add(best_int)
            else:
                fp.append(1)
                tp.append(0)
        ap[c] = average_precision(tp, fp, gt_count[c])
    valid = [v for v in ap.values() if not math.isnan(v)]
    return {"ap50_per_class": ap, "map50": float(sum(valid) / len(valid)) if valid else float("nan")}


def make_loader(cfg: dict[str, Any], split: str, batch_size: int, num_workers: int, nc: int, shuffle: bool = False) -> DataLoader:
    root = cfg["_root"]
    if split not in cfg:
        fallback = "val" if split == "test" and "val" in cfg else "train"
        split = fallback
    image_dir = resolve_split(root, cfg[split])
    return DataLoader(
        YoloDetectionDataset(image_dir, nc),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


def build_model(num_classes: int, pretrained: bool):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None if not pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision COCO weights; may download if absent.")
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--eval-data", action="append", default=[], type=Path, help="Additional YOLO data.yaml files to evaluate with the best clean-trained checkpoint.")
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = read_data_yaml(args.data)
    root = cfg["_root"]
    names = cfg.get("names", [])
    nc = int(cfg.get("nc", len(names)))
    if nc <= 0:
        raise ValueError("data.yaml must define nc or names.")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(2026)

    train_loader = make_loader(cfg, "train", args.batch_size, args.num_workers, nc, shuffle=True)
    val_loader = make_loader(cfg, "val", 1, args.num_workers, nc)
    test_loader = make_loader(cfg, "test", 1, args.num_workers, nc)

    model = build_model(nc + 1, args.pretrained).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    history = []
    best = {"epoch": -1, "val_map50": -1.0, "path": ""}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets, _paths in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if k != "image_id"} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, val_loader, device, nc, args.score_thr)
        ckpt = args.out / f"fasterrcnn_epoch_{epoch:03d}.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics, "names": names}, ckpt)
        row = {"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)), **val_metrics}
        history.append(row)
        if val_metrics["map50"] > best["val_map50"]:
            best = {"epoch": epoch, "val_map50": val_metrics["map50"], "path": str(ckpt)}

    if best["path"]:
        state = torch.load(best["path"], map_location=device)
        model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device, nc, args.score_thr)
    extra_evals: list[dict[str, Any]] = []
    for eval_yaml in args.eval_data:
        eval_cfg = read_data_yaml(eval_yaml)
        eval_names = eval_cfg.get("names", names)
        eval_nc = int(eval_cfg.get("nc", len(eval_names)))
        if eval_nc != nc:
            raise ValueError(f"{eval_yaml} has nc={eval_nc}, expected {nc} from training data.")
        eval_loader = make_loader(eval_cfg, args.eval_split, 1, args.num_workers, nc)
        metrics = evaluate(model, eval_loader, device, nc, args.score_thr)
        extra_evals.append({
            "data": str(eval_yaml),
            "split": args.eval_split,
            "metrics": metrics,
        })

    with (args.out / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "detector": "torchvision_fasterrcnn_mobilenet_v3_large_fpn",
        "data": str(args.data),
        "best_by_val_map50": best,
        "test_metrics": test_metrics,
        "extra_evals": extra_evals,
        "claim_boundary": "Protocol script only until this run is executed and added to provenance.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
