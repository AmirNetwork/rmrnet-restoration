from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "revised_loss_ablation"
RUN_ROOT = ROOT / "runs" / "revised_loss_ablation"
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

SCENARIO_KEY = "defocus"
SCENARIO_NAME = "defocus_medium"
TRAIN_ROOT = ROOT / "data" / "pcm_restoration_train"
VAL_ROOT = ROOT / "data" / "pcm_restoration_val"
TEST_ROOT = ROOT / "data" / "pcm_restoration_test"
VAL_YOLO = ROOT / "datasets" / "pcm_yolo_defocus_val" / "data.yaml"
TEST_YOLO = ROOT / "datasets" / "pcm_yolo_defocus_test" / "data.yaml"
PCM_DETECTOR = ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pcm_clean_80ep" / "weights" / "best.pt"


@dataclass(frozen=True)
class Variant:
    tag: str
    label: str
    flags: list[str] = field(default_factory=list)


VARIANTS: list[Variant] = [
    Variant(
        "base_only",
        "Base only",
        [
            "--code-source",
            "zero",
            "--no-defect-attention",
            "--block-type",
            "simple",
            "--attention-type",
            "none",
            "--conditioning",
            "film",
            "--aux-code-weight",
            "0",
        ],
    ),
    Variant(
        "code_supervision",
        "+ code supervision",
        [
            "--code-source",
            "metadata_fused",
            "--no-defect-attention",
            "--block-type",
            "simple",
            "--attention-type",
            "none",
            "--conditioning",
            "gated_basis",
            "--basis-sparsity-weight",
            "0",
            "--aux-code-weight",
            "0.05",
            "--metadata-dropout",
            "0.10",
            "--metadata-noise",
            "0.01",
        ],
    ),
    Variant(
        "attention",
        "+ guided attention",
        [
            "--code-source",
            "metadata_fused",
            "--block-type",
            "evidence",
            "--attention-type",
            "task",
            "--conditioning",
            "gated_basis",
            "--basis-sparsity-weight",
            "0",
            "--aux-code-weight",
            "0.05",
            "--metadata-dropout",
            "0.10",
            "--metadata-noise",
            "0.01",
        ],
    ),
    Variant(
        "detail_skip",
        "+ degradation-aware detail skip",
        [
            "--code-source",
            "metadata_fused",
            "--block-type",
            "evidence",
            "--attention-type",
            "task",
            "--conditioning",
            "gated_basis",
            "--basis-sparsity-weight",
            "0",
            "--aux-code-weight",
            "0.05",
            "--metadata-dropout",
            "0.10",
            "--metadata-noise",
            "0.01",
            "--detail-preserve",
            "--detail-gain",
            "0.12",
        ],
    ),
    Variant(
        "tdp",
        "+ defect-weighted TDP",
        [
            "--code-source",
            "metadata_fused",
            "--block-type",
            "evidence",
            "--attention-type",
            "task",
            "--conditioning",
            "gated_basis",
            "--basis-sparsity-weight",
            "0",
            "--aux-code-weight",
            "0.05",
            "--metadata-dropout",
            "0.10",
            "--metadata-noise",
            "0.01",
            "--detail-preserve",
            "--detail-gain",
            "0.12",
            "--use-task-losses",
            "--tdp-yolo-weights",
            str(PCM_DETECTOR),
            "--tdp-layers",
            "2,4",
            "--tdp-layer-weights",
            "0.5,1",
            "--detector-input-size",
            "256",
            "--lambda-tdp",
            "0.001",
            "--tdp-defect-mask-weight",
            "4.0",
            "--tdp-defect-mask-power",
            "1.5",
            "--lambda-jacobian",
            "0",
            "--lambda-active-contour",
            "0",
            "--lambda-detector-input-anchor",
            "0",
            "--lambda-evidence-nonregression",
            "0",
            "--cqmix-prob",
            "0",
            "--task-loss-warmup-epochs",
            "2",
        ],
    ),
    Variant(
        "tdp_cqmix",
        "+ TDP + CQMix",
        [
            "--code-source",
            "metadata_fused",
            "--block-type",
            "evidence",
            "--attention-type",
            "task",
            "--conditioning",
            "gated_basis",
            "--basis-sparsity-weight",
            "0",
            "--aux-code-weight",
            "0.05",
            "--metadata-dropout",
            "0.10",
            "--metadata-noise",
            "0.01",
            "--detail-preserve",
            "--detail-gain",
            "0.12",
            "--use-task-losses",
            "--tdp-yolo-weights",
            str(PCM_DETECTOR),
            "--tdp-layers",
            "2,4",
            "--tdp-layer-weights",
            "0.5,1",
            "--detector-input-size",
            "256",
            "--lambda-tdp",
            "0.001",
            "--tdp-defect-mask-weight",
            "4.0",
            "--tdp-defect-mask-power",
            "1.5",
            "--lambda-jacobian",
            "0",
            "--lambda-active-contour",
            "0",
            "--lambda-detector-input-anchor",
            "0",
            "--lambda-evidence-nonregression",
            "0",
            "--cqmix-prob",
            "0.5",
            "--task-loss-warmup-epochs",
            "2",
        ],
    ),
    Variant(
        "jacobian",
        "+ Jacobian stability",
        [
            "--code-source",
            "metadata_fused",
            "--block-type",
            "evidence",
            "--attention-type",
            "task",
            "--conditioning",
            "gated_basis",
            "--basis-sparsity-weight",
            "0",
            "--aux-code-weight",
            "0.05",
            "--metadata-dropout",
            "0.10",
            "--metadata-noise",
            "0.01",
            "--detail-preserve",
            "--detail-gain",
            "0.12",
            "--use-task-losses",
            "--tdp-yolo-weights",
            str(PCM_DETECTOR),
            "--tdp-layers",
            "2,4",
            "--tdp-layer-weights",
            "0.5,1",
            "--detector-input-size",
            "256",
            "--lambda-tdp",
            "0.001",
            "--tdp-defect-mask-weight",
            "4.0",
            "--tdp-defect-mask-power",
            "1.5",
            "--lambda-jacobian",
            "0.00002",
            "--jacobian-probes",
            "1",
            "--lambda-active-contour",
            "0",
            "--lambda-detector-input-anchor",
            "0",
            "--lambda-evidence-nonregression",
            "0",
            "--cqmix-prob",
            "0.5",
            "--task-loss-warmup-epochs",
            "2",
        ],
    ),
    Variant(
        "full_model",
        "full model",
        [
            "--code-source",
            "metadata_fused",
            "--block-type",
            "evidence",
            "--attention-type",
            "task",
            "--conditioning",
            "gated_basis",
            "--basis-sparsity-weight",
            "1.0",
            "--aux-code-weight",
            "0.05",
            "--metadata-dropout",
            "0.10",
            "--metadata-noise",
            "0.01",
            "--detail-preserve",
            "--detail-gain",
            "0.12",
            "--use-task-losses",
            "--tdp-yolo-weights",
            str(PCM_DETECTOR),
            "--tdp-layers",
            "2,4",
            "--tdp-layer-weights",
            "0.5,1",
            "--detector-input-size",
            "256",
            "--lambda-tdp",
            "0.001",
            "--tdp-defect-mask-weight",
            "4.0",
            "--tdp-defect-mask-power",
            "1.5",
            "--lambda-jacobian",
            "0.00002",
            "--jacobian-probes",
            "1",
            "--lambda-active-contour",
            "0",
            "--lambda-detector-input-anchor",
            "0.0005",
            "--lambda-evidence-nonregression",
            "0.02",
            "--lambda-detail-copy",
            "0.002",
            "--cqmix-prob",
            "0.5",
            "--task-loss-warmup-epochs",
            "2",
        ],
    ),
]


def run(cmd: list[str], *, log_path: Path | None = None) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            safe_line = line.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            try:
                print(safe_line, end="")
            except UnicodeEncodeError:
                print(safe_line.encode("ascii", errors="replace").decode("ascii"), end="")
            handle.write(line)
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_yaml_images(data_yaml: Path, split: str) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    value = Path(data.get(split, data.get("test", data.get("val", f"images/{split}"))))
    return value if value.is_absolute() else root / value


def train_variant(args: argparse.Namespace, variant: Variant, init: Path | None = None) -> Path:
    run_dir = RUN_ROOT / variant.tag
    if args.skip_train and (run_dir / f"rcadnet_epoch_{args.epochs:03d}.pth").exists():
        return run_dir
    cmd = [
        sys.executable,
        "train_rcadnet.py",
        "--data-root",
        str(TRAIN_ROOT),
        "--scenario",
        SCENARIO_NAME,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--patch-size",
        str(args.patch_size),
        "--lr",
        str(args.lr),
        "--width",
        str(args.width),
        "--device",
        args.device,
        "--out",
        str(run_dir),
        "--num-workers",
        str(args.num_workers),
        "--val-data-root",
        str(VAL_ROOT),
        "--val-scenario",
        SCENARIO_NAME,
        "--val-every",
        "1",
        "--save-every-epoch",
        "--seed",
        str(args.seed),
        "--edge-weight",
        "0.15",
        "--freq-weight",
        "0.05",
        "--defect-weight",
        "0.10",
        "--lambda-active-contour",
        "0",
        "--debug-first-batches",
        "0",
    ]
    if args.amp:
        cmd.append("--amp")
    if init is not None:
        cmd.extend(["--init-weights", str(init)])
    cmd.extend(variant.flags)
    run(cmd, log_path=EXP / "logs" / f"train_{variant.tag}.log")
    return run_dir


def restore_split(run_dir: Path, checkpoint: Path, variant: Variant, split: str) -> Path:
    source_yaml = VAL_YOLO if split == "val" else TEST_YOLO
    out = EXP / "restored" / split / variant.tag / checkpoint.stem
    yaml_path = out / "data.yaml"
    if yaml_path.exists():
        return yaml_path
    run(
        [
            sys.executable,
            "tools/restore_yolo_split.py",
            "--data",
            str(source_yaml),
            "--split",
            split,
            "--model",
            "rcadnet",
            "--scenario",
            SCENARIO_NAME,
            "--out",
            str(out),
            "--device",
            "cuda" if torch_cuda_requested() else "cpu",
            "--rcadnet-weights",
            str(checkpoint),
            "--rcadnet-code-source",
            "metadata",
        ],
        log_path=EXP / "logs" / f"restore_{variant.tag}_{split}_{checkpoint.stem}.log",
    )
    return yaml_path


def torch_cuda_requested() -> bool:
    return True


def eval_map(items: list[tuple[str, Path]], out_stem: Path, split: str, batch: int) -> Path:
    cmd = [
        sys.executable,
        "tools/eval_yolo_suite.py",
        "--weights",
        str(PCM_DETECTOR),
        "--imgsz",
        "640",
        "--batch",
        str(batch),
        "--device",
        "0",
        "--workers",
        "0",
        "--split",
        split,
        "--project",
            str(ROOT / "runs" / "detection_eval_revised_loss_ablation"),
        "--out",
        str(out_stem),
    ]
    for name, path in items:
        cmd.extend(["--item", f"{name}={path}"])
    run(cmd, log_path=EXP / "logs" / f"eval_{out_stem.name}.log")
    return out_stem.with_suffix(".csv")


def eval_per_class(name: str, data_yaml: Path, out_stem: Path, split: str, batch: int) -> Path:
    run(
        [
            sys.executable,
            "tools/eval_yolo_per_class_suite.py",
            "--weights",
            str(PCM_DETECTOR),
            "--item",
            f"{name}={data_yaml}",
            "--imgsz",
            "640",
            "--batch",
            str(batch),
            "--device",
            "0",
            "--workers",
            "0",
            "--split",
            split,
            "--project",
            str(ROOT / "runs" / "detection_eval_revised_loss_ablation_per_class"),
            "--out",
            str(out_stem),
        ],
        log_path=EXP / "logs" / f"per_class_{name}.log",
    )
    return out_stem.with_suffix(".csv")


def psnr_for_split(restored_yaml: Path) -> float:
    image_dir = resolve_yaml_images(restored_yaml, "test")
    gt_dir = TEST_ROOT / "scenarios" / SCENARIO_NAME / "gt"
    values: list[float] = []
    for restored_path in sorted(image_dir.iterdir()):
        if restored_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        gt_path = gt_dir / restored_path.name
        if not gt_path.exists():
            continue
        with Image.open(restored_path) as pred_image, Image.open(gt_path) as gt_image:
            pred = np.asarray(pred_image.convert("RGB"), dtype=np.float32) / 255.0
            gt = np.asarray(gt_image.convert("RGB"), dtype=np.float32) / 255.0
        if pred.shape != gt.shape:
            with Image.open(restored_path) as pred_image:
                pred = np.asarray(pred_image.convert("RGB").resize((gt.shape[1], gt.shape[0]), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
        mse = float(np.mean((pred - gt) ** 2))
        values.append(99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse))
    if not values:
        raise RuntimeError(f"No PSNR pairs found for {restored_yaml}")
    return float(np.mean(values))


def boundary_iou(name: str, data_yaml: Path) -> float:
    out = EXP / "boundary" / name
    run(
        [
            sys.executable,
            "tools/snake_boundary_metrics.py",
            "--data",
            str(data_yaml),
            "--split",
            "test",
            "--out",
            str(out),
            "--box-dir",
            str(ROOT / "datasets" / "pcm_yolo_defocus_test" / "labels" / "test"),
            "--gt-polygon-dir",
            str(ROOT / "datasets" / "pcm_yolo_defocus_test" / "labels" / "test"),
            "--classes",
            "all",
            "--max-images",
            "0",
            "--max-overlays",
            "8",
        ],
        log_path=EXP / "logs" / f"snake_{name}.log",
    )
    summary = json.loads((out / "snake_boundary_summary.json").read_text(encoding="utf-8"))
    class_rows = summary.get("classes", {})
    weighted_sum = 0.0
    objects = 0
    for stats in class_rows.values():
        n = int(stats.get("objects", 0))
        if "mean_gt_iou" in stats:
            weighted_sum += n * float(stats["mean_gt_iou"])
            objects += n
    return weighted_sum / objects if objects else 0.0


def choose_epoch(args: argparse.Namespace, variant: Variant, run_dir: Path) -> tuple[int, Path, float]:
    val_items: list[tuple[str, Path]] = []
    for epoch in range(1, args.epochs + 1):
        ckpt = run_dir / f"rcadnet_epoch_{epoch:03d}.pth"
        yaml_path = restore_split(run_dir, ckpt, variant, "val")
        val_items.append((f"{variant.tag}_ep{epoch:03d}", yaml_path))
    val_csv = eval_map(val_items, EXP / "validation" / f"{variant.tag}_epochs", "val", args.eval_batch)
    rows = read_csv(val_csv)
    best = max(rows, key=lambda row: (float(row["map50"]), -int(row["name"].split("ep")[-1])))
    epoch = int(best["name"].split("ep")[-1])
    return epoch, run_dir / f"rcadnet_epoch_{epoch:03d}.pth", float(best["map50"])


def summarize_variant(args: argparse.Namespace, variant: Variant, checkpoint: Path, selected_epoch: int, val_map50: float) -> dict[str, Any]:
    test_yaml = restore_split(checkpoint.parent, checkpoint, variant, "test")
    map_csv = eval_map([(variant.tag, test_yaml)], EXP / "test" / f"{variant.tag}_map", "test", args.eval_batch)
    map_row = read_csv(map_csv)[0]
    per_class_csv = eval_per_class(variant.tag, test_yaml, EXP / "test" / f"{variant.tag}_per_class", "test", args.eval_batch)
    crack_ap = 0.0
    for row in read_csv(per_class_csv):
        if row["class_name"].strip().lower() == "crack":
            crack_ap = float(row["map50"])
            break
    return {
        "variant_tag": variant.tag,
        "variant": variant.label,
        "selected_epoch": selected_epoch,
        "val_map50": val_map50,
        "mAP50": float(map_row["map50"]),
        "mAP50_95": float(map_row["map50_95"]),
        "precision": float(map_row["precision"]),
        "recall": float(map_row["recall"]),
        "PSNR": psnr_for_split(test_yaml),
        "Crack AP": crack_ap,
        "Boundary IoU": boundary_iou(variant.tag, test_yaml),
        "checkpoint": str(checkpoint),
        "data_yaml": str(test_yaml),
        "active_contour_loss": 0.0,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def write_latex_table(rows: list[dict[str, Any]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    best = {
        "mAP50": max(float(r["mAP50"]) for r in rows),
        "PSNR": max(float(r["PSNR"]) for r in rows),
        "Crack AP": max(float(r["Crack AP"]) for r in rows),
        "Boundary IoU": max(float(r["Boundary IoU"]) for r in rows),
    }

    def cell(row: dict[str, Any], key: str, digits: int) -> str:
        text = fmt(row[key], digits)
        return rf"\textbf{{{text}}}" if abs(float(row[key]) - best[key]) < 1e-12 else text

    body = "\n".join(
        (
            f"{row['variant']} & {cell(row, 'mAP50', 3)} & {cell(row, 'PSNR', 2)} & "
            f"{cell(row, 'Crack AP', 3)} & {cell(row, 'Boundary IoU', 3)} \\\\"
        )
        for row in rows
    )
    tex = rf"""\begin{{table}}[!t]
\centering
\caption{{Independent no-active-contour ablation on PCM defocus. Each row is trained with the same train split, random seed, validation-selection rule, and held-out test split. The revised rows use guided task attention, degradation-conditioned detail skipping, defect-weighted TDP, CQMix, and low-weight Jacobian/anchor/evidence regularization. Boundary IoU uses the fixed ground-truth-box active-contour audit; no train-time active-contour loss is used in any row.}}
\label{{tab:no_ac_ablation}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
Variant & \mapfifty & PSNR & Crack AP & Boundary IoU \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (TABLES / "table_no_active_contour_ablation.tex").write_text(tex, encoding="utf-8")


def write_bar_figure(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [str(row["variant"]).replace(" + ", "\n+ ") for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax1 = plt.subplots(figsize=(10.0, 4.6), dpi=190)
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, [float(r["mAP50"]) for r in rows], width, color="#4e79a7", label="mAP50")
    ax1.bar(x + width / 2, [float(r["Crack AP"]) for r in rows], width, color="#59a14f", label="Crack AP")
    ax2.plot(x, [float(r["PSNR"]) for r in rows], color="#e15759", marker="o", linewidth=2, label="PSNR")
    ax2.plot(x, [float(r["Boundary IoU"]) for r in rows], color="#f28e2b", marker="s", linewidth=2, label="Boundary IoU")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=24, ha="right", fontsize=8)
    ax1.set_ylabel("Detection AP")
    ax2.set_ylabel("PSNR / boundary IoU")
    ax1.set_title("Independent no-active-contour ablation on PCM defocus")
    ax1.grid(axis="y", color="#d8dee6", linewidth=0.7)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_no_ac_ablation.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final no-active-contour RMR-Net ablation suite.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scenario": SCENARIO_NAME,
        "train_root": str(TRAIN_ROOT),
        "val_root": str(VAL_ROOT),
        "test_root": str(TEST_ROOT),
        "detector": str(PCM_DETECTOR),
        "epochs_per_variant": args.epochs,
        "seed": args.seed,
        "active_contour_loss": 0.0,
        "variants": [variant.__dict__ for variant in VARIANTS],
    }
    (EXP / "ablation_protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        run_dir = train_variant(args, variant, None)
        selected_epoch, checkpoint, val_map50 = choose_epoch(args, variant, run_dir)
        row = summarize_variant(args, variant, checkpoint, selected_epoch, val_map50)
        rows.append(row)
        selection_rows.append(
            {
                "variant_tag": variant.tag,
                "variant": variant.label,
                "selected_epoch": selected_epoch,
                "val_map50": val_map50,
                "checkpoint": str(checkpoint),
                "active_contour_loss": 0.0,
            }
        )
        write_csv(EXP / "ablation_metrics_partial.csv", rows)
        write_csv(EXP / "selection_summary_partial.csv", selection_rows)

    write_csv(EXP / "ablation_metrics.csv", rows)
    write_csv(EXP / "selection_summary.csv", selection_rows)
    (EXP / "best_by_val_map.json").write_text(json.dumps(selection_rows[-1], indent=2), encoding="utf-8")
    write_latex_table(rows)
    write_bar_figure(rows)
    shutil.copy2(EXP / "ablation_metrics.csv", PAPER / "no_active_contour_ablation_metrics.csv")
    print(json.dumps({"rows": rows, "table": str(TABLES / "table_no_active_contour_ablation.tex")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
