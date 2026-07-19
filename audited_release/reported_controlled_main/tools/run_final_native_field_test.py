# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.tune_perception_gate import road_evidence_score  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


METHOD_LABELS = {
    "raw": "Raw native",
    "rmr_blind": "RMR-Net image-only",
    "rmr_metadata": "RMR-Net metadata",
    "nafnet": "NAFNet-road",
    "dfpir": "DFPIR",
    "demoe_auto": "DeMoE-auto",
    "demoe_scenario": "DeMoE-scenario",
    "instructir_generic": "InstructIR-generic",
    "instructir_metadata": "InstructIR-metadata",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final native-resolution geotagged field experiment. The script restores "
            "all annotated Sony cam1 frames without resizing, evaluates tiled YOLO "
            "detection, runs residual/gate sweeps, audits native sharpness, and writes "
            "paper-ready CSV/LaTeX/PNG assets."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("experiments/roboflow_geotagged_v5_native_real/native_real_yolo_newroad6/data.yaml"),
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("runs/detect/runs/detect/geotagged_yolov8_newroad/yolov8n_newroad_from_pcm40_150ep/weights/best.pt"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("experiments/roboflow_geotagged_v5_native_real/final_all49"),
    )
    parser.add_argument(
        "--models",
        default="raw,rmr_blind,rmr_metadata,nafnet,demoe_auto,demoe_scenario",
        help=(
            "Comma-separated restoration methods. Slow/heavy methods can be added with "
            "dfpir,instructir_generic,instructir_metadata when runtime/memory permits."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--restore-tile", type=int, default=768)
    parser.add_argument("--restore-overlap", type=int, default=96)
    parser.add_argument("--eval-tile", type=int, default=1024)
    parser.add_argument("--eval-overlap", type=int, default=256)
    parser.add_argument("--infer-imgsz", type=int, default=448)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--skip-restore", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--sharpness-full-limit", type=int, default=0, help="0 means the full cam1 pool.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)


def read_yaml(path: Path) -> tuple[Path, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return root, data


def split_dir(root: Path, data: dict[str, Any], split: str, kind: str) -> Path:
    split_value = str(data.get(split, data.get("val", f"images/{split}")))
    split_path = Path(split_value.replace("images", kind, 1))
    return split_path if split_path.is_absolute() else root / split_path


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def ensure_restore(args: argparse.Namespace, restored_root: Path) -> None:
    if args.skip_restore:
        return
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    run(
        [
            sys.executable,
            "tools/restore_native_yolo_split.py",
            "--data",
            args.data,
            "--split",
            "test",
            "--scenario",
            "native_real",
            "--out",
            restored_root,
            "--models",
            ",".join(models),
            "--device",
            args.device,
            "--tile",
            args.restore_tile,
            "--overlap",
            args.restore_overlap,
            "--skip-existing",
        ]
    )


def ensure_eval(args: argparse.Namespace, data_yaml: Path, out_dir: Path) -> None:
    if args.skip_eval and (out_dir / "summary.json").exists():
        return
    cmd = [
        sys.executable,
        "tools/eval_native_tiled_detector.py",
        "--data",
        data_yaml,
        "--weights",
        args.detector,
        "--out",
        out_dir,
        "--split",
        "test",
        "--tile",
        args.eval_tile,
        "--overlap",
        args.eval_overlap,
        "--infer-imgsz",
        args.infer_imgsz,
        "--conf",
        args.conf,
        "--nms-iou",
        args.nms_iou,
        "--device",
        "0" if args.device == "cuda" else args.device,
    ]
    if not args.no_overlays:
        cmd.append("--save-overlays")
    run(cmd)


def read_metrics(path: Path) -> dict[tuple[str, str, float], dict[str, float]]:
    rows: dict[tuple[str, str, float], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            threshold = float(row["threshold"])
            key = (row["metric"], row["mode"], threshold)
            rows[key] = {
                name: float(value) if value not in {"", None} else np.nan
                for name, value in row.items()
                if name not in {"metric", "mode", "threshold"}
            }
    return rows


def read_operating(path: Path) -> dict[tuple[float, str], dict[str, float]]:
    rows: dict[tuple[float, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (float(row["confidence"]), row["mode"])
            rows[key] = {
                name: float(value)
                for name, value in row.items()
                if name not in {"confidence", "mode"}
            }
    return rows


def mean_runtime(restore_summary: Path, model: str) -> float:
    if not restore_summary.exists():
        return 0.0
    values: list[float] = []
    with restore_summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == model:
                values.append(float(row.get("runtime_s", "0") or 0.0))
    return mean(values) if values else 0.0


def summarize_eval(eval_dir: Path, model: str, restore_summary: Path) -> dict[str, Any]:
    metrics = read_metrics(eval_dir / "native_tiled_metrics.csv")
    operating = read_operating(eval_dir / "operating_points.csv")
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    key10 = ("iou_greedy", "crack_group", 0.1)
    key25 = ("iou_greedy", "crack_group", 0.25)
    key50 = ("iou_greedy", "crack_group", 0.5)
    cov25 = ("gt_coverage_ioa", "crack_group", 0.25)
    op25 = operating.get((0.25, "crack_group"), {})
    row = {
        "tag": model,
        "method": METHOD_LABELS.get(model, model),
        "metadata": "yes" if model in {"rmr_metadata", "instructir_metadata"} else ("scenario" if model == "demoe_scenario" else "no"),
        "images": int(summary["images"]),
        "gt_boxes": int(summary["gt_boxes"]),
        "pred_per_image": float(summary["pred_boxes"]) / max(1, int(summary["images"])),
        "precision_iou10": metrics[key10]["precision"],
        "recall_iou10": metrics[key10]["recall"],
        "f1_iou10": metrics[key10]["f1"],
        "precision_iou25": metrics[key25]["precision"],
        "recall_iou25": metrics[key25]["recall"],
        "f1_iou25": metrics[key25]["f1"],
        "precision_iou50": metrics[key50]["precision"],
        "recall_iou50": metrics[key50]["recall"],
        "f1_iou50": metrics[key50]["f1"],
        "coverage_ioa25": metrics[cov25]["recall"],
        "false_pos_per_image_conf25": op25.get("false_positives_per_image", np.nan),
        "runtime_s_image": mean_runtime(restore_summary, model),
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_labels_metadata(source_root: Path, target_root: Path, split: str = "test") -> None:
    for kind in ("labels", "metadata"):
        src = source_root / kind / split
        dst = target_root / kind / split
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            for path in src.iterdir():
                if path.is_file():
                    shutil.copy2(path, dst / path.name)


def write_data_yaml(source_yaml: Path, target_root: Path, split: str = "test") -> Path:
    _source_root, cfg = read_yaml(source_yaml)
    data = {
        "path": str(target_root.resolve()).replace("\\", "/"),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "names": cfg["names"],
        "nc": cfg.get("nc", len(cfg["names"]) if isinstance(cfg["names"], list) else len(dict(cfg["names"]))),
    }
    out = target_root / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def blend_dataset(raw_yaml: Path, restored_yaml: Path, out_root: Path, eta: float) -> Path:
    raw_root, raw_cfg = read_yaml(raw_yaml)
    restored_root, restored_cfg = read_yaml(restored_yaml)
    raw_images = split_dir(raw_root, raw_cfg, "test", "images")
    restored_images = split_dir(restored_root, restored_cfg, "test", "images")
    out_image_dir = out_root / "images" / "test"
    if out_image_dir.exists():
        shutil.rmtree(out_image_dir)
    out_image_dir.mkdir(parents=True, exist_ok=True)
    restored_by_name = {p.name: p for p in list_images(restored_images)}
    for raw_path in list_images(raw_images):
        restored_path = restored_by_name.get(raw_path.name)
        if restored_path is None:
            continue
        with Image.open(raw_path) as raw_img, Image.open(restored_path) as res_img:
            raw = np.asarray(raw_img.convert("RGB"), dtype=np.float32)
            res = np.asarray(res_img.convert("RGB").resize(raw_img.size), dtype=np.float32)
        out = np.clip(raw + eta * (res - raw), 0.0, 255.0).astype(np.uint8)
        Image.fromarray(out).save(out_image_dir / raw_path.name, quality=95, subsampling=0)
    copy_labels_metadata(raw_root, out_root)
    return write_data_yaml(raw_yaml, out_root)


def gate_dataset(raw_yaml: Path, restored_yaml: Path, out_root: Path, threshold: float) -> tuple[Path, dict[str, int]]:
    raw_root, raw_cfg = read_yaml(raw_yaml)
    restored_root, restored_cfg = read_yaml(restored_yaml)
    raw_images = split_dir(raw_root, raw_cfg, "test", "images")
    restored_images = split_dir(restored_root, restored_cfg, "test", "images")
    out_image_dir = out_root / "images" / "test"
    if out_image_dir.exists():
        shutil.rmtree(out_image_dir)
    out_image_dir.mkdir(parents=True, exist_ok=True)
    restored_by_name = {p.name: p for p in list_images(restored_images)}
    rows = []
    counts = {"raw": 0, "restored": 0}
    for raw_path in list_images(raw_images):
        restored_path = restored_by_name.get(raw_path.name)
        if restored_path is None:
            continue
        delta = road_evidence_score(restored_path) - road_evidence_score(raw_path)
        use_restored = delta > threshold
        source = restored_path if use_restored else raw_path
        shutil.copy2(source, out_image_dir / raw_path.name)
        choice = "restored" if use_restored else "raw"
        counts[choice] += 1
        rows.append({"image": raw_path.name, "delta_score": delta, "threshold": threshold, "choice": choice})
    copy_labels_metadata(raw_root, out_root)
    write_csv(out_root / "gate_manifest.csv", rows)
    return write_data_yaml(raw_yaml, out_root), counts


def run_main_models(args: argparse.Namespace, restored_root: Path, eval_root: Path) -> list[dict[str, Any]]:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    restore_summary = restored_root / "restore_summary.csv"
    for model in models:
        data_yaml = restored_root / model / "data.yaml"
        if not data_yaml.exists():
            print(f"Skipping {model}: missing {data_yaml}", flush=True)
            continue
        out_dir = eval_root / model
        ensure_eval(args, data_yaml, out_dir)
        rows.append(summarize_eval(out_dir, model, restore_summary))
    return rows


def run_eta_sweep(args: argparse.Namespace, restored_root: Path, eval_root: Path) -> list[dict[str, Any]]:
    raw_yaml = restored_root / "raw" / "data.yaml"
    restored_yaml = restored_root / "rmr_metadata" / "data.yaml"
    etas = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    rows: list[dict[str, Any]] = []
    for eta in etas:
        name = f"rmr_eta_{str(eta).replace('.', 'p')}"
        data_yaml = blend_dataset(raw_yaml, restored_yaml, eval_root / "datasets" / name, eta)
        out_dir = eval_root / "eval" / name
        ensure_eval(args, data_yaml, out_dir)
        row = summarize_eval(out_dir, name, restored_root / "restore_summary.csv")
        row["eta"] = eta
        row["method"] = f"RMR-Net residual eta={eta:g}"
        rows.append(row)
    return rows


def run_tau_sweep(args: argparse.Namespace, restored_root: Path, eval_root: Path) -> list[dict[str, Any]]:
    raw_yaml = restored_root / "raw" / "data.yaml"
    restored_yaml = restored_root / "rmr_metadata" / "data.yaml"
    raw_root, raw_cfg = read_yaml(raw_yaml)
    restored_root_path, restored_cfg = read_yaml(restored_yaml)
    raw_images = split_dir(raw_root, raw_cfg, "test", "images")
    restored_images = split_dir(restored_root_path, restored_cfg, "test", "images")
    restored_by_name = {p.name: p for p in list_images(restored_images)}
    deltas = []
    for raw_path in list_images(raw_images):
        restored_path = restored_by_name.get(raw_path.name)
        if restored_path is not None:
            deltas.append(road_evidence_score(restored_path) - road_evidence_score(raw_path))
    values = np.asarray(deltas, dtype=np.float32)
    thresholds = [-1e9]
    thresholds.extend(float(np.percentile(values, q)) for q in [10, 25, 50, 75, 90])
    thresholds.append(1e9)
    rows: list[dict[str, Any]] = []
    for index, tau in enumerate(thresholds):
        name = f"rmr_tau_{index:02d}"
        data_yaml, counts = gate_dataset(raw_yaml, restored_yaml, eval_root / "datasets" / name, tau)
        out_dir = eval_root / "eval" / name
        ensure_eval(args, data_yaml, out_dir)
        row = summarize_eval(out_dir, name, restored_root / "restore_summary.csv")
        row["tau"] = tau
        row["raw_selected"] = counts["raw"]
        row["restored_selected"] = counts["restored"]
        if tau < -1e8:
            row["policy"] = "all restored"
        elif tau > 1e8:
            row["policy"] = "all raw"
        else:
            row["policy"] = f"tau={tau:.4f}"
        row["method"] = f"Gate {row['policy']}"
        rows.append(row)
    return rows


def sharpness_scores(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Image.open(path) as image:
            gray = image.convert("L")
            scale = min(1.0, 1024.0 / max(gray.size))
            if scale < 1.0:
                gray = gray.resize((int(gray.width * scale), int(gray.height * scale)))
            arr = np.asarray(gray, dtype=np.float32) / 255.0
        c = arr[1:-1, 1:-1]
        lap = arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] - 4.0 * c
        gx = arr[1:-1, 2:] - arr[1:-1, :-2]
        gy = arr[2:, 1:-1] - arr[:-2, 1:-1]
        rows.append(
            {
                "image": path.name,
                "laplacian_var": float(np.var(lap)),
                "tenengrad": float(np.mean(gx * gx + gy * gy)),
            }
        )
    return rows


def summarize_sharpness(args: argparse.Namespace, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_root, data_cfg = read_yaml(args.data)
    annotated = list_images(split_dir(data_root, data_cfg, "test", "images"))
    cam1 = ROOT / "geotagged" / "cam1"
    full_pool = sorted(p for p in cam1.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if args.sharpness_full_limit > 0:
        full_pool = full_pool[: args.sharpness_full_limit]
    rows_annotated = sharpness_scores(annotated)
    rows_full = sharpness_scores(full_pool)
    write_csv(out_dir / "sharpness_annotated49.csv", rows_annotated)
    write_csv(out_dir / "sharpness_full_cam1_pool.csv", rows_full)
    summary_rows = []
    for label, rows in [("Annotated 49", rows_annotated), ("Full cam1 pool", rows_full)]:
        for metric in ["laplacian_var", "tenengrad"]:
            vals = [float(r[metric]) for r in rows]
            summary_rows.append(
                {
                    "set": label,
                    "metric": metric,
                    "n": len(vals),
                    "mean": mean(vals),
                    "median": median(vals),
                    "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)),
                }
            )
    write_csv(out_dir / "sharpness_summary.csv", summary_rows)
    return summary_rows, rows_annotated


def fmt(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return str(x)
    if abs(val - round(val)) < 1e-12 and abs(val) < 1000:
        return str(int(round(val)))
    if abs(val) >= 1000:
        return f"{val:.1f}"
    return f"{val:.{digits}f}"


def write_tex_table(path: Path, caption: str, label: str, columns: list[str], headers: list[str], rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        vals = []
        for index, col in enumerate(columns):
            val = row.get(col, "")
            vals.append(str(val) if index == 0 else fmt(val))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_assets(out_dir: Path, main_rows: list[dict[str, Any]], eta_rows: list[dict[str, Any]], tau_rows: list[dict[str, Any]], sharpness_summary: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [row["method"] for row in main_rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - width, [row["precision_iou10"] for row in main_rows], width, label="Precision")
    ax.bar(x, [row["recall_iou10"] for row in main_rows], width, label="Recall")
    ax.bar(x + width, [row["f1_iou10"] for row in main_rows], width, label="F1")
    ax.set_ylabel("IoU 0.10 crack-group metric")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_geotagged_all49_prf.png", dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    etas = [row["eta"] for row in eta_rows]
    ax1.plot(etas, [row["precision_iou10"] for row in eta_rows], marker="o", label="Precision")
    ax1.plot(etas, [row["recall_iou10"] for row in eta_rows], marker="o", label="Recall")
    ax1.plot(etas, [row["f1_iou10"] for row in eta_rows], marker="o", label="F1")
    ax1.set_xlabel("Residual strength eta")
    ax1.set_ylabel("IoU 0.10 crack-group metric")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(etas, [row["false_pos_per_image_conf25"] for row in eta_rows], marker="s", color="tab:red", label="FP/image")
    ax2.set_ylabel("False positives / image at conf 0.25")
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, frameon=False, loc="upper center", ncol=4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_geotagged_eta_sweep.png", dpi=220)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    xs = np.arange(len(tau_rows))
    ax1.plot(xs, [row["precision_iou10"] for row in tau_rows], marker="o", label="Precision")
    ax1.plot(xs, [row["recall_iou10"] for row in tau_rows], marker="o", label="Recall")
    ax1.plot(xs, [row["f1_iou10"] for row in tau_rows], marker="o", label="F1")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([row["policy"] for row in tau_rows], rotation=25, ha="right")
    ax1.set_ylabel("IoU 0.10 crack-group metric")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.bar(xs, [row["restored_selected"] for row in tau_rows], alpha=0.2, color="tab:green", label="Restored selected")
    ax2.set_ylabel("Images sent to restorer")
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, frameon=False, loc="upper center", ncol=4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_geotagged_tau_sweep.png", dpi=220)
    plt.close(fig)

    sets = sorted({row["set"] for row in sharpness_summary})
    metrics = ["laplacian_var", "tenengrad"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric in zip(axes, metrics):
        vals = [next(row for row in sharpness_summary if row["set"] == s and row["metric"] == metric)["median"] for s in sets]
        ax.bar(sets, vals, color=["tab:blue", "tab:gray"])
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_native_sharpness_audit.png", dpi=220)
    plt.close(fig)


def make_failure_figure(raw_eval: Path, rmr_eval: Path, out_path: Path, limit: int = 4) -> None:
    raw_overlays = sorted((raw_eval / "overlays").glob("*_native_tiled.png"))
    rmr_by_name = {p.name: p for p in (rmr_eval / "overlays").glob("*_native_tiled.png")}
    pairs = [(p, rmr_by_name[p.name]) for p in raw_overlays if p.name in rmr_by_name][:limit]
    if not pairs:
        return
    thumbs = []
    for raw_path, rmr_path in pairs:
        row = []
        for path, title in [(raw_path, "Raw native"), (rmr_path, "RMR-Net metadata")]:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((720, 480))
                canvas = Image.new("RGB", (720, image.height + 36), "white")
                canvas.paste(image, (0, 36))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 8), title, fill=(0, 0, 0))
                row.append(canvas)
        width = sum(img.width for img in row)
        height = max(img.height for img in row)
        joined = Image.new("RGB", (width, height), "white")
        x = 0
        for img in row:
            joined.paste(img, (x, 0))
            x += img.width
        thumbs.append(joined)
    total_h = sum(img.height for img in thumbs)
    total_w = max(img.width for img in thumbs)
    atlas = Image.new("RGB", (total_w, total_h), "white")
    y = 0
    for img in thumbs:
        atlas.paste(img, (0, y))
        y += img.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    out_root = args.out_root
    restored_root = out_root / "restored" / "native_real"
    eval_root = out_root / "native_tiled_eval"
    paper_tables = ROOT / "paper_ieee_tits_rmrnet" / "tables"
    paper_figures = ROOT / "paper_ieee_tits_rmrnet" / "figures"
    out_root.mkdir(parents=True, exist_ok=True)

    ensure_restore(args, restored_root)
    main_rows = run_main_models(args, restored_root, eval_root)
    write_csv(out_root / "paper_metric_summary_all49.csv", main_rows)

    eta_rows = run_eta_sweep(args, restored_root, out_root / "eta_sweep")
    write_csv(out_root / "geotagged_eta_sweep.csv", eta_rows)

    tau_rows = run_tau_sweep(args, restored_root, out_root / "tau_sweep")
    write_csv(out_root / "geotagged_tau_sweep.csv", tau_rows)

    sharpness_summary, _ = summarize_sharpness(args, out_root / "sharpness_audit")
    plot_assets(out_root, main_rows, eta_rows, tau_rows, sharpness_summary)
    make_failure_figure(
        eval_root / "raw",
        eval_root / "rmr_metadata",
        out_root / "figures" / "fig_native_failure_modes.png",
    )

    # Copy final figure assets into the paper project.
    for figure in (out_root / "figures").glob("*.png"):
        shutil.copy2(figure, paper_figures / figure.name)

    write_tex_table(
        paper_tables / "table_geotagged_native_pilot.tex",
        (
            "High-quality Sony native-image field test on all 49 annotated cam1 frames. "
            "Images remain at their original 4752$\\times$3168 resolution; Roboflow labels "
            "are mapped back to native pixels and real EXIF/pose metadata is used where available. "
            "The relaxed IoU 0.10 metric diagnoses high-resolution detection support, while IoU "
            "0.25/0.50 and false positives expose the precision--recall tradeoff."
        ),
        "tab:geotagged_native_pilot",
        [
            "method",
            "images",
            "precision_iou10",
            "recall_iou10",
            "f1_iou10",
            "f1_iou25",
            "f1_iou50",
            "coverage_ioa25",
            "false_pos_per_image_conf25",
            "runtime_s_image",
        ],
        ["Method", "N", "P@.10", "R@.10", "F1@.10", "F1@.25", "F1@.50", "Cov@.25", "FP/img", "s/img"],
        main_rows,
    )
    write_tex_table(
        paper_tables / "table_geotagged_eta_sweep.tex",
        (
            "Residual-strength sweep on the high-quality Sony native-image field test. "
            "$\\eta=0$ is pass-through and $\\eta=1$ is full RMR-Net restoration. "
            "The sweep shows the deployable precision--recall tradeoff rather than claiming "
            "that one fixed restoration strength is universally best."
        ),
        "tab:geotagged_eta_sweep",
        [
            "method",
            "precision_iou10",
            "recall_iou10",
            "f1_iou10",
            "coverage_ioa25",
            "false_pos_per_image_conf25",
        ],
        ["Policy", "P@.10", "R@.10", "F1@.10", "Cov@.25", "FP/img"],
        eta_rows,
    )
    write_tex_table(
        paper_tables / "table_native_gate_sweep.tex",
        (
            "No-reference gate-threshold sweep on the Sony native-image field test. "
            "The gate compares road-evidence scores before and after restoration and sends "
            "a frame to the restorer only when the score gain exceeds $\\tau$."
        ),
        "tab:native_gate_sweep",
        ["policy", "raw_selected", "restored_selected", "precision_iou10", "recall_iou10", "f1_iou10", "false_pos_per_image_conf25"],
        ["Policy", "Raw", "Restored", "P@.10", "R@.10", "F1@.10", "FP/img"],
        tau_rows,
    )
    write_tex_table(
        paper_tables / "table_native_sharpness_audit.tex",
        (
            "Native-image sharpness audit. Laplacian variance and Tenengrad are computed "
            "on downscaled grayscale frames only for no-reference characterization; no labels "
            "or detector outputs are used. The annotated Sony subset is high quality rather "
            "than a severe natural-blur benchmark."
        ),
        "tab:native_sharpness_audit",
        ["set", "n", "mean", "median", "p10", "p90"],
        ["Set/metric", "N", "Mean", "Median", "P10", "P90"],
        [
            {
                "set": f"{row['set']} {row['metric']}",
                "n": row["n"],
                "mean": row["mean"],
                "median": row["median"],
                "p10": row["p10"],
                "p90": row["p90"],
            }
            for row in sharpness_summary
        ],
    )

    manifest = {
        "data": str(args.data),
        "detector": str(args.detector),
        "models": args.models,
        "images": main_rows[0]["images"] if main_rows else 0,
        "main_summary": str(out_root / "paper_metric_summary_all49.csv"),
        "eta_sweep": str(out_root / "geotagged_eta_sweep.csv"),
        "tau_sweep": str(out_root / "geotagged_tau_sweep.csv"),
        "sharpness": str(out_root / "sharpness_audit" / "sharpness_summary.csv"),
    }
    (out_root / "final_native_field_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
