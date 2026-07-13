from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

DET = ROOT / "runs/detection_eval_nativeblur_v14"
SNAKE_DET = ROOT / "runs/snake_nativeblur_v14"
SNAKE_GT = ROOT / "runs/snake_nativeblur_gtbox_v14"

MODELS = [
    ("native", "Native input"),
    ("rmr_blind", "RMR image-only"),
    ("nafnet", "NAFNet-road"),
    ("dfpir", "DFPIR"),
]

DATASETS = {
    "pothole": {
        "label": "IVCNZ native-blur",
        "csv": DET / "pothole_nativeblur640.csv",
        "n_images": 50,
        "snake_prefix": "pothole",
    },
    "pcm": {
        "label": "PCM native-blur",
        "csv": DET / "pcm_nativeblur.csv",
        "per_class": DET / "pcm_nativeblur_per_class.csv",
        "n_images": 80,
        "snake_prefix": "pcm",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_rows(dataset_key: str) -> dict[str, dict[str, float]]:
    rows = {}
    for row in read_csv(DATASETS[dataset_key]["csv"]):
        rows[row["name"]] = {
            "map50": float(row["map50"]),
            "map50_95": float(row["map50_95"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        }
    return rows


def crack_map50_rows() -> dict[str, float]:
    rows = {}
    for row in read_csv(DATASETS["pcm"]["per_class"]):
        if row["class_name"] == "crack":
            rows[row["eval_name"]] = float(row["map50"])
    return rows


def snake_summary(kind: str, dataset_key: str, model_key: str) -> dict:
    prefix = DATASETS[dataset_key]["snake_prefix"]
    root = SNAKE_GT if kind == "gt" else SNAKE_DET
    return read_json(root / f"{prefix}_{model_key}" / "snake_boundary_summary.json")


def snake_yield(kind: str, dataset_key: str, model_key: str) -> float:
    summary = snake_summary(kind, dataset_key, model_key)
    return float(summary["successes"]) / DATASETS[dataset_key]["n_images"]


def crack_gt_stats(model_key: str) -> tuple[float, float, float]:
    summary = snake_summary("gt", "pcm", model_key)
    stats = summary["classes"].get("crack", {})
    return (
        float(stats.get("successes", 0)) / DATASETS["pcm"]["n_images"],
        float(stats.get("mean_edge_alignment", 0.0)),
        float(stats.get("mean_contrast", 0.0)),
    )


def fmt(value: float, ndigits: int = 3) -> str:
    return f"{value:.{ndigits}f}"


def write_detection_table() -> None:
    crack = crack_map50_rows()
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Uncontrolled native-blur stress test. The images are real held-out frames selected by low no-reference sharpness; no synthetic blur, clean target, or controlled metadata is used. Restoration is image-only.}",
        r"\label{tab:native_blur_detection}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Source & \mapfifty & \mapall & Prec. & Recall \\",
        r"\midrule",
    ]
    for dataset_key in ["pothole", "pcm"]:
        rows = metric_rows(dataset_key)
        best = max(rows, key=lambda key: rows[key]["map50"])
        for idx, (model_key, label) in enumerate(MODELS):
            row = rows[model_key]
            dataset_label = DATASETS[dataset_key]["label"] if idx == 0 else ""
            map50 = fmt(row["map50"])
            if model_key == best:
                map50 = rf"\textbf{{{map50}}}"
            lines.append(
                f"{dataset_label} & {label} & {map50} & {fmt(row['map50_95'])} & "
                f"{fmt(row['precision'])} & {fmt(row['recall'])} \\\\"
            )
        if dataset_key == "pothole":
            lines.append(r"\addlinespace")
    lines += [
        r"\midrule",
        rf"PCM crack only & Native input & \textbf{{{fmt(crack['native'])}}} & -- & -- & -- \\",
        rf" & RMR image-only & {fmt(crack['rmr_blind'])} & -- & -- & -- \\",
        rf" & NAFNet-road & {fmt(crack['nafnet'])} & -- & -- & -- \\",
        rf" & DFPIR & {fmt(crack['dfpir'])} & -- & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_native_blur_detection.tex").write_text("\n".join(lines), encoding="utf-8")


def write_snake_table() -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Snake robustness on uncontrolled native-blur subsets. Detector-box yield uses frozen-YOLO predictions and therefore combines detection availability with contour validity. Fixed-GT-box yield gives every image source the same boxes and isolates whether the image supports a valid active contour.}",
        r"\label{tab:native_blur_snake}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Source & Det. yield & GT-box yield & Crack GT yield & Crack edge & Crack contrast \\",
        r"\midrule",
    ]
    for dataset_key in ["pothole", "pcm"]:
        gt_values = {key: snake_yield("gt", dataset_key, key) for key, _ in MODELS}
        best_gt = max(gt_values, key=gt_values.get)
        for idx, (model_key, label) in enumerate(MODELS):
            dataset_label = DATASETS[dataset_key]["label"] if idx == 0 else ""
            det_y = snake_yield("det", dataset_key, model_key)
            gt_y = gt_values[model_key]
            gt_text = fmt(gt_y, 2)
            if model_key == best_gt:
                gt_text = rf"\textbf{{{gt_text}}}"
            if dataset_key == "pcm":
                crack_y, crack_edge, crack_contrast = crack_gt_stats(model_key)
                crack_y_text = fmt(crack_y, 2)
                crack_edge_text = fmt(crack_edge, 2)
                crack_contrast_text = fmt(crack_contrast, 2)
            else:
                crack_y_text = crack_edge_text = crack_contrast_text = "--"
            lines.append(
                f"{dataset_label} & {label} & {fmt(det_y, 2)} & {gt_text} & "
                f"{crack_y_text} & {crack_edge_text} & {crack_contrast_text} \\\\"
            )
        if dataset_key == "pothole":
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_native_blur_snake.tex").write_text("\n".join(lines), encoding="utf-8")


def build_detection_figure() -> None:
    labels = [DATASETS[key]["label"].replace(" native-blur", "") for key in ["pothole", "pcm"]]
    x = np.arange(len(labels))
    width = 0.19
    colors = {
        "native": "#6B7280",
        "rmr_blind": "#E11D48",
        "nafnet": "#3B82F6",
        "dfpir": "#10B981",
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=240)
    for offset, (model_key, label) in zip([-1.5, -0.5, 0.5, 1.5], MODELS):
        values = [metric_rows(key)[model_key]["map50"] for key in ["pothole", "pcm"]]
        bars = ax.bar(x + offset * width, values, width, label=label, color=colors[model_key], edgecolor="white")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=7)
    ax.set_ylabel("mAP50")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.66)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20), fontsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.4)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_native_blur_detection.png", bbox_inches="tight")
    plt.close(fig)


def build_snake_figure() -> None:
    labels = [DATASETS[key]["label"].replace(" native-blur", "") for key in ["pothole", "pcm"]]
    x = np.arange(len(labels))
    width = 0.19
    colors = {
        "native": "#6B7280",
        "rmr_blind": "#E11D48",
        "nafnet": "#3B82F6",
        "dfpir": "#10B981",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=240, sharey=True)
    for ax, kind, title in [(axes[0], "det", "Detector boxes"), (axes[1], "gt", "Fixed GT boxes")]:
        for offset, (model_key, label) in zip([-1.5, -0.5, 0.5, 1.5], MODELS):
            values = [snake_yield(kind, key, model_key) for key in ["pothole", "pcm"]]
            bars = ax.bar(x + offset * width, values, width, label=label, color=colors[model_key], edgecolor="white")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", fontsize=6.5)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Accepted contours per image")
    axes[0].set_ylim(0, 3.05)
    axes[0].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(1.05, 1.22), fontsize=7.5)
    fig.tight_layout(pad=0.5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_native_blur_snake.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    write_detection_table()
    write_snake_table()
    build_detection_figure()
    build_snake_figure()
    print(
        json.dumps(
            {
                "tables": [
                    str(TABLES / "table_native_blur_detection.tex"),
                    str(TABLES / "table_native_blur_snake.tex"),
                ],
                "figures": [
                    str(FIGURES / "fig_native_blur_detection.png"),
                    str(FIGURES / "fig_native_blur_snake.png"),
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
