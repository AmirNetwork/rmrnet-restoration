"""Build paper assets for the v26 YOLO11s detector-strength audit."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
EXP = ROOT / "experiments" / "v26_yolo11s_eval"

POTH_BASE = EXP / "pothole_test_yolo11s_baselines.csv"
PCM_BASE = EXP / "pcm_test_yolo11s_baselines.csv"
POTH_CLEAN = EXP / "pothole_test_yolo11s_best.csv"
PCM_CLEAN = EXP / "pcm_test_yolo11s_best.csv"
POTH_TRAIN = ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pothole_clean_80ep" / "results.csv"
PCM_TRAIN = ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pcm_clean_80ep" / "results.csv"

SCENARIOS = ["motion", "defocus", "lowlight"]
SCENARIO_LABEL = {
    "motion": "motion blur",
    "defocus": "defocus",
    "lowlight": "low light",
}
METHOD_LABEL = {
    "input": "degraded",
    "dfpir": "DFPIR",
    "nafnet": "NAFNet-road",
    "rmrnet_v25": "RMR-Net",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_metric(value: float, bold: bool = False) -> str:
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def rows_by_scenario(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        name = row["name"]
        if "_" not in name:
            continue
        scenario, method = name.split("_", 1)
        grouped[scenario][method] = {
            "map50": float(row["map50"]),
            "map50_95": float(row["map50_95"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        }
    return grouped


def clean_metrics(path: Path) -> dict[str, float]:
    for row in read_csv(path):
        if row["name"] == "clean":
            return {
                "map50": float(row["map50"]),
                "map50_95": float(row["map50_95"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            }
    raise ValueError(f"No clean row in {path}")


def write_detection_table(
    out: Path,
    rows: list[dict[str, str]],
    clean: dict[str, float],
    caption: str,
    label: str,
    methods: list[str],
) -> None:
    grouped = rows_by_scenario(rows)
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Scenario & Input & mAP50 & mAP50--95 & Prec. & Rec. \\\\",
        "\\midrule",
        (
            "clean & clean & "
            f"{clean['map50']:.3f} & {clean['map50_95']:.3f} & "
            f"{clean['precision']:.3f} & {clean['recall']:.3f} \\\\"
        ),
    ]

    for scenario in SCENARIOS:
        available = grouped[scenario]
        winners = {}
        for metric in ["map50", "map50_95", "precision", "recall"]:
            winners[metric] = max((available[m][metric], m) for m in methods if m in available)[1]
        for method in methods:
            if method not in available:
                continue
            vals = available[method]
            lines.append(
                f"{SCENARIO_LABEL[scenario]} & {METHOD_LABEL[method]} & "
                f"{format_metric(vals['map50'], winners['map50'] == method)} & "
                f"{format_metric(vals['map50_95'], winners['map50_95'] == method)} & "
                f"{format_metric(vals['precision'], winners['precision'] == method)} & "
                f"{format_metric(vals['recall'], winners['recall'] == method)} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def write_training_table(out: Path) -> None:
    rows = []
    for dataset, train_path, data_yaml, clean_path in [
        ("IVCNZ pothole", POTH_TRAIN, ROOT / "datasets" / "pothole_yolo" / "data.yaml", POTH_CLEAN),
        ("PCM", PCM_TRAIN, ROOT / "datasets" / "road_damage_pcm_yolo" / "data.yaml", PCM_CLEAN),
    ]:
        train_rows = read_csv(train_path)
        best = max(train_rows, key=lambda row: float(row["metrics/mAP50(B)"]))
        clean = clean_metrics(clean_path)
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        base = data_yaml.parent
        counts = {}
        for split in ["train", "val", "test"]:
            img_dir = base / str(cfg[split])
            counts[split] = len([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        rows.append(
            [
                dataset,
                counts["train"],
                counts["val"],
                counts["test"],
                int(float(best["epoch"])),
                float(best["metrics/mAP50(B)"]),
                float(best["metrics/mAP50-95(B)"]),
                clean["map50"],
                clean["map50_95"],
            ]
        )

    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{YOLO11s detector-strength audit. Detectors are initialized from YOLO11s, fine-tuned for 80 epochs on clean training images, selected by validation mAP50, then frozen for all restoration comparisons.}",
        "\\label{tab:yolo11s_detector_audit}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Dataset & Tr. & Val & Test & Best ep. & Val mAP50 & Val mAP50--95 & Test mAP50 & Test mAP50--95 \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & "
            f"{row[5]:.3f} & {row[6]:.3f} & {row[7]:.3f} & {row[8]:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def detection_bar(rows: list[dict[str, str]], out: Path, title: str, methods: list[str]) -> None:
    grouped = rows_by_scenario(rows)
    colors = {
        "input": "#6b7280",
        "dfpir": "#3b82f6",
        "nafnet": "#10b981",
        "rmrnet_v25": "#d946ef",
    }
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(11.5, 3.2), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        values = [grouped[scenario][m]["map50"] for m in methods if m in grouped[scenario]]
        labels = [METHOD_LABEL[m] for m in methods if m in grouped[scenario]]
        xs = range(len(values))
        bars = ax.bar(xs, values, color=[colors[m] for m in methods if m in grouped[scenario]], width=0.72)
        ax.set_title(SCENARIO_LABEL[scenario], fontsize=10)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.0, max(0.55, max(values) + 0.08))
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Frozen YOLO11s mAP50")
    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary() -> None:
    poth = rows_by_scenario(read_csv(POTH_BASE))
    pcm = rows_by_scenario(read_csv(PCM_BASE))

    def gain(grouped: dict[str, dict[str, dict[str, float]]], scenario: str) -> float:
        return grouped[scenario]["rmrnet_v25"]["map50"] - grouped[scenario]["input"]["map50"]

    summary = {
        "detectors": {
            "pothole": {
                "weights": str(ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pothole_clean_80ep" / "weights" / "best.pt"),
                "best_epoch": 63,
                "validation_map50": 0.67315,
            },
            "pcm": {
                "weights": str(ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pcm_clean_80ep" / "weights" / "best.pt"),
                "best_epoch": 67,
                "validation_map50": 0.49373,
            },
        },
        "pothole_rmr_map50_gain_vs_degraded": {scenario: gain(poth, scenario) for scenario in SCENARIOS},
        "pcm_rmr_map50_gain_vs_degraded": {scenario: gain(pcm, scenario) for scenario in SCENARIOS},
        "claim_boundary": "RMR-Net wins motion/defocus on IVCNZ and all three PCM scenarios against available restored baselines under YOLO11s. Pothole low-light is a near tie with degraded input and should be described as a boundary case.",
    }
    (EXP / "V26_YOLO11S_DETECTOR_AUDIT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# V26 YOLO11s Detector-Strength Audit",
        "",
        "This audit replaces the weaker YOLOv8n detector-only story with stronger YOLO11s detectors fine-tuned for 80 epochs on the clean training splits. The restoration outputs are not re-selected on test data; the detectors are selected by validation mAP50 and frozen before all clean/degraded/restored test evaluation.",
        "",
        "## Main Findings",
        "",
        "- IVCNZ pothole: RMR-Net improves mAP50 by +0.327 under motion blur and +0.221 under defocus versus degraded input, and it beats DFPIR and NAFNet-road in those two scenarios.",
        "- IVCNZ low light: degraded input remains slightly highest in mAP50--95 and essentially tied with RMR-Net in mAP50, so this remains a claim boundary rather than a win.",
        "- PCM: RMR-Net improves mAP50 by +0.144 under motion blur, +0.215 under defocus, and +0.123 under low light versus degraded input, and it beats DFPIR and NAFNet-road in all three scenarios.",
        "",
        "## Reproducibility",
        "",
        "- Detector training command is implemented in `tools/train_yolo_detector.py`.",
        "- Pothole detector: `runs/detect/runs/yolo11s_v26/pothole_clean_80ep/weights/best.pt`.",
        "- PCM detector: `runs/detect/runs/yolo11s_v26/pcm_clean_80ep/weights/best.pt`.",
        "- Evaluation CSVs are in `experiments/v26_yolo11s_eval/`.",
        "",
    ]
    (EXP / "V26_YOLO11S_DETECTOR_AUDIT.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)

    poth_rows = read_csv(POTH_BASE)
    pcm_rows = read_csv(PCM_BASE)
    write_training_table(TABLES / "table_yolo11s_detector_audit.tex")
    write_detection_table(
        TABLES / "table_pothole_detection.tex",
        poth_rows,
        clean_metrics(POTH_CLEAN),
        "Held-out pothole detection with a stronger frozen YOLO11s detector. The detector is trained on clean images, selected by validation mAP50, and frozen for degraded/restored test evaluation.",
        "tab:pothole_detection",
        ["input", "dfpir", "nafnet", "rmrnet_v25"],
    )
    write_detection_table(
        TABLES / "table_pcm_detection.tex",
        pcm_rows,
        clean_metrics(PCM_CLEAN),
        "Held-out PCM pothole/crack/manhole detection with a stronger frozen YOLO11s detector. The detector is trained on clean images, selected by validation mAP50, and frozen for degraded/restored test evaluation.",
        "tab:pcm_detection",
        ["input", "dfpir", "nafnet", "rmrnet_v25"],
    )
    detection_bar(
        poth_rows,
        FIGURES / "fig_pothole_detection_recovery.png",
        "IVCNZ pothole detection recovery with frozen YOLO11s",
        ["input", "dfpir", "nafnet", "rmrnet_v25"],
    )
    detection_bar(
        pcm_rows,
        FIGURES / "fig_pcm_detection_recovery.png",
        "PCM road-damage detection recovery with frozen YOLO11s",
        ["input", "dfpir", "nafnet", "rmrnet_v25"],
    )
    write_summary()


if __name__ == "__main__":
    main()
