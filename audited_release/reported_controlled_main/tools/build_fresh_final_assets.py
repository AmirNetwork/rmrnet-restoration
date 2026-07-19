# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TAB = PAPER / "tables"
FIG = PAPER / "figures"


MODEL_NAMES = {
    "degraded": "Degraded",
    "dfpir": "DFPIR",
    "demoe_auto": "DeMoE-auto",
    "demoe_scenario": "DeMoE-scenario",
    "nafnet": "NAFNet-road",
    "rmr": "RMR-Net",
}

SCENARIO_NAMES = {
    "motion": "motion blur",
    "defocus": "defocus",
    "lowlight": "low light",
    "mixed": "mixed motion+low light",
}

SCENARIO_ORDER = ["motion", "defocus", "lowlight", "mixed"]
MODEL_ORDER = ["degraded", "dfpir", "demoe_auto", "demoe_scenario", "nafnet", "rmr"]


def read_json_rows(path: str | Path) -> dict[str, dict[str, float | str]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["name"]: row for row in rows}


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, bold: bool = False) -> str:
    text = f"{value:.3f}"
    return rf"\textbf{{{text}}}" if bold else text


def detection_rows(dataset: str) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    base = read_json_rows(ROOT / "experiments" / "fresh_final" / f"{dataset}_yolo11s.json")
    selected = read_csv_rows(ROOT / "experiments" / "fresh_final_selection" / f"{dataset}_test_rmrnet_selected.csv")
    selected_by_scenario: dict[str, dict[str, float]] = {}
    for row in selected:
        scenario = row["name"].split("_")[0]
        selected_by_scenario[scenario] = {
            "map50": float(row["map50"]),
            "map50_95": float(row["map50_95"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        }

    clean = {
        "map50": float(base["clean"]["map50"]),
        "map50_95": float(base["clean"]["map50_95"]),
        "precision": float(base["clean"]["precision"]),
        "recall": float(base["clean"]["recall"]),
    }
    scenario_rows: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in SCENARIO_ORDER:
        scenario_rows[scenario] = {}
        for model in MODEL_ORDER:
            if model == "rmr":
                scenario_rows[scenario][model] = selected_by_scenario[scenario]
            else:
                key = f"{scenario}_{model}"
                row = base[key]
                scenario_rows[scenario][model] = {
                    "map50": float(row["map50"]),
                    "map50_95": float(row["map50_95"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                }
    return clean, scenario_rows


def write_detection_table(dataset: str, out_name: str, label: str, caption: str) -> None:
    clean, scenarios = detection_rows(dataset)
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scenario & Input & mAP50 & mAP50--95 & Prec. & Rec. \\",
        r"\midrule",
        f"clean & clean & {fmt(clean['map50'])} & {fmt(clean['map50_95'])} & {fmt(clean['precision'])} & {fmt(clean['recall'])} \\\\",
    ]
    metrics = ["map50", "map50_95", "precision", "recall"]
    for scenario in SCENARIO_ORDER:
        best = {metric: max(scenarios[scenario][model][metric] for model in MODEL_ORDER) for metric in metrics}
        for model in MODEL_ORDER:
            row = scenarios[scenario][model]
            values = [fmt(row[m], abs(row[m] - best[m]) < 5e-7) for m in metrics]
            lines.append(f"{SCENARIO_NAMES[scenario]} & {MODEL_NAMES[model]} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (TAB / out_name).write_text("\n".join(lines), encoding="utf-8")


def write_crack_table() -> None:
    rows = read_csv_rows(ROOT / "experiments" / "fresh_final" / "pcm_yolo11s_per_class.csv")
    crack = {
        row["eval_name"]: {
            "map50": float(row["map50"]),
            "map50_95": float(row["map50_95"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        }
        for row in rows
        if row["class_name"] == "crack"
    }
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Crack-specific detection recovery on the PCM road-damage dataset. The refreshed results use the validation-mAP-selected RMR-Net checkpoint and the same frozen YOLO11s detector for every row.}",
        r"\label{tab:pcm_crack}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scenario & Input & Crack mAP50 & Crack mAP50--95 & Prec. & Rec. \\",
        r"\midrule",
    ]
    clean = crack["clean"]
    lines.append(f"clean & clean & {fmt(clean['map50'])} & {fmt(clean['map50_95'])} & {fmt(clean['precision'])} & {fmt(clean['recall'])} \\\\")
    metrics = ["map50", "map50_95", "precision", "recall"]
    for scenario in SCENARIO_ORDER:
        keys = [f"{scenario}_{model}" for model in MODEL_ORDER]
        best = {metric: max(crack[key][metric] for key in keys) for metric in metrics}
        for model in MODEL_ORDER:
            key = f"{scenario}_{model}"
            row = crack[key]
            values = [fmt(row[m], abs(row[m] - best[m]) < 5e-7) for m in metrics]
            lines.append(f"{SCENARIO_NAMES[scenario]} & {MODEL_NAMES[model]} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (TAB / "table_pcm_crack_detection.tex").write_text("\n".join(lines), encoding="utf-8")


def write_selection_table() -> None:
    poth = read_csv_rows(ROOT / "experiments" / "fresh_final_selection" / "pothole_val_selection_summary.csv")
    pcm = read_csv_rows(ROOT / "experiments" / "fresh_final_selection" / "pcm_val_selection_summary.csv")
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Validation-only checkpoint selection for the refreshed task-driven training runs. The held-out test split is evaluated only after this epoch choice is fixed.}",
        r"\label{tab:checkpoint_selection_sensitivity}",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Dataset & Rule & Epoch & Mean val mAP50 \\",
        r"\midrule",
    ]
    for dataset, rows in [("Pothole", poth), ("PCM", pcm)]:
        best = max(rows, key=lambda row: float(row["mean_val_map50"]))
        for row in rows:
            bold = row is best
            epoch = row["epoch"]
            score = fmt(float(row["mean_val_map50"]), bold)
            lines.append(f"{dataset} & validation mAP50 & {epoch} & {score} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TAB / "table_checkpoint_selection_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")


def detection_figure(dataset: str, out_name: str, title: str) -> None:
    _, scenarios = detection_rows(dataset)
    colors = {
        "degraded": "#9aa0a6",
        "dfpir": "#7e9cc9",
        "demoe_auto": "#b58acb",
        "demoe_scenario": "#8f73b7",
        "nafnet": "#d9a441",
        "rmr": "#1b8a7a",
    }
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.0), sharey=True)
    for ax, scenario in zip(axes, SCENARIO_ORDER):
        values = [scenarios[scenario][model]["map50"] for model in MODEL_ORDER]
        ax.bar(range(len(MODEL_ORDER)), values, color=[colors[m] for m in MODEL_ORDER], width=0.72)
        ax.set_title(SCENARIO_NAMES[scenario], fontsize=9)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_NAMES[m].replace("-road", "") for m in MODEL_ORDER], rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 0.7 if dataset == "pothole" else 0.55)
        for i, value in enumerate(values):
            ax.text(i, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=6.5)
    axes[0].set_ylabel("mAP50")
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / out_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    write_detection_table(
        "pothole",
        "table_pothole_detection.tex",
        "tab:pothole_detection",
        "Held-out pothole detection with a frozen YOLO11s detector. RMR-Net rows use the checkpoint selected by validation mAP50 before test evaluation. DeMoE-auto uses the official router; DeMoE-scenario uses known degradation routing.",
    )
    write_detection_table(
        "pcm",
        "table_pcm_detection.tex",
        "tab:pcm_detection",
        "Held-out PCM pothole/crack/manhole detection with a frozen YOLO11s detector. RMR-Net rows use validation-mAP checkpoint selection; mixed degradation combines motion blur and low light.",
    )
    write_crack_table()
    write_selection_table()
    detection_figure("pothole", "fig_pothole_detection_recovery.png", "Pothole detection recovery after restoration")
    detection_figure("pcm", "fig_pcm_detection_recovery.png", "PCM road-damage detection recovery after restoration")


if __name__ == "__main__":
    main()
