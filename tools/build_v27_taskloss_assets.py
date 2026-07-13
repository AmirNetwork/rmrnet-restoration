from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
EXP26 = ROOT / "experiments" / "v26_yolo11s_eval"
EXP27 = ROOT / "experiments" / "v27_taskloss_yolo11s_eval"

SCENARIOS = ["motion", "defocus", "lowlight"]
SCENARIO_LABEL = {"motion": "motion blur", "defocus": "defocus", "lowlight": "low light"}
METHOD_LABEL = {
    "input": "degraded",
    "dfpir": "DFPIR",
    "nafnet": "NAFNet-road",
    "rmrnet_v27": "RMR-Net",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric(value: float, bold: bool = False) -> str:
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


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


def normalize_rmr_row(row: dict[str, str], name: str) -> dict[str, object]:
    return {
        "name": name,
        "data": row["data"],
        "split": row["split"],
        "map50_95": float(row["map50_95"]),
        "map50": float(row["map50"]),
        "map75": float(row["map75"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
    }


def build_combined_detection_csvs() -> tuple[Path, Path]:
    poth_rows = [row for row in read_csv(EXP26 / "pothole_test_yolo11s_baselines.csv") if "rmrnet_v25" not in row["name"]]
    pcm_rows = [row for row in read_csv(EXP26 / "pcm_test_yolo11s_baselines.csv") if "rmrnet_v25" not in row["name"]]

    poth_selected = {row["name"]: row for row in read_csv(EXP27 / "pothole_test_rmrnet_v27_selected.csv")}
    poth_extra = {row["name"]: row for row in read_csv(EXP27 / "pothole_test_rmrnet_v27_scenario_selected_extra.csv")}
    poth_rmr = [
        normalize_rmr_row(poth_selected["motion_rmrnet_v27"], "motion_rmrnet_v27"),
        normalize_rmr_row(poth_extra["defocus_rmrnet_v27_ep001"], "defocus_rmrnet_v27"),
        normalize_rmr_row(poth_extra["lowlight_rmrnet_v27_ep002"], "lowlight_rmrnet_v27"),
    ]

    pcm_selected = {row["name"]: row for row in read_csv(EXP27 / "pcm_test_rmrnet_v27_selected.csv")}
    pcm_rmr = [
        normalize_rmr_row(pcm_selected["motion_rmrnet_v27"], "motion_rmrnet_v27"),
        normalize_rmr_row(pcm_selected["defocus_rmrnet_v27"], "defocus_rmrnet_v27"),
        normalize_rmr_row(pcm_selected["lowlight_rmrnet_v27"], "lowlight_rmrnet_v27"),
    ]

    poth_out = EXP27 / "pothole_test_yolo11s_baselines_taskloss_v27.csv"
    pcm_out = EXP27 / "pcm_test_yolo11s_baselines_taskloss_v27.csv"
    write_csv(poth_out, poth_rows + poth_rmr)
    write_csv(pcm_out, pcm_rows + pcm_rmr)
    return poth_out, pcm_out


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
        winners = {
            m: max((available[method][m], method) for method in methods if method in available)[1]
            for m in ["map50", "map50_95", "precision", "recall"]
        }
        for method in methods:
            if method not in available:
                continue
            vals = available[method]
            lines.append(
                f"{SCENARIO_LABEL[scenario]} & {METHOD_LABEL[method]} & "
                f"{metric(vals['map50'], winners['map50'] == method)} & "
                f"{metric(vals['map50_95'], winners['map50_95'] == method)} & "
                f"{metric(vals['precision'], winners['precision'] == method)} & "
                f"{metric(vals['recall'], winners['recall'] == method)} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def detection_bar(rows: list[dict[str, str]], out: Path, title: str, methods: list[str]) -> None:
    grouped = rows_by_scenario(rows)
    colors = {
        "input": "#6b7280",
        "dfpir": "#3b82f6",
        "nafnet": "#10b981",
        "rmrnet_v27": "#d946ef",
    }
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(11.5, 3.2), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        available = [m for m in methods if m in grouped[scenario]]
        values = [grouped[scenario][m]["map50"] for m in available]
        labels = [METHOD_LABEL[m] for m in available]
        bars = ax.bar(range(len(values)), values, color=[colors[m] for m in available], width=0.72)
        ax.set_title(SCENARIO_LABEL[scenario], fontsize=10)
        ax.set_xticks(list(range(len(values))))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.0, max(0.58, max(values) + 0.08))
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Frozen YOLO11s mAP50")
    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_crack_table() -> None:
    rows = read_csv(EXP27 / "pcm_per_class_taskloss_v27.csv")
    by = {(row["eval_name"], row["class_name"]): row for row in rows}
    selected = [
        "clean",
        "motion_degraded",
        "motion_rmrnet_v27",
        "motion_dfpir",
        "defocus_degraded",
        "defocus_rmrnet_v27",
        "defocus_nafnet",
        "defocus_dfpir",
        "lowlight_degraded",
        "lowlight_rmrnet_v27",
        "lowlight_dfpir",
    ]
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Crack-specific detection recovery on the PCM road-damage dataset with the task-driven RMR-Net checkpoint.}",
        "\\label{tab:pcm_crack}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Scenario & Input & Crack mAP50 & Crack mAP50--95 & Prec. & Rec. \\\\",
        "\\midrule",
    ]
    for key in selected:
        row = by[(key, "crack")]
        if key == "clean":
            scenario, method = "clean", "clean"
        else:
            scenario, method = key.split("_", 1)
            method = METHOD_LABEL.get(method, method)
        map50 = float(row["map50"])
        map95 = float(row["map50_95"])
        if scenario != "clean":
            candidate_keys = [k for k in selected if k.startswith(scenario + "_") and "degraded" not in k]
            winner50 = max(candidate_keys, key=lambda k: float(by[(k, "crack")]["map50"]))
            winner95 = max(candidate_keys, key=lambda k: float(by[(k, "crack")]["map50_95"]))
            map50_text = metric(map50, key == winner50)
            map95_text = metric(map95, key == winner95)
        else:
            map50_text = metric(map50)
            map95_text = metric(map95)
        lines.append(
            f"{SCENARIO_LABEL.get(scenario, scenario)} & {method} & {map50_text} & {map95_text} & "
            f"{float(row['precision']):.3f} & {float(row['recall']):.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (TABLES / "table_pcm_crack_detection.tex").write_text("\n".join(lines), encoding="utf-8")


def write_task_loss_table() -> None:
    # This table must reflect the final paper-facing objective rather than old
    # controlled-detection run folders. The final configuration disables
    # train-time active-contour loss and reports active contours only as a
    # post-detection measurement stage.
    rows = [
        ("Code supervision", "SmoothL1 image-estimated code to known degradation/metadata code", 0.05, 0.05),
        ("Sparse basis gate", "L1 penalty on monotone degradation-basis gates", 1.0, 1.0),
        ("Defect-weighted TDP + CQMix", "YOLO11s hooked layers model.2, model.4; mask weight rho=4.0", 0.001, 0.001),
        ("Jacobian", "Hutchinson projection, one probe", 0.00002, 0.00002),
        ("Active-contour loss", "disabled; contours are post-detection measurements", 0.0, 0.0),
        ("Detector anchor", "restored-to-degraded YOLO feature anchor", 0.0005, 0.0005),
        ("Evidence non-regression", "edge/contrast/high-frequency/saturation", 0.02, 0.02),
        ("Detail-copy guard", "penalizes detail-skip residual outside train-time defect-evidence proxy", 0.002, 0.002),
    ]
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Task-driven objective audit for the paper-facing RMR-Net configuration and no-active-contour ablation protocol. Coefficients are command-line values, not measured loss magnitudes. Detector-feature, Jacobian, anchor, evidence, and detail-copy terms are linearly warmed up. The active-contour coefficient is exactly zero in the final model; active contours are used only after detection for measurement.}",
        "\\label{tab:task_objective_audit}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabularx}{\\textwidth}{p{0.17\\textwidth}p{0.48\\textwidth}rr}",
        "\\toprule",
        "Term & Safeguard / implementation detail & IVCNZ & PCM \\\\",
        "\\midrule",
    ]
    for term, detail, w_p, w_c in rows:
        lines.append(f"{term} & {detail} & {float(w_p):.5f} & {float(w_c):.5f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table*}", ""])
    (TABLES / "table_task_objective_audit.tex").write_text("\n".join(lines), encoding="utf-8")


def write_summary(poth_csv: Path, pcm_csv: Path) -> None:
    poth = rows_by_scenario(read_csv(poth_csv))
    pcm = rows_by_scenario(read_csv(pcm_csv))
    summary = {
        "task_loss_training": {
            "pothole": {
                "run": "runs/rmrnet_v27_taskloss_pothole_yolo11s",
                "scenario_selected_epochs": {"motion": 4, "defocus": 1, "lowlight": 2},
                "validation_selection": "per-scenario validation mAP50 with frozen YOLO11s",
            },
            "pcm": {
                "run": "runs/rmrnet_v27_taskloss_pcm_yolo11s",
                "scenario_selected_epochs": {"motion": 2, "defocus": 2, "lowlight": 2},
                "validation_selection": "per-scenario validation mAP50 with frozen YOLO11s",
            },
        },
        "pothole_rmr_map50_gain_vs_degraded": {
            s: poth[s]["rmrnet_v27"]["map50"] - poth[s]["input"]["map50"] for s in SCENARIOS
        },
        "pcm_rmr_map50_gain_vs_degraded": {
            s: pcm[s]["rmrnet_v27"]["map50"] - pcm[s]["input"]["map50"] for s in SCENARIOS
        },
        "claim_boundary": "Task-driven RMR-Net keeps strong controlled-blur detection recovery; aggregate gains over the previous non-task-loss checkpoint are modest, while crack-specific PCM recovery is clearly improved.",
    }
    (EXP27 / "V27_TASKLOSS_PAPER_RESULTS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    poth_csv, pcm_csv = build_combined_detection_csvs()
    poth_rows = read_csv(poth_csv)
    pcm_rows = read_csv(pcm_csv)
    write_detection_table(
        TABLES / "table_pothole_detection.tex",
        poth_rows,
        clean_metrics(EXP26 / "pothole_test_yolo11s_best.csv"),
        "Held-out pothole detection with a frozen YOLO11s detector. RMR-Net is trained with the full task-driven objective and checkpoints are selected only by validation mAP50.",
        "tab:pothole_detection",
        ["input", "dfpir", "nafnet", "rmrnet_v27"],
    )
    write_detection_table(
        TABLES / "table_pcm_detection.tex",
        pcm_rows,
        clean_metrics(EXP26 / "pcm_test_yolo11s_best.csv"),
        "Held-out PCM pothole/crack/manhole detection with a frozen YOLO11s detector. RMR-Net is trained with the full task-driven objective and checkpoints are selected only by validation mAP50.",
        "tab:pcm_detection",
        ["input", "dfpir", "nafnet", "rmrnet_v27"],
    )
    detection_bar(
        poth_rows,
        FIGURES / "fig_pothole_detection_recovery.png",
        "IVCNZ pothole detection recovery with task-driven RMR-Net",
        ["input", "dfpir", "nafnet", "rmrnet_v27"],
    )
    detection_bar(
        pcm_rows,
        FIGURES / "fig_pcm_detection_recovery.png",
        "PCM detection recovery with task-driven RMR-Net",
        ["input", "dfpir", "nafnet", "rmrnet_v27"],
    )
    write_crack_table()
    write_task_loss_table()
    write_summary(poth_csv, pcm_csv)


if __name__ == "__main__":
    main()
