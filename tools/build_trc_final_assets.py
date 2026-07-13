from __future__ import annotations

"""Build Transportation Research Part C paper assets from the final 30-epoch run."""

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TRC = ROOT / "paper_trc_rmrnet"
TAB = TRC / "tables"
FIG = TRC / "figures"
IEEE = ROOT / "paper_ieee_tits_rmrnet"


MODEL_NAMES = {
    "degraded": "Degraded",
    "dfpir": "DFPIR",
    "demoe_auto": "DeMoE-auto",
    "demoe_scenario": "DeMoE-scenario",
    "nafnet": "NAFNet-road",
    "rmr": "RMR-Net",
}

SCENARIO_NAMES = {
    "motion": "Motion blur",
    "defocus": "Defocus blur",
    "lowlight": "Low light",
    "mixed": "Mixed motion + low light",
}

SCENARIOS = ["motion", "defocus", "lowlight", "mixed"]
MODELS = ["degraded", "dfpir", "demoe_auto", "demoe_scenario", "nafnet", "rmr"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt(value: float, digits: int = 3, bold: bool = False) -> str:
    text = f"{value:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def detection_rows(dataset: str) -> tuple[dict[str, float], dict[str, dict[str, dict[str, float]]]]:
    base_rows = {row["name"]: row for row in read_csv(ROOT / "experiments" / "fresh_final" / f"{dataset}_yolo11s.csv")}
    rmr_rows = {
        row["name"].replace("_rmr_selected", ""): row
        for row in read_csv(ROOT / "experiments" / "trc_final_30ep" / f"{dataset}_test_rmrnet_selected.csv")
    }
    clean = {metric: float(base_rows["clean"][metric]) for metric in ["map50", "map50_95", "precision", "recall"]}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in SCENARIOS:
        result[scenario] = {}
        for model in MODELS:
            if model == "rmr":
                row = rmr_rows[scenario]
            else:
                row = base_rows[f"{scenario}_{model}"]
            result[scenario][model] = {
                "map50": float(row["map50"]),
                "map50_95": float(row["map50_95"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            }
    return clean, result


def write_detection_table(dataset: str, out_name: str, caption: str, label: str) -> None:
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
        r"Scenario & Input & mAP50 & mAP50--95 & Precision & Recall \\",
        r"\midrule",
        "Clean & Native clean & "
        + " & ".join(fmt(clean[m]) for m in ["map50", "map50_95", "precision", "recall"])
        + r" \\",
    ]
    for scenario in SCENARIOS:
        best = {
            metric: max(scenarios[scenario][model][metric] for model in MODELS)
            for metric in ["map50", "map50_95", "precision", "recall"]
        }
        for model in MODELS:
            row = scenarios[scenario][model]
            values = [
                fmt(row[metric], bold=abs(row[metric] - best[metric]) < 1e-9)
                for metric in ["map50", "map50_95", "precision", "recall"]
            ]
            lines.append(f"{SCENARIO_NAMES[scenario]} & {MODEL_NAMES[model]} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    write_text(TAB / out_name, "\n".join(lines))


def write_crack_table() -> None:
    base = {
        row["eval_name"]: row
        for row in read_csv(ROOT / "experiments" / "fresh_final" / "pcm_yolo11s_per_class.csv")
        if row["class_name"] == "crack"
    }
    rmr = {
        row["eval_name"].replace("_rmr_selected", ""): row
        for row in read_csv(ROOT / "experiments" / "trc_final_30ep" / "pcm_test_rmrnet_selected_per_class.csv")
        if row["class_name"] == "crack"
    }
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Crack-specific PCM detection. The RMR-Net rows are from the 30-epoch checkpoint selected by validation mAP50; all other rows are executed baselines under the same detector protocol.}",
        r"\label{tab:pcm_crack}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scenario & Input & Crack mAP50 & Crack mAP50--95 & Precision & Recall \\",
        r"\midrule",
    ]
    clean = base["clean"]
    lines.append(
        "Clean & Native clean & "
        + " & ".join(fmt(float(clean[m])) for m in ["map50", "map50_95", "precision", "recall"])
        + r" \\"
    )
    for scenario in SCENARIOS:
        values_by_model = {}
        for model in MODELS:
            row = rmr[scenario] if model == "rmr" else base[f"{scenario}_{model}"]
            values_by_model[model] = {m: float(row[m]) for m in ["map50", "map50_95", "precision", "recall"]}
        best = {metric: max(values_by_model[model][metric] for model in MODELS) for metric in ["map50", "map50_95", "precision", "recall"]}
        for model in MODELS:
            row = values_by_model[model]
            vals = [fmt(row[m], bold=abs(row[m] - best[m]) < 1e-9) for m in ["map50", "map50_95", "precision", "recall"]]
            lines.append(f"{SCENARIO_NAMES[scenario]} & {MODEL_NAMES[model]} & " + " & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    write_text(TAB / "table_pcm_crack_detection.tex", "\n".join(lines))


def write_selection_table() -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Checkpoint provenance. Controlled test rows use the 30-epoch RMR-Net checkpoints selected only by validation mAP50. The native GT49 safety audit adds an identity-calibration stage using non-annotated cam1 frames that exclude the 49 GT49 test images.}",
        r"\label{tab:checkpoint_selection}",
        r"\small",
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"Dataset & Selected epoch & Selection / calibration rule \\",
        r"\midrule",
    ]
    labels = {"pothole": "IVCNZ pothole", "pcm": "PCM"}
    for dataset in ["pothole", "pcm"]:
        best = json.loads((ROOT / "experiments" / "trc_final_30ep" / f"{dataset}_best_by_val_map.json").read_text(encoding="utf-8"))
        lines.append(f"{labels[dataset]} & {best['epoch']} & mean validation mAP50 = {fmt(float(best['mean_val_map50']))} \\\\")
    lines.append(r"GT49 native field & -- & identity calibration on non-GT49 cam1 frames \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text(TAB / "table_checkpoint_selection.tex", "\n".join(lines))


def restoration_sources(dataset: str) -> list[dict[str, str]]:
    if dataset == "pothole":
        old = read_csv(ROOT / "runs" / "bench_pothole_test_rmr_metadata_naf_dfpir" / "metrics.csv")
        demoe_auto = read_csv(ROOT / "runs" / "bench_pothole_test_demoe_auto" / "metrics.csv")
        demoe_scenario = read_csv(ROOT / "runs" / "bench_pothole_test_demoe_scenario" / "metrics.csv")
        rmr = read_csv(ROOT / "runs" / "bench_trc_final_rmrnet_pothole_30ep" / "metrics.csv")
    else:
        old = read_csv(ROOT / "runs" / "bench_pcm_test_rmr_metadata_naf_dfpir" / "metrics.csv")
        demoe_auto = read_csv(ROOT / "runs" / "bench_pcm_test_demoe_auto" / "metrics.csv")
        demoe_scenario = read_csv(ROOT / "runs" / "bench_pcm_test_demoe_scenario" / "metrics.csv")
        rmr = read_csv(ROOT / "runs" / "bench_trc_final_rmrnet_pcm_30ep" / "metrics.csv")
    rows = []
    for row in rmr:
        new = dict(row)
        new["model"] = "RMR-Net"
        rows.append(new)
    for row in old:
        if row["model"] in {"NAFNet-road", "DFPIR-CVPR2025"}:
            rows.append(row)
    for row in demoe_auto:
        new = dict(row)
        new["model"] = "DeMoE-auto"
        rows.append(new)
    for row in demoe_scenario:
        new = dict(row)
        new["model"] = "DeMoE-scenario"
        rows.append(new)
    return rows


def write_restoration_table() -> None:
    all_rows: list[tuple[str, dict[str, str]]] = []
    for dataset in ["pothole", "pcm"]:
        for row in restoration_sources(dataset):
            all_rows.append((dataset, row))
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Full-reference restoration quality and CUDA runtime on held-out road-image test sets. RMR-Net rows are regenerated from the final 30-epoch checkpoints; baselines are the same executed DFPIR, DeMoE, and NAFNet runs used in the detector benchmark.}",
        r"\label{tab:restoration}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Dataset & Scenario & Model & PSNR & SSIM & ms/img \\",
        r"\midrule",
    ]
    pretty_model = {"DFPIR-CVPR2025": "DFPIR"}
    pretty_scenario = {
        "motion_horizontal_medium": "Motion",
        "defocus_medium": "Defocus",
        "lowlight_medium": "Low light",
    }
    for dataset in ["pothole", "pcm"]:
        for scenario in ["motion_horizontal_medium", "defocus_medium", "lowlight_medium"]:
            group = [row for ds, row in all_rows if ds == dataset and row["scenario"] == scenario]
            best_psnr = max(float(row["psnr"]) for row in group)
            best_ssim = max(float(row["ssim"]) for row in group)
            best_runtime = min(float(row["mean_runtime_ms"]) for row in group)
            order = ["RMR-Net", "NAFNet-road", "DFPIR-CVPR2025", "DeMoE-auto", "DeMoE-scenario"]
            for model in order:
                row = next(r for r in group if r["model"] == model)
                lines.append(
                    f"{dataset.upper()} & {pretty_scenario[scenario]} & {pretty_model.get(model, model)} & "
                    + " & ".join(
                        [
                            fmt(float(row["psnr"]), 2, abs(float(row["psnr"]) - best_psnr) < 1e-9),
                            fmt(float(row["ssim"]), 3, abs(float(row["ssim"]) - best_ssim) < 1e-9),
                            fmt(float(row["mean_runtime_ms"]), 1, abs(float(row["mean_runtime_ms"]) - best_runtime) < 1e-9),
                        ]
                    )
                    + r" \\"
                )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    write_text(TAB / "table_restoration.tex", "\n".join(lines))


def detection_figure(dataset: str, out_name: str, title: str) -> None:
    _, scenarios = detection_rows(dataset)
    colors = {
        "degraded": "#8c939c",
        "dfpir": "#4e79a7",
        "demoe_auto": "#b07aa1",
        "demoe_scenario": "#8064a2",
        "nafnet": "#f2b447",
        "rmr": "#0f8b8d",
    }
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.0), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        values = [scenarios[scenario][model]["map50"] for model in MODELS]
        ax.bar(range(len(MODELS)), values, color=[colors[m] for m in MODELS], width=0.72)
        ax.set_title(SCENARIO_NAMES[scenario], fontsize=9)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_NAMES[m].replace("-road", "") for m in MODELS], rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 0.68 if dataset == "pothole" else 0.58)
        for i, value in enumerate(values):
            ax.text(i, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=6)
    axes[0].set_ylabel("mAP50")
    fig.suptitle(title, fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / out_name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def copy_existing_figures() -> None:
    for name in [
        "fig_rmrnet_architecture.png",
        "fig_geotagged_all49_prf.png",
        "fig_geotagged_eta_sweep.png",
        "fig_geotagged_tau_sweep.png",
        "fig_native_failure_modes.png",
        "fig_fidelity_detection_correlation.png",
        "fig_snake_boundary_cross_model.png",
    ]:
        src = IEEE / "figures" / name
        if src.exists():
            shutil.copy2(src, FIG / name)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    write_detection_table(
        "pothole",
        "table_pothole_detection.tex",
        "Held-out pothole detection under controlled road-relevant degradation. RMR-Net uses the final 30-epoch checkpoint selected by validation mAP50; baselines use the same frozen YOLO11s detector.",
        "tab:pothole_detection",
    )
    write_detection_table(
        "pcm",
        "table_pcm_detection.tex",
        "Held-out PCM pothole/crack/manhole detection. RMR-Net uses the final 30-epoch checkpoint selected by validation mAP50; all rows use the same frozen YOLO11s detector.",
        "tab:pcm_detection",
    )
    write_crack_table()
    write_selection_table()
    write_restoration_table()
    detection_figure("pothole", "fig_pothole_detection_recovery.png", "Pothole detection recovery after restoration")
    detection_figure("pcm", "fig_pcm_detection_recovery.png", "PCM road-damage detection recovery after restoration")
    copy_existing_figures()


if __name__ == "__main__":
    main()
