"""Build final reviewer-readiness assets for the RMR-Net manuscript.

This pass adds executed polygon-contour accuracy, checkpoint-selection
sensitivity, model complexity, and cleaned claim-boundary tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
EXP = ROOT / "experiments" / "v30_submission_readiness"
SNAKE = ROOT / "runs" / "snake_polygon_accuracy_v30"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt(x: Any, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def tex_escape(text: str) -> str:
    return str(text).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def snake_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in ("motion", "defocus", "lowlight"):
        for source, label in (("degraded", "Degraded"), ("dfpir", "DFPIR"), ("nafnet", "NAFNet-road"), ("rmr", "RMR-Net")):
            path = SNAKE / f"pcm_{scenario}_{source}" / "snake_boundary_metrics.csv"
            data = read_csv(path)
            objects = len(data)
            successes = [r for r in data if r["success"].lower() == "true"]
            # Strict accuracy: failures receive zero overlap scores.
            ious = [float(r["gt_iou"]) if r["success"].lower() == "true" and r["gt_iou"] else 0.0 for r in data]
            dices = [float(r["gt_dice"]) if r["success"].lower() == "true" and r["gt_dice"] else 0.0 for r in data]
            bf1 = [float(r["gt_boundary_f1"]) if r["success"].lower() == "true" and r["gt_boundary_f1"] else 0.0 for r in data]
            chamfer = [float(r["gt_chamfer_px"]) for r in successes if r["gt_chamfer_px"]]
            haus = [float(r["gt_hausdorff_px"]) for r in successes if r["gt_hausdorff_px"]]
            crack_rows = [r for r in data if r["class_name"] == "crack"]
            crack_ious = [
                float(r["gt_iou"]) if r["success"].lower() == "true" and r["gt_iou"] else 0.0
                for r in crack_rows
            ]
            rows.append(
                {
                    "scenario": scenario,
                    "source": label,
                    "objects": objects,
                    "successes": len(successes),
                    "yield": len(successes) / max(objects, 1),
                    "mean_iou_all": float(np.mean(ious)) if ious else 0.0,
                    "mean_dice_all": float(np.mean(dices)) if dices else 0.0,
                    "mean_boundary_f1_all": float(np.mean(bf1)) if bf1 else 0.0,
                    "mean_chamfer_success": float(np.mean(chamfer)) if chamfer else 0.0,
                    "mean_hausdorff_success": float(np.mean(haus)) if haus else 0.0,
                    "crack_iou_all": float(np.mean(crack_ious)) if crack_ious else 0.0,
                }
            )
    return rows


def build_polygon_table_and_figure() -> list[dict[str, Any]]:
    rows = snake_rows()
    write_csv(EXP / "pcm_polygon_contour_accuracy.csv", rows)
    body = "\n".join(
        f"{tex_escape(r['scenario'])} & {tex_escape(r['source'])} & {r['objects']} & {fmt(r['yield'])} & "
        f"{fmt(r['mean_iou_all'])} & {fmt(r['mean_dice_all'])} & {fmt(r['mean_boundary_f1_all'])} & {fmt(r['crack_iou_all'])} \\\\"
        for r in rows
    )
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{PCM polygon contour-accuracy audit on a fixed 80-image held-out subset sampled with seed 2026. Every method receives the same ground-truth boxes; original YOLO-seg polygons are preserved for evaluation. Failed contours receive zero overlap, so IoU/Dice/BF1 combine measurement yield and accuracy.}}
\label{{tab:pcm_polygon_contour_accuracy}}
\scriptsize
\begin{{tabular}}{{llrrrrrr}}
\toprule
Scenario & Source & GT objs & Yield & IoU & Dice & BF1 & Crack IoU \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    write_text(TABLES / "table_pcm_polygon_contour_accuracy.tex", tex)

    scenarios = ["motion", "defocus", "lowlight"]
    sources = ["Degraded", "DFPIR", "NAFNet-road", "RMR-Net"]
    colors = {"Degraded": "#777777", "DFPIR": "#2ca02c", "NAFNet-road": "#ff7f0e", "RMR-Net": "#1f77b4"}
    x = np.arange(len(scenarios))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), dpi=180)
    for i, source in enumerate(sources):
        vals_iou = []
        vals_yield = []
        for scenario in scenarios:
            r = next(row for row in rows if row["scenario"] == scenario and row["source"] == source)
            vals_iou.append(r["mean_iou_all"])
            vals_yield.append(r["yield"])
        xpos = x + (i - 1.5) * width
        axes[0].bar(xpos, vals_iou, width, label=source, color=colors[source])
        axes[1].bar(xpos, vals_yield, width, label=source, color=colors[source])
    for ax, title, ylabel in zip(axes, ["Polygon IoU", "Accepted contour yield"], ["IoU, failures=0", "Success rate"]):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(["motion", "defocus", "low light"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Active-contour accuracy against preserved PCM polygons", y=1.02)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_pcm_polygon_contour_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def build_checkpoint_sensitivity() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, run_name, val_file in (
        ("IVCNZ pothole", "rmrnet_v27_taskloss_pothole_yolo11s", "pothole_val_selection_detail.csv"),
        ("PCM", "rmrnet_v27_taskloss_pcm_yolo11s", "pcm_val_selection_detail.csv"),
    ):
        hist = json.loads((ROOT / "runs" / run_name / "history.json").read_text(encoding="utf-8"))
        val = read_csv(ROOT / "experiments" / "v27_taskloss_yolo11s_eval" / val_file)
        psnr_by_epoch = {int(r["epoch"]): float(r["val_psnr"]) for r in hist}
        map_by_epoch: dict[int, list[float]] = {}
        scenario_best: dict[str, tuple[int, float]] = {}
        for r in val:
            epoch = int(r["epoch"])
            score = float(r["val_map50"])
            scenario = r["scenario"]
            map_by_epoch.setdefault(epoch, []).append(score)
            if scenario not in scenario_best or score > scenario_best[scenario][1]:
                scenario_best[scenario] = (epoch, score)
        mean_map = {epoch: float(np.mean(scores)) for epoch, scores in map_by_epoch.items()}
        psnr_epoch = max(psnr_by_epoch, key=psnr_by_epoch.get)
        map_epoch = max(mean_map, key=mean_map.get)
        # A simple validation-only composite normalizes PSNR and mean mAP to [0,1].
        ps = np.array([psnr_by_epoch[e] for e in sorted(psnr_by_epoch)], dtype=float)
        ms = np.array([mean_map[e] for e in sorted(psnr_by_epoch)], dtype=float)
        ps_n = (ps - ps.min()) / (ps.max() - ps.min() + 1e-9)
        ms_n = (ms - ms.min()) / (ms.max() - ms.min() + 1e-9)
        epochs = sorted(psnr_by_epoch)
        composite_epoch = epochs[int(np.argmax(ps_n + ms_n))]
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "criterion": "PSNR-selected",
                    "selected_epoch": str(psnr_epoch),
                    "val_psnr": psnr_by_epoch[psnr_epoch],
                    "mean_val_map50": mean_map[psnr_epoch],
                    "scenario_epochs": "-",
                },
                {
                    "dataset": dataset,
                    "criterion": "mean-mAP-selected",
                    "selected_epoch": str(map_epoch),
                    "val_psnr": psnr_by_epoch[map_epoch],
                    "mean_val_map50": mean_map[map_epoch],
                    "scenario_epochs": ", ".join(f"{k}:e{v[0]}" for k, v in sorted(scenario_best.items())),
                },
                {
                    "dataset": dataset,
                    "criterion": "PSNR+mAP composite",
                    "selected_epoch": str(composite_epoch),
                    "val_psnr": psnr_by_epoch[composite_epoch],
                    "mean_val_map50": mean_map[composite_epoch],
                    "scenario_epochs": "validation-only normalized sum",
                },
            ]
        )
    write_csv(EXP / "checkpoint_selection_sensitivity.csv", rows)
    body = "\n".join(
        f"{tex_escape(r['dataset'])} & {tex_escape(r['criterion'])} & {r['selected_epoch']} & "
        f"{fmt(r['val_psnr'], 2)} & {fmt(r['mean_val_map50'])} & {tex_escape(r['scenario_epochs'])} \\\\"
        for r in rows
    )
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Validation checkpoint-selection sensitivity for the task-driven fine-tune. The audit compares PSNR-selected, mean-mAP-selected, and normalized composite selection from already saved epoch checkpoints. Test labels are not used.}}
\label{{tab:checkpoint_selection_sensitivity}}
\scriptsize
\begin{{tabularx}}{{\textwidth}}{{llrrrX}}
\toprule
Dataset & Rule & Epoch & Val PSNR & Mean val mAP50 & Scenario-specific mAP epochs \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""
    write_text(TABLES / "table_checkpoint_selection_sensitivity.tex", tex)
    return rows


def build_complexity_table() -> dict[str, Any]:
    result = json.loads((EXP / "rmrnet_complexity_pcm_epoch002.json").read_text(encoding="utf-8"))
    tex = rf"""\begin{{table}}[!t]
\centering
\caption{{Deployed \rmr complexity on a 640$\times$360 road frame. Runtime and memory are measured on the Windows RTX 3050 workstation with CUDA; FLOPs count Conv2d/Linear multiply-adds as two FLOPs.}}
\label{{tab:rmrnet_complexity}}
\small
\begin{{tabular}}{{lr}}
\toprule
Metric & Value \\
\midrule
Parameters & {result['parameters'] / 1e6:.2f} M \\
Approx. FLOPs & {result['gflops_approx']:.2f} G \\
Mean runtime & {result['mean_runtime_ms']:.1f} ms \\
Median runtime & {result['median_runtime_ms']:.1f} ms \\
Peak GPU memory & {result['peak_gpu_memory_mb']:.1f} MB \\
Backend & {tex_escape(result['backend'])} ({tex_escape(result['device'])}) \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write_text(TABLES / "table_rmrnet_complexity.tex", tex)
    return result


def build_readiness_matrix() -> None:
    rows = [
        ("External road-damage benchmark", "protocol/code ready", "RDD2022 is documented and a converter is included; no RDD2022 run is claimed because the external dataset is not local."),
        ("Real road-damage telemetry", "pilot protocol ready", "No synchronized road-damage telemetry exists locally; a real-telemetry pilot-preparation script is included and the manuscript keeps KITTI real OXTS as controlled-blur evidence."),
        ("Polygon contour accuracy", "executed", "PCM raw polygons are preserved for an 80-image fixed-box audit with IoU, Dice, BF1, Chamfer, and Hausdorff."),
        ("Detector audit", "executed", "YOLO11s is the primary frozen detector; independently trained YOLOv8n checkpoints are used only for post-selection robustness."),
        ("DarkIR/GyroDeblurNet", "not executed", "Relevant baselines are cited but not assigned numbers without code/weights and training budget."),
        ("Ablation clarity", "executed", "Core deployed model, grouped train-time regularizers, and post-hoc Snake stage are separated."),
        ("Checkpoint sensitivity", "executed", "Validation PSNR/mAP/composite epoch-selection audit is reported."),
        ("Deployability metrics", "executed", "Params, approximate FLOPs, RTX runtime, and peak memory are reported."),
        ("Reproducibility artifact", "executed", "Scripts, configs, exact splits, provenance, and generated CSVs are included in the zip."),
        ("Front matter/baseline cleanup", "executed", "Placeholders removed from title metadata and internal-development baseline discussion removed."),
    ]
    body = "\n".join(f"{tex_escape(a)} & {tex_escape(b)} & {tex_escape(c)} \\\\" for a, b, c in rows)
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Final submission-readiness matrix. The table separates executed fixes from external-data items that cannot be honestly reported without new data collection or downloads.}}
\label{{tab:submission_readiness_matrix}}
\scriptsize
\begin{{tabularx}}{{\textwidth}}{{p{{0.23\textwidth}}p{{0.16\textwidth}}X}}
\toprule
Review item & Status & How it is handled \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""
    write_text(TABLES / "table_submission_readiness_matrix.tex", tex)
    write_csv(EXP / "submission_readiness_matrix.csv", [{"item": a, "status": b, "handling": c} for a, b, c in rows])


def update_provenance() -> None:
    path = PAPER / "RESULT_PROVENANCE_TABLE.csv"
    text = path.read_text(encoding="utf-8").rstrip()
    additions = [
        "table_pcm_polygon_contour_accuracy,PCM,polygon boundary audit,RMR-Net and baselines,Snake fixed-GT-box outputs,seeded 80-image held-out subset,as applicable,GT boxes fixed for all methods,not detector-dependent,experiments/v30_submission_readiness/pcm_polygon_contour_accuracy.csv,OK",
        "table_checkpoint_selection_sensitivity,IVCNZ/PCM,checkpoint sensitivity,RMR-Net v27 checkpoints,history and validation mAP CSVs,validation-only audit,synthetic proxy metadata,not a test selector,YOLO11s validation detectors,experiments/v30_submission_readiness/checkpoint_selection_sensitivity.csv,OK",
        "table_rmrnet_complexity,PCM,model complexity,RMR-Net checkpoint,runs/rmrnet_v27_taskloss_pcm_yolo11s/rcadnet_epoch_002.pth,not a selector,synthetic proxy metadata,all_restored,not applicable,experiments/v30_submission_readiness/rmrnet_complexity_pcm_epoch002.json,OK",
        "table_submission_readiness_matrix,all,review checklist,all artifacts,executed and not-executed boundaries,not a selector,as applicable,as applicable,as applicable,experiments/v30_submission_readiness/submission_readiness_matrix.csv,OK",
    ]
    for line in additions:
        if line not in text:
            text += "\n" + line
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    polygon = build_polygon_table_and_figure()
    selection = build_checkpoint_sensitivity()
    complexity = build_complexity_table()
    build_readiness_matrix()
    update_provenance()
    manifest = {
        "tables": [
            str(TABLES / "table_pcm_polygon_contour_accuracy.tex"),
            str(TABLES / "table_checkpoint_selection_sensitivity.tex"),
            str(TABLES / "table_rmrnet_complexity.tex"),
            str(TABLES / "table_submission_readiness_matrix.tex"),
        ],
        "figures": [str(FIGURES / "fig_pcm_polygon_contour_accuracy.png")],
        "csv": [
            str(EXP / "pcm_polygon_contour_accuracy.csv"),
            str(EXP / "checkpoint_selection_sensitivity.csv"),
            str(EXP / "submission_readiness_matrix.csv"),
        ],
        "json": [str(EXP / "rmrnet_complexity_pcm_epoch002.json")],
        "polygon_rows": len(polygon),
        "selection_rows": len(selection),
        "parameters": complexity["parameters"],
    }
    write_text(EXP / "V30_SUBMISSION_READINESS_MANIFEST.json", json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
