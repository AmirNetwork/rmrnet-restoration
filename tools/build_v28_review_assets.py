"""Build reviewer-response tables and figures for the RMR-Net manuscript."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "v28_review_robustness"
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(x: Any, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def scenario_label(dataset: str, scenario: str) -> str:
    prefix = "IVCNZ" if dataset.startswith("IVCNZ") else "PCM"
    return f"{prefix} {scenario}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_cross_detector_table(metrics: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in metrics:
        key = (row["detector"], row["dataset"], row["scenario"])
        grouped.setdefault(key, {})[row["state"]] = row

    rows = []
    for key in sorted(grouped, key=lambda k: (k[0], k[1], k[2])):
        det, dataset, scenario = key
        degraded = grouped[key]["degraded"]
        restored = grouped[key]["rmrnet"]
        delta = float(restored["map50"]) - float(degraded["map50"])
        rows.append((det, scenario_label(dataset, scenario), degraded["map50"], restored["map50"], delta))

    body = "\n".join(
        f"{det} & {label} & {fmt(deg)} & {fmt(res)} & {fmt(delta)} \\\\"
        for det, label, deg, res, delta in rows
    )
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Cross-detector held-out detection audit. YOLO11s is the primary frozen detector used for validation checkpoint selection; YOLOv8n is trained independently on clean images and used only as an auxiliary robustness audit. Values are Ultralytics test \mapfifty.}}
\label{{tab:cross_detector_audit}}
\scriptsize
\begin{{tabular}}{{llccc}}
\toprule
Detector & Test condition & Degraded & \rmr & $\Delta$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    write(TABLES / "table_cross_detector_audit.tex", tex)


def build_bootstrap_table(stats: list[dict[str, str]]) -> None:
    rows = []
    for row in stats:
        ci = f"[{fmt(row['delta_ci_low'])}, {fmt(row['delta_ci_high'])}]"
        p = "<0.005" if float(row["paired_bootstrap_p"]) < 0.005 else fmt(row["paired_bootstrap_p"])
        rows.append((row["detector"], scenario_label(row["dataset"], row["scenario"]), row["delta_ap50"], ci, p))
    body = "\n".join(
        f"{det} & {label} & {fmt(delta)} & {ci} & {p} \\\\"
        for det, label, delta, ci, p in rows
    )
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Paired bootstrap uncertainty for RMR-Net detection recovery. We resample held-out test images with replacement using the same bootstrap draw for degraded and restored outputs. The table reports the AP50 gain of \rmr over the degraded input using 400 bootstrap replicates; it is an uncertainty audit and is not used for model selection.}}
\label{{tab:bootstrap_detection_ci}}
\scriptsize
\begin{{tabular}}{{llccc}}
\toprule
Detector & Test condition & AP50 gain & 95\% CI & Paired bootstrap $p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    write(TABLES / "table_bootstrap_detection_ci.tex", tex)


def build_holm_table(stats: list[dict[str, str]]) -> None:
    tests = []
    for row in stats:
        p = float(row["paired_bootstrap_p"])
        tests.append(
            {
                "detector": row["detector"],
                "condition": scenario_label(row["dataset"], row["scenario"]),
                "delta": float(row["delta_ap50"]),
                "p": p,
            }
        )
    tests.sort(key=lambda row: row["p"])
    m = len(tests)
    running = 0.0
    for i, row in enumerate(tests):
        adjusted = min(1.0, (m - i) * row["p"])
        running = max(running, adjusted)
        row["holm"] = running
        row["significant"] = running < 0.05
    tests.sort(key=lambda row: (row["detector"], row["condition"]))

    def fmt_p(value: float) -> str:
        return "<0.005" if value < 0.005 else fmt(value)

    body = "\n".join(
        f"{row['detector']} & {row['condition']} & {fmt(row['delta'])} & {fmt_p(row['p'])} & {fmt_p(row['holm'])} & {'yes' if row['significant'] else 'no'} \\\\"
        for row in tests
    )
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Holm-corrected paired-bootstrap detection audit. The correction is applied across all degraded-versus-\rmr AP50 tests in Table~\ref{{tab:bootstrap_detection_ci}}.}}
\label{{tab:holm_bootstrap}}
\scriptsize
\begin{{tabular}}{{llrrrr}}
\toprule
Detector & Condition & AP50 gain & Raw $p$ & Holm $p$ & Sig. \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    write(TABLES / "table_holm_bootstrap.tex", tex)


def build_repro_table() -> None:
    rows = [
        ("RMR-Net restorer", "AdamW, CUDA AMP, patch size 128, batch size 1, width 40, seed 2026, validation PSNR logged every epoch."),
        ("Task-driven fine-tune", "No-active-contour final run from the validation-selected warm-start checkpoint; TDP/CQMix, Jacobian, detector anchor, evidence non-regression, sparse degradation gating, and supervised code learning are active after warmup."),
        ("Main detector", "YOLO11s fine-tuned on clean road-damage images for 80 epochs; batch 8; image size 640; seed 2026; validation-selected best checkpoint frozen for all degraded/restored tests."),
        ("Auxiliary detector audit", "YOLOv8n fine-tuned on the same clean splits for 40 epochs; batch 8; image size 640; seed 2026; not used for RMR-Net checkpoint selection."),
        ("Checkpoint selection", "RMR-Net checkpoints selected only by validation detector mAP50; held-out test mAP is never used for checkpoint or gate selection."),
        ("Uncertainty", "Paired bootstrap over held-out test images with 400 replicates and seed 2026; same resampled images are used for degraded and restored outputs."),
        ("Hardware/runtime", "Windows workstation, NVIDIA GeForce RTX 3050 6 GB, PyTorch 2.11.0+cu128; restoration and detector inference executed on CUDA."),
    ]
    body = "\n".join(f"{name} & {desc} \\\\" for name, desc in rows)
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Reproducibility-critical training and evaluation details. The entries summarize the executed code paths used for the reported tables and are mirrored by the released configuration and audit files.}}
\label{{tab:reproducibility_protocol}}
\scriptsize
\begin{{tabularx}}{{\textwidth}}{{p{{0.22\textwidth}}X}}
\toprule
Component & Executed protocol \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""
    write(TABLES / "table_reproducibility_protocol.tex", tex)


def build_delta_figure(stats: list[dict[str, str]]) -> None:
    order = [
        ("IVCNZ pothole", "motion blur"),
        ("IVCNZ pothole", "defocus"),
        ("IVCNZ pothole", "low light"),
        ("IVCNZ pothole", "mixed motion+low light"),
        ("PCM", "motion blur"),
        ("PCM", "defocus"),
        ("PCM", "low light"),
        ("PCM", "mixed motion+low light"),
    ]
    detectors = ["YOLO11s", "YOLOv8n"]
    lookup = {(r["detector"], r["dataset"], r["scenario"]): r for r in stats}
    labels = [
        scenario_label(dataset, scenario)
        .replace(" motion blur", "\nmotion")
        .replace(" defocus", "\ndefocus")
        .replace(" low light", "\nlow light")
        .replace(" mixed motion+low light", "\nmixed")
        for dataset, scenario in order
    ]
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=180)
    colors = {"YOLO11s": "#1f77b4", "YOLOv8n": "#2ca02c"}
    for i, det in enumerate(detectors):
        deltas = []
        lows = []
        highs = []
        for dataset, scenario in order:
            row = lookup[(det, dataset, scenario)]
            delta = float(row["delta_ap50"])
            deltas.append(delta)
            lows.append(delta - float(row["delta_ci_low"]))
            highs.append(float(row["delta_ci_high"]) - delta)
        xpos = x + (i - 0.5) * width
        ax.bar(xpos, deltas, width=width, label=det, color=colors[det], alpha=0.9)
        ax.errorbar(xpos, deltas, yerr=np.vstack([lows, highs]), fmt="none", ecolor="#222222", capsize=3, lw=1)
    ax.axhline(0.0, color="#444444", lw=1)
    ax.set_ylabel("AP50 gain over degraded input")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Paired bootstrap detection recovery across YOLO checkpoints")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_cross_detector_bootstrap_delta.png")
    plt.close(fig)


def update_provenance() -> None:
    provenance = PAPER / "RESULT_PROVENANCE_TABLE.csv"
    existing = provenance.read_text(encoding="utf-8").rstrip()
    additions = [
        "table_cross_detector_audit,IVCNZ/PCM,controlled and mixed scenarios,RMR-Net degraded-vs-restored,YOLO11s and YOLOv8n clean detectors,held-out test audit only; not checkpoint selection,synthetic proxy metadata,all_restored,YOLO11s primary and YOLOv8n auxiliary,experiments/v28_review_robustness/cross_detector_metrics.csv,OK",
        "table_bootstrap_detection_ci,IVCNZ/PCM,controlled and mixed scenarios,RMR-Net degraded-vs-restored,YOLO11s and YOLOv8n clean detectors,paired bootstrap held-out test audit,synthetic proxy metadata,all_restored,YOLO11s primary and YOLOv8n auxiliary,experiments/v28_review_robustness/bootstrap_ap50_deltas.csv,OK",
        "table_reproducibility_protocol,all,training/evaluation protocol,RMR-Net and detectors,config files and training recipes,not a performance selector,as applicable,as applicable,as applicable,paper_ieee_tits_rmrnet/tables/table_reproducibility_protocol.tex,OK",
    ]
    for line in additions:
        if line not in existing:
            existing += "\n" + line
    provenance.write_text(existing + "\n", encoding="utf-8")


def main() -> None:
    metrics = read_csv(EXP / "cross_detector_metrics.csv")
    stats = read_csv(EXP / "bootstrap_ap50_deltas.csv")
    build_cross_detector_table(metrics)
    build_bootstrap_table(stats)
    build_holm_table(stats)
    build_repro_table()
    build_delta_figure(stats)
    update_provenance()
    summary = {
        "tables": [
            str(TABLES / "table_cross_detector_audit.tex"),
            str(TABLES / "table_bootstrap_detection_ci.tex"),
            str(TABLES / "table_holm_bootstrap.tex"),
            str(TABLES / "table_reproducibility_protocol.tex"),
        ],
        "figure": str(FIGURES / "fig_cross_detector_bootstrap_delta.png"),
    }
    (EXP / "V28_REVIEW_ASSET_MANIFEST.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
