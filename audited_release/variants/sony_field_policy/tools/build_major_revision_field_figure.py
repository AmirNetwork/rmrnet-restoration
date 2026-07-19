#!/usr/bin/env python3
# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

"""Build the audited ILX-RD46 field-system figure from immutable CSV results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments" / "major_revision_evidence_20260715"
OUT = ROOT / "paper_automation_in_construction_rmrnet" / "figures" / "fig_ilx_audited_system_results.png"


def main() -> None:
    direct = pd.read_csv(EVIDENCE / "ilx_direct_single_view_metrics.csv")
    budget = pd.read_csv(EVIDENCE / "ilx_temporal_holdout_fixed_budget.csv")

    direct = direct[direct["run"].isin(["raw", "rmr_blind", "rmr_metadata", "demoe_auto", "instructir_generic"])]
    budget = budget[budget["run"].isin(["raw", "raw_plus_rmr", "raw_plus_nafnet", "raw_plus_dfpir", "raw_plus_demoe_auto", "raw_plus_instructir"])]

    colors = ["#4D4D4D", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    x = np.arange(len(direct))
    width = 0.36
    axes[0].bar(x - width / 2, direct["f1_iou10"], width, color="#0072B2", label="F1 at IoU 0.10")
    axes[0].bar(x + width / 2, direct["f1_iou50"], width, color="#E69F00", label="F1 at IoU 0.50")
    axes[0].set_xticks(x, ["Raw", "RMR image-only", "RMR context", "DeMoE-auto", "InstructIR"], rotation=22, ha="right")
    axes[0].set_ylim(0, 0.48)
    axes[0].set_ylabel("F1")
    axes[0].set_title("Direct single-view preprocessing")
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(budget))
    axes[1].bar(x - width / 2, budget["f1_iou10"], width, color="#0072B2", label="F1 at IoU 0.10")
    axes[1].bar(x + width / 2, budget["coverage"], width, color="#009E73", label="Relaxed coverage")
    axes[1].set_xticks(x, ["Raw", "Raw+RMR", "Raw+NAFNet", "Raw+DFPIR", "Raw+DeMoE", "Raw+InstructIR"], rotation=22, ha="right")
    axes[1].set_ylim(0, 0.48)
    axes[1].set_title("Chronological holdout, fixed 56-prediction budget")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7, rotation=90)

    fig.suptitle("ILX-RD46 pilot: restoration and raw-preserving deployment are distinct", fontsize=12, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=240, facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
