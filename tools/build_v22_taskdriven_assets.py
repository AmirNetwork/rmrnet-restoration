from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper_ieee_tits_rmrnet" / "figures"


def grouped_bar(path: Path, title: str, scenarios: list[str], series: dict[str, list[float]]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    labels = list(series.keys())
    x = np.arange(len(scenarios))
    width = 0.18
    colors = ["#a8b0b7", "#809bce", "#f4a261", "#139b75"]

    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfd")
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2.0) * width
        bars = ax.bar(x + offset, series[label], width, label=label, color=colors[idx], edgecolor="white", linewidth=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.008,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                rotation=90,
                color="#26323a",
            )
    ax.set_title(title, fontsize=12.2, weight="bold", pad=10)
    ax.set_ylabel("Frozen YOLO mAP50")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=12, ha="right")
    ax.set_ylim(0, max(max(values) for values in series.values()) + 0.10)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", ncols=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    grouped_bar(
        FIG / "fig_pothole_detection_recovery.png",
        "Pothole detection recovery after v22 task-driven training",
        ["motion", "defocus", "low light"],
        {
            "degraded": [0.042, 0.037, 0.355],
            "NAFNet-road": [0.056, 0.116, 0.317],
            "DFPIR": [0.141, 0.056, 0.266],
            "RMR-Net": [0.242, 0.176, 0.383],
        },
    )
    grouped_bar(
        FIG / "fig_taskdriven_v22_audit.png",
        "Detector-selected v22 audit",
        ["IVCNZ motion", "IVCNZ defocus", "IVCNZ low", "PCM motion", "PCM defocus", "PCM low"],
        {
            "degraded": [0.042, 0.037, 0.355, 0.197, 0.077, 0.321],
            "previous": [0.236, 0.165, 0.369, 0.309, 0.270, 0.415],
            "v21 audit": [0.238, 0.168, 0.346, 0.293, 0.261, 0.413],
            "v22 selected": [0.242, 0.176, 0.383, 0.310, 0.259, 0.414],
        },
    )
    grouped_bar(
        FIG / "fig_taskdriven_v23_composite_audit.png",
        "Composite-loss v23 audit",
        ["IVCNZ motion", "IVCNZ defocus", "IVCNZ low", "PCM motion", "PCM defocus", "PCM low"],
        {
            "degraded": [0.042, 0.037, 0.355, 0.197, 0.077, 0.321],
            "current paper": [0.242, 0.176, 0.383, 0.310, 0.270, 0.415],
            "v23 direct": [0.231, 0.160, 0.339, 0.301, 0.273, 0.410],
            "v23 policy": [0.231, 0.160, 0.373, 0.301, 0.273, 0.410],
        },
    )


if __name__ == "__main__":
    main()
