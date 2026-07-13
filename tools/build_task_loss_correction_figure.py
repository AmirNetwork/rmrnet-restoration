from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper_ieee_tits_rmrnet" / "figures" / "fig_task_loss_corrections.png"


def add_box(ax, xy, wh, title, body, face, edge):
    x, y = xy
    w, h = wh
    rect = Rectangle((x, y), w, h, linewidth=1.6, edgecolor=edge, facecolor=face)
    ax.add_patch(rect)
    ax.text(x + 0.025, y + h - 0.04, title, fontsize=10.2, weight="bold", color="#1d2930", va="top")
    ax.text(x + 0.025, y + h - 0.105, body, fontsize=7.8, color="#293941", va="top", linespacing=1.08)


def main() -> None:
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.035,
        0.955,
        "Task-loss safeguards for detector-aware restoration",
        fontsize=16,
        weight="bold",
        color="#18242b",
    )
    ax.text(
        0.035,
        0.905,
        "The objective keeps restoration fidelity dominant while preventing geometry collapse, texture flattening, and detector-feature shortcuts.",
        fontsize=10.5,
        color="#45535a",
    )

    failures = [
        (
            "Dirac collapse",
            "Earlier TDAC could reduce loss by\npushing phi away from the zero\nlevel set, making delta(phi) vanish.",
        ),
        (
            "Texture flattening",
            "The regional term backpropagated into\nrestored pixels and encouraged\npiecewise-constant road patches.",
        ),
        (
            "Detector shortcut",
            "YOLO-feature matching can reward\ncoherent high-frequency artifacts\nthat do not generalize to test frames.",
        ),
    ]
    fixes = [
        (
            "SDF-stable TDAC",
            "Add Eikonal regularization,\na small Dirac floor, and bounded\nspatial lambda maps.",
        ),
        (
            "Texture-safe region term",
            "Detach restored intensity inside the\nChan-Vese region energy so TDAC\ntrains geometry, not flat images.",
        ),
        (
            "CQMix TDP",
            "Randomly mix clean/restored patches\ninside the frozen YOLO feature loss\nto break adversarial coherence.",
        ),
    ]

    y_positions = [0.645, 0.385, 0.125]
    for idx, y in enumerate(y_positions):
        add_box(ax, (0.045, y), (0.35, 0.215), failures[idx][0], failures[idx][1], "#fff4ef", "#c86a4a")
        add_box(ax, (0.61, y), (0.35, 0.215), fixes[idx][0], fixes[idx][1], "#eff8f2", "#3f8b5d")
        ax.add_patch(
            FancyArrowPatch(
                (0.41, y + 0.108),
                (0.595, y + 0.108),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.7,
                color="#31444d",
            )
        )

    for y in y_positions:
        ax.text(0.50, y + 0.14, "safeguard", ha="center", va="center", fontsize=9.3, weight="bold", color="#31444d")

    ax.text(
        0.045,
        0.055,
        "Smoke verification: TDAC gradients flow to phi/lambda maps while restored-pixel gradients from the region term are detached; "
        "CQMix TDP and Jacobian losses are finite on CUDA.",
        fontsize=9,
        color="#4e5b62",
    )

    fig.savefig(FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print({"figure": str(FIG), "bytes": FIG.stat().st_size})


if __name__ == "__main__":
    main()
