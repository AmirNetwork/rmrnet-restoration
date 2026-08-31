#!/usr/bin/env python3
"""Compose the CRID collection-platform and trajectory figure.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

The source images are retained without generative editing. Text, borders, and
layout remain vector objects in the generated PDF.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_automation_in_construction_rmrnet" / "source_assets"
OUTPUT = ROOT / "paper_ieee_tits_trace_r" / "figures" / "fig_trace_crid_collection.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    configure_style()
    platform = Image.open(args.source / "sony_sensor_suite.jpg").convert("RGB")
    trajectory = Image.open(args.source / "sony_cam1_trajectory.png").convert("RGB")

    # Crop the portrait photograph around the camera and roof mount. The map
    # is retained in full because its route geometry is part of the audit.
    photo = platform.crop((80, 0, platform.width - 30, min(platform.height, 760)))
    fig = plt.figure(figsize=(7.16, 2.85), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 2.45), wspace=0.035)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes[0].imshow(photo)
    axes[1].imshow(trajectory)
    titles = ("(a) Sony ILX-LR1 roof mount", "(b) Synchronized survey trajectory")
    for axis, title in zip(axes, titles, strict=True):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title, loc="left", fontsize=7.6, fontweight="bold", pad=3)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(0.65)
    axes[0].text(
        0.03,
        0.035,
        "camera and calibrated housing",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        bbox={"facecolor": "white", "edgecolor": "#111111", "linewidth": 0.45, "pad": 2.0, "alpha": 0.92},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02, dpi=320)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
