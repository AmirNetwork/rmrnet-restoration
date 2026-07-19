# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the effective paper-facing RMR-Net method configuration.")
    parser.add_argument("--config", default="configs/rmrnet_headline.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = cfg["model"]
    training = cfg["training"]
    selection = cfg["selection"]

    uses = [
        f"{model['name']} with {model['backbone_name']} backbone",
        f"width {model['width']} and {model['block_type']} residual blocks",
        f"{model['attention_type']} attention",
        f"{model['conditioning']} conditioning with {model['code_source']} code",
        "metadata-conditioned inference with image-only fallback",
        "bounded evidence-preserving detail skip",
        "restoration, edge, frequency, defect-weighted, visibility, and code-supervision losses",
        "YOLO-feature TDP loss with CQMix",
        "Hutchinson Jacobian regularization",
        "stabilized train-time active-contour regularization",
        "detector-input feature anchoring and road-evidence non-regression",
        f"frozen {cfg['detection']['detector_family']} detector evaluation at {cfg['detection']['imgsz']} px",
    ]

    does_not_use = []
    task_losses = training.get("task_driven_composite_losses", {})
    if not task_losses.get("enabled_for_headline_tables", False):
        does_not_use.extend(
            [
                "YOLO-feature TDP loss for headline table selection",
                "Hutchinson Jacobian penalty for headline table selection",
                "train-time active-contour loss for headline table selection",
                "phase-2 detector adaptation in the main comparison",
            ]
        )
    if selection["deployment_policy"] == "all_restored_for_controlled_degradation_tables":
        does_not_use.append("mixed residual deployment policy in controlled-degradation headline tables")

    print("The headline RMR-Net uses:")
    for item in uses:
        print(f"  - {item}")

    print("\nThe headline RMR-Net does not use:")
    for item in does_not_use:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
