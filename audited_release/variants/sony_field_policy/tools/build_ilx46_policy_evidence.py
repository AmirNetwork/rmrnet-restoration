# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Build leakage-safe direct and raw-preserving evidence for ILX-RD46.

Residual strength is selected on the first chronological half.  The second
half is evaluated once.  Every two-view method uses the same primary-family
NMS and is capped at the raw-view prediction count, separating restoration
evidence from the trivial benefit of emitting more detections.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_yolo26_coordinate_gt46 import (
    box_iou,
    corpus_ap,
    evaluate_one,
    greedy_match,
    load_gt,
    load_predictions,
    prf,
    success_count,
)


PAPER = ROOT / "paper_automation_in_construction_rmrnet"
ANNOTATIONS = ROOT / "road-defect-seg-9junedata.coco-segmentation" / "train" / "_annotations.coco.json"
RMR_CANDIDATES = tuple(f"current_metadata_eta{eta}" for eta in ("0p05", "0p10", "0p25", "0p50", "1p00"))
DIRECT_LABELS = {
    "raw": "Raw native",
    "current_metadata_eta1p00": "RMR-Net metadata, direct",
    "nafnet": "NAFNet-road",
    "dfpir": "DFPIR",
    "demoe_auto": "DeMoE-auto",
    "demoe_scenario": "DeMoE-scenario",
    "instructir_generic": "InstructIR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def subset(items: dict[str, list[dict]], images: list[str]) -> dict[str, list[dict]]:
    return {image: items.get(image, []) for image in images}


def threshold(items: dict[str, list[dict]], confidence: float) -> dict[str, list[dict]]:
    return {image: [item for item in rows if float(item["conf"]) >= confidence] for image, rows in items.items()}


def cap(items: dict[str, list[dict]], budget: int) -> dict[str, list[dict]]:
    ranked = sorted((item for rows in items.values() for item in rows), key=lambda item: float(item["conf"]), reverse=True)[:budget]
    output: dict[str, list[dict]] = {}
    for item in ranked:
        output.setdefault(str(item["image"]), []).append(item)
    return output


def fuse(
    raw: dict[str, list[dict]],
    second: dict[str, list[dict]],
    *,
    confidence: float = 0.08,
    nms_iou: float = 0.20,
) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for image in sorted(set(raw) | set(second)):
        candidates = [item for item in raw.get(image, []) + second.get(image, []) if float(item["conf"]) >= confidence]
        kept: list[dict] = []
        for item in sorted(candidates, key=lambda candidate: float(candidate["conf"]), reverse=True):
            duplicate = any(
                old["primary"] == item["primary"] and box_iou(old["box"], item["box"]) >= nms_iou
                for old in kept
            )
            if not duplicate:
                kept.append(item)
        output[image] = kept
    return output


def compact(name: str, gt: dict[str, list[dict]], preds: dict[str, list[dict]]) -> tuple[dict[str, Any], list[dict]]:
    summary, class_rows = evaluate_one(name, gt, preds)
    row10 = next(row for row in summary if row["mode"] == "primary" and float(row["iou"]) == 0.10)
    row50 = next(row for row in summary if row["mode"] == "primary" and float(row["iou"]) == 0.50)
    coverage = next(row for row in summary if row["mode"] == "primary_success")
    return (
        {
            "run": name,
            "images": int(row10["images"]),
            "gt": int(row10["gt"]),
            "pred": int(row10["pred"]),
            "precision_iou10": float(row10["precision"]),
            "recall_iou10": float(row10["recall"]),
            "f1_iou10": float(row10["f1"]),
            "f1_iou50": float(row50["f1"]),
            "ap_iou10": corpus_ap(gt, preds, 0.10, "primary"),
            "ap_iou50": corpus_ap(gt, preds, 0.50, "primary"),
            "coverage": float(coverage["recall"]),
        },
        class_rows,
    )


def paired_bootstrap(
    gt: dict[str, list[dict]],
    raw: dict[str, list[dict]],
    proposed: dict[str, list[dict]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    images = sorted(gt)
    rng = random.Random(seed)
    f1_deltas: list[float] = []
    coverage_deltas: list[float] = []
    for _ in range(samples):
        sampled = rng.choices(images, k=len(images))
        metrics: list[tuple[float, float]] = []
        for predictions in (raw, proposed):
            tp = fp = fn = covered = total = 0
            for image in sampled:
                a, b, c = greedy_match(gt[image], predictions.get(image, []), 0.10, "primary")
                tp += a
                fp += b
                fn += c
                covered += success_count(gt[image], predictions.get(image, []))
                total += len(gt[image])
            metrics.append((prf(tp, fp, fn)[2], covered / total if total else 0.0))
        f1_deltas.append(metrics[1][0] - metrics[0][0])
        coverage_deltas.append(metrics[1][1] - metrics[0][1])

    def summarize(values: list[float], prefix: str) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            f"{prefix}_delta": float(np.mean(array)),
            f"{prefix}_lo": float(np.quantile(array, 0.025)),
            f"{prefix}_hi": float(np.quantile(array, 0.975)),
            f"{prefix}_p": min(1.0, 2.0 * min(float(np.mean(array <= 0)), float(np.mean(array >= 0)))),
        }

    return {**summarize(f1_deltas, "f1"), **summarize(coverage_deltas, "coverage")}


def add_holm_adjustment(rows: list[dict[str, Any]], field: str, output_field: str) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index][field]))
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(rows[index][field]))
        running = max(running, adjusted)
        rows[index][output_field] = running


def direct_table(rows: list[dict[str, Any]]) -> str:
    body = [
        f"{row['method']} & {row['pred']} & {row['ap_iou10']:.3f} & {row['precision_iou10']:.3f} & "
        f"{row['recall_iou10']:.3f} & {row['f1_iou10']:.3f} & {row['f1_iou50']:.3f} & {row['coverage']:.3f} \\\\"
        for row in rows
    ]
    return "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\caption{Direct single-view results on the 44 ILX-RD46 frames containing the common detector taxonomy.}",
            r"\label{tab:ilx_direct}",
            r"\scriptsize",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"View & Pred. & AP@.10 & P@.10 & R@.10 & F1@.10 & F1@.50 & Relaxed coverage \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


def policy_table(rows: list[dict[str, Any]]) -> str:
    body = [
        f"{row['method']} & {row['pred']} & {row['ap_iou10']:.3f} & {row['precision_iou10']:.3f} & "
        f"{row['recall_iou10']:.3f} & {row['f1_iou10']:.3f} & {row['coverage']:.3f} \\\\"
        for row in rows
    ]
    return "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\caption{Chronological ILX-RD46 evaluation with every two-view system capped at the raw prediction count.}",
            r"\label{tab:ilx_temporal_holdout}",
            r"\scriptsize",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"System & Pred. & AP@.10 & P@.10 & R@.10 & F1@.10 & Relaxed coverage \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    experiment = args.experiment.resolve()
    prediction_root = experiment / "detections"
    out = (args.out or experiment / "policy_evidence").resolve()
    out.mkdir(parents=True, exist_ok=True)
    gt, class_counts = load_gt(ANNOTATIONS)
    ordered = sorted(gt)
    calibration_images = ordered[: len(ordered) // 2]
    evaluation_images = ordered[len(ordered) // 2 :]
    gt_calibration = subset(gt, calibration_images)
    gt_evaluation = subset(gt, evaluation_images)

    required = {"raw", *DIRECT_LABELS, *RMR_CANDIDATES}
    missing = sorted(name for name in required if not (prediction_root / name).exists())
    if missing:
        raise RuntimeError(f"Missing ILX detector outputs: {missing}")
    predictions = {name: load_predictions(prediction_root / name) for name in required}

    calibration_rows = []
    for name in RMR_CANDIDATES:
        row, _ = compact(name, gt_calibration, subset(predictions[name], calibration_images))
        calibration_rows.append(row)
    selected_rmr = max(calibration_rows, key=lambda row: (row["f1_iou10"], row["coverage"], -row["pred"]))["run"]
    write_csv(out / "residual_policy_calibration.csv", calibration_rows)

    direct_names = dict(DIRECT_LABELS)
    direct_names[selected_rmr] = f"RMR-Net metadata, calibrated ({selected_rmr.split('eta')[-1]})"
    direct_rows: list[dict[str, Any]] = []
    direct_class_rows: list[dict[str, Any]] = []
    for name, label in direct_names.items():
        row, class_rows = compact(name, gt, predictions[name])
        row["method"] = label
        direct_rows.append(row)
        direct_class_rows.extend({**item, "method": label} for item in class_rows)
    write_csv(out / "direct_single_view_metrics.csv", direct_rows)
    write_csv(out / "direct_single_view_per_class.csv", direct_class_rows)

    raw_eval = subset(predictions["raw"], evaluation_images)
    raw_calibration = subset(predictions["raw"], calibration_images)
    raw_row, raw_class = compact("raw", gt_evaluation, raw_eval)
    raw_calibration_row, _ = compact("raw_calibration", gt_calibration, raw_calibration)
    raw_row["method"] = "Raw native"
    raw_row["selected_on"] = "none"
    budget = int(raw_row["pred"])
    policy_rows = [raw_row]
    matched_precision_rows = [
        {
            **raw_row,
            "selected_confidence": "detector default",
            "target_calibration_precision": raw_calibration_row["precision_iou10"],
        }
    ]
    policy_class_rows = [{**item, "method": "Raw native"} for item in raw_class]
    comparison_names = {
        selected_rmr: "Raw + RMR-Net",
        "nafnet": "Raw + NAFNet-road",
        "dfpir": "Raw + DFPIR",
        "demoe_auto": "Raw + DeMoE-auto",
        "demoe_scenario": "Raw + DeMoE-scenario",
        "instructir_generic": "Raw + InstructIR",
    }
    if (prediction_root / "weak_gamma090").exists():
        predictions["weak_gamma090"] = load_predictions(prediction_root / "weak_gamma090")
        comparison_names["weak_gamma090"] = "Raw + weak photometric"
    if (prediction_root / "raw_tta").exists():
        predictions["raw_tta"] = load_predictions(prediction_root / "raw_tta")
        comparison_names["raw_tta"] = "Raw + raw TTA"

    bootstrap_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    for index, (name, label) in enumerate(comparison_names.items()):
        second_calibration = subset(predictions[name], calibration_images)
        second_eval = subset(predictions[name], evaluation_images)
        fused_calibration = fuse(raw_calibration, second_calibration)
        fused = fuse(raw_eval, second_eval)
        capped = cap(fused, budget)
        row, class_rows = compact(f"raw_plus_{name}", gt_evaluation, capped)
        row["method"] = label
        row["selected_on"] = "first chronological half" if name == selected_rmr else "fixed protocol"
        row["prediction_budget"] = budget
        policy_rows.append(row)
        policy_class_rows.extend({**item, "method": label} for item in class_rows)
        bootstrap_rows.append(
            {
                "comparison": f"{label} minus raw",
                **paired_bootstrap(gt_evaluation, raw_eval, capped, samples=args.bootstrap, seed=args.seed + index),
            }
        )
        confidence_candidates: list[tuple[float, dict[str, Any]]] = []
        for confidence in np.linspace(0.05, 0.90, 86):
            candidate, _ = compact(
                f"{name}_calibration_{confidence:.2f}",
                gt_calibration,
                threshold(fused_calibration, float(confidence)),
            )
            confidence_candidates.append((float(confidence), candidate))
        selected_confidence, _ = min(
            confidence_candidates,
            key=lambda item: (
                abs(item[1]["precision_iou10"] - raw_calibration_row["precision_iou10"]),
                -item[1]["recall_iou10"],
                item[1]["pred"],
            ),
        )
        matched, _ = compact(
            f"raw_plus_{name}_matched_precision",
            gt_evaluation,
            threshold(fused, selected_confidence),
        )
        matched.update(
            {
                "method": label,
                "selected_confidence": selected_confidence,
                "target_calibration_precision": raw_calibration_row["precision_iou10"],
            }
        )
        matched_precision_rows.append(matched)
        for confidence in np.linspace(0.05, 0.90, 35):
            curve, _ = compact(label, gt_evaluation, threshold(fused, float(confidence)))
            pr_rows.append(
                {
                    "system": label,
                    "confidence": float(confidence),
                    "precision_iou10": curve["precision_iou10"],
                    "recall_iou10": curve["recall_iou10"],
                    "f1_iou10": curve["f1_iou10"],
                    "pred": curve["pred"],
                }
            )
    add_holm_adjustment(bootstrap_rows, "f1_p", "f1_holm_p")
    add_holm_adjustment(bootstrap_rows, "coverage_p", "coverage_holm_p")
    write_csv(out / "chronological_fixed_budget_metrics.csv", policy_rows)
    write_csv(out / "chronological_fixed_budget_per_class.csv", policy_class_rows)
    write_csv(out / "chronological_matched_precision_metrics.csv", matched_precision_rows)
    write_csv(out / "chronological_bootstrap.csv", bootstrap_rows)
    write_csv(out / "chronological_pr_curves.csv", pr_rows)

    tables = PAPER / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "table_ilx_direct_audited.tex").write_text(direct_table(direct_rows), encoding="utf-8")
    (tables / "table_ilx_temporal_holdout.tex").write_text(policy_table(policy_rows), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.2, 4.3), constrained_layout=True)
    labels = [row["method"] for row in policy_rows]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, [row["f1_iou10"] for row in policy_rows], width, label="F1@0.10", color="#0072B2")
    ax.bar(x + width / 2, [row["coverage"] for row in policy_rows], width, label="Relaxed coverage", color="#E69F00")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Raw-preserving ILX-RD46 systems at a matched prediction budget")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2)
    figure = PAPER / "figures" / "fig_ilx46_policy_comparison.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=240, facecolor="white")
    plt.close(fig)

    manifest = {
        "status": "PASS",
        "frames_total": 46,
        "frames_common_taxonomy": len(gt),
        "class_counts": dict(class_counts),
        "ordering": "lexicographic filename order, equivalent to capture time",
        "calibration_frames": calibration_images,
        "evaluation_frames": evaluation_images,
        "selected_rmr_view": selected_rmr,
        "selection_metric": "calibration F1@0.10, then relaxed coverage, then fewer predictions",
        "test_labels_used_for_selection": False,
        "fusion": {"confidence_floor": 0.08, "primary_family_nms_iou": 0.20},
        "prediction_budget": budget,
        "synthetic_degradation_added": False,
        "direct_and_system_results_separated": True,
    }
    (out / "policy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
