"""Build reviewer-completion audits from the executed RMR-Net artifacts.

This script does not invent new benchmark numbers.  It turns existing runs into
traceable manuscript assets: dataset/split counts, Holm-corrected bootstrap
rows, contour-yield confidence intervals, fidelity-vs-detection correlation,
and a claim/evidence boundary table.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
EXP = ROOT / "experiments" / "v29_review_completion"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BOOTSTRAP_REPS = 800
BOOTSTRAP_SEED = 2026


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def split_dir(data_yaml: Path, split: str) -> Path:
    data = read_yaml(data_yaml)
    root = Path(data["path"])
    value = data.get(split) or data.get("val")
    return root / value


def label_dir_for_images(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    return Path(*parts)


def image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def split_stats(name: str, data_yaml: Path) -> list[dict[str, Any]]:
    data = read_yaml(data_yaml)
    names_raw = data["names"]
    if isinstance(names_raw, dict):
        class_names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        class_names = {i: str(v) for i, v in enumerate(names_raw)}
    rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        img_dir = split_dir(data_yaml, split)
        imgs = image_files(img_dir)
        labels = label_dir_for_images(img_dir)
        counts: Counter[str] = Counter()
        boxes = 0
        for img in imgs:
            lab = labels / f"{img.stem}.txt"
            if not lab.exists():
                continue
            for line in lab.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                counts[class_names.get(cls, str(cls))] += 1
                boxes += 1
        rows.append(
            {
                "dataset": name,
                "split": split,
                "images": len(imgs),
                "boxes": boxes,
                "class_counts": ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)) or "none",
            }
        )
    return rows


def stress_subset_stats(name: str, data_yaml: Path) -> dict[str, Any]:
    data = read_yaml(data_yaml)
    names_raw = data["names"]
    if isinstance(names_raw, dict):
        class_names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        class_names = {i: str(v) for i, v in enumerate(names_raw)}
    img_dir = split_dir(data_yaml, "test")
    imgs = image_files(img_dir)
    labels = label_dir_for_images(img_dir)
    counts: Counter[str] = Counter()
    boxes = 0
    for img in imgs:
        lab = labels / f"{img.stem}.txt"
        if not lab.exists():
            continue
        for line in lab.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            counts[class_names.get(cls, str(cls))] += 1
            boxes += 1
    return {
        "dataset": name,
        "split": "held-out stress",
        "images": len(imgs),
        "boxes": boxes,
        "class_counts": ", ".join(f"{k}:{counts[k]}" for k in sorted(counts)) or "none",
    }


def build_split_audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(split_stats("IVCNZ pothole", ROOT / "datasets" / "pothole_yolo" / "data.yaml"))
    rows.extend(split_stats("PCM road damage", ROOT / "datasets" / "road_damage_pcm_yolo" / "data.yaml"))
    rows.append(stress_subset_stats("IVCNZ native-blur subset", ROOT / "datasets" / "pothole_yolo_nativeblur_test" / "data.yaml"))
    rows.append(stress_subset_stats("PCM native-blur subset", ROOT / "datasets" / "pcm_yolo_nativeblur_test" / "data.yaml"))
    write_csv(EXP / "dataset_split_audit.csv", rows)
    body = "\n".join(
        f"{tex_escape(r['dataset'])} & {r['split']} & {r['images']} & {r['boxes']} & {tex_escape(r['class_counts'])} \\\\"
        for r in rows
    )
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Dataset and split audit for the executed road-damage experiments. Counts are read from the YOLO labels used by the training and evaluation scripts. The native-blur rows are stress subsets and are not used to train the detectors.}}
\label{{tab:dataset_split_audit}}
\scriptsize
\begin{{tabularx}}{{\textwidth}}{{llrrX}}
\toprule
Dataset & Split & Images & Boxes & Class counts \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""
    write_text(TABLES / "table_dataset_split_audit.tex", tex)
    return rows


def holm_rows() -> list[dict[str, Any]]:
    stats = read_csv(ROOT / "experiments" / "v28_review_robustness" / "bootstrap_ap50_deltas.csv")
    ordered = sorted(stats, key=lambda r: float(r["paired_bootstrap_p"]))
    m = len(ordered)
    running = 0.0
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ordered):
        raw = float(row["paired_bootstrap_p"])
        adj = min(1.0, max(running, raw * (m - idx)))
        running = adj
        rows.append(
            {
                "detector": row["detector"],
                "dataset": row["dataset"],
                "scenario": row["scenario"],
                "delta_ap50": float(row["delta_ap50"]),
                "raw_p": raw,
                "holm_p": adj,
                "significant_0p05": adj < 0.05,
            }
        )
    write_csv(EXP / "holm_corrected_detection_bootstrap.csv", rows)
    body = "\n".join(
        f"{r['detector']} & {tex_escape(r['dataset'])} {tex_escape(r['scenario'])} & {fmt(r['delta_ap50'])} & "
        f"{'<0.005' if r['raw_p'] < 0.005 else fmt(r['raw_p'])} & "
        f"{'<0.005' if r['holm_p'] < 0.005 else fmt(r['holm_p'])} & "
        f"{'yes' if r['significant_0p05'] else 'no'} \\\\"
        for r in rows
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
    write_text(TABLES / "table_holm_bootstrap.tex", tex)
    return rows


SNAKE_SPECS = [
    ("Pothole motion", "degraded", "pothole_motion_degraded", ROOT / "datasets" / "pothole_yolo_motion_test" / "images" / "test"),
    ("Pothole motion", "NAFNet-road", "pothole_motion_nafnet", ROOT / "datasets" / "pothole_yolo_motion_test" / "images" / "test"),
    ("Pothole motion", "DFPIR", "pothole_motion_dfpir", ROOT / "datasets" / "pothole_yolo_motion_test" / "images" / "test"),
    ("Pothole motion", "RMR-Net", "pothole_motion_rmr", ROOT / "datasets" / "pothole_yolo_motion_test" / "images" / "test"),
    ("PCM defocus", "degraded", "pcm_defocus_degraded", ROOT / "datasets" / "pcm_yolo_defocus_test" / "images" / "test"),
    ("PCM defocus", "NAFNet-road", "pcm_defocus_nafnet", ROOT / "datasets" / "pcm_yolo_defocus_test" / "images" / "test"),
    ("PCM defocus", "DFPIR", "pcm_defocus_dfpir", ROOT / "datasets" / "pcm_yolo_defocus_test" / "images" / "test"),
    ("PCM defocus", "RMR-Net", "pcm_defocus_rmr", ROOT / "datasets" / "pcm_yolo_defocus_test" / "images" / "test"),
    ("PCM low light", "degraded", "pcm_lowlight_degraded", ROOT / "datasets" / "pcm_yolo_lowlight_test" / "images" / "test"),
    ("PCM low light", "NAFNet-road", "pcm_lowlight_nafnet", ROOT / "datasets" / "pcm_yolo_lowlight_test" / "images" / "test"),
    ("PCM low light", "DFPIR", "pcm_lowlight_dfpir", ROOT / "datasets" / "pcm_yolo_lowlight_test" / "images" / "test"),
    ("PCM low light", "RMR-Net", "pcm_lowlight_rmr", ROOT / "datasets" / "pcm_yolo_lowlight_test" / "images" / "test"),
]


def per_image_success(csv_path: Path, image_dir: Path) -> dict[str, int]:
    counts = {p.name: 0 for p in image_files(image_dir)}
    if not csv_path.exists():
        return counts
    for row in read_csv(csv_path):
        if row.get("success", "").lower() == "true":
            counts[row["image"]] = counts.get(row["image"], 0) + 1
    return counts


def bootstrap_mean(values: list[float], rng: random.Random) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    means = []
    for _ in range(BOOTSTRAP_REPS):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(float(vals[idx].mean()))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(low), float(high)


def build_contour_bootstrap() -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for setting, source, folder, image_dir in SNAKE_SPECS:
        csv_path = ROOT / "runs" / "snake_compare_v13" / folder / "snake_boundary_metrics.csv"
        counts = per_image_success(csv_path, image_dir)
        values = list(counts.values())
        mean, low, high = bootstrap_mean(values, rng)
        grouped[setting][source] = values
        rows.append(
            {
                "setting": setting,
                "source": source,
                "accepted_per_image": mean,
                "ci_low": low,
                "ci_high": high,
                "images": len(values),
                "accepted_total": int(sum(values)),
            }
        )
    write_csv(EXP / "contour_yield_bootstrap.csv", rows)
    display_order = [("Pothole motion", "degraded"), ("Pothole motion", "DFPIR"), ("Pothole motion", "NAFNet-road"), ("Pothole motion", "RMR-Net"),
                     ("PCM defocus", "degraded"), ("PCM defocus", "DFPIR"), ("PCM defocus", "NAFNet-road"), ("PCM defocus", "RMR-Net"),
                     ("PCM low light", "degraded"), ("PCM low light", "DFPIR"), ("PCM low light", "NAFNet-road"), ("PCM low light", "RMR-Net")]
    lookup = {(r["setting"], r["source"]): r for r in rows}
    body = "\n".join(
        f"{tex_escape(setting)} & {tex_escape(source)} & {lookup[(setting, source)]['accepted_total']} & "
        f"{fmt(lookup[(setting, source)]['accepted_per_image'])} & "
        f"[{fmt(lookup[(setting, source)]['ci_low'])}, {fmt(lookup[(setting, source)]['ci_high'])}] \\\\"
        for setting, source in display_order
    )
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Bootstrap uncertainty for detector-guided active-contour measurement yield. Accepted contours/image is bootstrapped over held-out images; images with no detector box contribute zero yield. This evaluates measurement availability, not pixel-mask segmentation accuracy.}}
\label{{tab:contour_bootstrap_ci}}
\scriptsize
\begin{{tabular}}{{llrrc}}
\toprule
Setting & Source & Accepted & Accepted/img & 95\% CI \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    write_text(TABLES / "table_contour_bootstrap_ci.tex", tex)
    return rows


RESTORATION_ROWS = [
    ("Pothole", "motion", "RMR-Net", 25.40),
    ("Pothole", "motion", "NAFNet-road", 24.76),
    ("Pothole", "motion", "DFPIR", 26.44),
    ("Pothole", "defocus", "RMR-Net", 24.72),
    ("Pothole", "defocus", "NAFNet-road", 24.58),
    ("Pothole", "defocus", "DFPIR", 25.98),
    ("Pothole", "lowlight", "RMR-Net", 29.16),
    ("Pothole", "lowlight", "NAFNet-road", 31.61),
    ("Pothole", "lowlight", "DFPIR", 16.26),
    ("PCM", "motion", "RMR-Net", 27.78),
    ("PCM", "motion", "NAFNet-road", 25.50),
    ("PCM", "motion", "DFPIR", 28.65),
    ("PCM", "defocus", "RMR-Net", 26.94),
    ("PCM", "defocus", "NAFNet-road", 25.88),
    ("PCM", "defocus", "DFPIR", 28.43),
    ("PCM", "lowlight", "RMR-Net", 32.12),
    ("PCM", "lowlight", "NAFNet-road", 34.03),
    ("PCM", "lowlight", "DFPIR", 17.41),
]


def detection_lookup() -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    files = [
        ("Pothole", ROOT / "experiments" / "v27_taskloss_yolo11s_eval" / "pothole_test_yolo11s_baselines_taskloss_v27.csv"),
        ("PCM", ROOT / "experiments" / "v27_taskloss_yolo11s_eval" / "pcm_test_yolo11s_baselines_taskloss_v27.csv"),
    ]
    method_map = {
        "dfpir": "DFPIR",
        "nafnet": "NAFNet-road",
        "rmrnet_v27": "RMR-Net",
    }
    for dataset, path in files:
        for row in read_csv(path):
            name = row["name"]
            parts = name.split("_")
            if len(parts) < 2:
                continue
            scenario = parts[0]
            suffix = "_".join(parts[1:])
            method = method_map.get(suffix)
            if method is None:
                continue
            lookup[(dataset, scenario, method)] = float(row["map50"])
    return lookup


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if len(xx) < 2 or np.std(xx) == 0 or np.std(yy) == 0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def kendall_tau(x: list[float], y: list[float]) -> float:
    concordant = 0
    discordant = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = concordant + discordant
    return 0.0 if denom == 0 else (concordant - discordant) / denom


def build_correlation_assets() -> list[dict[str, Any]]:
    det = detection_lookup()
    rows: list[dict[str, Any]] = []
    for dataset, scenario, method, psnr in RESTORATION_ROWS:
        map50 = det[(dataset, scenario, method)]
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "method": method,
                "psnr": psnr,
                "map50": map50,
            }
        )
    write_csv(EXP / "fidelity_detection_pairs.csv", rows)
    psnr = [float(r["psnr"]) for r in rows]
    map50 = [float(r["map50"]) for r in rows]
    all_s = spearman(psnr, map50)
    all_k = kendall_tau(psnr, map50)
    summary_rows: list[dict[str, Any]] = [
        {"subset": "all methods/scenarios", "pairs": len(rows), "spearman_rho": all_s, "kendall_tau": all_k}
    ]
    for dataset in ("Pothole", "PCM"):
        subset = [r for r in rows if r["dataset"] == dataset]
        summary_rows.append(
            {
                "subset": dataset,
                "pairs": len(subset),
                "spearman_rho": spearman([float(r["psnr"]) for r in subset], [float(r["map50"]) for r in subset]),
                "kendall_tau": kendall_tau([float(r["psnr"]) for r in subset], [float(r["map50"]) for r in subset]),
            }
        )
    write_csv(EXP / "fidelity_detection_correlation.csv", summary_rows)
    body = "\n".join(
        f"{tex_escape(r['subset'])} & {r['pairs']} & {fmt(r['spearman_rho'])} & {fmt(r['kendall_tau'])} \\\\"
        for r in summary_rows
    )
    tex = rf"""\begin{{table}}[!t]
\centering
\caption{{Correlation between full-reference PSNR and downstream YOLO11s mAP50 across the executed restoration methods. Weak or inconsistent rank correlation supports the central claim that image fidelity and ITS task utility must both be reported.}}
\label{{tab:fidelity_detection_correlation}}
\small
\begin{{tabular}}{{lrrr}}
\toprule
Subset & Pairs & Spearman $\rho$ & Kendall $\tau$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write_text(TABLES / "table_fidelity_detection_correlation.tex", tex)

    colors = {"RMR-Net": "#1f77b4", "NAFNet-road": "#ff7f0e", "DFPIR": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=180)
    for method in colors:
        xs = [r["psnr"] for r in rows if r["method"] == method]
        ys = [r["map50"] for r in rows if r["method"] == method]
        ax.scatter(xs, ys, label=method, s=54, color=colors[method], edgecolor="white", linewidth=0.8)
    for r in rows:
        ax.annotate(r["scenario"][0].upper(), (r["psnr"], r["map50"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("YOLO11s mAP50")
    ax.set_title("Fidelity and detection utility are not interchangeable")
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_fidelity_detection_correlation.png")
    plt.close(fig)
    return rows


def build_claim_and_protocol_tables() -> None:
    claim_rows = [
        ("Real-world validity", "partly executed", "Native-blur stress test, unseen-source PCM transfer, and KITTI raw OXTS metadata are included; RDD2022 conversion code is provided but no RDD2022 run is claimed."),
        ("Cross-detector generalization", "executed within YOLO family", "Primary YOLO11s and independently trained YOLOv8n clean-image detectors are evaluated after checkpoints are fixed; YOLOv8n is used only as a post-selection robustness audit."),
        ("Full ablation package", "claim-bounded", "Metadata, image-only, detail-skip, and grouped task-loss audits are reported; individual loss-term 3-5 seed ablations are not claimed."),
        ("Hyperparameter search", "partly executed", "Validation-only sweeps cover residual gate and detail-skip gain; broader metadata dropout/code-jitter/loss grids are released as planned config scope, not headline evidence."),
        ("Stronger baselines", "claim-bounded", "Executed baselines are degraded input, official DFPIR, official DeMoE-auto and DeMoE-scenario, NAFNet-road, and image-only RMR-Net; unrun Restormer/DarkIR/InstructIR are cited but not assigned numbers."),
        ("Better splits", "partly executed", "Train/val/test labels are audited, KITTI is sequence-level, native-blur subsets are held out, and PCM transfer is leave-source-out; repeated 3-5 seed splits remain future work."),
        ("Statistical rigor", "executed for AP/yield", "Paired bootstrap CIs, Holm correction, and contour-yield bootstrap are generated from held-out images; no mixed-effects or McNemar claim is made."),
        ("Boundary evaluation", "executed with limits", "Detector-box and fixed-GT-box Snake yield are reported with object geometry; PCM raw polygons are preserved for a seeded fixed-box IoU/Dice/BF1/Chamfer/Hausdorff audit."),
        ("Correlation analysis", "executed", "Spearman and Kendall correlations between PSNR and YOLO11s mAP50 are reported."),
        ("Reproducibility release", "executed", "Configs, seeds, provenance CSV, checkpoint-selection rules, hardware, and commands are included in the repository package."),
        ("Writing cleanup", "executed", "The manuscript separates deployed core model, train-time regularizers, and post-hoc boundary recognition."),
        ("Ethics and governance", "executed", "A deployment-governance table covers retention, audit logs, privacy, pass-through failures, and human review."),
    ]
    claim_body = "\n".join(
        f"{tex_escape(item)} & {tex_escape(status)} & {tex_escape(evidence)} \\\\"
        for item, status, evidence in claim_rows
    )
    claim_tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Claim and evidence boundary audit. Rows marked claim-bounded are deliberately not promoted as fully solved experiments; they constrain the contribution so that every quantitative claim remains tied to executed artifacts.}}
\label{{tab:claim_evidence_audit}}
\scriptsize
\begin{{tabularx}}{{\textwidth}}{{p{{0.19\textwidth}}p{{0.13\textwidth}}X}}
\toprule
Reviewer concern & Status & Evidence boundary in this submission \\
\midrule
{claim_body}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""
    write_text(TABLES / "table_claim_evidence_audit.tex", claim_tex)

    ablation_rows = [
        ("Metadata fusion", "image-only vs metadata", "Table~\\ref{tab:metadata_ablation}"),
        ("Detail-preserving skip", "validation detail-gain sweep", "Table~\\ref{tab:v25_detail_audit}"),
        ("Task-evidence attention", "included in deployed core; not separately removed", "claim grouped with core model"),
        ("Detector-feature loss", "included in composite task-loss group", "Table~\\ref{tab:task_objective_audit}"),
        ("Jacobian regularizer", "included in composite task-loss group", "Table~\\ref{tab:task_objective_audit}"),
        ("Active-contour regularizer", "included in composite task-loss group", "Table~\\ref{tab:task_objective_audit}"),
        ("Detector anchor", "included in composite task-loss group", "Table~\\ref{tab:task_objective_audit}"),
        ("Evidence non-regression", "included in composite task-loss group", "Table~\\ref{tab:task_objective_audit}"),
        ("Residual pass-through gate", "validation-only policy; native-blur claim boundary", "Table~\\ref{tab:native_blur_detection}"),
    ]
    ablation_body = "\n".join(
        f"{tex_escape(comp)} & {tex_escape(scope)} & {ref} \\\\"
        for comp, scope, ref in ablation_rows
    )
    ablation_tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Ablation and hyperparameter scope. This table prevents component overload by distinguishing isolated ablations from grouped train-time regularizers. The paper does not claim one-by-one causality for every low-weight regularizer.}}
\label{{tab:ablation_hparam_scope}}
\scriptsize
\begin{{tabularx}}{{\textwidth}}{{p{{0.25\textwidth}}p{{0.38\textwidth}}X}}
\toprule
Component & Executed isolation/sweep & Where reported \\
\midrule
{ablation_body}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""
    write_text(TABLES / "table_ablation_hparam_scope.tex", ablation_tex)

    governance_rows = [
        ("Raw-image retention", "Store only the minimum inspection interval needed for audit; allow hashed frame IDs in derived CSVs."),
        ("Privacy", "Blur or discard faces/plates before long-term storage when road-inspection law or fleet policy requires it."),
        ("Audit logs", "Record model checkpoint, detector checkpoint, gate decision, metadata availability, and restoration policy for each inspected segment."),
        ("Pass-through failure mode", "If degradation confidence is low or metadata is missing, prefer native-image detection rather than blind enhancement."),
        ("False restoration confidence", "Flag outputs with strong clipping, low edge evidence, or failed active contours for human review."),
        ("Operational use", "Treat contours as measurement aids for maintenance triage, not as sole legal or safety decisions."),
    ]
    gov_body = "\n".join(f"{tex_escape(k)} & {tex_escape(v)} \\\\" for k, v in governance_rows)
    gov_tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Deployment governance checklist for restoration-assisted road inspection. These items are implementation controls rather than performance metrics.}}
\label{{tab:deployment_governance}}
\scriptsize
\begin{{tabularx}}{{\textwidth}}{{p{{0.22\textwidth}}X}}
\toprule
Issue & Recommended control \\
\midrule
{gov_body}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""
    write_text(TABLES / "table_deployment_governance.tex", gov_tex)

    write_csv(
        EXP / "claim_evidence_audit.csv",
        [{"item": item, "status": status, "evidence": evidence} for item, status, evidence in claim_rows],
    )


def update_provenance() -> None:
    path = PAPER / "RESULT_PROVENANCE_TABLE.csv"
    text = path.read_text(encoding="utf-8").rstrip()
    additions = [
        "table_dataset_split_audit,IVCNZ/PCM/native subsets,split audit,label counts,local YOLO labels,not a selector,none,none,none,experiments/v29_review_completion/dataset_split_audit.csv,OK",
        "table_holm_bootstrap,IVCNZ/PCM,controlled detection bootstrap,RMR-Net vs degraded,YOLO11s/YOLOv8n held-out tests,Holm correction over executed AP tests,synthetic proxy metadata,all_restored,YOLO11s and YOLOv8n,experiments/v29_review_completion/holm_corrected_detection_bootstrap.csv,OK",
        "table_contour_bootstrap_ci,IVCNZ/PCM,boundary measurement,RMR-Net and baselines,Snake CSV outputs,bootstrap over held-out images,as applicable,detector-box Snake measurement,YOLO detector boxes,experiments/v29_review_completion/contour_yield_bootstrap.csv,OK",
        "table_fidelity_detection_correlation,IVCNZ/PCM,PSNR-vs-mAP correlation,RMR-Net/NAFNet-road/DFPIR,main restoration and detection CSVs,post-hoc analysis only,as applicable,all_restored,YOLO11s,experiments/v29_review_completion/fidelity_detection_correlation.csv,OK",
        "table_claim_evidence_audit,all,claim boundary,all reported components,executed artifacts and limitations,not a performance selector,as applicable,as applicable,as applicable,experiments/v29_review_completion/claim_evidence_audit.csv,OK",
        "table_deployment_governance,all,deployment governance,system controls,manuscript checklist,not a performance selector,as applicable,pass-through gate as needed,as applicable,paper_ieee_tits_rmrnet/tables/table_deployment_governance.tex,OK",
    ]
    for line in additions:
        if line not in text:
            text += "\n" + line
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    split_rows = build_split_audit()
    holm = holm_rows()
    contour = build_contour_bootstrap()
    corr_pairs = build_correlation_assets()
    build_claim_and_protocol_tables()
    update_provenance()
    manifest = {
        "tables": [
            str(TABLES / "table_dataset_split_audit.tex"),
            str(TABLES / "table_holm_bootstrap.tex"),
            str(TABLES / "table_contour_bootstrap_ci.tex"),
            str(TABLES / "table_fidelity_detection_correlation.tex"),
            str(TABLES / "table_claim_evidence_audit.tex"),
            str(TABLES / "table_ablation_hparam_scope.tex"),
            str(TABLES / "table_deployment_governance.tex"),
        ],
        "figures": [str(FIGURES / "fig_fidelity_detection_correlation.png")],
        "csv": [
            str(EXP / "dataset_split_audit.csv"),
            str(EXP / "holm_corrected_detection_bootstrap.csv"),
            str(EXP / "contour_yield_bootstrap.csv"),
            str(EXP / "fidelity_detection_pairs.csv"),
            str(EXP / "fidelity_detection_correlation.csv"),
            str(EXP / "claim_evidence_audit.csv"),
        ],
        "counts": {
            "split_rows": len(split_rows),
            "holm_rows": len(holm),
            "contour_rows": len(contour),
            "correlation_pairs": len(corr_pairs),
        },
        "seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPS,
    }
    write_text(EXP / "V29_REVIEW_COMPLETION_MANIFEST.json", json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
