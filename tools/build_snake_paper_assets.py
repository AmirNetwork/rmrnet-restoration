from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
RUN_ROOT = ROOT / "runs/snake_compare_v13"

MODELS = [
    ("degraded", "Degraded"),
    ("dfpir", "DFPIR"),
    ("demoe_auto", "DeMoE-auto"),
    ("demoe_scenario", "DeMoE-scenario"),
    ("nafnet", "NAFNet-road"),
    ("rmr", "RMR-Net"),
]

TASKS = {
    "pothole_motion": {
        "label": "Pothole motion",
        "image_dir": ROOT / "datasets/pothole_yolo_motion_test/images/test",
        "variants": {
            "degraded": ROOT / "datasets/pothole_yolo_motion_test/images/test",
            "nafnet": ROOT / "datasets/pothole_yolo_motion_test_nafnet/images/test",
            "dfpir": ROOT / "datasets/pothole_yolo_motion_test_dfpir/images/test",
            "demoe_auto": ROOT / "datasets/pothole_yolo_motion_test_demoe_auto/images/test",
            "demoe_scenario": ROOT / "datasets/pothole_yolo_motion_test_demoe_scenario/images/test",
            "rmr": ROOT / "datasets/pothole_yolo_motion_test_rmrnet_revised/images/test",
        },
        "n_images": 187,
    },
    "pcm_defocus": {
        "label": "PCM defocus",
        "image_dir": ROOT / "datasets/pcm_yolo_defocus_test/images/test",
        "variants": {
            "degraded": ROOT / "datasets/pcm_yolo_defocus_test/images/test",
            "nafnet": ROOT / "datasets/pcm_yolo_defocus_test_nafnet/images/test",
            "dfpir": ROOT / "datasets/pcm_yolo_defocus_test_dfpir/images/test",
            "demoe_auto": ROOT / "datasets/pcm_yolo_defocus_test_demoe_auto/images/test",
            "demoe_scenario": ROOT / "datasets/pcm_yolo_defocus_test_demoe_scenario/images/test",
            "rmr": ROOT / "datasets/pcm_yolo_defocus_test_rmrnet_revised/images/test",
        },
        "n_images": 302,
    },
    "pcm_lowlight": {
        "label": "PCM low light",
        "image_dir": ROOT / "datasets/pcm_yolo_lowlight_test/images/test",
        "variants": {
            "degraded": ROOT / "datasets/pcm_yolo_lowlight_test/images/test",
            "nafnet": ROOT / "datasets/pcm_yolo_lowlight_test_nafnet/images/test",
            "dfpir": ROOT / "datasets/pcm_yolo_lowlight_test_dfpir/images/test",
            "demoe_auto": ROOT / "datasets/pcm_yolo_lowlight_test_demoe_auto/images/test",
            "demoe_scenario": ROOT / "datasets/pcm_yolo_lowlight_test_demoe_scenario/images/test",
            "rmr": ROOT / "datasets/pcm_yolo_lowlight_test_rmrnet_revised/images/test",
        },
        "n_images": 302,
    },
}


def run_dir(task: str, model_key: str) -> Path:
    return RUN_ROOT / f"{task}_{model_key}"


def read_summary(task: str, model_key: str) -> dict:
    path = run_dir(task, model_key) / "snake_boundary_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(task: str, model_key: str) -> list[dict[str, str]]:
    path = run_dir(task, model_key) / "snake_boundary_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def success_bool(row: dict[str, str]) -> bool:
    return str(row.get("success", "")).lower() == "true"


def overall_edge_alignment(task: str, model_key: str) -> float:
    rows = [row for row in read_metrics(task, model_key) if success_bool(row)]
    values = [float(row["edge_alignment"]) for row in rows if row.get("edge_alignment")]
    return float(np.mean(values)) if values else 0.0


def class_stats(task: str, model_key: str, class_name: str) -> dict:
    return read_summary(task, model_key)["classes"].get(
        class_name,
        {
            "objects": 0,
            "successes": 0,
            "success_rate": 0.0,
            "mean_edge_alignment": 0.0,
            "mean_contrast": 0.0,
        },
    )


def fmt_int(value: int | float) -> str:
    return f"{int(round(value))}"


def fmt_float(value: float, ndigits: int = 2) -> str:
    return f"{value:.{ndigits}f}"


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def maybe_bold(text: str, condition: bool) -> str:
    return rf"\textbf{{{text}}}" if condition else text


def write_boundary_comparison_table() -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Detector-guided active-contour boundary recognition after each restoration method. Accepted contours/image is the operational measurement yield: the number of detector boxes that produced a non-collapsed, non-leaking contour divided by the number of test images. This is more informative than pass rate alone because a method can have a high pass rate after producing very few detector boxes. Bold marks the best measurement yield within each degradation setting.}",
        r"\label{tab:snake_boundary_comparison}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Setting & Image source & Boxes & Accepted & Accepted/img & Pass (\%) & Edge align. \\",
        r"\midrule",
    ]
    for task_key, task in TASKS.items():
        yields = {}
        rows = {}
        for model_key, model_name in MODELS:
            summary = read_summary(task_key, model_key)
            n_images = int(task["n_images"])
            accepted = int(summary["successes"])
            measurement_yield = accepted / max(n_images, 1)
            yields[model_key] = measurement_yield
            rows[model_key] = {
                "model": model_name,
                "boxes": int(summary["objects"]),
                "accepted": accepted,
                "yield": measurement_yield,
                "pass": float(summary["success_rate"]),
                "edge": overall_edge_alignment(task_key, model_key),
            }
        best_key = max(yields, key=yields.get)
        for idx, (model_key, _) in enumerate(MODELS):
            row = rows[model_key]
            setting = task["label"] if idx == 0 else ""
            lines.append(
                f"{setting} & {row['model']} & {row['boxes']} & "
                f"{maybe_bold(fmt_int(row['accepted']), model_key == best_key)} & "
                f"{maybe_bold(fmt_float(row['yield'], 2), model_key == best_key)} & "
                f"{fmt_pct(row['pass'])} & {fmt_float(row['edge'], 2)} \\\\"
            )
        if task_key != list(TASKS.keys())[-1]:
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_snake_boundary_comparison.tex").write_text("\n".join(lines), encoding="utf-8")


def write_crack_measurement_table() -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Crack-specific boundary-recognition audit on the PCM test set. The crack path uses ridge-enhanced active contours inside YOLO crack boxes. Bold marks the highest accepted crack-contour yield.}",
        r"\label{tab:snake_crack_measurement}",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Setting & Source & Boxes & Accepted & Acc./img & Pass (\%) & Edge \\",
        r"\midrule",
    ]
    for task_key in ["pcm_defocus", "pcm_lowlight"]:
        n_images = int(TASKS[task_key]["n_images"])
        yields = {}
        rows = {}
        for model_key, model_name in MODELS:
            stats = class_stats(task_key, model_key, "crack")
            accepted = int(stats["successes"])
            measurement_yield = accepted / max(n_images, 1)
            yields[model_key] = measurement_yield
            rows[model_key] = {
                "model": model_name,
                "boxes": int(stats["objects"]),
                "accepted": accepted,
                "yield": measurement_yield,
                "pass": float(stats["success_rate"]),
                "edge": float(stats["mean_edge_alignment"]),
            }
        best_key = max(yields, key=yields.get)
        for idx, (model_key, _) in enumerate(MODELS):
            row = rows[model_key]
            setting = TASKS[task_key]["label"] if idx == 0 else ""
            lines.append(
                f"{setting} & {row['model']} & {row['boxes']} & "
                f"{maybe_bold(fmt_int(row['accepted']), model_key == best_key)} & "
                f"{maybe_bold(fmt_float(row['yield'], 3), model_key == best_key)} & "
                f"{fmt_pct(row['pass'])} & {fmt_float(row['edge'], 2)} \\\\"
            )
        if task_key != "pcm_lowlight":
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_snake_crack_measurement.tex").write_text("\n".join(lines), encoding="utf-8")


def build_yield_figure() -> None:
    labels = [TASKS[key]["label"] for key in TASKS]
    x = np.arange(len(labels))
    width = 0.19
    colors = {
        "degraded": "#6B7280",
        "dfpir": "#10B981",
        "demoe_auto": "#7f6dba",
        "demoe_scenario": "#b279a2",
        "nafnet": "#3B82F6",
        "rmr": "#E11D48",
    }
    fig, ax = plt.subplots(figsize=(8.8, 3.4), dpi=240)
    offsets = np.arange(len(MODELS)) - (len(MODELS) - 1) / 2.0
    for offset, (model_key, model_name) in zip(offsets, MODELS):
        values = [
            read_summary(task_key, model_key)["successes"] / TASKS[task_key]["n_images"]
            for task_key in TASKS
        ]
        bars = ax.bar(
            x + offset * width,
            values,
            width,
            label=model_name,
            color=colors[model_key],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_ylabel("Accepted contours per image")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.24), fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.4)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_snake_boundary_yield.png", bbox_inches="tight")
    plt.close(fig)


def crop_box_for_image(csv_path: Path, image_name: str) -> tuple[int, int, int, int] | None:
    if not csv_path.exists():
        return None
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["image"] == image_name and success_bool(row):
                rows.append(row)
    if not rows:
        return None
    x1 = min(int(float(row["crop_x1"])) for row in rows)
    y1 = min(int(float(row["crop_y1"])) for row in rows)
    x2 = max(int(float(row["crop_x2"])) for row in rows)
    y2 = max(int(float(row["crop_y2"])) for row in rows)
    return x1, y1, x2, y2


def accepted_count(task: str, model_key: str, image_name: str, class_name: str | None = None) -> int:
    count = 0
    for row in read_metrics(task, model_key):
        if row["image"] != image_name or not success_bool(row):
            continue
        if class_name is not None and row["class_name"] != class_name:
            continue
        count += 1
    return count


def read_panel(
    path: Path,
    target_w: int,
    target_h: int,
    crop_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if crop_box is not None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = crop_box
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        pad_x = max(24, int(round(0.40 * bw)))
        pad_y = max(24, int(round(0.55 * bh)))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)
        image = image[y1 : y2 + 1, x1 : x2 + 1]
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    panel = np.full((target_h, target_w, 3), 248, dtype=np.uint8)
    y0 = (target_h - resized.shape[0]) // 2
    x0 = (target_w - resized.shape[1]) // 2
    panel[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return panel


def image_or_overlay(task: str, model_key: str, image_name: str) -> Path:
    overlay = run_dir(task, model_key) / "overlays" / f"{Path(image_name).stem}_snake.png"
    if overlay.exists():
        return overlay
    candidate = TASKS[task]["variants"][model_key] / image_name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find overlay or source image for {task}/{model_key}/{image_name}")


def add_panel_label(panel: np.ndarray, label: str, sublabel: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 44), (255, 255, 255), -1)
    cv2.putText(panel, label, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, sublabel, (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA)


def build_cross_model_figure() -> None:
    examples = [
        ("Pothole / motion", "pothole_motion", "img-1171.jpg", None),
        ("Crack / defocus", "pcm_defocus", "vlcsnap-2025-02-26-20h36m12s955.jpg", "crack"),
    ]
    panel_w, panel_h = 270, 190
    gutter = 12
    header_h = 34
    row_label_w = 0
    width = len(MODELS) * panel_w + (len(MODELS) - 1) * gutter + row_label_w
    height = len(examples) * panel_h + (len(examples) - 1) * gutter + header_h
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for col, (_, model_name) in enumerate(MODELS):
        x = row_label_w + col * (panel_w + gutter)
        cv2.putText(canvas, model_name, (x + 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 1, cv2.LINE_AA)
    for row_idx, (row_title, task, image_name, class_name) in enumerate(examples):
        crop_box = crop_box_for_image(run_dir(task, "rmr") / "snake_boundary_metrics.csv", image_name)
        y = header_h + row_idx * (panel_h + gutter)
        for col_idx, (model_key, model_name) in enumerate(MODELS):
            x = row_label_w + col_idx * (panel_w + gutter)
            path = image_or_overlay(task, model_key, image_name)
            panel = read_panel(path, panel_w, panel_h, crop_box)
            count = accepted_count(task, model_key, image_name, class_name)
            add_panel_label(panel, row_title if col_idx == 0 else model_name, f"accepted contours: {count}")
            canvas[y : y + panel_h, x : x + panel_w] = panel
    FIGURES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURES / "fig_snake_boundary_cross_model.png"), canvas)


def build_refinement_atlas() -> None:
    selected = [
        ("pothole_motion", "img-1171.jpg", "motion potholes"),
        ("pothole_motion", "img-721.jpg", "multiple potholes"),
        ("pothole_motion", "img-790.jpg", "wet pothole rims"),
        ("pcm_defocus", "vlcsnap-2025-02-26-20h36m12s955.jpg", "defocus cracks"),
        ("pcm_defocus", "20250219_164919.jpg", "patch/crack edges"),
        ("pcm_lowlight", "vlcsnap-2025-02-26-20h37m43s267.jpg", "low-light cracks"),
    ]
    panel_w, panel_h = 420, 236
    gutter = 14
    canvas = np.full((2 * panel_h + gutter, 3 * panel_w + 2 * gutter, 3), 255, dtype=np.uint8)
    for idx, (task, image_name, label) in enumerate(selected):
        r, c = divmod(idx, 3)
        y = r * (panel_h + gutter)
        x = c * (panel_w + gutter)
        crop_box = crop_box_for_image(run_dir(task, "rmr") / "snake_boundary_metrics.csv", image_name)
        panel = read_panel(image_or_overlay(task, "rmr", image_name), panel_w, panel_h, crop_box)
        add_panel_label(panel, label, f"RMR-Net accepted: {accepted_count(task, 'rmr', image_name)}")
        canvas[y : y + panel_h, x : x + panel_w] = panel
    FIGURES.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGURES / "fig_snake_boundary_refinement.png"), canvas)


def main() -> None:
    write_boundary_comparison_table()
    write_crack_measurement_table()
    build_yield_figure()
    build_cross_model_figure()
    build_refinement_atlas()
    print(
        json.dumps(
            {
                "tables": [
                    str(TABLES / "table_snake_boundary_comparison.tex"),
                    str(TABLES / "table_snake_crack_measurement.tex"),
                ],
                "figures": [
                    str(FIGURES / "fig_snake_boundary_yield.png"),
                    str(FIGURES / "fig_snake_boundary_cross_model.png"),
                    str(FIGURES / "fig_snake_boundary_refinement.png"),
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
