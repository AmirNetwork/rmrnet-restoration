from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


SCENARIOS = {
    "motion": "motion_horizontal_medium",
    "defocus": "defocus_medium",
    "lowlight": "lowlight_medium",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def latex_table(headers: list[str], rows: list[list[str]], caption: str, label: str, align: str | None = None) -> str:
    if align is None:
        align = "ll" + "r" * max(len(headers) - 2, 0)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def detection_metrics(root: Path) -> list[dict]:
    names = {
        "clean": "clean_test_metrics.json",
        "motion/degraded": "motion_test_metrics.json",
        "motion/RCAD-Net": "motion_test_rcadnet_metrics.json",
        "motion/RCAD-Net++": "motion_test_rcadnetpp_metrics.json",
        "motion/DFPIR": "motion_test_dfpir_metrics.json",
        "defocus/degraded": "defocus_test_metrics.json",
        "defocus/RCAD-Net": "defocus_test_rcadnet_metrics.json",
        "defocus/RCAD-Net++": "defocus_test_rcadnetpp_metrics.json",
        "defocus/DFPIR": "defocus_test_dfpir_metrics.json",
        "lowlight/degraded": "lowlight_test_metrics.json",
        "lowlight/RCAD-Net": "lowlight_test_rcadnet_metrics.json",
        "lowlight/RCAD-Net++": "lowlight_test_rcadnetpp_metrics.json",
        "lowlight/DFPIR": "lowlight_test_dfpir_metrics.json",
    }
    rows = []
    for key, filename in names.items():
        path = root / filename
        if not path.exists():
            continue
        metric = read_json(path)
        if key == "clean":
            scenario, method = "clean", "clean"
        else:
            scenario, method = key.split("/")
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "map50": metric["map50"],
                "map50_95": metric["map50_95"],
                "precision": metric["precision"],
                "recall": metric["recall"],
            }
        )
    return rows


def restoration_metrics(root: Path) -> list[dict]:
    rows = []
    sources = [
        ("RCAD-Net", root / "runs" / "bench_pothole_restoration_test_rcadnet" / "metrics.csv"),
        ("RCAD-Net++", root / "runs" / "bench_pothole_restoration_test_rcadnetpp_dfpir" / "metrics.csv"),
        ("DFPIR", root / "runs" / "bench_pothole_restoration_test_rcadnetpp_dfpir" / "metrics.csv"),
    ]
    for label, path in sources:
        if not path.exists():
            continue
        for row in read_csv(path):
            model = row["model"]
            if label == "RCAD-Net++" and model != "RCAD-Net":
                continue
            if label == "DFPIR" and "DFPIR" not in model:
                continue
            if label == "RCAD-Net" and model != "RCAD-Net":
                continue
            rows.append(
                {
                    "scenario": row["scenario"].replace("_medium", "").replace("_horizontal", ""),
                    "method": label,
                    "psnr": float(row["psnr"]),
                    "ssim": float(row["ssim"]),
                    "runtime": float(row["mean_runtime_ms"]),
                }
            )
    return rows


def kodak_metrics(root: Path) -> list[dict]:
    path = root / "runs" / "bench_kodak24_rcadnetpp_dfpir" / "metrics.csv"
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        method = "RCAD-Net++" if row["model"] == "RCAD-Net" else "DFPIR"
        rows.append(
            {
                "scenario": row["scenario"],
                "method": method,
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "runtime": float(row["mean_runtime_ms"]),
            }
        )
    return rows


def write_detection_table(rows: list[dict], out: Path) -> None:
    body = []
    for row in rows:
        body.append(
            [
                row["scenario"],
                row["method"],
                fmt(row["map50"], 3),
                fmt(row["map50_95"], 3),
                fmt(row["precision"], 3),
                fmt(row["recall"], 3),
            ]
        )
    tex = latex_table(
        ["Scenario", "Input", "mAP50", "mAP50--95", "Prec.", "Rec."],
        body,
        "Held-out pothole detection after degradation and restoration. The detector is frozen for every row.",
        "tab:detection_test",
    )
    (out / "table_detection_test.tex").write_text(tex, encoding="utf-8")


def write_restoration_table(rows: list[dict], out: Path) -> None:
    body = []
    for row in rows:
        body.append([row["scenario"], row["method"], fmt(row["psnr"], 2), fmt(row["ssim"], 3), fmt(row["runtime"], 1)])
    tex = latex_table(
        ["Scenario", "Model", "PSNR", "SSIM", "ms/img"],
        body,
        "Held-out full-reference restoration and runtime on RTX 3050 at 320 px long side.",
        "tab:restoration_test",
    )
    (out / "table_restoration_test.tex").write_text(tex, encoding="utf-8")


def write_kodak_table(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    body = []
    for row in rows:
        scenario = row["scenario"].replace("_medium", "").replace("_", " ")
        body.append([scenario, row["method"], fmt(row["psnr"], 2), fmt(row["ssim"], 3), fmt(row["runtime"], 1)])
    tex = latex_table(
        ["Scenario", "Model", "PSNR", "SSIM", "ms/img"],
        body,
        "Kodak-24 synthetic restoration appendix benchmark. This tests generic natural-image transfer rather than road-task detection.",
        "tab:kodak24_appendix",
    )
    (out / "table_kodak24_appendix.tex").write_text(tex, encoding="utf-8")


def write_baseline_table(out: Path) -> None:
    rows = [
        ["DFPIR", "CVPR 2025", "all-in-one, 5D", "official code+weights", "run"],
        ["Restormer", "CVPR 2022 oral", "motion, defocus, denoise, derain", "official code+weights", "recommended"],
        ["NAFNet", "ECCV 2022", "efficient deblur/denoise", "official code+weights", "recommended"],
        ["MPRNet", "CVPR 2021", "multi-stage deblur/derain/denoise", "official code+weights", "recommended"],
        ["FFTformer", "CVPR 2023", "frequency-domain deblurring", "official code+weights", "recommended"],
        ["DarkIR", "CVPR 2025", "low-light blur/noise/enhancement", "official code+weights", "recommended for low light"],
        ["InstructIR", "ECCV 2024", "prompted all-in-one restoration", "code+HF weights", "optional"],
    ]
    tex = latex_table(
        ["Baseline", "Venue", "Coverage", "Availability", "Status"],
        rows,
        "Reviewer-facing baseline pool. The current executed baseline is DFPIR; remaining rows are integration targets with public implementations.",
        "tab:baseline_pool",
        align="lllll",
    )
    (out / "table_baseline_pool.tex").write_text(tex, encoding="utf-8")


def detection_bar(rows: list[dict], out: Path) -> None:
    scenarios = ["motion", "defocus", "lowlight"]
    methods = ["degraded", "RCAD-Net", "RCAD-Net++", "DFPIR"]
    values = {(r["scenario"], r["method"]): r["map50"] for r in rows}
    x = range(len(scenarios))
    width = 0.19
    colors = ["#8a8f98", "#2f6f9f", "#159a74", "#b5533c"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    for i, method in enumerate(methods):
        ax.bar([v + (i - 1.5) * width for v in x], [values.get((s, method), 0) for s in scenarios], width, label=method, color=colors[i])
    ax.set_ylabel("YOLO mAP50")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["motion", "defocus", "low light"])
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(out / "fig_detection_map50_test.png", bbox_inches="tight")
    plt.close(fig)


def runtime_tradeoff(det_rows: list[dict], res_rows: list[dict], out: Path) -> None:
    det = {(r["scenario"], r["method"]): r["map50"] for r in det_rows}
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=200)
    markers = {"RCAD-Net": "o", "RCAD-Net++": "s", "DFPIR": "^"}
    colors = {"motion": "#666666", "defocus": "#2f6f9f", "lowlight": "#159a74"}
    for row in res_rows:
        scenario = "lowlight" if row["scenario"] == "lowlight" else row["scenario"]
        method = row["method"]
        if method not in markers:
            continue
        y = det.get((scenario, method))
        if y is None:
            continue
        ax.scatter(row["runtime"], y, s=70, marker=markers[method], color=colors.get(scenario, "#222222"), edgecolor="white", linewidth=0.8)
        ax.text(row["runtime"] + 4, y, f"{method} {scenario}", fontsize=7, va="center")
    ax.set_xlabel("Restoration runtime (ms/image)")
    ax.set_ylabel("Downstream mAP50")
    ax.set_xscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_runtime_detection_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def architecture_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.0), dpi=220)
    ax.axis("off")
    boxes = [
        ("Road image\n$I_d$", (0.05, 0.58), "#e8ecef"),
        ("Blind degradation\nencoder", (0.24, 0.76), "#d7ece6"),
        ("Scenario / sensor\ncode (optional)", (0.24, 0.40), "#f2e2d4"),
        ("Fused road\ncondition code", (0.45, 0.58), "#dce8f6"),
        ("Defect-edge\nattention", (0.63, 0.76), "#efe0ec"),
        ("FiLM-conditioned\nlightweight U-Net", (0.63, 0.40), "#dce8f6"),
        ("Restored road\nimage $I_r$", (0.84, 0.58), "#e8ecef"),
        ("Frozen YOLO\nfor task eval", (0.84, 0.18), "#f4e7c5"),
    ]
    for text, (x, y), color in boxes:
        rect = plt.Rectangle((x, y), 0.14, 0.16, facecolor=color, edgecolor="#24323d", linewidth=1.2, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x + 0.07, y + 0.08, text, ha="center", va="center", fontsize=8.5)
    arrows = [
        ((0.19, 0.66), (0.24, 0.84)),
        ((0.38, 0.84), (0.45, 0.66)),
        ((0.38, 0.48), (0.45, 0.66)),
        ((0.19, 0.66), (0.63, 0.84)),
        ((0.59, 0.66), (0.63, 0.48)),
        ((0.77, 0.48), (0.84, 0.66)),
        ((0.91, 0.58), (0.91, 0.34)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#24323d", lw=1.2))
    ax.text(0.45, 0.18, "Training losses: L1 + edge + FFT + defect-weighted reconstruction + auxiliary code loss", ha="center", fontsize=8.5)
    ax.text(0.45, 0.08, "Novelty: task-driven road restoration that conditions on degradation evidence and preserves pothole/crack boundaries.", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig_rcadnetpp_architecture.png", bbox_inches="tight")
    plt.close(fig)


def center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = image.size
    tw, th = size
    left = max((w - tw) // 2, 0)
    top = max((h - th) // 2, 0)
    return image.crop((left, top, min(left + tw, w), min(top + th, h))).resize(size, Image.Resampling.BICUBIC)


def make_panel(root: Path, scenario: str, out: Path) -> None:
    input_dir = root / "data" / "pothole_restoration_test" / "scenarios" / scenario / "input"
    gt_dir = root / "data" / "pothole_restoration_test" / "scenarios" / scenario / "gt"
    tag = "motion" if "motion" in scenario else "lowlight" if "lowlight" in scenario else "defocus"
    rcad = root / f"datasets/pothole_yolo_{tag}_test_rcadnet/images/test"
    rcadpp = root / f"datasets/pothole_yolo_{tag}_test_rcadnetpp/images/test"
    dfpir = root / f"datasets/pothole_yolo_{tag}_test_dfpir/images/test"
    name = sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[0]
    paths = [
        ("degraded", input_dir / name),
        ("RCAD-Net", rcad / name),
        ("RCAD-Net++", rcadpp / name),
        ("DFPIR", dfpir / name),
        ("ground truth", gt_dir / name),
    ]
    tile = (260, 180)
    label_h = 28
    panel = Image.new("RGB", (tile[0] * len(paths), tile[1] + label_h), "white")
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (label, path) in enumerate(paths):
        with Image.open(path) as image:
            crop = center_crop(image.convert("RGB"), tile)
        panel.paste(crop, (i * tile[0], label_h))
        draw.text((i * tile[0] + 8, 6), label, fill=(20, 20, 20), font=font)
    panel.save(out / f"qual_{scenario}.png")


def write_summary(det_rows: list[dict], res_rows: list[dict], out: Path) -> None:
    by = {(r["scenario"], r["method"]): r for r in det_rows}
    lines = [
        "# Paper Asset Summary",
        "",
        "Main final method: **RCAD-Net++**, which adds a blind image-derived degradation-code estimator to the original scenario-conditioned RCAD-Net.",
        "",
        "Strongest held-out test detection facts:",
        "",
    ]
    for scenario in ["motion", "defocus", "lowlight"]:
        degraded = by[(scenario, "degraded")]["map50"]
        ours = by[(scenario, "RCAD-Net++")]["map50"]
        dfpir = by[(scenario, "DFPIR")]["map50"]
        lines.append(f"- {scenario}: degraded {degraded:.3f}, RCAD-Net++ {ours:.3f}, DFPIR {dfpir:.3f}.")
    lines += [
        "",
        "The paper should state the tradeoff clearly: RCAD-Net++ is optimized for downstream road-defect evidence, not for maximizing generic PSNR in every blur family.",
    ]
    (out / "PAPER_ASSET_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_root = root / "paper_assets" / "rcadnetpp_2026-05-24"
    tables = out_root / "tables"
    figures = out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    det_rows = detection_metrics(root / "runs" / "detection_eval")
    res_rows = restoration_metrics(root)
    kodak_rows = kodak_metrics(root)
    write_detection_table(det_rows, tables)
    write_restoration_table(res_rows, tables)
    write_kodak_table(kodak_rows, tables)
    write_baseline_table(tables)
    detection_bar(det_rows, figures)
    runtime_tradeoff(det_rows, res_rows, figures)
    architecture_figure(figures)
    for scenario in SCENARIOS.values():
        make_panel(root, scenario, figures)
    write_summary(det_rows, res_rows, out_root)
    (out_root / "detection_test.json").write_text(json.dumps(det_rows, indent=2), encoding="utf-8")
    (out_root / "restoration_test.json").write_text(json.dumps(res_rows, indent=2), encoding="utf-8")
    (out_root / "kodak24_appendix.json").write_text(json.dumps(kodak_rows, indent=2), encoding="utf-8")
    print(out_root)


if __name__ == "__main__":
    main()
