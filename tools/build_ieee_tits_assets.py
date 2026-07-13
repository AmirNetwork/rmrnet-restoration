from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "paper_assets" / "rcadnetpp_2026-05-24"
OUT = ROOT / "paper_assets" / "ieee_tits_rcadnetpp_2026-05-24"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"


def load_json(name: str) -> list[dict]:
    return json.loads((ASSET_ROOT / name).read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]], wide: bool = False) -> None:
    env = "table*" if wide else "table"
    align = "ll" + "r" * max(0, len(headers) - 2)
    lines = [
        f"\\begin{{{env}}}[!t]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\small",
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def table_detection(rows: list[dict]) -> None:
    body = []
    order = [
        ("clean", "clean"),
        ("motion", "degraded"),
        ("motion", "RCAD-Net"),
        ("motion", "RCAD-Net++"),
        ("motion", "DFPIR"),
        ("defocus", "degraded"),
        ("defocus", "RCAD-Net"),
        ("defocus", "RCAD-Net++"),
        ("defocus", "DFPIR"),
        ("lowlight", "degraded"),
        ("lowlight", "RCAD-Net"),
        ("lowlight", "RCAD-Net++"),
        ("lowlight", "DFPIR"),
    ]
    by = {(r["scenario"], r["method"]): r for r in rows}
    for key in order:
        row = by[key]
        body.append([row["scenario"], row["method"], fmt(row["map50"]), fmt(row["map50_95"]), fmt(row["precision"]), fmt(row["recall"])])
    write_table(
        TAB_DIR / "table_detection_ieee.tex",
        "Held-out pothole detection under controlled road-monitoring degradations. The YOLO detector is frozen for all degraded and restored inputs.",
        "tab:detection_ieee",
        ["Scenario", "Input", "mAP50", "mAP50--95", "Prec.", "Rec."],
        body,
        wide=True,
    )


def table_restoration(rows: list[dict]) -> None:
    body = [[r["scenario"], r["method"], fmt(r["psnr"], 2), fmt(r["ssim"]), fmt(r["runtime"], 1)] for r in rows]
    write_table(
        TAB_DIR / "table_restoration_ieee.tex",
        "Full-reference road restoration and runtime on the held-out test split at 320-pixel long side.",
        "tab:restoration_ieee",
        ["Scenario", "Model", "PSNR", "SSIM", "ms/img"],
        body,
        wide=False,
    )


def table_kodak(rows: list[dict]) -> None:
    body = [[r["scenario"].replace("_", " "), r["method"], fmt(r["psnr"], 2), fmt(r["ssim"]), fmt(r["runtime"], 1)] for r in rows]
    write_table(
        TAB_DIR / "table_kodak_ieee.tex",
        "Kodak-24 generic restoration sanity check. This table clarifies that RCAD-Net++ is road-task-oriented rather than a universal natural-image restoration replacement.",
        "tab:kodak_ieee",
        ["Scenario", "Model", "PSNR", "SSIM", "ms/img"],
        body,
        wide=True,
    )


def table_baselines() -> None:
    rows = [
        ["DFPIR", "CVPR 2025", "All-in-one restoration", "Executed"],
        ["RCAD-Net", "Ablation", "No blind degradation encoder", "Executed"],
        ["RCAD-Net++", "Proposed", "Blind degradation code + defect attention", "Executed"],
        ["Restormer", "CVPR 2022", "Motion/defocus/denoise/derain", "Planned"],
        ["NAFNet", "ECCV 2022", "Efficient deblur/denoise", "Planned"],
        ["FFTformer", "CVPR 2023", "Frequency-domain deblurring", "Planned"],
        ["DarkIR", "CVPR 2025", "Low-light restoration", "Planned"],
        ["InstructIR", "ECCV 2024", "Instruction-guided all-in-one restoration", "Optional"],
    ]
    write_table(
        TAB_DIR / "table_baseline_ieee.tex",
        "Baseline and ablation matrix for the current submission package and extension plan.",
        "tab:baseline_ieee",
        ["Method", "Venue/Role", "Coverage", "Status"],
        rows,
        wide=True,
    )


def detection_recovery(rows: list[dict]) -> None:
    by = {(r["scenario"], r["method"]): r["map50"] for r in rows}
    scenarios = ["motion", "defocus", "lowlight"]
    labels = ["Motion blur", "Defocus", "Low light"]
    methods = ["degraded", "RCAD-Net", "RCAD-Net++", "DFPIR"]
    colors = ["#8c939c", "#3b78a8", "#139b75", "#b95745"]
    x = range(len(scenarios))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.25), dpi=240)
    for i, method in enumerate(methods):
        ax.bar([v + (i - 1.5) * width for v in x], [by[(s, method)] for s in scenarios], width, label=method, color=colors[i])
    clean = by[("clean", "clean")]
    ax.axhline(clean, color="#222222", linestyle="--", linewidth=1.0)
    ax.text(2.42, clean + 0.01, "clean detector", fontsize=8, va="bottom")
    ax.set_ylabel("Frozen YOLO mAP50")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.66)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ieee_detection_recovery.png", bbox_inches="tight")
    plt.close(fig)


def its_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=240)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        ("Vehicle/roadside\nRGB camera", 0.04, 0.58, 0.16, 0.22, "#e7edf3"),
        ("Road degradations\nblur, vibration,\nlow light", 0.26, 0.58, 0.18, 0.22, "#f4e8d8"),
        ("RCAD-Net++\nedge restoration", 0.51, 0.58, 0.18, 0.22, "#dfeff0"),
        ("Road-defect\nperception", 0.75, 0.58, 0.18, 0.22, "#e9e5f2"),
        ("Autonomous vehicle\nrisk awareness", 0.17, 0.16, 0.22, 0.18, "#eef3df"),
        ("Maintenance\nprioritization", 0.58, 0.16, 0.22, 0.18, "#eef3df"),
    ]
    for text, x, y, w, h, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#26343f", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
    arrows = [
        ((0.20, 0.69), (0.26, 0.69)),
        ((0.44, 0.69), (0.51, 0.69)),
        ((0.69, 0.69), (0.75, 0.69)),
        ((0.84, 0.58), (0.30, 0.34)),
        ((0.84, 0.58), (0.69, 0.34)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.2, color="#26343f"))
    ax.text(0.51, 0.48, "restoration judged by downstream defect mAP + latency", ha="center", fontsize=8.5, color="#26343f")
    ax.text(0.51, 0.04, "T-ITS positioning: robust perception for connected/autonomous road monitoring and infrastructure management", ha="center", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ieee_its_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = image.size
    tw, th = size
    left = max((w - tw) // 2, 0)
    top = max((h - th) // 2, 0)
    return image.crop((left, top, min(left + tw, w), min(top + th, h))).resize(size, Image.Resampling.BICUBIC)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def qualitative_zoom(scenario: str, tag: str, title: str) -> None:
    input_dir = ROOT / "data" / "pothole_restoration_test" / "scenarios" / scenario / "input"
    gt_dir = ROOT / "data" / "pothole_restoration_test" / "scenarios" / scenario / "gt"
    rcadpp_dir = ROOT / f"datasets/pothole_yolo_{tag}_test_rcadnetpp/images/test"
    dfpir_dir = ROOT / f"datasets/pothole_yolo_{tag}_test_dfpir/images/test"
    name = sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[0]
    items = [
        ("Degraded", input_dir / name),
        ("RCAD-Net++", rcadpp_dir / name),
        ("DFPIR", dfpir_dir / name),
        ("Clean", gt_dir / name),
    ]
    tile = (260, 170)
    zoom = (260, 120)
    pad = 10
    label_h = 34
    title_h = 34
    width = len(items) * tile[0] + (len(items) + 1) * pad
    height = title_h + label_h + tile[1] + pad + zoom[1] + pad
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    f_title = font(18)
    f_label = font(15)
    draw.text((pad, 8), title, fill=(20, 20, 20), font=f_title)
    for i, (label, path) in enumerate(items):
        x = pad + i * (tile[0] + pad)
        y = title_h + label_h
        with Image.open(path) as image:
            full = center_crop(image.convert("RGB"), tile)
            z = center_crop(image.convert("RGB"), zoom)
        panel.paste(full, (x, y))
        panel.paste(z, (x, y + tile[1] + pad))
        draw.text((x + 6, title_h + 8), label, fill=(20, 20, 20), font=f_label)
        draw.rectangle((x, y, x + tile[0] - 1, y + tile[1] - 1), outline=(210, 210, 210), width=1)
        draw.rectangle((x, y + tile[1] + pad, x + zoom[0] - 1, y + tile[1] + pad + zoom[1] - 1), outline=(180, 36, 31), width=2)
    panel.save(FIG_DIR / f"fig_ieee_qual_{tag}_zoom.png")


def tradeoff(rows_det: list[dict], rows_res: list[dict]) -> None:
    det = {(r["scenario"], r["method"]): r["map50"] for r in rows_det}
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=240)
    method_style = {"RCAD-Net": ("o", "#3b78a8"), "RCAD-Net++": ("s", "#139b75"), "DFPIR": ("^", "#b95745")}
    for row in rows_res:
        scenario = row["scenario"]
        method = row["method"]
        if method not in method_style:
            continue
        y = det.get((scenario, method))
        if y is None:
            continue
        marker, color = method_style[method]
        ax.scatter(row["runtime"], y, s=70, marker=marker, color=color, edgecolor="white", linewidth=0.8)
        ax.text(row["runtime"] * 1.05, y, f"{method} / {scenario}", fontsize=7.5, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Restoration latency (ms/image, log scale)")
    ax.set_ylabel("Frozen detector mAP50")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ieee_latency_map_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def task_teaser(rows_det: list[dict]) -> None:
    """Create a compact IEEE-style summary figure for the main manuscript."""
    by = {(r["scenario"], r["method"]): r["map50"] for r in rows_det}
    examples = [
        ("motion_horizontal_medium", "motion", "Motion blur"),
        ("defocus_medium", "defocus", "Defocus"),
        ("lowlight_medium", "lowlight", "Low light"),
    ]
    methods = [
        ("Degraded", "input", "#8c939c"),
        ("RCAD-Net++", "rcadnetpp", "#139b75"),
        ("DFPIR", "dfpir", "#b95745"),
        ("Clean", "gt", "#26343f"),
    ]

    tile = (190, 116)
    pad = 12
    left_w = 4 * tile[0] + 5 * pad
    right_w = 440
    header_h = 78
    row_h = tile[1] + 48
    width = left_w + right_w + 3 * pad
    height = header_h + len(examples) * row_h + pad
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    f_title = font(25)
    f_sub = font(14)
    f_label = font(14)
    f_small = font(12)

    draw.text((pad, 14), "Task-driven road restoration for intelligent transportation systems", fill=(20, 28, 34), font=f_title)
    draw.text(
        (pad, 46),
        "RCAD-Net++ targets defect evidence and edge latency, then evaluates whether frozen road perception recovers.",
        fill=(70, 78, 86),
        font=f_sub,
    )

    for j, (name, _, color) in enumerate(methods):
        x = pad + j * (tile[0] + pad)
        draw.text((x + 4, header_h - 20), name, fill=color, font=f_label)

    for i, (scenario, tag, label) in enumerate(examples):
        y = header_h + i * row_h
        input_dir = ROOT / "data" / "pothole_restoration_test" / "scenarios" / scenario / "input"
        gt_dir = ROOT / "data" / "pothole_restoration_test" / "scenarios" / scenario / "gt"
        rcadpp_dir = ROOT / f"datasets/pothole_yolo_{tag}_test_rcadnetpp/images/test"
        dfpir_dir = ROOT / f"datasets/pothole_yolo_{tag}_test_dfpir/images/test"
        name = sorted(p.name for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[0]
        paths = {
            "input": input_dir / name,
            "gt": gt_dir / name,
            "rcadnetpp": rcadpp_dir / name,
            "dfpir": dfpir_dir / name,
        }
        draw.text((pad, y + 6), label, fill=(20, 28, 34), font=f_label)
        for j, (_, key, color) in enumerate(methods):
            x = pad + j * (tile[0] + pad)
            with Image.open(paths[key]) as image:
                crop = center_crop(image.convert("RGB"), tile)
            panel.paste(crop, (x, y + 30))
            draw.rectangle((x, y + 30, x + tile[0] - 1, y + tile[1] + 29), outline=color, width=2)

    chart_x = left_w + 2 * pad
    chart_y = header_h + 16
    draw.text((chart_x, chart_y - 36), "Held-out pothole detection mAP50", fill=(20, 28, 34), font=f_label)
    draw.line((chart_x, chart_y + 5, chart_x + 365, chart_y + 5), fill=(205, 210, 214), width=1)
    bar_h = 18
    gap = 9
    scale_w = 335
    chart_methods = [("degraded", "#8c939c"), ("RCAD-Net++", "#139b75"), ("DFPIR", "#b95745")]
    for i, (_, tag, label) in enumerate(examples):
        base_y = chart_y + 26 + i * 116
        draw.text((chart_x, base_y - 20), label, fill=(20, 28, 34), font=f_small)
        for j, (method, color) in enumerate(chart_methods):
            value = by[(tag, method)]
            y0 = base_y + j * (bar_h + gap)
            draw.rectangle((chart_x, y0, chart_x + scale_w, y0 + bar_h), fill=(238, 241, 243))
            draw.rectangle((chart_x, y0, chart_x + int(scale_w * value / 0.60), y0 + bar_h), fill=color)
            draw.text((chart_x + scale_w + 8, y0 - 1), f"{value:.3f}", fill=(20, 28, 34), font=f_small)
            draw.text((chart_x + 6, y0 + 2), method, fill=(255, 255, 255), font=f_small)

    clean = by[("clean", "clean")]
    draw.text((chart_x, height - 42), f"Clean detector reference: {clean:.3f} mAP50", fill=(20, 28, 34), font=f_small)
    draw.text((chart_x, height - 24), "Latency: RCAD-Net++ 42-47 ms/img; DFPIR 402-419 ms/img at 320 px.", fill=(70, 78, 86), font=f_small)
    panel.save(FIG_DIR / "fig_ieee_task_teaser.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    detection = load_json("detection_test.json")
    restoration = load_json("restoration_test.json")
    kodak = load_json("kodak24_appendix.json")
    table_detection(detection)
    table_restoration(restoration)
    table_kodak(kodak)
    table_baselines()
    detection_recovery(detection)
    tradeoff(detection, restoration)
    task_teaser(detection)
    its_pipeline()
    qualitative_zoom("defocus_medium", "defocus", "Defocus restoration for road-defect perception")
    qualitative_zoom("lowlight_medium", "lowlight", "Low-light restoration for road-defect perception")
    qualitative_zoom("motion_horizontal_medium", "motion", "Motion-blur restoration for road-defect perception")
    print(OUT)


if __name__ == "__main__":
    main()
