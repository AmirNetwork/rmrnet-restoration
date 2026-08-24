from __future__ import annotations

"""Build CRID complete-INS overlays and a temporally sampled paper atlas.

The TRACE-R column follows the frozen guarded dual-view policy. Frames that
fail the label-free evidence gate display the unmodified native image and its
native detections.  Frames that pass display the restored image and the fused
native/restored detections.  This keeps the visual artifact synchronized with
the detector input policy evaluated in the paper.
"""

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_crid46_sequence_disjoint_comparison import (  # noqa: E402
    load_compact_predictions,
    split_names,
    subset_gt,
)


EXPERIMENT = ROOT / "experiments/crid46_complete_ins_semantic_tdp_public_v8_tilefix_val_20260812"
OPERATING = ROOT / "experiments/crid46_complete_ins_semantic_tdp_public_v8_operating_points_20260812"
GUARDED = ROOT / "experiments/crid46_tilefix_guarded_policy_20260812"
SPLIT = ROOT / "datasets/crid46_direct_sbg_external_detector_20260811"
LABEL_ROOT = ROOT / "datasets/gt46_sony_classbalanced_20260801"
NATIVE = LABEL_ROOT / "images"
BASELINES = ROOT / "experiments/final20260810_crid46_native_restoration"
OUT = ROOT / "experiments/final_release_20260812_tilefix/crid_complete_ins_overlays"
PAPER_FIGURE = ROOT / "paper_automation_in_construction_rmrnet/figures/fig_crid46_direct_sbg_atlas.png"
SUPPLEMENT_FIGURE = ROOT / "paper_automation_in_construction_rmrnet/figures/fig_crid46_direct_sbg_all13.png"

METHODS = (
    ("raw", "Raw", NATIVE),
    ("nafnet", "NAFNet", BASELINES / "nafnet/images/test"),
    ("dfpir", "DFPIR", BASELINES / "dfpir/images/test"),
    ("demoe_auto", "DeMoE", BASELINES / "demoe_auto/images/test"),
    ("instructir", "InstructIR", BASELINES / "instructir_generic/images/test"),
    ("rmr_guarded_dual_view", "TRACE-R", None),
)

GT_COLOR = (245, 190, 45)
PRED_COLOR = (20, 176, 125)
INK = (25, 35, 43)
MUTED = (75, 88, 98)
WHITE = (255, 255, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf") if bold else Path("C:/Windows/Fonts/calibri.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def by_stem(folder: Path) -> dict[str, Path]:
    result = {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if not result:
        raise FileNotFoundError(f"No images in {folder}")
    return result


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    width: int,
    text: str | None = None,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    if text:
        text_font = font(max(18, width * 4), True)
        bounds = draw.textbbox((0, 0), text, font=text_font)
        text_w = bounds[2] - bounds[0]
        text_h = bounds[3] - bounds[1]
        top = max(0, y1 - text_h - width * 2)
        draw.rectangle((x1, top, x1 + text_w + width * 2, y1), fill=color)
        draw.text((x1 + width, top), text, fill=WHITE, font=text_font)


def annotate(
    image: Image.Image,
    gt: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    box_width = max(4, round(min(result.size) / 500))
    for item in gt:
        draw_box(draw, tuple(item["box"]), GT_COLOR, box_width, item["primary"])
    for item in predictions:
        if float(item["conf"]) < threshold:
            continue
        draw_box(
            draw,
            tuple(item["box"]),
            PRED_COLOR,
            box_width,
            f"{item['primary']} {float(item['conf']):.2f}",
        )
    return result


def decision_note(method: str, name: str, decisions: dict[str, dict[str, Any]]) -> str | None:
    if method != "rmr_guarded_dual_view":
        return None
    policy = decisions[name]["policy_output"]
    return "native pass-through" if policy == "native" else "restored + native evidence"


def crop_for(items: list[dict[str, Any]], size: tuple[int, int], aspect: float) -> tuple[int, int, int, int]:
    width, height = size
    if not items:
        return (0, 0, width, height)
    x1 = min(float(item["box"][0]) for item in items)
    y1 = min(float(item["box"][1]) for item in items)
    x2 = max(float(item["box"][2]) for item in items)
    y2 = max(float(item["box"][3]) for item in items)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    crop_w = max((x2 - x1) * 2.0, width * 0.32)
    crop_h = max((y2 - y1) * 2.0, height * 0.32)
    if crop_w / crop_h < aspect:
        crop_w = crop_h * aspect
    else:
        crop_h = crop_w / aspect
    crop_w, crop_h = min(crop_w, width), min(crop_h, height)
    left = min(max(center_x - crop_w / 2, 0), width - crop_w)
    top = min(max(center_y - crop_h / 2, 0), height - crop_h)
    return round(left), round(top), round(left + crop_w), round(top + crop_h)


def make_atlas(
    names: list[str],
    *,
    image_roots: dict[str, dict[str, Path]],
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    thresholds: dict[str, float],
    decisions: dict[str, dict[str, Any]],
    output: Path,
    cell: tuple[int, int],
    legend: str,
) -> None:
    labels = ["Ground truth"] + [display for _, display, _ in METHODS]
    gap_x, gap_y = 12, 48
    left, top = 145, 68
    page_w = left + len(labels) * (cell[0] + gap_x)
    page_h = top + len(names) * (cell[1] + gap_y) + 32
    page = Image.new("RGB", (page_w, page_h), WHITE)
    draw = ImageDraw.Draw(page)
    for column, label in enumerate(labels):
        x = left + column * (cell[0] + gap_x)
        draw.text((x + 4, 18), label, fill=INK, font=font(22, True))
    for row, name in enumerate(names):
        y = top + row * (cell[1] + gap_y)
        draw.text((10, y + 8), f"Frame {row + 1}", fill=INK, font=font(21, True))
        draw.text((10, y + 36), f"{len(ground_truth[name])} labels", fill=MUTED, font=font(17))
        native = Image.open(image_roots["raw"][Path(name).stem]).convert("RGB")
        crop = crop_for(ground_truth[name], native.size, cell[0] / cell[1])
        gt_panel = annotate(native, ground_truth[name], [], 1.0).crop(crop).resize(cell, Image.Resampling.LANCZOS)
        page.paste(gt_panel, (left, y))
        draw.rectangle((left, y, left + cell[0] - 1, y + cell[1] - 1), outline=(180, 190, 198), width=2)
        for method_index, (method, _, _) in enumerate(METHODS, start=1):
            source = Image.open(image_roots[method][Path(name).stem]).convert("RGB")
            panel = annotate(
                source,
                ground_truth[name],
                predictions[method].get(name, []),
                thresholds[method],
            )
            panel = panel.crop(crop).resize(cell, Image.Resampling.LANCZOS)
            x = left + method_index * (cell[0] + gap_x)
            page.paste(panel, (x, y))
            draw.rectangle((x, y, x + cell[0] - 1, y + cell[1] - 1), outline=(180, 190, 198), width=2)
            note = decision_note(method, name, decisions)
            if note:
                note_font = font(15, True)
                bounds = draw.textbbox((0, 0), note, font=note_font)
                note_w = bounds[2] - bounds[0]
                draw.rectangle((x + 5, y + 5, x + note_w + 15, y + 28), fill=(255, 255, 255))
                draw.text((x + 10, y + 7), note, fill=INK, font=note_font)
        draw.text(
            (left, y + cell[1] + 8),
            legend,
            fill=MUTED,
            font=font(16),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, dpi=(300, 300), quality=95)


def main() -> None:
    names = split_names("test", SPLIT)
    ground_truth = subset_gt(names, LABEL_ROOT)
    operating_rows = json.loads(
        (OPERATING / "frozen_operating_points_before_test.json").read_text(encoding="utf-8")
    )["operating_points"]
    thresholds = {row["method"]: float(row["threshold"]) for row in operating_rows}
    guarded_freeze = json.loads(
        (GUARDED / "frozen_policy_before_supportive_test.json").read_text(encoding="utf-8")
    )
    thresholds["rmr_guarded_dual_view"] = float(
        guarded_freeze["detector_operating_point"]["threshold"]
    )
    prediction_maps = {}
    for method, _, _ in METHODS:
        path = (
            GUARDED / "supportive_predictions/rmr_guarded_dual_view.csv"
            if method == "rmr_guarded_dual_view"
            else EXPERIMENT / "test_predictions" / f"{method}.csv"
        )
        prediction_maps[method] = load_compact_predictions(path)

    decision_rows = json.loads(
        (GUARDED / "supportive_predictions/rmr_guarded_dual_view.json").read_text(encoding="utf-8")
    )["decisions"]
    decisions = {row["image"]: row for row in decision_rows}
    native_images = by_stem(NATIVE)
    restored_images = by_stem(EXPERIMENT / "current_rmr_restored/test")
    image_roots = {}
    for method, _, folder in METHODS:
        if method != "rmr_guarded_dual_view":
            image_roots[method] = by_stem(folder)
            continue
        image_roots[method] = {
            Path(name).stem: (
                native_images[Path(name).stem]
                if decisions[name]["policy_output"] == "native"
                else restored_images[Path(name).stem]
            )
            for name in names
        }

    OUT.mkdir(parents=True, exist_ok=True)
    for method, display, _ in METHODS:
        method_out = OUT / method
        method_out.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = Image.open(image_roots[method][Path(name).stem]).convert("RGB")
            overlay = annotate(
                source,
                ground_truth[name],
                prediction_maps[method].get(name, []),
                thresholds[method],
            )
            note = decision_note(method, name, decisions)
            if note:
                draw = ImageDraw.Draw(overlay)
                note_font = font(30, True)
                bounds = draw.textbbox((0, 0), note, font=note_font)
                note_w = bounds[2] - bounds[0]
                draw.rectangle((20, 20, 50 + note_w, 70), fill=WHITE)
                draw.text((35, 28), note, fill=INK, font=note_font)
            overlay.save(method_out / name, quality=92, subsampling=0)
        print(f"{display}: {len(names)} native overlays")

    gt_out = OUT / "ground_truth"
    gt_out.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = Image.open(image_roots["raw"][Path(name).stem]).convert("RGB")
        annotate(source, ground_truth[name], [], 1.0).save(
            gt_out / name, quality=92, subsampling=0
        )

    # First, middle, and last frames are fixed by temporal position and do not
    # depend on annotations, predictions, or restoration quality.
    selected = [names[0], names[len(names) // 2], names[-1]]
    common_rank_floor = float(guarded_freeze["constants"]["restored_confidence_floor"])
    common_thresholds = {method: common_rank_floor for method, _, _ in METHODS}
    make_atlas(
        selected,
        image_roots=image_roots,
        ground_truth=ground_truth,
        predictions=prediction_maps,
        thresholds=common_thresholds,
        decisions=decisions,
        output=PAPER_FIGURE,
        cell=(430, 286),
        legend=(
            "Yellow: annotation   Green: ranked detector evidence at common "
            f"confidence >= {common_rank_floor:.3f}"
        ),
    )
    make_atlas(
        names,
        image_roots=image_roots,
        ground_truth=ground_truth,
        predictions=prediction_maps,
        thresholds=thresholds,
        decisions=decisions,
        output=SUPPLEMENT_FIGURE,
        cell=(285, 190),
        legend="Yellow: annotation   Green: prediction at validation-selected operating threshold",
    )
    (OUT / "figure_manifest.json").write_text(
        json.dumps(
            {
                "selection": "first, middle, and last temporal frames; independent of labels and predictions",
                "main_figure_common_confidence_floor": common_rank_floor,
                "metadata": "complete synchronized Sony EXIF and 200 Hz SBG INS packet",
                "trace_r_field_expert_checkpoint": "runs/rmrnet_crid_complete_ins_semantic_tdp_public_stage_v8_20260812/rcadnet_epoch_001.pth",
                "trace_r_policy": "validation-frozen guarded dual-view policy",
                "trace_r_policy_freeze": str((GUARDED / "frozen_policy_before_supportive_test.json").relative_to(ROOT)),
                "native_tile_normalization": "actual accumulated weight clamped only by numerical epsilon",
                "trace_r_frame_decisions": decisions,
                "paper_frames": selected,
                "all_frames": names,
                "thresholds_selected_on_validation": thresholds,
                "ground_truth_color": "yellow",
                "prediction_color": "green",
                "native_overlays_preserve_4752x3168_resolution": True,
                "paper_figure": str(PAPER_FIGURE.relative_to(ROOT)),
                "supplement_figure": str(SUPPLEMENT_FIGURE.relative_to(ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
