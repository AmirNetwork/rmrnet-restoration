# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Write the experiment-specific metadata mapping and information-budget audit.

This file is deliberately descriptive: every row corresponds to code executed
by ``rcadnet/scenario_codes.py`` or the KITTI preparation script. It prevents
synthetic degradation parameters, raw telemetry, derived physical priors, and
route context from being presented as interchangeable metadata.
"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "major_revision_metadata_mapping"
TABLE = ROOT / "paper_automation_in_construction_rmrnet" / "tables" / "table_metadata_mapping.tex"


ROWS = [
    {
        "experiment": "IVCNZ and PCM",
        "field": "scenario family and severity; generated blur/noise/exposure fields when stored",
        "unit": "category; px; deg; ms; unitless",
        "normalization": "scenario levels 0.30/0.60/0.90; blur length / 25; angle projected by |cos|, |sin|; scores clipped to [0,1]",
        "target_code": "motion_x, motion_y, random motion, defocus, noise, low light, JPEG, strength",
        "source": "known synthetic corruption parameters; never called real sensor metadata",
        "inference": "available only because the controlled benchmark declares its corruption",
    },
    {
        "experiment": "KITTI raw-OXTS",
        "field": "vehicle speed, local angular rate, acceleration, exposure",
        "unit": "m/s; rad/s; m/s^2; ms",
        "normalization": "s = clip(v / 18), y = clip(|yaw| / 0.09), a = clip(||a|| / 4.5), e = clip(t / 24); motion = clip(0.12 + 0.46 s sqrt(e) + 0.25 y + 0.17 a)",
        "target_code": "coarse motion magnitude and camera-plane direction proxy",
        "source": "synchronized OXTS packets and declared exposure; derived blur length/angle removed",
        "inference": "yes, from the capture platform",
    },
    {
        "experiment": "KITTI full prior",
        "field": "OXTS motion, exposure, camera-02 intrinsics, representative road depth/ROI",
        "unit": "SI units and pixels",
        "normalization": "pinhole optical-flow displacement; length/25 and direction projected by |cos|, |sin|",
        "target_code": "motion_x, motion_y, strength",
        "source": "deterministic physical proxy; 4.5 m depth, y=0.92H, blur scale 1.6; no target/detector access",
        "inference": "yes if calibration and the declared depth proxy are available",
    },
    {
        "experiment": "ILX-RD46 pilot",
        "field": "EXIF exposure, ISO and brightness; geotag, pose and uncertainty",
        "unit": "ms; ISO; EV; deg; m; uncertainty units",
        "normalization": "0.25-ms exposure and brightness-derived illumination score; route/pose fields are retained as context and are not converted to a blur kernel",
        "target_code": "illumination and context audit only; the reliability gate suppresses unsupported motion conditioning",
        "source": "project-collected Sony EXIF and filename-matched coordinate records",
        "inference": "yes; interpreted only as a sharp-image safety test",
    },
]


def tex_escape(value: object) -> str:
    """Escape audit text before inserting it into a LaTeX tabular cell."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "metadata_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)
    (OUT / "metadata_mapping.json").write_text(json.dumps(ROWS, indent=2), encoding="utf-8")

    body = "\n".join(
        f"{tex_escape(row['experiment'])} & {tex_escape(row['field'])} & {tex_escape(row['unit'])} & "
        f"{tex_escape(row['normalization'])} & {tex_escape(row['target_code'])} & "
        f"{tex_escape(row['source'])} & {tex_escape(row['inference'])} \\\\"  # noqa: E501
        for row in ROWS
    )
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(
        """\\begin{table*}[!t]
\\centering
\\caption{Experiment-specific conditioning information. Synthetic parameters, physical telemetry, derived priors, and route context are not treated as equivalent.}
\\label{tab:metadata_mapping}
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\begin{tabular}{p{0.09\\textwidth}p{0.15\\textwidth}p{0.07\\textwidth}p{0.18\\textwidth}p{0.13\\textwidth}p{0.17\\textwidth}p{0.10\\textwidth}}
\\toprule
Experiment & Fields & Units & Normalization/mapping & Code coordinates & Calibration source & At inference? \\\\
\\midrule
"""
        + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(OUT / "metadata_mapping.csv"), "table": str(TABLE)}, indent=2))


if __name__ == "__main__":
    main()
