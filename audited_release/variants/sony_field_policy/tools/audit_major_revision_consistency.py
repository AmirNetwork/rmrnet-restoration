# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Fail the release if paper headlines and audited result artifacts diverge."""

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_automation_in_construction_rmrnet"
MASTER = ROOT / "experiments" / "major_revision_evidence_20260715" / "master_controlled_results.csv"
OUT = ROOT / "experiments" / "major_revision_evidence_20260715" / "release_consistency_audit.json"

CHECKPOINTS = {
    "IVCNZ": (
        ROOT / "runs" / "trc_final_rmrnet_pothole_30ep" / "rcadnet_epoch_028.pth",
        "3c6a1a8e582639fade7c5cae9cbb301e3f2987d600c06584adacb14c1ab538dd",
    ),
    "PCM": (
        ROOT / "runs" / "trc_final_rmrnet_pcm_30ep" / "rcadnet_epoch_028.pth",
        "e580d2bf0bb8cc3319afbcca3b3d1cb96ef340667497f445b6685c2ebbe7eec7",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(rows: list[dict[str, str]], dataset: str, scenario: str, method: str, key: str = "map50") -> float:
    row = next(r for r in rows if r["dataset"] == dataset and r["scenario"] == scenario and r["method"] == method)
    return float(row[key])


def main() -> None:
    rows = read_csv(MASTER)
    manuscript = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    provenance = (PAPER / "RESULT_PROVENANCE_TABLE.csv").read_text(encoding="utf-8")
    checks: list[dict[str, object]] = []

    for dataset, (path, expected) in CHECKPOINTS.items():
        actual = sha256(path)
        checks.append({"check": f"{dataset} checkpoint SHA256", "ok": actual == expected, "actual": actual, "expected": expected})

    expected_headlines = {
        "IVCNZ motion degraded": metric(rows, "IVCNZ", "motion", "Degraded"),
        "IVCNZ motion RMR": metric(rows, "IVCNZ", "motion", "RMR-Net"),
        "PCM defocus degraded": metric(rows, "PCM", "defocus", "Degraded"),
        "PCM defocus RMR": metric(rows, "PCM", "defocus", "RMR-Net"),
    }
    for name, value in expected_headlines.items():
        token = f"{value:.3f}"
        checks.append({"check": name, "ok": token in manuscript, "token": token})

    # The main controlled tables are generated from MASTER. Verify every
    # dataset/scenario/method row, not only the four values quoted in prose.
    scenario_names = {
        "motion": "motion blur",
        "defocus": "defocus",
        "lowlight": "low light",
        "mixed": "mixed motion+low light",
    }
    table_paths = {
        "IVCNZ": PAPER / "tables" / "table_pothole_detection.tex",
        "PCM": PAPER / "tables" / "table_pcm_detection.tex",
    }
    for dataset, path in table_paths.items():
        normalized = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", path.read_text(encoding="utf-8"))
        for row in rows:
            if row["dataset"] != dataset or row["scenario"] == "clean":
                continue
            expected = (
                f"{scenario_names[row['scenario']]} & {row['method']} & "
                f"{float(row['map50']):.3f} & {float(row['map50_95']):.3f} & "
                f"{float(row['precision']):.3f} & {float(row['recall']):.3f}"
            )
            checks.append(
                {
                    "check": f"{dataset}/{row['scenario']}/{row['method']} table row",
                    "ok": expected in normalized,
                    "expected": expected,
                }
            )

    # Paper configuration must reflect the fixed 30-epoch checkpoints.
    task_table = (PAPER / "tables" / "table_task_objective_audit.tex").read_text(encoding="utf-8")
    checks.extend(
        [
            {"check": "effective sparse basis coefficient", "ok": "0.00010 & 0.00010" in task_table},
            {"check": "active contour excluded from reported objective", "ok": "active-contour loss weight is zero" in manuscript},
            {"check": "headline detail cap", "ok": "\\eta_d=0.12" in manuscript},
            {"check": "no obsolete headline detail cap", "ok": "\\eta_d=0.20" not in manuscript},
        ]
    )

    # Once field evidence exists, bind its generated rows to the table sources.
    field_csv = OUT.parent / "ilx_direct_single_view_metrics.csv"
    field_table = PAPER / "tables" / "table_ilx_direct_audited.tex"
    if field_csv.exists() and field_table.exists():
        field_text = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", field_table.read_text(encoding="utf-8"))
        for row in read_csv(field_csv):
            expected = (
                f"{row['method']} & {int(float(row['pred']))} & {float(row['precision_iou10']):.3f} & "
                f"{float(row['recall_iou10']):.3f} & {float(row['f1_iou10']):.3f} & "
                f"{float(row['f1_iou50']):.3f} & {float(row['coverage']):.3f}"
            )
            checks.append({"check": f"ILX direct/{row['method']}", "ok": expected in field_text, "expected": expected})

    stale = [
        "runs/rmrnet_noac_pothole_yolo11s/rcadnet_epoch_004.pth",
        "runs/rmrnet_noac_pcm_yolo11s/rcadnet_epoch_001.pth",
        "experiments/no_active_contour_fullrun/pothole_test_noac_selected.csv",
        "experiments/no_active_contour_fullrun/pcm_test_noac_selected.csv",
    ]
    for token in stale:
        checks.append({"check": f"stale provenance absent: {token}", "ok": token not in provenance})

    required = [
        "experiments/trc_final_30ep/pothole_test_rmrnet_selected.csv",
        "experiments/trc_final_30ep/pcm_test_rmrnet_selected.csv",
        "global epoch 28",
        "rcadnet_epoch_028.pth",
    ]
    for token in required:
        checks.append({"check": f"required provenance present: {token}", "ok": token in provenance})

    report = {"ok": all(bool(c["ok"]) for c in checks), "checks": checks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
