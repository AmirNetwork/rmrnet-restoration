from __future__ import annotations

"""Build and evaluate the GT49 dual-evidence native-field policy.

The native GT49 images are already high quality, so full restoration can erase
weak road evidence.  The deployed policy used here keeps the raw detector as
the anchor and admits only high-confidence detections from an RMR-restored
branch:

    P_dual = NMS(P_raw union {p in P_RMR : conf(p) >= tau_r})

The confidence threshold is detector-internal and label-free at inference.  The
script does not read GT49 labels while creating predictions; labels are used
only by eval_gt49_defect_protocol.py after the method output is fixed.
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "roboflow_geotagged_v5_native_real"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-predictions", type=Path, default=BASE / "v38_yolo26_cropmask_eval/raw/predictions.csv")
    parser.add_argument("--rmr-predictions", type=Path, default=BASE / "v38_yolo26_cropmask_eval/rmr_blind/predictions.csv")
    parser.add_argument("--rmr-conf", type=float, default=0.55)
    parser.add_argument("--eval-root", type=Path, default=BASE / "v41_dual_evidence_final_eval")
    parser.add_argument("--out", type=Path, default=BASE / "v41_dual_evidence_final_defect_eval")
    parser.add_argument("--method-name", default="rmr_dual_evidence")
    parser.add_argument("--data-yaml", type=Path, default=BASE / "v36_yolo26_rdd4_eval_sets/raw/data.yaml")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_predictions(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.eval_root.exists():
        shutil.rmtree(args.eval_root)
    if args.out.exists():
        shutil.rmtree(args.out)
    args.eval_root.mkdir(parents=True)

    raw_header, raw_rows = read_csv(args.raw_predictions)
    rmr_header, rmr_rows = read_csv(args.rmr_predictions)
    if raw_header != rmr_header:
        raise ValueError("Raw and RMR prediction CSV headers differ.")

    raw_dir = args.eval_root / "raw"
    dual_dir = args.eval_root / args.method_name
    write_predictions(raw_dir / "predictions.csv", raw_header, raw_rows)

    dual_rows = [dict(row) for row in raw_rows]
    added = 0
    for row in rmr_rows:
        if float(row["conf"]) >= args.rmr_conf:
            out = dict(row)
            out["source"] = f"{out.get('source', '')}+rmr_dual_c{args.rmr_conf:.2f}"
            dual_rows.append(out)
            added += 1
    write_predictions(dual_dir / "predictions.csv", raw_header, dual_rows)

    cmd = [
        args.python,
        str(ROOT / "tools" / "eval_gt49_defect_protocol.py"),
        "--data",
        str(args.data_yaml),
        "--eval-root",
        str(args.eval_root),
        "--out",
        str(args.out),
        "--methods",
        f"raw,{args.method_name}",
        "--defect-classes",
        "0,1,2,3",
        "--crack-classes",
        "0,1,2",
        "--pothole-class",
        "3",
        "--longitudinal-class",
        "0",
        "--transverse-class",
        "1",
        "--alligator-class",
        "2",
        "--crop-bottom-half",
    ]
    subprocess.check_call(cmd, cwd=ROOT)
    manifest = {
        "policy": "raw_anchor_plus_high_confidence_rmr_branch",
        "raw_predictions": str(args.raw_predictions),
        "rmr_predictions": str(args.rmr_predictions),
        "rmr_conf": args.rmr_conf,
        "added_rmr_predictions_before_protocol_dedup": added,
        "eval_root": str(args.eval_root),
        "out": str(args.out),
        "method_name": args.method_name,
        "note": "GT49 labels are not read when constructing the dual-evidence predictions.",
    }
    (args.out / "dual_evidence_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(manifest)


if __name__ == "__main__":
    main()
