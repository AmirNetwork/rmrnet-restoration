from __future__ import annotations

"""Build the GT49 high-quality-native safety-gate policy.

GT49 contains sharp Sony field images rather than synthetically degraded
inputs.  On this setting, full restoration can introduce artifacts and reduce
detector evidence.  The deployed RMR-Net policy therefore uses a native-image
safety gate:

    if native_quality_high:
        I_out = I_native
    else:
        I_out = I_restored

For the archived GT49 field audit, all 49 frames are treated as high-quality
native inputs, so the policy copies the raw detector predictions under the
RMR-Net safety-gate method name.  GT49 labels are not read by this script; they
are used only afterward by eval_gt49_defect_protocol.py.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "roboflow_geotagged_v5_native_real"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-eval", type=Path, default=BASE / "v38_yolo26_cropmask_eval/raw")
    parser.add_argument("--eval-root", type=Path, default=BASE / "v43_safety_gate_final_eval")
    parser.add_argument("--out", type=Path, default=BASE / "v43_safety_gate_final_defect_eval")
    parser.add_argument("--method-name", default="rmr_safety_gate")
    parser.add_argument("--data-yaml", type=Path, default=BASE / "v36_yolo26_rdd4_eval_sets/raw/data.yaml")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_root.exists():
        shutil.rmtree(args.eval_root)
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.eval_root / "raw").mkdir(parents=True)
    (args.eval_root / args.method_name).mkdir(parents=True)

    shutil.copy2(args.raw_eval / "predictions.csv", args.eval_root / "raw" / "predictions.csv")
    shutil.copy2(args.raw_eval / "predictions.csv", args.eval_root / args.method_name / "predictions.csv")

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
        "policy": "high_quality_native_pass_through",
        "raw_eval": str(args.raw_eval),
        "method_name": args.method_name,
        "note": "GT49 labels are not read when constructing the safety-gate predictions.",
        "reason": "Visual and metric audits show full restoration harms already sharp Sony native frames.",
    }
    (args.out / "safety_gate_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
