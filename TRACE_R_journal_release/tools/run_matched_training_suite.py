#!/usr/bin/env python3
# TRACE-R release integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Run the restartable, matched-budget controlled-restoration training suite.

The suite uses one shared training stream and common optimization objective for
RMR-P, NAFNet, DeMoE, DFPIR, and InstructIR. Each model receives the same
effective batch size, crop distribution, detector features, optimizer-step
count, learning-rate schedule, and checkpoint cadence. Public pretraining is
retained and disclosed; this script equalizes *target-domain adaptation*, not
the models' historically different pretraining corpora.

No validation or test data are opened here. A separate validation-only stage
selects checkpoints after all training jobs complete.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("rmrp", "nafnet", "nafnet_meta", "demoe", "dfpir", "instructir")


def parse_tagged_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected MODEL=PATH")
    model, raw = value.split("=", 1)
    model = model.strip().lower()
    if model not in DEFAULT_MODELS:
        raise argparse.ArgumentTypeError(f"Unknown model: {model}")
    return model, Path(raw).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--effective-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--detector-supervised-weight", type=float, default=0.0)
    parser.add_argument("--tdp-weight", type=float, default=0.08)
    parser.add_argument("--tdp-input-size", type=int, default=320)
    parser.add_argument("--tdp-warmup-epochs", type=int, default=3)
    parser.add_argument("--defect-crop-probability", type=float, default=0.6)
    parser.add_argument("--rmrp-state-weight", type=float, default=0.50)
    parser.add_argument("--rmrp-physical-weight", type=float, default=0.20)
    parser.add_argument("--rmrp-new-module-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--rmrp-metadata-dropout", type=float, default=0.35)
    parser.add_argument("--rmrp-metadata-noise", type=float, default=0.015)
    parser.add_argument(
        "--rmrp-metadata-mismatch-probability", type=float, default=0.10
    )
    parser.add_argument("--rmrp-metadata-curriculum-epochs", type=int, default=0)
    parser.add_argument(
        "--rmrp-metadata-curriculum-ramp-epochs", type=int, default=0
    )
    parser.add_argument("--rmrp-route-teacher-epochs", type=int, default=0)
    parser.add_argument("--rmrp-route-teacher-ramp-epochs", type=int, default=0)
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="Resume an audited partial model directory from its latest saved epoch.",
    )
    parser.add_argument(
        "--init-override",
        type=parse_tagged_path,
        action="append",
        help="Use one explicit initialization for MODEL in this restartable stage.",
    )
    return parser.parse_args()


def model_args(model: str, init_overrides: dict[str, Path]) -> list[str]:
    if model in init_overrides:
        args = ["--init-weights", str(init_overrides[model])]
        if model == "instructir":
            args += ["--lm-head-weights", "weights/instructir/lm_instructir-7d.pt"]
        full_precision_backbone = model == "dfpir" or (
            model == "rmrp" and "DFPIR" in init_overrides[model].name.upper()
        )
        args += ["--micro-batch-size", "1" if full_precision_backbone else "2"]
        return args
    if model == "rmrp":
        return [
            "--micro-batch-size", "2",
            "--init-weights", "runs/rmrnet_ivcnz_calibrated_v2_motion_mixed_full_v1_20260729/rcadnet_epoch_001.pth",
            "--init-weights", "runs/rmrnet_pcm_calibrated_v2_lowlight_mixed_full_v2_20260728/rcadnet_epoch_001.pth",
        ]
    if model == "nafnet":
        return [
            "--micro-batch-size", "2",
            "--init-weights", "runs/major_revision_sequence_disjoint_v1_nafnet30_20260716_pothole_30ep/nafnet_epoch_030.pth",
            "--init-weights", "runs/major_revision_sequence_disjoint_v1_nafnet30_20260716_pcm_30ep/nafnet_epoch_030.pth",
        ]
    if model == "nafnet_meta":
        return [
            "--micro-batch-size", "2",
            "--init-weights", "runs/major_revision_sequence_disjoint_v1_nafnet30_20260716_pothole_30ep/nafnet_epoch_030.pth",
            "--init-weights", "runs/major_revision_sequence_disjoint_v1_nafnet30_20260716_pcm_30ep/nafnet_epoch_030.pth",
        ]
    if model == "demoe":
        return ["--micro-batch-size", "2", "--init-weights", "weights/demoe/DeMoE.pt"]
    if model == "dfpir":
        # Full-precision micro-batches prevent the NaNs observed for DFPIR in
        # float16; accumulation preserves the shared effective batch of four.
        return [
            "--micro-batch-size", "1",
            "--init-weights",
            "weights/dfpir/DFPIR-5D-pn31.29-0.8889_pr37.62-0.9779_ph31.64-0.9794_pb28.82-0.8734_pl23.82-0.8428_avr30.64-0.9125.pth.tar",
        ]
    if model == "instructir":
        return [
            "--micro-batch-size", "2",
            "--init-weights", "weights/instructir/im_instructir-7d.pt",
            "--lm-head-weights", "weights/instructir/lm_instructir-7d.pt",
        ]
    raise ValueError(model)


def shared_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-root", "ivcnz=data/pothole_restoration_practical_sensor_calibrated_v2_train",
        "--data-root", "pcm=data/pcm_restoration_practical_sensor_calibrated_v2_train",
        "--detector", "ivcnz=runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pothole_clean_80ep/weights/best.pt",
        "--detector", "pcm=runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pcm_clean_80ep/weights/best.pt",
        "--defect-label-root", "ivcnz=datasets/pothole_yolo_sequence_disjoint_v1/labels/train",
        "--defect-label-root", "pcm=datasets/road_damage_pcm_yolo_sequence_disjoint_v1/labels/train",
        "--epochs", str(args.epochs),
        "--samples-per-epoch", str(args.samples_per_epoch),
        "--effective-batch-size", str(args.effective_batch_size),
        "--save-every", str(args.save_every),
        "--seed", str(args.seed),
        "--lr", str(args.lr),
        "--min-lr-ratio", str(args.min_lr_ratio),
        "--weight-decay", str(args.weight_decay),
        "--edge-weight", str(args.edge_weight),
        "--tdp-weight", str(args.tdp_weight),
        "--detector-supervised-weight", str(args.detector_supervised_weight),
        "--tdp-input-size", str(args.tdp_input_size),
        "--tdp-warmup-epochs", str(args.tdp_warmup_epochs),
        "--defect-crop-probability", str(args.defect_crop_probability),
    ]


def write_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def latest_checkpoint(run_dir: Path, model: str) -> Path:
    candidates = sorted(run_dir.glob(f"{model}_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No resumable checkpoint in {run_dir}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    init_overrides = dict(args.init_override or [])
    out_root = (ROOT / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "suite_state.json"
    state = {
        "protocol": "matched target-domain adaptation v1",
        "test_split_used": False,
        "models": {},
        "started_unix": time.time(),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    for model in args.models:
        run_dir = out_root / model
        complete = run_dir / "TRAINING_COMPLETE.json"
        if complete.exists():
            state["models"][model] = {"status": "complete", "run_dir": str(run_dir)}
            write_state(state_path, state)
            print(f"[suite] skip complete model={model}", flush=True)
            continue
        resume_checkpoint: Path | None = None
        if run_dir.exists() and args.resume_partial:
            resume_checkpoint = latest_checkpoint(run_dir, model)
        elif run_dir.exists():
            raise RuntimeError(
                f"Partial directory requires audit before restart: {run_dir}. "
                "Move it aside or resume from its last verified checkpoint."
            )

        command = [
            sys.executable,
            "-u",
            "train_matched_restorer.py",
            "--model", model,
            *shared_args(args),
            *model_args(model, init_overrides),
            "--out", str(run_dir),
        ]
        if resume_checkpoint is not None:
            command += ["--resume-from", str(resume_checkpoint)]
        if model == "rmrp":
            command += [
                "--state-weight", str(args.rmrp_state_weight),
                "--physical-weight", str(args.rmrp_physical_weight),
                "--new-module-lr-multiplier",
                str(args.rmrp_new_module_lr_multiplier),
                "--metadata-dropout", str(args.rmrp_metadata_dropout),
                "--metadata-noise", str(args.rmrp_metadata_noise),
                "--metadata-mismatch-probability",
                str(args.rmrp_metadata_mismatch_probability),
                "--metadata-curriculum-epochs",
                str(args.rmrp_metadata_curriculum_epochs),
                "--metadata-curriculum-ramp-epochs",
                str(args.rmrp_metadata_curriculum_ramp_epochs),
                "--route-teacher-epochs", str(args.rmrp_route_teacher_epochs),
                "--route-teacher-ramp-epochs",
                str(args.rmrp_route_teacher_ramp_epochs),
            ]
        state["models"][model] = {
            "status": "running",
            "run_dir": str(run_dir),
            "command": command,
            "started_unix": time.time(),
            "resume_checkpoint": (
                str(resume_checkpoint) if resume_checkpoint is not None else None
            ),
        }
        write_state(state_path, state)
        print(f"[suite] start model={model}", flush=True)
        log_path = out_root / f"{model}.log"
        with log_path.open("a" if resume_checkpoint is not None else "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code != 0 or not complete.exists():
            state["models"][model].update(
                {"status": "failed", "return_code": return_code, "ended_unix": time.time()}
            )
            write_state(state_path, state)
            raise RuntimeError(f"Training failed for {model}; inspect {log_path}")
        state["models"][model].update(
            {"status": "complete", "return_code": return_code, "ended_unix": time.time()}
        )
        write_state(state_path, state)

    state["status"] = "complete"
    state["ended_unix"] = time.time()
    write_state(state_path, state)
    print(f"[suite] complete: {out_root}", flush=True)


if __name__ == "__main__":
    main()
