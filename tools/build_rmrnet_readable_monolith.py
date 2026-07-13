"""Build a readable one-file RMR-Net paper script.

Unlike the base64 giant file, this generator writes a normal Python runner and
then appends embedded source files as readable commented code sections:

    # === BEGIN EMBEDDED FILE: rcadnet/model.py ===
    #|import torch
    #|...
    # === END EMBEDDED FILE: rcadnet/model.py ===

The generated script can extract those sections back into a working project and
delegate training/evaluation/paper commands to the extracted files.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rmrnet_paper_monolith_readable.py"

INCLUDE_DIRS = ["models", "rcadnet", "losses", "tools", "baselines", "configs"]
TOP_LEVEL = [
    "benchmark_adapter_rcadnet.py",
    "benchmark_unified_restoration.py",
    "infer_rcadnet.py",
    "train_rcadnet.py",
    "train_rmrnet.py",
    "train_road_baseline.py",
    "rmrnet_paper_onefile.py",
    "README.md",
    "EXPERIMENT_PROTOCOL.md",
    "EXPERIMENT_RESULTS.md",
    "BASELINE_REVIEW_MATRIX.md",
    "REPRODUCIBILITY_ARTIFACTS_README.md",
    "requirements-windows-gpu.txt",
    "requirements-detection-extra.txt",
    "requirements-dfpir-extra.txt",
]
PAPER_DIR = ROOT / "paper_ieee_tits_rmrnet"
RESULT_DIRS = [ROOT / "experiments" / "v29_review_completion", ROOT / "experiments" / "v30_submission_readiness"]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".tex",
    ".bib",
    ".bst",
    ".cls",
}
EXCLUDE_NAMES = {"__pycache__", ".git", ".venv"}
EXCLUDE_FILES = {OUT.name, "rmrnet_paper_allinone_giant.py"}


RUNNER = r'''#!/usr/bin/env python
r"""Readable monolithic RMR-Net paper runner.

This file is intentionally written in normal Python style. The runnable code is
at the top. The embedded project files are plain-text commented sections at the
bottom, so you can search/read/debug them without decoding base64.

Important: datasets and large pretrained weights are not embedded. Run this file
inside the project folder, or extract it into a folder where data/, datasets/,
weights/, yolo11s.pt, and related checkpoints are available.

Common CMD use:

    python rmrnet_paper_monolith_readable.py check
    python rmrnet_paper_monolith_readable.py smoke
    python rmrnet_paper_monolith_readable.py train-rmr --out runs\my_rmrnet --epochs 30
    python rmrnet_paper_monolith_readable.py assets
    python rmrnet_paper_monolith_readable.py paper

Fresh folder use:

    python rmrnet_paper_monolith_readable.py extract --target C:\path\to\project --force
    cd /d C:\path\to\project
    python rmrnet_paper_monolith_readable.py check
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


BEGIN = "# === BEGIN EMBEDDED FILE: "
END = "# === END EMBEDDED FILE: "
DEFAULT_TARGET = "."


def script_path() -> Path:
    return Path(__file__).resolve()


def parse_embedded_files() -> dict[str, str]:
    """Read the commented source sections embedded in this file."""
    files: dict[str, list[str]] = {}
    current: str | None = None
    for raw in script_path().read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith(BEGIN):
            current = raw[len(BEGIN) :].rsplit(" ===", 1)[0]
            files[current] = []
            continue
        if raw.startswith(END):
            current = None
            continue
        if current is not None and raw.startswith("#|"):
            files[current].append(raw[2:])
    return {name: "\n".join(lines) + "\n" for name, lines in files.items()}


def extract_files(target: Path, force: bool = False, code_only: bool = False) -> None:
    target = target.resolve()
    embedded = parse_embedded_files()
    written = 0
    skipped = 0
    for rel, text in embedded.items():
        if code_only and rel.startswith(("paper_ieee_tits_rmrnet/", "experiments/")):
            continue
        dst = target / rel
        if dst.exists() and not force:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        written += 1
    print(f"extracted={written} skipped_existing={skipped} target={target}")


def run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    print("\n[run]", printable, flush=True)
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode:
        raise SystemExit(proc.returncode)


def delegate(target: Path, command: str, rest: list[str], dry_run: bool = False) -> None:
    runner = target / "rmrnet_paper_onefile.py"
    if not runner.exists():
        raise SystemExit(f"Missing {runner}. Run extract first or use --extract-first.")
    run([sys.executable, str(runner), command, *rest], target, dry_run=dry_run)


def common_extract(args: argparse.Namespace) -> Path:
    target = Path(args.target).resolve()
    if getattr(args, "extract_first", False):
        extract_files(target, force=getattr(args, "force", False), code_only=getattr(args, "code_only", False))
    return target


def cmd_list(_: argparse.Namespace) -> None:
    files = parse_embedded_files()
    for rel, text in sorted(files.items()):
        print(f"{rel}\t{len(text.encode('utf-8'))} bytes")
    print(f"files={len(files)}")


def cmd_extract(args: argparse.Namespace) -> None:
    extract_files(Path(args.target), force=args.force, code_only=args.code_only)


def cmd_env(_: argparse.Namespace) -> None:
    root = Path.cwd()
    print(f"cd /d {root}")
    print(r".venv\Scripts\activate")
    print("python rmrnet_paper_monolith_readable.py check")
    print("")
    print("If the environment does not exist:")
    print("python -m venv .venv")
    print(r".venv\Scripts\activate")
    print("pip install -r requirements-windows-gpu.txt")
    print("pip install -r requirements-detection-extra.txt")


def cmd_check(args: argparse.Namespace) -> None:
    target = common_extract(args)
    print(f"target={target}")
    print(f"python={sys.executable}")
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch_check_failed={exc}")
    if not args.no_compile:
        run(
            [
                sys.executable,
                "-m",
                "compileall",
                "models",
                "rcadnet",
                "losses",
                "baselines",
                "tools",
                "train_rcadnet.py",
                "train_rmrnet.py",
                "train_road_baseline.py",
            ],
            target,
            dry_run=args.dry_run,
        )


def cmd_delegate(args: argparse.Namespace) -> None:
    target = common_extract(args)
    mapping = {
        "train-rmr": "train",
        "eval-yolo": "eval",
    }
    delegate_name = mapping.get(args.subcommand, args.subcommand)
    delegate(target, delegate_name, args.rest, dry_run=args.dry_run)


def cmd_train_nafnet(args: argparse.Namespace) -> None:
    target = common_extract(args)
    rest = args.rest or [
        "--data-root",
        "data/pothole_restoration",
        "--scenario",
        "motion_horizontal_medium",
        "--model",
        "nafnet",
        "--epochs",
        "30",
        "--batch-size",
        "2",
        "--patch-size",
        "192",
        "--width",
        "32",
        "--device",
        "cuda",
        "--out",
        "runs/nafnet_monolith_default",
        "--num-workers",
        "0",
    ]
    run([sys.executable, "train_road_baseline.py", *rest], target, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Readable one-file RMR-Net paper runner.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("list", help="List embedded source files.")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("env", help="Print environment setup commands.")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("extract", help="Extract embedded source files.")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--force", action="store_true")
    p.add_argument("--code-only", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("check", help="Check CUDA and compile extracted/current code.")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--extract-first", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--code-only", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_check)

    for name in ["smoke", "train-rmr", "restore", "eval-yolo", "snake", "profile", "assets", "papercheck", "package", "paper"]:
        p = sub.add_parser(name, help=f"Run/delegate {name}. Remaining args are passed through.")
        p.add_argument("--target", default=DEFAULT_TARGET)
        p.add_argument("--extract-first", action="store_true")
        p.add_argument("--force", action="store_true")
        p.add_argument("--code-only", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("rest", nargs=argparse.REMAINDER)
        p.set_defaults(func=cmd_delegate)

    p = sub.add_parser("train-nafnet", help="Train the NAFNet road baseline.")
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--extract-first", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--code-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_train_nafnet)

    return parser


def main() -> None:
    start = time.time()
    args = build_parser().parse_args()
    args.func(args)
    print(f"\n[done] {args.subcommand} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
'''


def should_include(path: Path) -> bool:
    if path.name in EXCLUDE_FILES:
        return False
    if any(part in EXCLUDE_NAMES for part in path.parts):
        return False
    return path.suffix.lower() in TEXT_SUFFIXES


def collect() -> list[Path]:
    files: list[Path] = []
    for rel in TOP_LEVEL:
        path = ROOT / rel
        if path.exists() and should_include(path):
            files.append(path)
    for rel in INCLUDE_DIRS:
        root = ROOT / rel
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file() and should_include(p))
    if PAPER_DIR.exists():
        files.extend(p for p in PAPER_DIR.rglob("*") if p.is_file() and should_include(p))
    for root in RESULT_DIRS:
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file() and should_include(p))
    return sorted({p.resolve(): p for p in files}.values(), key=lambda p: p.relative_to(ROOT).as_posix())


def append_embedded(lines: list[str], path: Path) -> None:
    rel = path.relative_to(ROOT).as_posix()
    lines.append("")
    lines.append(f"# === BEGIN EMBEDDED FILE: {rel} ===")
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        lines.append("#|" + line)
    if text.endswith("\n"):
        lines.append("#|")
    lines.append(f"# === END EMBEDDED FILE: {rel} ===")


def main() -> None:
    files = collect()
    lines = [RUNNER, "", "# Embedded project source files follow as readable commented code."]
    for path in files:
        append_embedded(lines, path)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "files": len(files), "bytes": OUT.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
