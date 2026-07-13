"""Build a monolithic all-in-one RMR-Net paper runner.

The generated file embeds project source files and selected paper/result assets
as base64 payloads, then exposes a single command-line interface. This is meant
for a user who wants one giant Python file that can extract the code and run the
paper workflow from Command Prompt.
"""

from __future__ import annotations

import base64
import hashlib
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rmrnet_paper_allinone_giant.py"

INCLUDE_DIRS = [
    "models",
    "rcadnet",
    "losses",
    "tools",
    "baselines",
    "configs",
]

TOP_LEVEL_FILES = [
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
    "requirements-windows-gpu.txt",
    "requirements-detection-extra.txt",
    "requirements-dfpir-extra.txt",
]

PAPER_SUFFIXES = {".tex", ".bib", ".bst", ".cls", ".md", ".csv"}
PAPER_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
RESULT_DIRS = [
    "experiments/v29_review_completion",
    "experiments/v30_submission_readiness",
    "runs/snake_polygon_accuracy_v30",
]

EXCLUDE_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "supplementary_rmrnet_realmeta.pdf",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pth", ".pt", ".zip", ".pdf"}


RUNNER = r'''
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


DEFAULT_SCENARIOS = ["motion_horizontal_medium", "defocus_medium", "lowlight_medium"]
DEFAULT_TRAIN_ROOTS = ["data/pothole_restoration", "data/pcm_restoration_train"]
DEFAULT_PAPER_DIR = "paper_ieee_tits_rmrnet"


def here() -> Path:
    return Path(__file__).resolve().parent


def target_path(target: str | Path, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(target).resolve() / p


def cmd_text(cmd) -> str:
    out = []
    for x in cmd:
        s = str(x)
        out.append(f'"{s}"' if " " in s else s)
    return " ".join(out)


def run(cmd, cwd: Path, dry_run: bool = False):
    print("\n[run]", cmd_text(cmd), flush=True)
    if dry_run:
        return 0
    proc = subprocess.run([str(x) for x in cmd], cwd=str(cwd))
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc.returncode


def write_embedded_file(root: Path, rel: str, payload: str, sha256: str, force: bool) -> bool:
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return False
    data = base64.b64decode(payload.encode("ascii"))
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise RuntimeError(f"Embedded payload checksum mismatch for {rel}")
    dst.write_bytes(data)
    return True


def extract_all(target: Path, force: bool = False, code_only: bool = False) -> None:
    target = target.resolve()
    count = 0
    skipped = 0
    for rel, meta in EMBEDDED_FILES.items():
        kind = meta.get("kind", "code")
        if code_only and kind not in {"code", "config", "doc"}:
            continue
        wrote = write_embedded_file(target, rel, meta["b64"], meta["sha256"], force)
        if wrote:
            count += 1
        else:
            skipped += 1
    print(f"extracted={count} skipped_existing={skipped} target={target}")


def ensure(paths, cwd: Path):
    missing = []
    for p in paths:
        pp = Path(p)
        full = pp if pp.is_absolute() else cwd / pp
        if not full.exists():
            missing.append(str(full))
    if missing:
        raise SystemExit("Missing required paths:\n" + "\n".join("  - " + x for x in missing))


def maybe_extract(args):
    if getattr(args, "extract_first", False):
        extract_all(Path(args.target), force=getattr(args, "force", False), code_only=getattr(args, "code_only", False))


def python_cmd(cwd: Path, *parts):
    return [sys.executable, *parts]


def cmd_list(args):
    for rel, meta in sorted(EMBEDDED_FILES.items()):
        print(f"{rel}\t{meta.get('kind','code')}\t{meta['size']} bytes")
    print(f"files={len(EMBEDDED_FILES)}")


def cmd_extract(args):
    extract_all(Path(args.target), force=args.force, code_only=args.code_only)


def cmd_check(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    print(f"target={cwd}")
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
        run(python_cmd(cwd, "-m", "compileall", "models", "rcadnet", "losses", "baselines", "tools", "train_rcadnet.py", "train_rmrnet.py", "train_road_baseline.py"), cwd, args.dry_run)


def add_train_common(p):
    p.add_argument("--target", default=".")
    p.add_argument("--extract-first", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--data-root", action="append")
    p.add_argument("--scenario", action="append")
    p.add_argument("--out", default="runs/rmrnet_allinone_run")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=192)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")


def rmr_train_command(args, smoke=False):
    roots = args.data_root or DEFAULT_TRAIN_ROOTS
    scenarios = args.scenario or DEFAULT_SCENARIOS
    cmd = [sys.executable, "train_rmrnet.py"]
    for root in roots:
        cmd += ["--data-root", root]
    for sc in scenarios:
        cmd += ["--scenario", sc]
    cmd += [
        "--epochs", str(1 if smoke else args.epochs),
        "--batch-size", str(args.batch_size),
        "--patch-size", str(args.patch_size),
        "--width", str(args.width),
        "--device", args.device,
        "--out", args.out,
        "--num-workers", str(args.num_workers),
        "--code-source", "metadata_fused",
        "--block-type", "evidence",
        "--attention-type", "task",
        "--conditioning", "gated_basis",
        "--detail-preserve",
        "--detail-gain", "0.25",
        "--metadata-dropout", "0.3",
        "--metadata-noise", "0.05",
        "--edge-weight", "0.1",
        "--freq-weight", "0.05",
        "--defect-weight", "0.15",
        "--visibility-weight", "0.05",
        "--use-task-losses",
        "--use-tdac-head",
        "--tdp-yolo-weights", "yolo11s.pt",
        "--lambda-tdp", "0.02",
        "--lambda-jacobian", "0.001",
        "--lambda-active-contour", "0.005",
        "--lambda-detector-input-anchor", "0.005",
        "--lambda-evidence-nonregression", "0.01",
        "--task-loss-warmup-epochs", "3",
        "--grad-clip", "1.0",
        "--save-every-epoch",
    ]
    cmd.append("--amp" if args.amp else "--no-amp")
    if smoke:
        cmd.append("--smoke-test")
    return cmd


def cmd_smoke(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    ensure(args.data_root or DEFAULT_TRAIN_ROOTS, cwd)
    run(rmr_train_command(args, smoke=True), cwd, args.dry_run)


def cmd_train_rmr(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    ensure(args.data_root or DEFAULT_TRAIN_ROOTS, cwd)
    run(rmr_train_command(args, smoke=False), cwd, args.dry_run)


def cmd_train_nafnet(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    roots = args.data_root or ["data/pothole_restoration"]
    scenarios = args.scenario or DEFAULT_SCENARIOS
    for root in roots:
        for sc in scenarios:
            out = Path(args.out) / f"{Path(root).name}_{sc}"
            cmd = [
                sys.executable, "train_road_baseline.py",
                "--data-root", root,
                "--scenario", sc,
                "--model", "nafnet",
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--patch-size", str(args.patch_size),
                "--width", str(args.width),
                "--device", args.device,
                "--out", str(out),
                "--num-workers", str(args.num_workers),
            ]
            run(cmd, cwd, args.dry_run)


def cmd_restore(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    ensure([args.data], cwd)
    cmd = [
        sys.executable, "tools/restore_yolo_split.py",
        "--data", args.data,
        "--split", args.split,
        "--model", args.model,
        "--scenario", args.scenario,
        "--out", args.out,
        "--device", args.device,
    ]
    if args.weights:
        cmd += [{"rcadnet": "--rcadnet-weights", "dfpir": "--dfpir-weights", "nafnet": "--nafnet-weights"}[args.model], args.weights]
    if args.model == "rcadnet":
        cmd += ["--rcadnet-code-source", args.code_source]
    if args.dfpir_clip:
        cmd.append("--dfpir-clip")
    if args.residual_strength != 1.0:
        cmd += ["--residual-strength", str(args.residual_strength)]
    run(cmd, cwd, args.dry_run)


def cmd_eval(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    ensure([args.weights], cwd)
    if not args.item:
        raise SystemExit("Use --item name=path/to/data.yaml at least once")
    cmd = [
        sys.executable, "tools/eval_yolo_suite.py",
        "--weights", args.weights,
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", args.device,
        "--workers", str(args.workers),
        "--split", args.split,
        "--out", args.out,
    ]
    for item in args.item:
        cmd += ["--item", item]
    run(cmd, cwd, args.dry_run)


def cmd_snake(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    ensure([args.data], cwd)
    cmd = [
        sys.executable, "tools/snake_boundary_metrics.py",
        "--data", args.data,
        "--split", args.split,
        "--out", args.out,
        "--iterations", str(args.iterations),
        "--max-overlays", str(args.max_overlays),
    ]
    for flag, value in [("--box-dir", args.box_dir), ("--gt-polygon-dir", args.gt_polygon_dir), ("--classes", args.classes)]:
        if value:
            cmd += [flag, value]
    if args.max_images:
        cmd += ["--max-images", str(args.max_images)]
    if args.sample_seed:
        cmd += ["--sample-seed", str(args.sample_seed)]
    run(cmd, cwd, args.dry_run)


def cmd_profile(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    ensure([args.weights], cwd)
    cmd = [
        sys.executable, "tools/profile_rmrnet_complexity.py",
        "--weights", args.weights,
        "--out", args.out,
        "--height", str(args.height),
        "--width", str(args.width),
        "--runs", str(args.runs),
        "--warmup", str(args.warmup),
        "--device", args.device,
    ]
    run(cmd, cwd, args.dry_run)


ASSET_SCRIPTS = [
    "tools/build_rmrnet_ieee_assets.py",
    "tools/build_v27_taskloss_assets.py",
    "tools/build_v28_review_assets.py",
    "tools/build_v29_review_completion_assets.py",
    "tools/build_v30_submission_readiness_assets.py",
    "tools/build_snake_paper_assets.py",
    "tools/build_native_blur_assets.py",
    "tools/build_kitti_realmeta_assets.py",
    "tools/build_kitti_realism_audit_assets.py",
]


def cmd_assets(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    scripts = ASSET_SCRIPTS
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        scripts = [s for s in scripts if Path(s).stem in wanted or Path(s).name in wanted]
    for script in scripts:
        if not (cwd / script).exists():
            if args.skip_missing:
                print(f"[skip missing] {script}")
                continue
            raise SystemExit(f"Missing asset script: {cwd / script}")
        run([sys.executable, script], cwd, args.dry_run)


def manuscript_check(paper: Path):
    import re
    tex = paper / "manuscript.tex"
    if not tex.exists():
        raise SystemExit(f"Missing manuscript: {tex}")
    text = tex.read_text(encoding="utf-8", errors="replace")
    missing = []
    for m in re.finditer(r"\\input\{([^}]+)\}", text):
        p = paper / f"{m.group(1)}.tex"
        if not p.exists():
            missing.append(str(p))
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        p = paper / m.group(1)
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise SystemExit("Missing manuscript inputs:\n" + "\n".join(missing))
    print("All manuscript inputs and figures exist.")


def zip_folder(src: Path, dst: Path, exclude_names=None):
    exclude_names = set(exclude_names or [])
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in src.rglob("*"):
            if file.is_file() and file.name not in exclude_names:
                zf.write(file, file.relative_to(src))


def cmd_package(args):
    maybe_extract(args)
    cwd = Path(args.target).resolve()
    paper = cwd / args.paper_dir
    manuscript_check(paper)
    main_zip = cwd / args.main_zip
    supp_zip = cwd / args.supp_zip
    if args.dry_run:
        print(f"Would zip {paper} -> {main_zip}")
        print(f"Would zip supplement/provenance -> {supp_zip}")
        return
    zip_folder(paper, main_zip, exclude_names={"supplementary_rmrnet_realmeta.pdf"})
    temp = cwd / "_allinone_supplement_stage"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    for name in ["SUPPLEMENTARY_README.md", "supplementary_rmrnet_realmeta.pdf"]:
        src = paper / name
        if src.exists():
            shutil.copy2(src, temp / name)
    for rel in ["experiments/v29_review_completion", "experiments/v30_submission_readiness"]:
        src = cwd / rel
        if src.exists():
            dst = temp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)
    zip_folder(temp, supp_zip)
    print(f"{main_zip} {main_zip.stat().st_size / (1024*1024):.2f} MB")
    print(f"{supp_zip} {supp_zip.stat().st_size / (1024*1024):.2f} MB")


def cmd_paper(args):
    maybe_extract(args)
    cmd_check(argparse.Namespace(target=args.target, no_compile=False, dry_run=args.dry_run, extract_first=False))
    cmd_assets(argparse.Namespace(target=args.target, only="", skip_missing=True, dry_run=args.dry_run, extract_first=False))
    manuscript_check(Path(args.target).resolve() / args.paper_dir)
    cmd_package(args)


def build_parser():
    p = argparse.ArgumentParser(description="Giant embedded all-in-one runner for the RMR-Net paper.")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("list", help="List embedded files.")
    q.set_defaults(func=cmd_list)

    q = sub.add_parser("extract", help="Extract embedded code/assets to a folder.")
    q.add_argument("--target", default=".")
    q.add_argument("--force", action="store_true")
    q.add_argument("--code-only", action="store_true")
    q.set_defaults(func=cmd_extract)

    q = sub.add_parser("check", help="Check/extract/compile project code.")
    q.add_argument("--target", default=".")
    q.add_argument("--extract-first", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--no-compile", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=cmd_check)

    q = sub.add_parser("smoke", help="Run one RMR-Net smoke-test batch.")
    add_train_common(q)
    q.set_defaults(func=cmd_smoke)

    q = sub.add_parser("train-rmr", help="Train RMR-Net.")
    add_train_common(q)
    q.set_defaults(func=cmd_train_rmr)

    q = sub.add_parser("train-nafnet", help="Train NAFNet road baseline.")
    add_train_common(q)
    q.set_defaults(func=cmd_train_nafnet)

    q = sub.add_parser("restore", help="Restore one YOLO split.")
    q.add_argument("--target", default=".")
    q.add_argument("--extract-first", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--data", required=True)
    q.add_argument("--split", default="test")
    q.add_argument("--model", choices=["rcadnet", "dfpir", "nafnet"], required=True)
    q.add_argument("--scenario", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--weights", default="")
    q.add_argument("--device", default="cuda")
    q.add_argument("--code-source", default="metadata", choices=["scenario", "metadata", "blind"])
    q.add_argument("--dfpir-clip", action="store_true")
    q.add_argument("--residual-strength", type=float, default=1.0)
    q.set_defaults(func=cmd_restore)

    q = sub.add_parser("eval-yolo", help="Evaluate YOLO on one or more data.yaml files.")
    q.add_argument("--target", default=".")
    q.add_argument("--extract-first", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--weights", default="yolo11s.pt")
    q.add_argument("--item", action="append")
    q.add_argument("--out", required=True)
    q.add_argument("--imgsz", type=int, default=640)
    q.add_argument("--batch", type=int, default=8)
    q.add_argument("--device", default="0")
    q.add_argument("--workers", type=int, default=0)
    q.add_argument("--split", default="test")
    q.set_defaults(func=cmd_eval)

    q = sub.add_parser("snake", help="Run active-contour metrics.")
    q.add_argument("--target", default=".")
    q.add_argument("--extract-first", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--data", required=True)
    q.add_argument("--split", default="test")
    q.add_argument("--out", required=True)
    q.add_argument("--box-dir", default="")
    q.add_argument("--gt-polygon-dir", default="")
    q.add_argument("--classes", default="")
    q.add_argument("--max-images", type=int, default=0)
    q.add_argument("--sample-seed", type=int, default=0)
    q.add_argument("--iterations", type=int, default=45)
    q.add_argument("--max-overlays", type=int, default=10)
    q.set_defaults(func=cmd_snake)

    q = sub.add_parser("profile", help="Profile RMR-Net.")
    q.add_argument("--target", default=".")
    q.add_argument("--extract-first", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--weights", required=True)
    q.add_argument("--out", default="experiments/allinone/rmrnet_complexity.json")
    q.add_argument("--height", type=int, default=360)
    q.add_argument("--width", type=int, default=640)
    q.add_argument("--runs", type=int, default=30)
    q.add_argument("--warmup", type=int, default=8)
    q.add_argument("--device", default="cuda")
    q.set_defaults(func=cmd_profile)

    q = sub.add_parser("assets", help="Generate all paper tables/figures from available results.")
    q.add_argument("--target", default=".")
    q.add_argument("--extract-first", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--only", default="")
    q.add_argument("--skip-missing", action="store_true")
    q.set_defaults(func=cmd_assets)

    q = sub.add_parser("package", help="Package paper zips.")
    q.add_argument("--target", default=".")
    q.add_argument("--extract-first", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--paper-dir", default=DEFAULT_PAPER_DIR)
    q.add_argument("--main-zip", default="paper_ieee_tits_rmrnet_allinone_main.zip")
    q.add_argument("--supp-zip", default="paper_ieee_tits_rmrnet_allinone_supplement.zip")
    q.set_defaults(func=cmd_package)

    q = sub.add_parser("paper", help="Compile code, regenerate assets, verify manuscript, and package zips.")
    q.add_argument("--target", default=".")
    q.add_argument("--extract-first", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--paper-dir", default=DEFAULT_PAPER_DIR)
    q.add_argument("--main-zip", default="paper_ieee_tits_rmrnet_allinone_main.zip")
    q.add_argument("--supp-zip", default="paper_ieee_tits_rmrnet_allinone_supplement.zip")
    q.set_defaults(func=cmd_paper)

    return p


def main():
    os.chdir(here())
    start = time.time()
    args = build_parser().parse_args()
    args.func(args)
    print(f"\n[done] {args.cmd} in {time.time() - start:.1f}s")
'''


def should_include(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_NAMES:
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    if path.name == OUT.name:
        return False
    return True


def kind_for(rel: str) -> str:
    if rel.startswith("paper_ieee_tits_rmrnet/"):
        if "/figures/" in rel:
            return "paper_figure"
        if "/tables/" in rel:
            return "paper_table"
        return "paper_source"
    if rel.startswith("experiments/") or rel.startswith("runs/"):
        return "result"
    if rel.startswith("configs/"):
        return "config"
    if rel.endswith((".md", ".txt", ".csv", ".json", ".yaml", ".yml")):
        return "doc"
    return "code"


def collect_files() -> list[Path]:
    files: list[Path] = []
    for rel in TOP_LEVEL_FILES:
        p = ROOT / rel
        if p.exists() and should_include(p):
            files.append(p)
    for directory in INCLUDE_DIRS:
        root = ROOT / directory
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and should_include(p):
                files.append(p)
    paper = ROOT / "paper_ieee_tits_rmrnet"
    if paper.exists():
        for p in paper.rglob("*"):
            if not p.is_file() or not should_include(p):
                continue
            suffix = p.suffix.lower()
            if suffix in PAPER_SUFFIXES or suffix in PAPER_IMAGE_SUFFIXES:
                files.append(p)
    for directory in RESULT_DIRS:
        root = ROOT / directory
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and should_include(p):
                if p.suffix.lower() in {".csv", ".json", ".txt", ".md"}:
                    files.append(p)
    unique = sorted({p.resolve(): p for p in files}.values(), key=lambda x: x.relative_to(ROOT).as_posix())
    return unique


def wrap_b64(data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return "\n".join(textwrap.wrap(encoded, 76))


def main() -> None:
    files = collect_files()
    lines = [
        '"""Giant embedded RMR-Net paper code runner.',
        "",
        "This file was generated by tools/build_rmrnet_allinone_giant.py.",
        "It embeds the project code, selected paper source/assets, and small result",
        "artifacts needed to regenerate the paper tables/figures from local data.",
        '"""',
        "",
        RUNNER,
        "",
        "EMBEDDED_FILES = {",
    ]
    manifest = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        manifest.append({"path": rel, "size": len(data), "sha256": sha, "kind": kind_for(rel)})
        lines.append(f"    {rel!r}: {{")
        lines.append(f"        'kind': {kind_for(rel)!r},")
        lines.append(f"        'size': {len(data)},")
        lines.append(f"        'sha256': {sha!r},")
        lines.append("        'b64': (")
        for chunk in wrap_b64(data).splitlines():
            lines.append(f"            {chunk!r}")
        lines.append("        ),")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append(f"EMBEDDED_MANIFEST = {json.dumps(manifest, indent=2)!r}")
    lines.append("")
    lines.append('if __name__ == "__main__":')
    lines.append("    main()")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "files": len(files), "bytes": OUT.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
