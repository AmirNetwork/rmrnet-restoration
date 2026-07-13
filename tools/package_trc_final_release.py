from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAMP = "20260708"

PAPER_DIR = ROOT / "paper_trc_rmrnet"
PAPER_ZIP = ROOT / f"rmrnet_trc_final_paper_source_{STAMP}.zip"
CODE_ZIP = ROOT / f"rmrnet_trc_final_code_release_{STAMP}.zip"
EVIDENCE_ZIP = ROOT / f"rmrnet_trc_final_evidence_outputs_{STAMP}.zip"

STAGE = ROOT / "_trc_release_stage"


CODE_ROOT_FILES = [
    "README.md",
    "YOLO26_NATIVE_FIELD_DETECTOR_TECHNICAL_NOTE.md",
    "run_yolo26_nz_hybrid.ps1",
    "requirements-windows-gpu.txt",
    "requirements-detection-extra.txt",
    "requirements-dfpir-extra.txt",
    "requirements-demoe-extra.txt",
    "train_rcadnet.py",
    "train_rmrnet.py",
    "train_road_baseline.py",
    "benchmark_adapter_rcadnet.py",
    "benchmark_unified_restoration.py",
]

CODE_DIRS = [
    "models",
    "rcadnet",
    "losses",
    "baselines",
    "configs",
]

TOOL_FILES = [
    "tools/run_trc_final_30ep.py",
    "tools/build_trc_final_assets.py",
    "tools/build_gt49_defect_first_assets.py",
    "tools/build_gt49_safety_gate_policy.py",
    "tools/build_gt49_native_evidence_enhancer.py",
    "tools/build_trc_architecture_figure.py",
    "tools/package_trc_final_release.py",
    "tools/restore_yolo_split.py",
    "tools/restore_native_yolo_split.py",
    "tools/eval_yolo_suite.py",
    "tools/eval_yolo_per_class_suite.py",
    "tools/eval_native_tiled_detector.py",
    "tools/eval_gt49_defect_protocol.py",
    "tools/build_gt49_yolo26_segmentation_overlays.py",
    "tools/infer_nz_yolo26_hybrid_native.py",
    "tools/make_degraded_yolo_split.py",
    "tools/make_restoration_from_yolo.py",
    "tools/make_synthetic_restoration_data.py",
    "tools/prepare_pothole_yolo.py",
    "tools/prepare_pcm_yolo.py",
    "tools/prepare_geotagged_cam1_metadata.py",
    "tools/run_final_native_field_test.py",
    "tools/run_v32_revised_controlled_eval.py",
    "tools/prepare_newroad_primary_yolo.py",
    "tools/prepare_gt49_primary_eval_sets.py",
    "tools/prepare_newroad_rdd4_yolo.py",
    "tools/prepare_gt49_rdd4_eval_sets.py",
    "tools/snake_boundary_metrics.py",
]

THIRD_PARTY_PATCH_FILES = [
    "third_party/InstructIR-main/text/models.py",
]

SELECTED_DATASET_DIRS = [
    "pothole_yolo_motion_val_rmrnet_trc30_ep028",
    "pothole_yolo_motion_test_rmrnet_trc30_ep028",
    "pothole_yolo_defocus_val_rmrnet_trc30_ep028",
    "pothole_yolo_defocus_test_rmrnet_trc30_ep028",
    "pothole_yolo_lowlight_val_rmrnet_trc30_ep028",
    "pothole_yolo_lowlight_test_rmrnet_trc30_ep028",
    "pothole_yolo_mixed_val_rmrnet_trc30_ep028",
    "pothole_yolo_mixed_test_rmrnet_trc30_ep028",
    "pcm_yolo_motion_val_rmrnet_trc30_ep028",
    "pcm_yolo_motion_test_rmrnet_trc30_ep028",
    "pcm_yolo_defocus_val_rmrnet_trc30_ep028",
    "pcm_yolo_defocus_test_rmrnet_trc30_ep028",
    "pcm_yolo_lowlight_val_rmrnet_trc30_ep028",
    "pcm_yolo_lowlight_test_rmrnet_trc30_ep028",
    "pcm_yolo_mixed_val_rmrnet_trc30_ep028",
    "pcm_yolo_mixed_test_rmrnet_trc30_ep028",
]


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return "__pycache__" in parts or path.suffix in {".pyc", ".pyo"}


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    for item in src.rglob("*"):
        if item.is_dir() or should_skip(item):
            continue
        rel = item.relative_to(src)
        copy_file(item, dst / rel)


def zip_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in sorted(src.rglob("*")):
            if item.is_dir() or should_skip(item):
                continue
            archive.write(item, item.relative_to(src.parent))


def package_paper() -> None:
    zip_dir(PAPER_DIR, PAPER_ZIP)


def package_code() -> None:
    code_stage = STAGE / "rmrnet_trc_final_code_release"
    if code_stage.exists():
        shutil.rmtree(code_stage)
    code_stage.mkdir(parents=True)

    for rel in CODE_ROOT_FILES:
        copy_file(ROOT / rel, code_stage / rel)
    for rel in CODE_DIRS:
        copy_tree(ROOT / rel, code_stage / rel)
    for rel in TOOL_FILES:
        copy_file(ROOT / rel, code_stage / rel)
    for rel in THIRD_PARTY_PATCH_FILES:
        if (ROOT / rel).exists():
            copy_file(ROOT / rel, code_stage / rel)
    detector_package = ROOT / "incoming_yolo26_fined_tuned_main_20260706" / "yolo26-fined-tuned-main"
    if detector_package.exists():
        copy_tree(detector_package, code_stage / "detectors" / "yolo26_cropmask_package")

    readme = code_stage / "RUN_TRC_FINAL_EXPERIMENTS.md"
    readme.write_text(
        "# RMR-Net TRC Final Experiment Runner\n\n"
        "Activate the Windows GPU environment from the repository root:\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\activate\n"
        "```\n\n"
        "Download the native-field detector checkpoint before rerunning GT49:\n\n"
        "```powershell\n"
        "python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TamAko783/YOLO26s_RDD_FRDC_Distilled_v2', filename='YOLO26s_RDD_FRDC_Distilled_v2.pt', local_dir='weights/yolo26s_rdd_frdc_distilled_v2')\"\n"
        "```\n\n"
        "Run the final controlled benchmark workflow:\n\n"
        "```powershell\n"
        "python tools\\run_trc_final_30ep.py --epochs 30 --device cuda\n"
        "python tools\\build_trc_final_assets.py\n"
        "python tools\\prepare_gt49_rdd4_eval_sets.py --input-root experiments\\roboflow_geotagged_v5_native_real\\v32_final_gt49_allmethods --out experiments\\roboflow_geotagged_v5_native_real\\v36_yolo26_rdd4_eval_sets\n"
        "$methods = @('raw','rmr_blind','rmr_metadata','rmr_metadata_gated','nafnet','dfpir','demoe_auto','demoe_scenario','instructir_generic','instructir_metadata')\n"
        "foreach ($m in $methods) { python tools\\eval_native_tiled_detector.py --data experiments\\roboflow_geotagged_v5_native_real\\v36_yolo26_rdd4_eval_sets\\$m\\data.yaml --weights weights\\yolo26s_rdd_frdc_distilled_v2\\YOLO26s_RDD_FRDC_Distilled_v2.pt --out experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_eval\\$m --strategy hybrid --full-imgsz 1536 --full-conf 0.5 --full-iou 0.78 --tile 2048 --overlap 640 --center-margin 256 --infer-imgsz 1536 --conf 0.5 --branch-iou 0.78 --tile-fusion-conf 0.5 --nms-iou 0.50 --crop-bottom-half --road-filter --crack-classes 0,1,2 --device 0 }\n"
        "$etaMethods = @('rmr_eta_0p0','rmr_eta_0p1','rmr_eta_0p25','rmr_eta_0p5','rmr_eta_0p75','rmr_eta_1p0')\n"
        "foreach ($m in $etaMethods) { python tools\\eval_native_tiled_detector.py --data experiments\\roboflow_geotagged_v5_native_real\\v36_yolo26_rdd4_eta_sets\\$m\\data.yaml --weights weights\\yolo26s_rdd_frdc_distilled_v2\\YOLO26s_RDD_FRDC_Distilled_v2.pt --out experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_eta_eval\\$m --strategy hybrid --full-imgsz 1536 --full-conf 0.5 --full-iou 0.78 --tile 2048 --overlap 640 --center-margin 256 --infer-imgsz 1536 --conf 0.5 --branch-iou 0.78 --tile-fusion-conf 0.5 --nms-iou 0.50 --crop-bottom-half --road-filter --crack-classes 0,1,2 --device 0 }\n"
        "$tauMethods = @('rmr_tau_00','rmr_tau_01','rmr_tau_02','rmr_tau_03','rmr_tau_04','rmr_tau_05','rmr_tau_06')\n"
        "foreach ($m in $tauMethods) { python tools\\eval_native_tiled_detector.py --data experiments\\roboflow_geotagged_v5_native_real\\v36_yolo26_rdd4_tau_sets\\$m\\data.yaml --weights weights\\yolo26s_rdd_frdc_distilled_v2\\YOLO26s_RDD_FRDC_Distilled_v2.pt --out experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_tau_eval\\$m --strategy hybrid --full-imgsz 1536 --full-conf 0.5 --full-iou 0.78 --tile 2048 --overlap 640 --center-margin 256 --infer-imgsz 1536 --conf 0.5 --branch-iou 0.78 --tile-fusion-conf 0.5 --nms-iou 0.50 --crop-bottom-half --road-filter --crack-classes 0,1,2 --device 0 }\n"
        "python tools\\eval_gt49_defect_protocol.py --data experiments\\roboflow_geotagged_v5_native_real\\v36_yolo26_rdd4_eval_sets\\raw\\data.yaml --eval-root experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_eval --out experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_defect_eval --defect-classes 0,1,2,3 --crack-classes 0,1,2 --pothole-class 3 --longitudinal-class 0 --transverse-class 1 --alligator-class 2 --crop-bottom-half\n"
        "python tools\\eval_gt49_defect_protocol.py --data experiments\\roboflow_geotagged_v5_native_real\\v36_yolo26_rdd4_eval_sets\\raw\\data.yaml --eval-root experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_eta_eval --out experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_eta_defect_eval --defect-classes 0,1,2,3 --crack-classes 0,1,2 --pothole-class 3 --longitudinal-class 0 --transverse-class 1 --alligator-class 2 --crop-bottom-half\n"
        "python tools\\eval_gt49_defect_protocol.py --data experiments\\roboflow_geotagged_v5_native_real\\v36_yolo26_rdd4_eval_sets\\raw\\data.yaml --eval-root experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_tau_eval --out experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_tau_defect_eval --defect-classes 0,1,2,3 --crack-classes 0,1,2 --pothole-class 3 --longitudinal-class 0 --transverse-class 1 --alligator-class 2 --crop-bottom-half\n"
        "python tools\\build_gt49_safety_gate_policy.py --python python\n"
        "python tools\\build_gt49_defect_first_assets.py\n"
        "python tools\\build_gt49_yolo26_segmentation_overlays.py --conf 0.10 --panel-width 420 --max-atlas-images 12\n"
        "python tools\\build_gt49_yolo26_segmentation_overlays.py --methods raw,rmr_eta_0p25,nafnet,instructir_generic --conf 0.10 --panel-width 560 --max-atlas-images 6 --out experiments\\roboflow_geotagged_v5_native_real\\v38_yolo26_cropmask_paper_examples\n"
        "python tools\\build_trc_architecture_figure.py\n"
        "```\n\n"
        "The selected checkpoints are chosen by validation mAP50 only; test images are restored and evaluated after selection. "
        "The native GT49 field audit uses the frozen YOLO26s RDD-FRDC checkpoint from Hugging Face, "
        "with the modified crop/mask protocol, a lower-road native crop, full-crop inference, 2048-pixel center-safe tiles, and a fixed road-surface gate. The paper table then applies the defect-first protocol: "
        "non-defect classes are ignored, GT/prediction matching is performed per image, crack subtypes are grouped for primary localization, and crack-fragment duplicates are suppressed for reporting. "
        "The known-GT success column is recall-oriented for the manually annotated GT49 audit: a same-primary prediction recovers a label if IoU >= 0.10, GT coverage >= 25%, or the GT center is inside the prediction. "
        "GT49 labels are not used to update detector weights.\n",
        encoding="utf-8",
    )
    zip_dir(code_stage, CODE_ZIP)


def copy_run_artifacts(run_name: str, out: Path) -> None:
    run_dir = ROOT / "runs" / run_name
    target = out / "runs" / run_name
    for item in run_dir.iterdir():
        if item.is_dir():
            continue
        if item.suffix == ".pth" and item.name != "rcadnet_epoch_028.pth":
            continue
        copy_file(item, target / item.name)


def copy_native_field_evidence(out: Path) -> None:
    native_root = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v32_final_gt49_allmethods"
    target = out / "experiments" / "roboflow_geotagged_v5_native_real" / "v32_final_gt49_allmethods"
    if not native_root.exists():
        return

    for name in [
        "paper_metric_summary_all49.csv",
        "geotagged_eta_sweep.csv",
        "geotagged_tau_sweep.csv",
        "final_native_field_manifest.json",
    ]:
        src = native_root / name
        if src.exists():
            copy_file(src, target / name)

    sharpness = native_root / "sharpness_audit" / "sharpness_summary.csv"
    if sharpness.exists():
        copy_file(sharpness, target / "sharpness_audit" / "sharpness_summary.csv")

    eval_root = native_root / "native_tiled_eval"
    if eval_root.exists():
        for eval_dir in eval_root.iterdir():
            if not eval_dir.is_dir():
                continue
            for name in ["summary.json", "native_tiled_metrics.csv", "operating_points.csv", "predictions.csv"]:
                src = eval_dir / name
                if src.exists():
                    copy_file(src, target / "native_tiled_eval" / eval_dir.name / name)

    native_exp_root = ROOT / "experiments" / "roboflow_geotagged_v5_native_real"
    for dirname in [
        "v38_yolo26_cropmask_defect_eval",
        "v38_yolo26_cropmask_eta_defect_eval",
        "v38_yolo26_cropmask_tau_defect_eval",
        "v43_safety_gate_final_eval",
        "v43_safety_gate_final_defect_eval",
    ]:
        defect_eval = native_exp_root / dirname
        if defect_eval.exists():
            defect_target = out / "experiments" / "roboflow_geotagged_v5_native_real" / dirname
            for name in [
                "defect_protocol_summary.csv",
                "defect_protocol_class_metrics.csv",
                "defect_protocol_type_confusion.csv",
                "defect_protocol_manifest.json",
                "safety_gate_manifest.json",
                "predictions.csv",
            ]:
                src = defect_eval / name
                if src.exists():
                    copy_file(src, defect_target / name)

    detector_eval = native_exp_root / "v38_yolo26_cropmask_eval"
    if detector_eval.exists():
        for eval_dir in detector_eval.iterdir():
            if not eval_dir.is_dir():
                continue
            for name in ["summary.json", "native_tiled_metrics.csv", "operating_points.csv", "predictions.csv"]:
                src = eval_dir / name
                if src.exists():
                    copy_file(
                        src,
                        out
                        / "experiments"
                        / "roboflow_geotagged_v5_native_real"
                        / "v38_yolo26_cropmask_eval"
                        / eval_dir.name
                        / name,
                    )

    safety_eval = native_exp_root / "v43_safety_gate_final_eval"
    if safety_eval.exists():
        for eval_dir in safety_eval.iterdir():
            if not eval_dir.is_dir():
                continue
            for name in ["predictions.csv"]:
                src = eval_dir / name
                if src.exists():
                    copy_file(
                        src,
                        out
                        / "experiments"
                        / "roboflow_geotagged_v5_native_real"
                        / "v43_safety_gate_final_eval"
                        / eval_dir.name
                        / name,
                    )

    visual_overlays = native_exp_root / "v38_yolo26_cropmask_visual_overlays"
    if visual_overlays.exists():
        copy_tree(visual_overlays, out / "experiments" / "roboflow_geotagged_v5_native_real" / "v38_yolo26_cropmask_visual_overlays")

    paper_examples = native_exp_root / "v38_yolo26_cropmask_paper_examples"
    if paper_examples.exists():
        copy_tree(paper_examples, out / "experiments" / "roboflow_geotagged_v5_native_real" / "v38_yolo26_cropmask_paper_examples")

    for audit_name in ["gt49_policy_visual_audit.jpg", "gt49_restoration_visual_audit.jpg"]:
        audit_path = native_exp_root / audit_name
        if audit_path.exists():
            copy_file(audit_path, out / "experiments" / "roboflow_geotagged_v5_native_real" / audit_name)

    detector_run = ROOT / "runs" / "detect" / "runs" / "detect" / "yolo26s_rdd4_gt49_field" / "finetune_1536_120ep"
    detector_target = out / "runs" / "detect" / "yolo26s_rdd4_gt49_field" / "finetune_1536_120ep"
    for name in ["results.csv", "args.yaml"]:
        src = detector_run / name
        if src.exists():
            copy_file(src, detector_target / name)


def package_evidence() -> None:
    evidence_stage = STAGE / "rmrnet_trc_final_evidence_outputs"
    if evidence_stage.exists():
        shutil.rmtree(evidence_stage)
    evidence_stage.mkdir(parents=True)

    copy_tree(ROOT / "experiments" / "trc_final_30ep", evidence_stage / "experiments" / "trc_final_30ep")
    copy_tree(
        ROOT / "experiments" / "trc_revised_identity_final",
        evidence_stage / "experiments" / "trc_revised_identity_final",
    )
    copy_native_field_evidence(evidence_stage)
    copy_tree(PAPER_DIR / "tables", evidence_stage / "paper_trc_rmrnet" / "tables")
    copy_tree(PAPER_DIR / "figures", evidence_stage / "paper_trc_rmrnet" / "figures")
    copy_file(PAPER_DIR / "RESULT_PROVENANCE_TABLE.csv", evidence_stage / "paper_trc_rmrnet" / "RESULT_PROVENANCE_TABLE.csv")

    for run_name in ["trc_final_rmrnet_pothole_30ep", "trc_final_rmrnet_pcm_30ep"]:
        copy_run_artifacts(run_name, evidence_stage)

    for bench_name in ["bench_trc_final_rmrnet_pothole_30ep", "bench_trc_final_rmrnet_pcm_30ep"]:
        copy_tree(ROOT / "runs" / bench_name, evidence_stage / "runs" / bench_name)

    for bench_name in ["bench_trc_revised_identity_pothole", "bench_trc_revised_identity_pcm"]:
        bench_path = ROOT / "runs" / bench_name
        if bench_path.exists():
            copy_tree(bench_path, evidence_stage / "runs" / bench_name)

    for name in SELECTED_DATASET_DIRS:
        copy_tree(ROOT / "datasets" / name, evidence_stage / "datasets" / name)

    (evidence_stage / "README_EVIDENCE.md").write_text(
        "# RMR-Net TRC Evidence Outputs\n\n"
        "This package contains the selected controlled validation/test summaries, "
        "the selected epoch checkpoints, revised identity-calibration audit CSV/JSON outputs, the GT49 native-field hybrid-detector summaries, and paper figures/tables. "
        "GT49 defect matching is image-safe and the known-GT success column uses the tolerant same-primary-class recovery rule documented in `YOLO26_NATIVE_FIELD_DETECTOR_TECHNICAL_NOTE.md`. "
        "The full 30-epoch checkpoint folders are intentionally not duplicated here; they remain in the workspace under `runs/`.\n",
        encoding="utf-8",
    )
    zip_dir(evidence_stage, EVIDENCE_ZIP)


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    package_paper()
    package_code()
    package_evidence()
    for path in [PAPER_ZIP, CODE_ZIP, EVIDENCE_ZIP]:
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
