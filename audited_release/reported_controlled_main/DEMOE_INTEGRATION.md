# DeMoE Integration Notes

This project now includes a benchmark adapter for **DeMoE: Towards Unified Image
Deblurring using a Mixture-of-Experts Decoder**. DeMoE is valuable for this paper
because it is a recent all-in-one deblurring method that explicitly routes blur
types through a mixture-of-experts decoder. That makes it a stronger comparison
than a single-task motion-only deblurring network when our benchmark mixes motion
blur, defocus, low light, vibration-like blur, and native blur.

## Local Code Path

- Official repo copy: `third_party/DeMoE-main/`
- Adapter: `baselines/demoe_adapter.py`
- Full-reference benchmark: `benchmark_unified_restoration.py --model demoe`
- YOLO split restoration: `tools/restore_yolo_split.py --model demoe`
- Extra dependency note: `requirements-demoe-extra.txt`

The official DeMoE inference script uses a Linux-style `torchrun` flow. The local
adapter imports the released architecture directly and runs it as a normal
PyTorch module on Windows/CUDA.

Local compatibility patch: `third_party/DeMoE-main/archs/DeMoE.py` now calls
`F.softmax(class_weights_0, dim=1)` instead of relying on PyTorch's deprecated
implicit dimension selection. This preserves the intended class-router softmax
and removes a runtime warning on the current CUDA PyTorch build.

## Weights

Official DeMoE comparisons require the released checkpoint. The adapter will not
silently run random weights.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demoe-extra.txt
New-Item -ItemType Directory -Force weights\demoe
# Download DeMoE.pt from the official repository's OneDrive link.
# Place it at:
# weights\demoe\DeMoE.pt
```

Use `--demoe-smoke` only to test wiring. Smoke results must never be reported as
baseline numbers.

## Scenario-to-Task Mapping

DeMoE supports automatic routing plus manual task labels. In this benchmark,
`--demoe-task auto` uses DeMoE's own router. `--demoe-task scenario` maps the
known degradation scenario to the corresponding manual DeMoE task:

| local scenario family | DeMoE task |
|---|---|
| `defocus_*` | `defocus` |
| `lowlight_*`, `low_light_*` | `low_light` |
| `motion_*`, `vibration_*`, `shake_*` | `synth_global_motion` |
| `native_*`, `real_*`, KITTI-style controlled real-drive blur | `global_motion` |
| unknown/mixed | `auto` |

For the main paper, run both:

1. `--demoe-task auto`, measuring DeMoE as a complete all-in-one system.
2. `--demoe-task scenario`, measuring restoration quality when blur type is
   known. This mirrors the metadata question in RMR-Net: how much benefit comes
   from knowing the degradation?

## Benchmark Commands

Full-reference restoration:

```powershell
.\.venv\Scripts\python.exe benchmark_unified_restoration.py --data-root data\pothole_restoration --scenario motion_horizontal_medium --scenario defocus_medium --scenario lowlight_medium --model rcadnet --model dfpir --model nafnet --model demoe --rcadnet-weights runs\best_rmrnet\best.pth --dfpir-weights weights\dfpir\latest.pth.tar --nafnet-weights runs\nafnet_road\nafnet_last.pth --demoe-weights weights\demoe\DeMoE.pt --demoe-task auto --device cuda --out runs\unified_with_demoe_auto
```

Downstream YOLO detection:

```powershell
.\.venv\Scripts\python.exe tools\restore_yolo_split.py --data datasets\pcm_yolo_motion_test\data.yaml --split val --model demoe --scenario motion_horizontal_medium --demoe-weights weights\demoe\DeMoE.pt --demoe-task auto --out datasets\pcm_yolo_motion_test_demoe_auto --device cuda
.\.venv\Scripts\python.exe tools\eval_yolo_suite.py --weights yolo11s.pt --item demoe=datasets\pcm_yolo_motion_test_demoe_auto\data.yaml --imgsz 640 --device cuda --out runs\demoe_yolo_eval
```

## What DeMoE Changes In Our Experimental Story

DeMoE is a good reminder that reviewers will ask whether a conditional road
restorer is being compared against modern all-in-one deblurring rather than only
older task-specific baselines. It also gives us two useful protocol additions:

- **Router audit:** report automatic versus manual task routing for DeMoE, and
  metadata versus image-only routing for RMR-Net.
- **Out-of-distribution audit:** include native blur and held-out road-source
  tests, because an all-in-one method can look strong on known degradation types
  but still fail when the blur distribution shifts.

The narrative should stay precise: DeMoE is an external all-in-one deblurring
baseline; RMR-Net is a road-damage, metadata-aware, detector-and-boundary-driven
restoration pipeline. DeMoE can challenge the restoration side of the claim,
while the YOLO and active-contour evaluations test whether restored images are
useful for ITS perception.
