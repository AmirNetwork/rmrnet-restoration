# TRACE-R reproducibility guide

Author: Amir Ghorbani (`amir.ghorbani@rmit.edu.au`)

This guide separates executable code, third-party assets, and retained
evidence. TRACE-R does not redistribute datasets or third-party checkpoints.

## 1. Environment

Create the Windows CUDA environment from the repository root:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-windows-gpu.txt
pip install -r requirements-experiments.txt
pip install -r requirements-detection-extra.txt
pip install -r requirements-demoe-extra.txt
pip install -r requirements-dfpir-extra.txt
```

Verify CUDA before any timing or model run:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Third-party projects

Place the official repositories under `third_party/`:

- `third_party/DeMoE-main`
- `third_party/DFPIR-main`
- `third_party/InstructIR-main`

Obtain pretrained files from their official releases. Upstream licenses and
links are listed in `THIRD_PARTY_LICENSES.md`.

## 3. Controlled data

Prepare source/sequence-disjoint IVCNZ and PCM folders with the project data
builders. The reported training roots are:

- `data/pothole_restoration_practical_sensor_calibrated_v2_train`
- `data/pcm_restoration_practical_sensor_calibrated_v2_train`

The validation roots use the matching
`*_practical_sensor_calibrated_v2_<cause>` names for motion, defocus, low light,
and mixed motion-low-light. Public sidecars contain only the 82-value observable
packet. Exact synthetic renderer values can create training supervision but are
not router inputs.

## 4. Matched training

Use `tools/run_matched_training_suite.py` or invoke
`train_matched_restorer.py` for each baseline with the same data roots,
detectors, seed, and budget. The final continuation has 70 epochs and 4,096
optimizer steps per restorer. Its common objective is:

`L = L_charbonnier + 0.15 L_gradient + 0.20 L_TDP + 0.10 L_detector`.

Do not select checkpoints on a test split. The reported expert hashes are in
`configs/trace_r_journal_release.json`.

## 5. TRACE-R validation

```bat
python tools\validate_tracer_expert_fusion.py ^
  --control correct ^
  --control unavailable ^
  --control cross_condition_shuffled ^
  --out E:\TRACE_R_experiments\controlled_validation
```

The script rejects test-labelled inputs, hashes every checkpoint, records route
counts, and resumes completed stages. Selection is the unweighted mean of IVCNZ
and PCM validation mAP50.

## 6. CRID field policy

CRID keeps the native 4752 x 3168 images, synchronized Sony EXIF, and 200 Hz
SBG measurements. The 12-frame earlier temporal block fixes detector operating
points and the native/restored policy. The later 13-frame block is supportive,
not a newly sealed test. `models/tracer_detection_policy.py` implements the
label-free policy without reading annotations.

## 7. Paper assets

`tools/build_tracer_journal_assets.py` refuses a ledger unless its status is
`FROZEN_VALIDATION_ONLY` and `test_split_used` is false. It performs no
training, inference, model selection, or metric optimization.

```bat
python tools\build_tracer_journal_assets.py
```

The historical `rmrp` ledger key and some filenames are intentionally retained
to preserve hashes from the executed earlier-study pipeline. The public class
and CLI name for the journal method is TRACE-R.
