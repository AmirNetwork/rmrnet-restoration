# Data and Weights

The GitHub repository contains code, configs, compact result artifacts, and the small frozen YOLO26 field-detector checkpoint used by the G46 coordinate-wrapper audit. Full datasets, restored image outputs, and large experiment folders are not committed because of size and licensing.

## Controlled Restoration Data

Expected paired-restoration format:

```text
data_root/
  scenarios/
    scenario_name/
      input/
      gt/
```

Each `input` image must have a clean counterpart in `gt` with the same filename.

## IVCNZ and PCM Splits

The paper uses controlled road-damage restoration splits derived from the local IVCNZ-style pothole data and PCM road-damage data. Scripts in `tools/` prepare YOLO-format detector splits and paired restoration folders:

- `tools/prepare_pothole_yolo.py`
- `tools/prepare_pcm_yolo.py`
- `tools/make_degraded_yolo_split.py`
- `tools/make_synthetic_restoration_data.py`

## RDD2022

RDD2022 is not bundled. Use:

```powershell
python tools\prepare_rdd2022_yolo.py --help
```

Then download the dataset from its official source and follow the generated split instructions.

## G46 Sony Native Field Audit

The G46 audit expects:

- original Sony `cam1` native images;
- Roboflow COCO labels mapped back to original filenames;
- `precise_cam1_coords.csv` containing pose/geotag metadata;
- `Yolo26_coordinate/YOLO26s_RDD_FRDC_Distilled_v2.pt`.

The repository includes the coordinate detector wrapper and checkpoint used in the local paper run. If you prefer to download the checkpoint directly:

```powershell
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='TamAko783/YOLO26s_RDD_FRDC_Distilled_v2', filename='YOLO26s_RDD_FRDC_Distilled_v2.pt', local_dir='Yolo26_coordinate')"
```

## Baseline Weights

Third-party restoration baselines are not redistributed here unless their files are already small and local. Use each official project for weights and licensing:

- DFPIR: https://github.com/TxpHome/DFPIR
- DeMoE: https://github.com/cidautai/DeMoE
- NAFNet: https://github.com/megvii-research/NAFNet
- InstructIR: official project/model release where available

Place downloaded weights in a local `weights/` folder or pass explicit paths to the scripts.
