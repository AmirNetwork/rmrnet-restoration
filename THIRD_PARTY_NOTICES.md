# Third-Party Notices

This repository contains integration code for several external projects. Their original licenses and citation requirements remain in force.

## YOLO26 Field Detector

The native-field detector checkpoint is `YOLO26s_RDD_FRDC_Distilled_v2.pt`, from:

https://huggingface.co/TamAko783/YOLO26s_RDD_FRDC_Distilled_v2

The manuscript cites the model card and describes it as a YOLO26s RDD-FRDC distilled checkpoint trained on unified RDD-style road-defect data with teacher distillation. Check the Hugging Face model card for its current license and usage terms before redistribution or deployment.

## Ultralytics YOLO

The detector training/evaluation scripts use Ultralytics APIs. See:

https://docs.ultralytics.com/

## Restoration Baselines

The repository includes adapters and scripts for external baselines. Use their official repositories for weights and license terms:

- DFPIR: https://github.com/TxpHome/DFPIR
- DeMoE: https://github.com/cidautai/DeMoE
- NAFNet: https://github.com/megvii-research/NAFNet

## Datasets

Datasets are not redistributed in this repository. Download them from their official sources and follow their licenses.
