# RMR-Net IEEE T-ITS Code Release Package

This archive contains the code paths used by the final IEEE T-ITS manuscript package:

- RMR-Net/RCADNet model code: models/, rcadnet/
- Training entry points: train_rcadnet.py, train_rmrnet.py
- Restoration/evaluation utilities: tools/
- Baseline wrappers/configs: baselines/, configs/
- Final G46 detector script and weight: Yolo26_coordinate/
- Final G46 paper-result CSVs: results/gt46_yolo26_coordinate_final/

Recommended Windows setup from the project root:

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-windows-gpu.txt
pip install ultralytics opencv-python pandas pillow matplotlib scikit-image scipy tqdm

Quick integrity check:

python -m compileall models rcadnet tools train_rcadnet.py train_rmrnet.py Yolo26_coordinate

Final G46 detector protocol:
Use Yolo26_coordinate/Yolo26_coordinate_revised.py with YOLO26s_RDD_FRDC_Distilled_v2.pt. The final paper reports the revised G46 native-resolution field test with no road crop, no G46 detector fine-tuning, no synthetic degradation, and original image resolution preserved.
