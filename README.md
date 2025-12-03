# SSD (Raman Imaging)

Overview
-	This repository contains Python code for the SSD project used in Raman imaging experiments. It includes training/evaluation scripts, network definitions, utility functions, and sample datasets (MAT files). The repository is organized to support experiments on both simulated and real data.

Repository structure
-	`func.py` : utility functions and helpers used across scripts.
-	`model.py` : model wrapper and entry points for building/initializing networks.
-	`metric.py` : image quality metrics (SSIM / PSNR, etc.).
-	`main_RealData.py` : runner script to run experiments on real data (loads `Data/Real_Data`).
-	`main_SimuData.py` : runner script to run experiments on simulated data (loads `Data/Simu_Data`).
-	`Networks/` : network architecture modules (`Network.py`, `basicblock.py`, `common.py`).
-	`Results/` : output folder where trained models, logs, and result images are saved.
-	`Data/` : contains `.mat` files used by the project. Do not commit large datasets to the repo; store them externally if needed.
-	`test_PSRN_SSIM_SAM.m` : MATLAB script for additional testing/metrics.

Requirements
-	Python 3.8+ recommended (Windows tested). 
-	Common Python packages used by this repository (install with pip):

```
pip install numpy scipy matplotlib h5py scikit-image
```

- If the project uses PyTorch, install a matching version for your CUDA/Python configuration. Example (CPU-only):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- If you use conda, create an env:

```
conda create -n ssd_env python=3.9
conda activate ssd_env
pip install -r requirements.txt  # if you create one
```

Data
-	Place real or simulated `.mat` files inside `Data/Real_Data/` or `Data/Simu_Data/` as required by the main scripts. Example files included: `Cell_Paramecium.mat`, `Chessboard_data.mat`.

How to run
-	Run the real-data pipeline (PowerShell):

```powershell
cd "\SSD"
python main_RealData.py
```

-	Run the simulated-data pipeline:

```powershell
python main_SimuData.py
```

Notes
-	Check `func.py` and `model.py` for configurable parameters (data paths, model hyperparameters). Modify them or add a simple CLI wrapper to pass parameters at runtime.
-	The MATLAB test `test_PSRN_SSIM_SAM.m` requires MATLAB and reads/writes `.mat` files in `Results/`.

Suggestions and next steps
-	Add a `requirements.txt` created from your environment (run `pip freeze > requirements.txt`).
-	Add usage examples showing expected input shapes and sample outputs (images) inside `Results/`.
-	Add a short CONTRIBUTING or developer notes file if you expect others to run/extend experiments.

Contact
-	If you want changes or more details (e.g., example outputs, CLI flags, or adding a `requirements.txt`), tell me and I will update the README.

License
-	Add a license file if you plan to share publicly. Currently no license is included.
