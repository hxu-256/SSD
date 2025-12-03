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
-	Check configurable parameters (data paths, model hyperparameters). 
-	The MATLAB test `test_PSRN_SSIM_SAM.m` requires MATLAB and reads/writes `.mat` files in `Results/`.

