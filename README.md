# Physics-Informed Grey-Box Framework for Static SHM
**A transferable Python methodology for static Structural Health Monitoring (SHM) under incomplete data.**

This repository contains the `heritageshm` Python library and a guided Jupyter Notebook pipeline. It provides a complete workflow for processing static structural monitoring data (e.g., inclinometers, strain gauges, displacement sensors), handling missing data outages, and performing residual-based anomaly detection using interpretable machine learning.

*Note: Documented examples and validation datasets (e.g., the medieval urban walls of Gubbio) will be added to an `/examples` directory in future updates.*

---

## 📖 Methodology Overview
This project presents a generic, two-phase workflow that moves from raw sensor cleaning to operational monitoring using a grey-box paradigm.

**Phase A: Historical Model Building**
1. **Data Preprocessing & Gap Diagnosis:** Synchronization of on-site structural sensors with external environmental proxy data (e.g., satellite reanalysis or local weather stations). Includes gap taxonomy classification (MCAR, MAR, MNAR).
2. **Physics-Informed Regressor Selection:** Engle-Granger cointegration testing to validate the long-run equilibrium between structural response and environmental proxies, followed by the generation of thermal-lag features.
3. **Compact Imputation Benchmark:** Evaluation of baseline and advanced gap-filling models (Gaussian Process Regression, Bayesian Dynamic Linear Models, attention-based BiLSTM, and XGBoost) to reconstruct contiguous data outages while preserving structural interpretability.

**Phase B: Operational Monitoring**
4. **Grey-Box Decomposition:** Separation of structural trend, environmental seasonality, and structural residuals using `NeuralProphet` (combining autoregressive memory with exogenous environmental regressors).
5. **Residual-Based Anomaly Detection:** Application of EWMA and CUSUM control charts on strictly stationary structural residuals to trigger operational alarms, effectively suppressing environmentally driven false positives.

---

## 📂 Project Structure

```text
/heritage_shm_project
│
├── /data                    # Unprocessed, interim, and fully processed datasets
│   ├── /raw                 # Expected location for raw files (ignored by Git)
│   │   ├── /sensor          # Raw sensor files (e.g., .adc)
│   │   └── /proxies         # Environment proxies in .csv format (e.g., ERA5)
│   ├── /interim             # Intermediate cached data
│   │   ├── /sensor          # Cleaned and standardized sensor DataFrames
│   │   └── /aligned         # Synchronized datasets (sensor + proxies)
│   └── /processed           # Imputed and decomposed time series
│
├── /outputs                 # Generated artifacts (ignored by Git)
│   ├── /figures             # High-res output plots
│   ├── /tables              # Exported CSV metrics and tables
│   └── /models              # Saved model weights
│
├── /heritageshm             # Core Python Library
│   ├── dataloader.py        # I/O operations
│   ├── preprocessing.py     # Alignment and spline upsampling
│   ├── diagnostics.py       # Gap taxonomy and cointegration testing
│   ├── features.py          # Thermal inertia feature generation
│   ├── imputation.py        # Benchmark wrappers
│   ├── decomposition.py     # NeuralProphet configuration
│   ├── monitoring.py        # EWMA/CUSUM control logic
│   └── viz.py               # Seaborn/matplotlib wrappers for visualization
│
├── 00_Sensor_Preprocessing.ipynb
├── 01_Data_Quality_and_Gaps.ipynb
├── 02_Proxy_Validation_and_Lags.ipynb
├── 03_Imputation_Benchmark.ipynb
└── 04_GreyBox_Decomposition_and_Monitoring.ipynb
```

---

## 🚀 Installation & Execution

### 1. Environment Setup
This project uses Conda for environment management to ensure reproducibility. 
```bash
# Create the environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate heritageshm_env
```

### 2. Supplying the Data
Due to size and privacy limits, the `/data/` and `/outputs/` directories are tracked locally but ignored by Git. To run this pipeline:
1. Place your target static sensor data into `/data/raw/sensor/`.
2. Place your environmental proxy data into `/data/raw/proxies/` as CSV files.

### 3. Running the Pipeline
The methodology is intended to be executed sequentially via the provided Jupyter Notebooks. The UI-agnostic design of the `heritageshm` library allows these tools to be easily integrated into standalone dashboard interfaces in the future.

Launch the interactive environment:
```bash
python -m jupyterlab
```
Execute the notebooks in order from `00` to `04` to replicate the full analytical workflow.