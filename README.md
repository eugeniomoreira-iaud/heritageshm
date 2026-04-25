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
3. **Compact Imputation Benchmark:** Implementation of an XGBoost virtual sensing model with conformal bootstrap calibration to reconstruct contiguous data outages while preserving structural interpretability.

**Phase B: Operational Monitoring**
4. **Grey-Box Decomposition:** Separation of structural trend, environmental seasonality, and structural residuals using `NeuralProphet` (combining autoregressive memory with exogenous environmental regressors).
5. **Residual-Based Anomaly Detection:** Application of EWMA and CUSUM control charts on strictly stationary structural residuals to trigger operational alarms, effectively suppressing environmentally driven false positives.

---

## 📂 Project Structure

```text
heritageshm/                        # Repository root
│
├── heritageshm/                    # Core Python library
│   ├── dataloader.py               # I/O operations (.adc sensors, .csv proxies)
│   ├── preprocessing.py            # Alignment and spline upsampling
│   ├── diagnostics.py              # Gap taxonomy (MCAR/MAR/MNAR) and cointegration
│   ├── features.py                 # Thermal inertia lagged-feature generation
│   ├── imputation.py               # Benchmark imputation wrappers
│   ├── decomposition.py            # NeuralProphet configuration and wrappers
│   ├── monitoring.py               # EWMA/CUSUM control chart logic
│   └── viz.py                      # Seaborn/matplotlib visualization utilities
│
├── data/                           # Datasets — tracked locally, ignored by Git
│   ├── raw/
│   │   ├── sensor/                 # Raw sensor files (e.g., .adc)
│   │   └── proxies/                # Environmental proxy files (e.g., ERA5 CSV)
│   ├── interim/
│   │   ├── sensor/                 # Cleaned and standardized sensor DataFrames
│   │   └── aligned/                # Synchronized sensor + proxy datasets
│   └── processed/                  # Imputed and decomposed time series
│
├── outputs/                        # Generated artifacts — tracked locally, ignored by Git
│   ├── figures/                    # High-resolution output plots
│   ├── tables/                     # Exported CSV metrics and tables
│   └── models/                     # Saved model weights
│
├── auxiliary/                      # Supplementary datasets and secondary notebooks
│
├── 00_Sensor_Preprocessing.ipynb   # Notebook 00 — raw data extraction and cleaning
├── 00_Sensor_Preprocessing.py      # Jupytext paired source (edit this, not the .ipynb)
├── 01_Data_Quality_and_Gaps.ipynb
├── 01_Data_Quality_and_Gaps.py
├── 02_Proxy_Validation_and_Lags.ipynb
├── 02_Proxy_Validation_and_Lags.py
├── 03_Imputation_Benchmark.ipynb
├── 03_Imputation_Benchmark.py
├── 04_GreyBox_Decomposition_and_Monitoring.ipynb
├── 04_GreyBox_Decomposition_and_Monitoring.py
│
├── auto_watcher.py                 # Jupytext file watcher (auto-syncs .py → .ipynb)
├── auto_watcher instructions.md    # Instructions for using the auto-watcher
├── jupytext.toml                   # Jupytext pairing configuration
├── environment.yml                 # Conda environment specification
├── file_structure.md               # Extended file structure reference
├── CLAUDE.md                       # Instructions for AI coding agents
└── LICENSE
```

> **Note on Jupytext pairing:** Every `.ipynb` notebook has a paired `.py` file (in `py:percent` format). **Always edit the `.py` file**, never the `.ipynb` directly. Use `auto_watcher.py` or the Jupytext CLI to sync changes back to the notebook. See `auto_watcher instructions.md` for details.

---

## 🚀 Installation & Execution

### 1. Create the Conda Environment

The environment is fully specified in `environment.yml`. To create it from scratch:

```bash
conda env create -f environment.yml
conda activate neuralprophet_env
```

This installs all required dependencies, including `neuralprophet`, `xgboost`, `jupytext`, `statsmodels`, and the full scientific Python stack.

### 2. Update an Existing Environment

If the `environment.yml` has changed (e.g., new packages were added) and you want to sync your local environment without recreating it:

```bash
conda activate neuralprophet_env
conda env update -f environment.yml --prune
```

The `--prune` flag removes any packages that are no longer listed in the file, keeping the environment consistent with the specification.

To fully recreate the environment from scratch (e.g., after a major update or to resolve conflicts):

```bash
conda deactivate
conda env remove -n neuralprophet_env
conda env create -f environment.yml
conda activate neuralprophet_env
```

### 3. Export Your Current Environment

To save the exact state of your working environment (including resolved dependency versions) for reproducibility:

```bash
# Export full specification with exact versions (for exact reproduction)
conda env export -n neuralprophet_env > environment_frozen.yml

# Export only explicitly installed packages (portable across platforms)
conda env export -n neuralprophet_env --from-history > environment.yml
```

Commit `environment.yml` to Git to keep the specification up to date.

### 4. Supplying the Data

Due to size and privacy limits, the `data/` and `outputs/` directories are tracked locally but ignored by Git. To run this pipeline:

1. Place your target static sensor data into `data/raw/sensor/`.
2. Place your environmental proxy data into `data/raw/proxies/` as CSV files.

### 5. Running the Pipeline

The methodology is executed sequentially via the Jupyter Notebooks. Launch JupyterLab:

```bash
jupyter lab
```

Execute the notebooks in order from `00` to `04` to replicate the full analytical workflow:

| Notebook | Description |
|---|---|
| `00_Sensor_Preprocessing` | Raw data extraction, cleaning, and standardization |
| `01_Data_Quality_and_Gaps` | Proxy alignment and gap characterization (MCAR/MAR/MNAR) |
| `02_Proxy_Validation_and_Lags` | Cointegration testing and thermal-lag feature engineering |
| `03_Imputation_Benchmark` | XGBoost virtual sensing and uncertainty quantification |
| `04_GreyBox_Decomposition_and_Monitoring` | NeuralProphet decomposition and EWMA/CUSUM anomaly detection |
