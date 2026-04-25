# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and other AI coding agents when working with code in this repository.

## Overview
This repository implements a **Physics-Informed Grey-Box Framework for Static Structural Health Monitoring (SHM)**. It provides a transferable Python methodology for processing structural sensor data (e.g., inclinometers, strain gauges) and environmental proxies (e.g., ERA5-Land skin temperature, solar radiation) to detect structural anomalies using interpretable machine learning. The `heritageshm` library is not installed as a package — it is imported from the repo root via `sys.path` manipulation in every notebook (see Import Pattern below).

---

## Development Commands

### Environment Setup
All development must be performed within the Conda environment specified in `environment.yml`. The environment name is `neuralprophet_env`.

```bash
# Create from scratch
conda env create -f environment.yml
conda activate neuralprophet_env

# Update an existing environment after changes to environment.yml
conda activate neuralprophet_env
conda env update -f environment.yml --prune
```

Key packages in the environment: `neuralprophet` (via pip), `xgboost`, `scikit-learn`, `statsmodels`, `jupytext`, `watchdog`, and the full scientific Python stack (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`).

### Running the Pipeline
The analysis is driven by a sequential Jupyter Notebook pipeline. Execute the following notebooks in order:

1. `00_Sensor_Preprocessing.ipynb` — Raw data extraction and cleaning.
2. `01_Data_Quality_and_Gaps.ipynb` — Proxy alignment and gap characterization.
3. `02_Proxy_Validation_and_Lags.ipynb` — Thermal lag screening and feature engineering.
4. `03_Imputation_Benchmark.ipynb` — XGBoost virtual sensing and uncertainty quantification.
5. `04_GreyBox_Decomposition_and_Monitoring.ipynb` — NeuralProphet decomposition and anomaly detection.

To launch the interactive environment:
```bash
jupyter lab
```

---

## Jupytext Pairing and the Auto-Watcher

### The Pairing Rule (CRITICAL)
Every Jupyter Notebook (`.ipynb`) is strictly paired with a human-readable Python script (`.py`) in `py:percent` format, configured globally in `jupytext.toml`.

**NEVER edit `.ipynb` files directly.** The underlying JSON structure is heavily prone to corruption during automated edits. Always perform all reading, reasoning, and modifications on the paired `.py` file. The user handles syncing `.py` edits back to `.ipynb` locally.

Paired files:
- `00_Sensor_Preprocessing.ipynb` ↔ `00_Sensor_Preprocessing.py`
- `01_Data_Quality_and_Gaps.ipynb` ↔ `01_Data_Quality_and_Gaps.py`
- `02_Proxy_Validation_and_Lags.ipynb` ↔ `02_Proxy_Validation_and_Lags.py`
- `03_Imputation_Benchmark.ipynb` ↔ `03_Imputation_Benchmark.py`
- `04_GreyBox_Decomposition_and_Monitoring.ipynb` ↔ `04_GreyBox_Decomposition_and_Monitoring.py`

### Auto-Watcher (`auto_watcher.py`)
The repo includes `auto_watcher.py`, a file-system watcher that automatically runs `jupytext --sync` whenever a `.py` or `.ipynb` file is saved, keeping pairs in sync without manual intervention. It requires the `watchdog` package.

To start the watcher (in a dedicated terminal, before editing notebooks):
```powershell
# Windows — activate environment first
& "C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1"
conda activate neuralprophet_env
python auto_watcher.py
```

Leave this terminal running in the background during the session. See `auto_watcher instructions.md` for full details.

---

## Architecture

### Core Library (`heritageshm/`)
The `heritageshm` package contains the functional logic. **Changes to any module affect all notebooks** — never modify function signatures without updating all call sites in the `.py` notebook files.

| Module | Role | Key outputs |
|---|---|---|
| `dataloader.py` | Ingestion of `.adc` sensor files and `.csv` proxy files; saving interim data | Pandas DataFrames |
| `preprocessing.py` | Signal cleaning, resampling, and multi-proxy alignment onto the sensor index | Aligned DataFrame |
| `diagnostics.py` | Gap taxonomy (MCAR/MAR/MNAR), ADF, Engle-Granger cointegration, Ljung-Box tests; gap histogram figure and gap statistics table | Figure `.png`, stats `.csv` |
| `features.py` | Physically motivated thermal inertia lagged-feature generation | Feature DataFrame |
| `imputation.py` | Wrappers for benchmark gap-filling models (XGBoost virtual sensing with conformal bootstrap) | Imputed series, uncertainty bounds |
| `decomposition.py` | NeuralProphet grey-box decomposition: trend, seasonality, AR memory, exogenous regressors | Decomposed DataFrame, model object |
| `monitoring.py` | EWMA and CUSUM control chart implementation for residual-based anomaly detection | Control chart figure, alarm table |
| `viz.py` | Seaborn/Matplotlib visualization utilities; `apply_theme()` for consistent plot styling | Figure objects |

### Import Pattern
The `heritageshm` library is not installed as a package. Every notebook and standalone script must include the following path injection at the top before importing from the library:

```python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from heritageshm.dataloader import load_preprocessed_sensor
# ... other imports
```

When generating new notebook cells or scripts, always include this block.

### Data Flow
1. **Raw Data**: `data/raw/sensor/` (`.adc` files) and `data/raw/proxies/` (`.csv` files).
2. **Interim Data**: Cleaned/aligned datasets in `data/interim/sensor/` and `data/interim/aligned/`.
3. **Processed Data**: Final feature matrices and imputed series in `data/processed/`.
4. **Outputs**: Plots in `outputs/figures/`, metrics in `outputs/tables/`, models in `outputs/models/`.

> **Important:** `data/` and `outputs/` are gitignored and do not exist in the cloned repository. They must be created locally and populated with data before running any notebook. A `FileNotFoundError` on these paths is expected behaviour on a fresh clone — it is not a code bug.

### Output Artifact Naming Convention
All saved figures and tables follow a consistent naming pattern:
```
{notebook_id}_{artifact_id}_{station}_{description}.{ext}
```
Examples: `01_01_st02_gap_histogram.png`, `03_02_st02_imputation_metrics.csv`.

When generating new output cells, follow this convention. The station identifier is controlled by the `TARGET_STATION` parameter at the top of each notebook.

### Primary User Parameter: `TARGET_STATION`
All notebooks are parameterized around `TARGET_STATION` (e.g., `'st02'`). This string controls which sensor file is loaded and which output files are named. It is always defined near the top of each notebook as a clearly marked user input. When modifying notebooks, never hardcode a station identifier — always reference `TARGET_STATION`.
