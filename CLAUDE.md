# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This repository implements a **Physics-Informed Grey-Box Framework for Static Structural Health Monitoring (SHM)**. It provides a methodology for processing sensor data (e.g., inclinometers, strain gauges) and environmental proxies (e.g., temperature, radiation) to detect structural anomalies using interpretable machine learning.

## Development Commands

### Environment Setup
All development should be performed within the Conda environment specified in `environment.yml`.
```bash
conda env create -f environment.yml
conda activate heritageshm_env
```

### Running the Pipeline
The analysis is driven by a sequential Jupyter Notebook pipeline. To run the full workflow, execute the following notebooks in order:
1. `00_Sensor_Preprocessing.ipynb` - Raw data extraction and cleaning.
2. `01_Data_Quality_and_Gaps.ipynb` - Proxy alignment and gap characterization.
3. `02_Proxy_Validation_and_Lags.ipynb` - Thermal lag screening and feature engineering.
4. `03_Imputation_Benchmark.ipynb` - Comparison of gap-filling models (GPR, BDLM, BiLSTM, XGBoost).
5. `04_GreyBox_shutdown_and_Monitoring.ipynb` - NeuralProphet decomposition and anomaly detection.

To launch the interactive environment:
```bash
python -m jupyterlab
```

## Architecture

### Core Library (`heritageshm/`)
The `heritageshm` package contains the functional logic:
- `dataloader.py`: Handles ingestion of `.adc` sensor files and `.csv` proxy files.
- `preprocessing.py`: Implements signal cleaning, alignment, and temperature compensation.
- `diagnostics.py`: Performs gap taxonomy (MCAR/MAR/MNAR) and statistical tests (Cointegration, ADF, Ljung-Box).
- `features.py`: Generates physically motivated lagged features (thermal inertia).
- `imputation.py`: Wrappers for advanced gap-filling models.
- `decomposition.py`: Wrappers for `NeuralProphet` grey-box decomposition.
- `monitoring.py`: Implementation of EWMA and CUSUM control charts.
- `viz.py`: Visualization utilities for SHM data.

### Data Flow
1. **Raw Data**: Located in `data/raw/sensor/` (.adc) and `data/raw/proxies/` (.csv).
2. **Interim Data**: Cleaned/aligned datasets stored in `data/interim/sensor/` and `data/interim/aligned/`.
3. **Processed Data**: Final feature matrices and imputed series in `data/processed/`.
4. **Outputs**: Plots in `outputs/figures/`, metrics in `outputs/tables/`, and models in `outputs/models/`.

*Note: `data/` and `outputs/` are large and should not be committed to Git.*

## Code Style and Conventions

### Paired .py Files
Jupyter Notebooks are now paired with human-readable `.py` files using Jupytext. These `.py` files are the primary source for code review and editing. Always refer to these files when making changes to the pipeline logic.

Paired files:
- `00_Sensor_Preprocessing.ipynb` ↔ `00_Sensor_Preprocessing.py`
- `01_Data_Quality_and_Gaps.ipynb` ↔ `01_Data_Quality_and_Gaps.py`
- `02_Proxy_Validation_and_Lags.ipynb` ↔ `02_Proxy_Validation_and_Lags.py`
- `03_Imputation_Benchmark.ipynb` ↔ `03_Imputation_Benchmark.py`
- `04_GreyBox_shutdown_and_Monitoring.ipynb` ↔ `04_GreyBox_Decomposition_and_Monitoring.py`

When making changes, edit the `.py` files directly. The notebook synchronization will be handled manually later.
