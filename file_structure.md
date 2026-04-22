# Project File Structure

This file provides an overview of the repository's components and their purposes.

## Core Pipeline (Jupyter Notebooks)
- `00_Sensor_Preprocessing.ipynb`: Initial sensor data cleaning and synchronization.
- `01_Data_Quality_and_Gaps.ipynb`: Identification and classification of data gaps.
- `02_Proxy_Validation_and_Lags.ipynb`: Validation of environmental proxies and thermal lag features.
- `03_Imputation_Benchmark.ipynb`: Benchmarking gap-filling models.
- `04_GreyBox_Decomposition_and_Monitoring.ipynb`: Structural decomposition and anomaly detection.

## Core Library (`heritageshm`)
- `heritageshm/`: The main Python package for the SHM framework.
    - `dataloader.py`: Data I/O and loading.
    - `preprocessing.py`: Alignment and upsampling.
    - `diagnostics.py`: Gap taxonomy and cointegration.
    - `features.py`: Feature engineering (thermal inertia).
    - `imputation.py`: Imputation model wrappers.
    - `decomposition.py`: NeuralProphet-based decomposition logic.
    - `monitoring.py`: Control chart-based anomaly detection (EWMA/CUSUM).
    - `viz.py`: Visualization utilities.

## Data and Outputs
- `data/`: Storage for raw, interim, and processed datasets.
- `outputs/`: Directory for generated figures, tables, and trained models.
- `auxiliary/`: Supplementary datasets and secondary analysis notebooks.

## Configuration and Documentation
- `README.md`: Project documentation and overview.
- `LICENSE`: Legal license information.
- `environment.yml`: Conda environment specification.
