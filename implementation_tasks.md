# Implementation Roadmap

This document outlines the pending development tasks required to complete the `heritageshm` framework as described in the `README.md`.

## 1. Core Library Implementation

### `heritageshm/imputation.py`
The current implementation is only a placeholder. We need to implement the benchmark suite for gap-filling models.
- [ ] Implement **Gaussian Process Regression (GPR)** wrapper.
- [ ] Implement **Bayesian Dynamic Linear Models (BDLM)** wrapper.
- [ ] Implement **attention-based BiLSTM** architecture and training wrapper.
- [ ] Implement **XGBoost** regression wrapper for time-series imputation.
- [ ] Create a unified `ImputationBenchmark` class to run and compare these models.

### `heritageshm/decomposition.py`
- [ ] Integrate `NeuralProphet` to handle the grey-box decomposition (Trend + Seasonality + Regressors).
- [ ] Implement logic for extracting and saving the structural residuals (the "innovation" component).
- [ ] Ensure seamless integration with the `features.py` generated lags.

### `heritageshm/monitoring.py`
- [ ] Implement **EWMA (Exponentially Weighted Moving Average)** control chart logic.
- [ ] Implement **CUSUM (Cumulative Sum)** control chart logic.
- [ ] Add alerting/thresholding logic to trigger alarms when residuals exceed control limits.

## 2. Pipeline (Jupyter Notebooks) Implementation

### `00_Sensor_Preprocessing.ipynb`
- [ ] Verify end-to-end execution with current `data/raw/sensor` structure.
- [ ] Refine cleaning parameters for different sensor types.

### `01_Data_Quality_and_Gaps.ipynb`
- [ ] Ensure alignment logic handles varying proxy frequencies (e.g., hourly vs daily).
- [ ] Validate gap taxonomy output accuracy.

### `02_Proxy_Validation_and_Lags.ipynb`
- [ ] Ensure feature matrix generation and saving as `.parquet` is robust.
- [ ] Verify cross-correlation plotting for large proxy sets.

### `03_Imputation_Benchmark.ipynb`
- [ ] Implement the experimental loop: Load feature matrix $\rightarrow$ Run Imputation Models $\rightarrow$ Evaluate Error.
- [ ] Implement error metric calculation (RMSE, MAE, MSE) across all models.
- [ ] Generate comparison plots (e.im. error distribution by model).
- [ ] Save benchmark results to `outputs/tables/`.

### `04_GreyBox_Decomposition_and_Monitoring.ipynb`
- [ ] Implement the training pipeline: Load Features $\rightarrow$ Fit `NeuralProphet` $\rightarrow$ Extract Residuals.
- [ ] Implement the monitoring loop: Apply EWMA/CUSUM $\rightarrow$ Identify Anomalies.
- [ ] Generate final monitoring dashboards (plots of residuals vs control limits).

## 3. Examples & Documentation

### Examples & Validation
- [ ] Create the `/examples` directory.
- [ ] Prepare and add the **Gubbio dataset** (as mentioned in README) as a primary validation case.
- [ ] Create an example Jupyter Notebook in `/examples` demonstrating the end-to-end pipeline.

### Configuration and Documentation
- [ ] Finalize `environment.yml` for complete reproducibility.
- [ ] Ensure `README.md` accurately reflects the final directory structure and usage.
