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

---

## Coding Conventions

### Notebook Structure Standard (Notebooks 01–04)

Notebooks 01 through 04 form the **transferable core pipeline**. They are designed to work with any static structural sensor dataset and any compatible environmental proxy dataset — not only the Gubbio case. When writing or modifying these notebooks, preserve this generality: never hardcode site-specific values, units, or assumptions.

Every notebook in the 01–04 sequence must follow this internal structure:

**1. Title cell (Markdown)**
The first cell must be a Markdown title block containing:
- The notebook number and descriptive title as an H1 heading.
- A one-paragraph statement of the notebook's goal within the overall pipeline.
- A numbered list of the steps the notebook executes, matching the step headers used throughout.

Example pattern (from Notebook 01):
```markdown
# Notebook 01 · Data Quality, Proxies, and Gap Characterization

This notebook executes **Phase A, Step 1** of the `heritageshm` pipeline:

1. **Sensor Loading** — Load the preprocessed sensor CSV from Notebook 00.
2. **Proxy Loading** — Load environmental proxy data and select relevant columns.
3. **Alignment** — Resample and synchronize proxies onto the sensor index.
4. **Gap Characterization** — Classify missing data and diagnose gap taxonomy.
5. **Save** — Export the aligned dataset to `/data/interim/aligned/`.
```

**2. Imports cell (Code)**
One code cell containing all imports and `apply_theme()`. No logic, no parameters.

**3. Step cells (alternating Markdown + Code)**
Each step consists of:
- A Markdown header cell (`## Step N · Name`) containing:
  - A plain-language description of what the step does.
  - A **Parameter Tuning Guidance** subsection (`### Parameter Tuning Guidance`) that documents every user-facing parameter in the following code cell: its name, purpose, accepted values, default, and effect on downstream steps.
- One or more code cells implementing the step by calling library functions. Complex logic must not be written inline — it belongs in a module.

**4. Save / export cell**
The final step must always save all outputs required by the next notebook. Saves must be gated on a success boolean (e.g., `if step_ok: save(...)`) so that partial failures do not silently produce corrupt files.

**5. No inline complex logic**
Notebook cells must not contain multi-function algorithms, statistical tests, or model fitting code written directly in the cell. All such logic must be encapsulated in a function in the appropriate `heritageshm/` module and called from the notebook. A cell should read like an orchestration script, not an implementation.

---

### Notebook 00 — Special Status

`00_Sensor_Preprocessing.ipynb` is intentionally **ad hoc** and site-specific. It encodes the particular raw data format, column layout, file extension, and compensation coefficients of the author's on-site inclinometer system (`.adc` files, tab-separated, temperature-compensation coefficient). It is not expected to be transferable without modification. When working on Notebook 00:
- Do not attempt to generalise it to match the 01–04 pattern.
- Parameters such as `COMP_COEFF`, `STATIONS`, `SEPARATOR`, and `FILE_EXT` are site-specific and must be updated by the user for a new deployment.
- Its output (`{station}_preprocessed.csv` in `data/interim/sensor/`) is the standard entry point for the transferable pipeline starting at Notebook 01.

---

### Expected Dataset Contracts (Notebooks 01–04)

The transferable pipeline (01–04) expects two input datasets with the following structure. Any new sensor or proxy dataset must conform to these contracts before being fed into the pipeline.

**Sensor dataset** (`data/interim/sensor/{station}_preprocessed.csv`)
- Produced by Notebook 00 (or any equivalent preprocessing step).
- A CSV file with a `datetime` column parseable as a `DatetimeIndex`.
- At minimum one structural response column (e.g., `absinc` for absolute inclination). Column name is user-configurable via `TARGET_COL`.
- Regular or near-regular time steps (gaps allowed; the pipeline handles them). Typical resolution: hourly or sub-hourly.
- Units: SI or consistent engineering units. No requirement for specific units, but units must be consistent across the full series.
- No pre-imputation expected: the pipeline assumes this file contains NaN where data is missing.

**Proxy dataset** (`data/raw/proxies/{name}.csv`)
- A CSV file with a datetime column named `datetime (UTC)`, parseable as a `DatetimeIndex`.
- One or more environmental variable columns. Column names and units are user-configurable via `PROXY_COLS`.
- Must cover the full temporal window of the sensor dataset (checked explicitly in Notebook 01 before alignment).
- Typical sources: ERA5-Land reanalysis (skin temperature, solar radiation), local weather station exports, or any reanalysis provider (e.g., Oikolab). The pipeline does not assume a specific source.
- Resolution: hourly or finer. The pipeline resamples to `TARGET_FREQ` during alignment.
- Metadata columns (coordinates, model name, elevation, UTC offset) are automatically dropped during loading in Notebook 01 if listed in `META_COLS`.

---

### Module Coding Standard (`heritageshm/`)

**Module-level docstring**
Every `.py` file in `heritageshm/` must begin with a module-level docstring that states: (1) the module name, (2) its responsibility within the pipeline, and (3) its primary inputs and outputs. Example:

```python
"""
Module: diagnostics.py
Handles gap taxonomy characterization, cointegration testing,
and residual diagnostics (ADF, Ljung-Box).
"""
```

**Function-level docstrings (mandatory)**
Every public function must have a NumPy-style or Google-style docstring containing:
- A one-line summary of what the function does.
- `Parameters` section: name, type, description, and default for every argument.
- `Returns` section: type and description of every return value.
- Any side effects (e.g., saves a file, prints to stdout) must be noted.

Example pattern (already used in `diagnostics.py`):
```python
def test_cointegration(df, target_col, proxy_col, alpha=0.05):
    """
    Performs the Engle-Granger two-step cointegration test to validate
    the physical long-run equilibrium between the structural response and the proxy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both variables.
    target_col : str
        The structural sensor column (e.g., 'absinc').
    proxy_col : str
        The environmental proxy column (e.g., 'skin_temperature (degC)').
    alpha : float, optional
        Significance level for the test. Default 0.05.

    Returns
    -------
    is_cointegrated : bool
        True if the null hypothesis of no cointegration is rejected.
    p_value : float
        P-value from the Engle-Granger test.
    """
```

**No logic in notebooks**
If a block of code in a notebook cell is longer than ~10 lines, performs a statistical test, trains a model, or produces a figure, it must be refactored into a named function in the appropriate module. The notebook cell then becomes a single function call with named arguments.

**No hardcoded paths in modules**
Module functions must never hardcode file paths. All paths must be passed as arguments. Modules are path-agnostic; notebooks control all I/O paths.

**No side effects without explicit opt-in**
Functions that save files or produce plots must have an optional parameter (e.g., `save_plot_path=None`) that defaults to no side effect. The notebook controls whether outputs are saved.
