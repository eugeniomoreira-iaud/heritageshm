# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: neuralprophet_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 00 · Sensor Preprocessing
#
# **Goal:** Extract, clean, and temperature-compensate raw on-site sensor data.
#
# **Steps:**
# 1. **File Inspection** — Examine one raw file before loading to confirm delimiter and column layout.
# 2. **Data Loading** — Read all raw sensor files from `/data/raw/sensor/`.
# 3. **Signal Cleaning and Compensation** — Remove power-loss rows, filter physical outliers,
#    and apply the on-board temperature compensation coefficient.
# 4. **Compensation Visualisation** — Plot raw vs. compensated signal and the applied correction.
# 5. **Save** — Export cleaned, standardized datasets to `/data/interim/sensor/`.

# %% [markdown]
# ## Import libraries

# %%
import sys
import os
import glob
sys.path.insert(0, os.path.abspath('..'))

from heritageshm.dataloader import inspect_raw_file, load_sensor_directory, organize_sensor_data
from heritageshm.preprocessing import process_station
from heritageshm.viz import apply_theme, plot_compensation_comparison

apply_theme(context='notebook')

# %% [markdown]
# ## Step 1 · File Inspection
#
# Inspect one representative raw file to confirm the delimiter, column count, and date format
# before loading the entire directory.

# %%
RAW_FOLDER   = 'data/raw/sensor'
FILE_EXT     = '.adc'

sample_files = glob.glob(os.path.join(RAW_FOLDER, f'*{FILE_EXT}'))
if sample_files:
    file_info = inspect_raw_file(sample_files[0])
else:
    print(f'No {FILE_EXT} files found in {RAW_FOLDER}. Check RAW_FOLDER and FILE_EXT.')
    file_info = {}

# %% [markdown]
# ## Step 2 · Load Raw Sensor Files
#
# Read all files in the raw sensor directory and organise them into per-station DataFrames.

# %%
SEPARATOR = '\t'
DECIMAL_COMMA = True
HEADER = None

STATIONS = {
    'st01': ['charge', 'temp', 'hum', 'absinc'],
    'st02': ['charge', 'temp', 'hum', 'absinc'],
    'st03': ['charge', 'temp', 'hum', 'absinc'],
}



df_raw = load_sensor_directory(
    folder_path=RAW_FOLDER,
    extension=FILE_EXT,
    sep=SEPARATOR,
    header=HEADER,
    column_names=None,
    date_col=0,
    time_col=1,
    save_combined=False,
)

print(f'\nLoaded dataset shape : {df_raw.shape}')
print(f'Date range : {df_raw.index.min()} → {df_raw.index.max()}')

stations_dict = organize_sensor_data(df_raw, STATIONS)
print(f'Organised {len(stations_dict)} station(s): {list(stations_dict.keys())}')

# %% [markdown]
# ## Step 3 · Signal Cleaning and Temperature Compensation
#
# For each station, `process_station()` executes three operations in sequence:
# 1. Removes rows where battery charge equals zero (power-loss events).
# 2. Drops readings outside the physical validity range defined in `OUTLIER_THRESHOLDS`.
# 3. Applies linear temperature compensation using `COMP_COEFF`.
#    The raw signal is preserved as `{SIGNAL_COL}_raw` in the saved CSV to support
#    later visualisation; only the compensated column is returned to the pipeline.

# %%
SIGNAL_COL       = 'absinc'
TEMP_COL         = 'temp'
COMP_COEFF       = 0.005      # mdeg · °C⁻¹ · 10⁻³ — update from calibration sheet
SPIKE_THRESHOLD  = 500.0       # mdeg — maximum allowed |Δ absinc| between samples
OUTPUT_DIR       = 'data/interim/sensor'


processed = {}

for st, df_st in stations_dict.items():
    df_clean, _ = process_station(
        st=st,
        df_st=df_st,
        signal_col=SIGNAL_COL,
        temp_col=TEMP_COL,
        comp_coeff=COMP_COEFF,
        spike_threshold=SPIKE_THRESHOLD,
        output_dir=OUTPUT_DIR,
    )
    processed[st] = df_clean

# %% [markdown]
# ## Step 4 · Compensation Visualisation
#
# `plot_compensation_comparison()` reads directly from the saved interim CSV so it can be
# called independently at any time — including outside this notebook — to inspect any station.
# Modify `VIZ_STATION` (or `viz_file` and `viz_col` below) to inspect a different station or column.

# %%
VIZ_STATION = 'st02'
viz_file    = os.path.join(OUTPUT_DIR, f'{VIZ_STATION}_preprocessed.csv')
viz_col     = SIGNAL_COL   # column to plot; change here if a different signal is needed

plot_compensation_comparison(
    file_path=viz_file,
    signal_col=viz_col,
    save_plot=True,
    save_path='outputs/figures',
    filename='00_compensation_comparison',
)

# %% [markdown]
# ## Step 5 · Dataset Preview
#
# Quick sanity check on the cleaned output for each station.

# %%
from IPython.display import display
for st, df_clean in processed.items():
    print(f'--- {st} ---  shape: {df_clean.shape}  |  '
          f'{df_clean.index.min()} → {df_clean.index.max()}')
    display(df_clean.head(3))
    print()
