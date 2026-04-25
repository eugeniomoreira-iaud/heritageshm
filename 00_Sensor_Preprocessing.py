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
# 3. **Signal Cleaning and Compensation** — Remove power-loss rows, apply a gap-aware spike
#    filter (differences across NaN / zero-charge boundaries are excluded), and apply the
#    on-board temperature compensation coefficient.
# 4. **Compensation Visualisation** — Plot raw vs. compensated signal and the applied
#    correction, with dropped values highlighted.
# 5. **Save** — Export cleaned, standardized datasets to `/data/interim/sensor/`.

# %% [markdown]
# ## Import libraries

# %%
import sys
import os
import glob
from IPython.display import display
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
#
# ### Parameter Tuning Guidance
#
# | Parameter | Purpose | Accepted values | Default | Notes |
# |---|---|---|---|---|
# | `RAW_FOLDER` | Path to the directory containing raw sensor files | Any valid path string | `'data/raw/sensor'` | Relative to repo root |
# | `FILE_EXT` | Extension of the raw sensor files | Any string (e.g. `'.adc'`, `'.csv'`) | `'.adc'` | Must match exactly (case-sensitive on Linux) |

# %%
RAW_FOLDER = 'data/raw/sensor'
FILE_EXT   = '.adc'

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
#
# ### Parameter Tuning Guidance
#
# | Parameter | Purpose | Accepted values | Default | Notes |
# |---|---|---|---|---|
# | `SEPARATOR` | Column delimiter in the raw files | `'\\t'` (tab), `';'`, `','` | `'\\t'` | Use the value reported by Step 1 inspection |
# | `DECIMAL_COMMA` | Whether values use a comma as decimal separator | `True` / `False` | `True` | Set `True` for European formats (e.g. `3,14`); `False` for Anglo (e.g. `3.14`) |
# | `HEADER` | Row index of the header, or `None` if no header | `0`, `None` | `None` | Use the value reported by Step 1 inspection |
# | `STATIONS` | Mapping of station IDs to their ordered column names | `dict[str, list[str]]` | see below | `None` entries skip a column position; order must match the file layout |

# %%
SEPARATOR     = '\t'
DECIMAL_COMMA = True
HEADER        = None

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
    decimal_comma=DECIMAL_COMMA,
    column_names=None,
    date_col=0,
    time_col=1,
    save_combined=False,
)

print(f'\nLoaded dataset shape : {df_raw.shape}')
print(f'Date range : {df_raw.index.min()} \u2192 {df_raw.index.max()}')

stations_dict = organize_sensor_data(df_raw, STATIONS)
print(f'Organised {len(stations_dict)} station(s): {list(stations_dict.keys())}')

# %% [markdown]
# ## Step 3 · Signal Cleaning and Temperature Compensation
#
# For each station, `process_station()` executes three operations in sequence:
# 1. **Gap-aware spike filter:** computes |\u0394y| only on *valid consecutive pairs*
#    (pairs where neither the current nor the previous sample is NaN or zero).
#    This prevents gap boundaries from being mistakenly flagged as spikes — a common
#    issue in datasets with frequent data outages.
# 2. **Power-loss removal:** drops rows where battery charge equals zero.
# 3. **Temperature compensation:** applies linear thermal correction using `COMP_COEFF`
#    and normalizes the series so it starts at 0.  The raw signal is preserved as
#    `{SIGNAL_COL}_raw` in the saved CSV to support later visualisation.
#
# ### Parameter Tuning Guidance
#
# | Parameter | Purpose | Accepted values | Default | Notes |
# |---|---|---|---|---|
# | `SIGNAL_COL` | Structural response column to clean and compensate | Any string matching a column in `STATIONS` | `'absinc'` | Must exist in every station defined in `STATIONS` |
# | `TEMP_COL` | On-board temperature column used for compensation | Any string | `'temp'` | Must exist in every station |
# | `COMP_COEFF` | Thermal compensation coefficient (mdeg \u00b7 \u00b0C\u207b\u00b9 \u00b7 10\u207b\u00b3) | `float` | `0.005` | Update from your sensor calibration sheet; larger values = stronger correction |
# | `SPIKE_THRESHOLD` | Max allowed |\u0394y| between *valid consecutive* samples (mdeg) | `float > 0` | `500.0` | Only differences between non-NaN, non-zero adjacent samples are checked; set conservatively large first and review the dropped CSV before tightening |
# | `OUTPUT_DIR` | Destination directory for interim CSVs | Any valid path string | `'data/interim/sensor'` | Created automatically if it does not exist |

# %%
SIGNAL_COL       = 'absinc'
TEMP_COL         = 'temp'
COMP_COEFF       = 0.005       # mdeg \u00b7 \u00b0C\u207b\u00b9 \u00b7 10\u207b\u00b3 \u2014 update from calibration sheet
SPIKE_THRESHOLD  = 500.0       # mdeg \u2014 applied only to valid consecutive pairs (gap-aware)
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
# `plot_compensation_comparison()` reads directly from the saved interim CSV so it can
# be called independently at any time — including outside this notebook — to inspect
# any station.
#
# Dropped rows are overlaid on the signal panel in distinct colours:
# **crimson** = spike removals; **darkviolet** = power-loss removals.
#
# ### Parameter Tuning Guidance
#
# | Parameter | Purpose | Accepted values | Default | Notes |
# |---|---|---|---|---|
# | `VIZ_STATION` | Station to inspect | Any key in `STATIONS` | `'st02'` | Changes which interim CSV is loaded |
# | `VIZ_START` / `VIZ_END` | Date zoom window | `'YYYY-MM-DD'` string or `None` | `None` | Set both to `None` to show the full series; narrow the window to inspect specific events |
# | `DOT_SIZE` | Marker size for the main signal scatter | `float > 0` | `2` | Reduce for dense datasets; increase for sparse ones |
# | `DROPPED_DOT_SIZE` | Marker size for dropped-value overlay | `float > 0` or `None` | `None` (= `DOT_SIZE * 4`) | Set explicitly to control visibility of dropped points relative to the main cloud |

# %%
VIZ_STATION       = 'st02'
VIZ_START         = '2021-01-01'    # e.g. '2020-06-01' \u2014 or None for no lower bound
VIZ_END           = '2021-02-01'    # e.g. '2021-03-31' \u2014 or None for no upper bound
DOT_SIZE          = 2
DROPPED_DOT_SIZE  = None            # None = DOT_SIZE * 4; set a float to override

viz_file     = os.path.join(OUTPUT_DIR, f'{VIZ_STATION}_preprocessed.csv')
dropped_file = os.path.join(OUTPUT_DIR, f'{VIZ_STATION}_dropped.csv')

plot_compensation_comparison(
    file_path=viz_file,
    signal_col=SIGNAL_COL,
    dropped_path=dropped_file,
    date_start=VIZ_START,
    date_end=VIZ_END,
    dot_size=DOT_SIZE,
    dropped_dot_size=DROPPED_DOT_SIZE,
    save_plot=True,
    save_path='outputs/figures',
    filename=f'00_01_{VIZ_STATION}_compensation_comparison',
)

# %% [markdown]
# ## Step 5 · Dataset Preview
#
# Quick sanity check on the cleaned output for each station.

# %%
for st, df_clean in processed.items():
    print(f'--- {st} ---  shape: {df_clean.shape}  |  '
          f'{df_clean.index.min()} \u2192 {df_clean.index.max()}')
    display(df_clean.head(3))
    print()
