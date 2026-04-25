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
# # Meteosystem Italy — Proxy Data Download and Preparation
#
# **Pipeline position**: Auxiliary notebook. Runs *before* Notebook 01.
#
# This notebook downloads hourly weather data from the Meteosystem website
# (www.meteosystem.com) for a specified Italian station, assembles the monthly
# CSVs into a single DataFrame, standardises column names and the datetime index,
# and saves the result to `data/raw/proxies/` in the exact format expected by
# Notebook 01.
#
# ## Output contract (required by Notebook 01)
#
# | Requirement | Detail |
# |---|---|
# | **File location** | `data/raw/proxies/{OUTPUT_NAME}.csv` |
# | **First column (index)** | `datetime` — ISO 8601, timezone-naive UTC |
# | **Index name** | `datetime` |
# | **Column names** | English, with units in parentheses — e.g. `temperature (degC)` |
# | **Metadata columns** | Removed before saving |
# | **Values** | Numeric only; non-numeric entries coerced to `NaN` |
#
# ## Steps
# 1. **Configuration** — Set station URL, date range, and output name.
# 2. **Download** — Fetch monthly CSVs from Meteosystem and concatenate.
# 3. **Inspect raw columns** — Print the raw column list from the source.
# 4. **Standardise** — Parse datetime index, rename columns, drop metadata.
# 5. **Validate** — Check coverage, continuity, and missing-value rate.
# 6. **Save** — Write to `data/raw/proxies/{OUTPUT_NAME}.csv`.

# %%
import os
import io
import calendar
import time

import pandas as pd
import requests
import matplotlib.pyplot as plt
from IPython.display import display

# %% [markdown]
# ## Step 1 · Configuration
#
# ### Parameter Tuning Guidance
#
# | Parameter | Type | Description |
# |---|---|---|
# | `STATION_SLUG` | `str` | Subdirectory in the Meteosystem URL (e.g. `'gubbio'`). Change for a different station. |
# | `START_YEAR` | `int` | First year to download (inclusive). |
# | `END_YEAR` | `int` | Last year to download (inclusive). |
# | `OUTPUT_NAME` | `str` | Base filename written to `data/raw/proxies/`. Notebook 01 reads this path via `PROXY_FILE`. |
# | `COLUMN_RENAME` | `dict` | Maps raw Meteosystem column headers to standardised English names. Inspect Step 3 output and update this dict if the source schema changes. |
# | `META_COLS` | `list` | Raw column names that carry metadata (station name, coordinates, etc.) and must be dropped before saving. |
# | `TZ_SOURCE` | `str` or `None` | IANA timezone of the raw data (e.g. `'Europe/Rome'`). Set to `None` if already UTC or timezone-naive. The index is always saved timezone-naive (UTC). |
# | `REQUEST_DELAY` | `float` | Seconds to wait between requests. Keep ≥ 1.0 to avoid overloading the server. |

# %%
# === USER INPUT ===
STATION_SLUG = 'gubbio'               # Meteosystem station subdirectory
START_YEAR   = 2020
END_YEAR     = 2022
OUTPUT_NAME  = 'meteosystem_gubbio'   # → saved as data/raw/proxies/meteosystem_gubbio.csv

# Rename map: raw Meteosystem header → standardised name used by Notebook 01
# Keys must match EXACTLY what the source CSV returns (check case and spacing).
# Run Step 3 first to inspect the raw column names, then fill / correct these entries.
COLUMN_RENAME = {
    # --- datetime ---
    'Data/Ora':                     'datetime',

    # --- temperature ---
    'Temp. Aria (°C)':              'temperature (degC)',
    'Temp. Rugiada (°C)':           'dewpoint_temperature (degC)',
    'Temp. Bulbo Umido (°C)':       'wetbulb_temperature (degC)',

    # --- humidity ---
    'Umidità Relativa (%)':         'relative_humidity (%)',

    # --- radiation ---
    'Radiazione Solare (W/m²)':     'surface_solar_radiation (W/m^2)',

    # --- wind ---
    'Velocità Vento (m/s)':         'wind_speed (m/s)',
    'Direzione Vento (°)':          'wind_direction (deg)',

    # --- precipitation ---
    'Precipitazione (mm)':          'total_precipitation (mm)',

    # --- pressure ---
    'Pressione (hPa)':              'pressure (hPa)',
}

# Columns that carry metadata and must be dropped (use raw names, before renaming).
# Inspect Step 3 output and fill in any metadata columns (e.g. ['Stazione', 'Lat', 'Lon']).
META_COLS = []

TZ_SOURCE     = 'Europe/Rome'   # Raw data timezone. Set to None if already UTC/naive.
REQUEST_DELAY = 1.0             # seconds between HTTP requests
# ==================

BASE_URL    = f'https://www.meteosystem.com/dati/{STATION_SLUG}/csv.php'
OUTPUT_PATH = f'data/raw/proxies/{OUTPUT_NAME}.csv'

os.makedirs('data/raw/proxies', exist_ok=True)
print(f'Station  : {STATION_SLUG}')
print(f'Range    : {START_YEAR} – {END_YEAR}')
print(f'Output   : {OUTPUT_PATH}')

# %% [markdown]
# ## Step 2 · Download Monthly CSVs
#
# Iterates month by month, constructs the Meteosystem query URL, and accumulates
# each monthly chunk. Failures are logged but do not abort the loop — a summary
# of failed months is printed at the end.

# %%
all_chunks    = []
failed_months = []

for year in range(START_YEAR, END_YEAR + 1):
    yy = str(year)[-2:]
    for month in range(1, 13):
        _, last_day = calendar.monthrange(year, month)
        mm = f'{month:02d}'

        params = {
            'gg2': '01',               'mm2': mm, 'aa2': yy,
            'gg':  f'{last_day:02d}',  'mm':  mm, 'aa':  yy,
        }

        print(f'  Downloading {mm}/{year} … ', end='', flush=True)

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            text = response.text.strip()
            if not text:
                print('no data — skipped.')
                continue

            chunk = pd.read_csv(io.StringIO(text), sep=';')

            if chunk.empty:
                print('empty CSV — skipped.')
                continue

            all_chunks.append(chunk)
            print(f'OK  ({len(chunk)} rows)')

        except Exception as exc:
            print(f'FAILED: {exc}')
            failed_months.append(f'{mm}/{year}')

        time.sleep(REQUEST_DELAY)

print(f'\nDownload complete.  Chunks collected: {len(all_chunks)}')
if failed_months:
    print(f'Failed months ({len(failed_months)}): {", ".join(failed_months)}')

# %% [markdown]
# ## Step 3 · Inspect Raw Columns
#
# Print the raw column headers returned by the source so you can update
# `COLUMN_RENAME` and `META_COLS` in Step 1 if needed.
# **Review this output before proceeding to Step 4.**

# %%
assert all_chunks, (
    'No data was retrieved. '
    'Check the station URL, date range, and network access.'
)

df_raw = pd.concat(all_chunks, ignore_index=True)
df_raw = df_raw.drop_duplicates()

print(f'Raw shape: {df_raw.shape}')
print(f'\nRaw columns ({len(df_raw.columns)}):')
for i, col in enumerate(df_raw.columns):
    print(f'  [{i:2d}]  {repr(col)}')

display(df_raw.head(3))

# %% [markdown]
# ## Step 4 · Standardise — Datetime Index, Column Names, Metadata Removal
#
# ### Parameter Tuning Guidance
#
# | Parameter | Where | Description |
# |---|---|---|
# | `COLUMN_RENAME` | Step 1 | Update if raw column names differ from the default keys. |
# | `META_COLS` | Step 1 | Add any metadata column names found in the Step 3 output. |
# | `TZ_SOURCE` | Step 1 | Set to the IANA timezone of the raw timestamps (e.g. `'Europe/Rome'`). The output index is always timezone-naive UTC. |
#
# **Do not perform unit conversions here.**  Notebook 01 operates on the raw
# physical units as they arrive. Convert only if the source scale is non-standard
# (e.g. if relative humidity is in % but your pipeline expects 0–1 fraction).

# %%
df = df_raw.copy()

# --- Drop metadata columns ---
cols_to_drop = [c for c in META_COLS if c in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f'Dropped metadata columns: {cols_to_drop}')

# --- Rename known columns ---
rename_applicable = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
df = df.rename(columns=rename_applicable)

unmapped = [c for c in df.columns if c not in rename_applicable.values() and c != 'datetime']
if unmapped:
    print(f'\n⚠ Unmapped columns (kept as-is, review COLUMN_RENAME): {unmapped}')

# --- Parse and set datetime index ---
assert 'datetime' in df.columns, (
    "No 'datetime' column found after renaming. "
    "Ensure COLUMN_RENAME maps the raw date column to 'datetime'."
)

df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')

n_unparsed = df['datetime'].isnull().sum()
if n_unparsed:
    print(f'\n⚠ {n_unparsed} rows had unparseable datetime values — dropped.')
    df = df.dropna(subset=['datetime'])

# --- Timezone conversion → UTC naive ---
if TZ_SOURCE is not None:
    df['datetime'] = (
        df['datetime']
        .dt.tz_localize(TZ_SOURCE, ambiguous='infer', nonexistent='shift_forward')
        .dt.tz_convert('UTC')
        .dt.tz_localize(None)   # strip tzinfo → timezone-naive UTC
    )
    print(f'Timezone: {TZ_SOURCE} → UTC (timezone-naive)')
else:
    if hasattr(df['datetime'].dtype, 'tz') and df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    print('Timezone: assumed UTC — no conversion applied.')

df = df.set_index('datetime')
df.index.name = 'datetime'
df = df.sort_index()

# --- Coerce all remaining columns to numeric ---
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f'\nStandardised shape  : {df.shape}')
print(f'Index range         : {df.index.min()}  →  {df.index.max()}')
print(f'Index dtype         : {df.index.dtype}')
print(f'\nFinal columns ({len(df.columns)}):')
for col in df.columns:
    miss = df[col].isnull().mean() * 100
    print(f'  {col:<45}  {miss:5.1f}% missing')

display(df.head(3))

# %% [markdown]
# ## Step 5 · Validate
#
# Checks that the output satisfies the Notebook 01 proxy contract:
# - Index is a sorted, timezone-naive `DatetimeIndex`
# - No column is entirely `NaN`
# - Dataset covers the requested date range

# %%
validation_ok = True

assert isinstance(df.index, pd.DatetimeIndex), 'Index is not a DatetimeIndex.'

if not df.index.is_monotonic_increasing:
    print('⚠ Index not sorted — sorting now.')
    df = df.sort_index()

if df.index.tz is not None:
    print('⚠ Index has timezone info — stripping to naive UTC.')
    df.index = df.index.tz_localize(None)

all_nan = [c for c in df.columns if df[c].isnull().all()]
if all_nan:
    print(f'⚠ Entirely-NaN columns dropped: {all_nan}')
    df = df.drop(columns=all_nan)

expected_start = pd.Timestamp(f'{START_YEAR}-01-01')
expected_end   = pd.Timestamp(f'{END_YEAR}-12-31')
if df.index.min() > expected_start:
    print(f'⚠ Dataset starts {df.index.min()}, later than expected {expected_start}.')
    validation_ok = False
if df.index.max() < expected_end:
    print(f'⚠ Dataset ends {df.index.max()}, earlier than expected {expected_end}.')
    validation_ok = False

if validation_ok:
    print('✓ All validation checks passed.')
else:
    print('\n⚠ Some checks failed — review warnings above before saving.')

print(f'\nFinal dataset: {df.shape[0]:,} rows × {df.shape[1]} columns')

# Quick overview plot
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if numeric_cols:
    n_plot = min(len(numeric_cols), 4)
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols[:n_plot]):
        ax.plot(df.index, df[col], linewidth=0.5, color='steelblue')
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(
        f'Meteosystem {STATION_SLUG.capitalize()} — quick overview', fontsize=10
    )
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Step 6 · Save
#
# Saves the standardised proxy dataset to `data/raw/proxies/{OUTPUT_NAME}.csv`.
# The datetime index is written as the first column so Notebook 01 can load it
# with `pd.read_csv(PROXY_FILE, index_col=0, parse_dates=True)`.
#
# In Notebook 01, set:
# ```python
# PROXY_FILE = 'data/raw/proxies/{OUTPUT_NAME}.csv'
# ```

# %%
if validation_ok:
    df.to_csv(OUTPUT_PATH, index=True)
    print(f'✓ Proxy saved to : {OUTPUT_PATH}')
    print(f'  Shape          : {df.shape}')
    print(f'  Index range    : {df.index.min()}  →  {df.index.max()}')
    print(f'\nIn Notebook 01, set:')
    print(f"  PROXY_FILE = '{OUTPUT_PATH}'")
else:
    print('Save skipped — resolve validation warnings above before saving.')
    print('To force-save despite warnings, replace `if validation_ok:` with `if True:`.')
