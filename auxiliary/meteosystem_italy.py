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
# This notebook downloads 30-minute weather data from the Meteosystem website
# (www.meteosystem.com) for a specified Italian station, assembles the monthly
# CSVs into a single DataFrame, parses the datetime index from the two source
# columns (`Data` + `Ora`), and saves the result to `data/raw/proxies/` so that
# Notebook 01 can load it with `pd.read_csv(PROXY_FILE, index_col=0, parse_dates=True)`.
#
# ## Source schema (Meteosystem Gubbio)
#
# | Raw column | Type | Notes |
# |---|---|---|
# | `Data` | date string `d/m/yyyy` | Combined with `Ora` to form the datetime index |
# | `Ora` | time string `H:MM` | Combined with `Data` |
# | `Temp` | float °C | Air temperature |
# | `Min` | float °C | Minimum temperature in period |
# | `Max` | float °C | Maximum temperature in period |
# | `Umid` | int % | Relative humidity |
# | `Dew pt` | float °C | Dew-point temperature |
# | `Vento` | float m/s | Wind speed |
# | `Dir` | string (e.g. `NE`) | Wind direction — categorical, **dropped** |
# | `Raffica` | float m/s | Wind gust speed |
# | `Dir Raff.` | string | Gust direction — categorical, **dropped** |
# | `Press` | float hPa | Atmospheric pressure |
# | `Pioggia` | float mm | Precipitation |
# | `Int.Pio.` | float | Precipitation intensity |
# | `Rad.Sol.` | int W/m² | Solar radiation |
#
# ## Output contract (required by Notebook 01)
#
# | Requirement | Detail |
# |---|---|
# | **File location** | `data/raw/proxies/{OUTPUT_NAME}.csv` |
# | **First column (index)** | `datetime` — ISO 8601, timezone-naive UTC |
# | **Index name** | `datetime` |
# | **Values** | Numeric only; categorical columns dropped |
#
# ## Steps
# 1. **Configuration** — Set station URL, date range, output name, and cache settings.
# 2. **Download** — Fetch missing monthly CSVs from Meteosystem; already-cached months are skipped.
# 3. **Assemble** — Load all cached chunk files from disk and concatenate into `df_raw`.
# 4. **Standardise** — Combine datetime columns, drop categoricals, coerce to numeric.
# 5. **Validate and plot** — Check coverage, sorted index, missing-value rate, internal gaps, and outliers.
# 6. **Save** — Write the final proxy CSV if all validations pass.
#
# > **Re-run workflow:** After the first full run of Step 2, you can freely modify and
# > re-run Steps 3–6 without re-downloading any data. To download only newly added months,
# > extend `END_YEAR` or `START_YEAR` and re-run Step 2 — already-cached months are skipped
# > automatically. To force a full re-download, set `FORCE_REDOWNLOAD = True` in Step 1.
#
# > **Jupytext pairing note:** This notebook is paired with `meteosystem_italy.ipynb`
# > via the `formats: ipynb,py:percent` header above. The root `jupytext.toml`
# > applies globally to all notebooks in the repository, including those in `auxiliary/`.
# > Always edit this `.py` file and sync to the notebook via `jupytext --sync` or
# > `auto_watcher.py`. Never edit the `.ipynb` directly.

# %% [markdown]
# ## Imports

# %%
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import io
import calendar
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from IPython.display import display
from tqdm.auto import tqdm

from heritageshm.viz import apply_theme, plot_proxy_overview

apply_theme()

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
# | `OUTPUT_NAME` | `str` | Base filename written to `data/raw/proxies/`. Set `PROXY_FILE` in Notebook 01 to this path. |
# | `COLUMNS_TO_DROP` | `list` | Columns to remove before saving. Used for categorical columns (`Dir`, `Dir Raff.`) that cannot be used as numeric proxy variables. Add any other unwanted columns here. |
# | `TZ_SOURCE` | `str` or `None` | IANA timezone of the raw timestamps. The output index is always timezone-naive UTC. Set to `None` if the source is already UTC. |
# | `REQUEST_DELAY` | `float` | Seconds to wait between requests. Keep ≥ 1.0 (minimum — do not lower). |
# | `FORCE_REDOWNLOAD` | `bool` | If `True`, re-download every month even if its chunk file already exists in `CACHE_DIR`. Useful after correcting a download bug or updating the station URL. Default: `False`. |

# %%
# === USER INPUT ===
STATION_SLUG     = 'gubbio'               # Meteosystem station subdirectory
START_YEAR       = 2020
END_YEAR         = 2022
OUTPUT_NAME      = 'meteosystem_gubbio'   # → saved as data/raw/proxies/meteosystem_gubbio.csv
FORCE_REDOWNLOAD = False                  # True = ignore cache, re-download everything

# Categorical columns to drop (not coercible to numeric, not useful as proxy variables)
COLUMNS_TO_DROP = ['Dir', 'Dir Raff.']

TZ_SOURCE     = 'Europe/Rome'   # Raw timestamps are local Italian time (CET/CEST)
REQUEST_DELAY = 1.0             # seconds between HTTP requests (minimum — do not lower)
# ==================

BASE_URL    = f'https://www.meteosystem.com/dati/{STATION_SLUG}/csv.php'
OUTPUT_PATH = f'data/raw/proxies/{OUTPUT_NAME}.csv'
CACHE_DIR   = Path(f'data/raw/proxies/_cache/{STATION_SLUG}')

os.makedirs('data/raw/proxies', exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f'Station      : {STATION_SLUG}')
print(f'Range        : {START_YEAR} – {END_YEAR}')
print(f'Output       : {OUTPUT_PATH}')
print(f'Cache dir    : {CACHE_DIR}')
print(f'Force reload : {FORCE_REDOWNLOAD}')

# %% [markdown]
# ## Step 2 · Download Monthly CSVs
#
# Iterates month by month, constructs the Meteosystem query URL, and saves each
# monthly response as an individual CSV file in `CACHE_DIR` (one file per month,
# named `{year}_{month:02d}.csv`). Months whose cache file already exists are
# **skipped** unless `FORCE_REDOWNLOAD = True`.
#
# This means you only ever download each month once. To fetch newly available
# months, just extend `END_YEAR` and re-run this step — previously downloaded
# months are untouched. Steps 3–6 read from the cache and are always safe to
# re-run independently.

# %%
failed_months = []
n_downloaded  = 0
n_skipped     = 0

yesterday = date.today() - timedelta(days=1)

# Collect all valid year/month pairs that are not entirely in the future
months_to_process = [
    (year, month)
    for year in range(START_YEAR, END_YEAR + 1)
    for month in range(1, 13)
    if date(year, month, 1) <= yesterday
]

pbar = tqdm(months_to_process, desc='Downloading monthly data')
for year, month in pbar:
    mm       = f'{month:02d}'
    label    = f'{mm}/{year}'
    cache_fp = CACHE_DIR / f'{year}_{mm}.csv'

    # --- Cache hit: skip download ---
    if cache_fp.exists() and not FORCE_REDOWNLOAD:
        n_skipped += 1
        pbar.set_postfix_str(f'{label} [cached]')
        continue

    # --- Cache miss: download ---
    yy = str(year)[-2:]
    _, last_day = calendar.monthrange(year, month)
    month_end   = date(year, month, last_day)
    if month_end > yesterday:
        last_day = yesterday.day

    params = {
        'gg2': '01',               'mm2': mm, 'aa2': yy,
        'gg':  f'{last_day:02d}',  'mm':  mm, 'aa':  yy,
    }

    pbar.set_postfix_str(f'Fetching {label}')

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        text = response.text.strip()
        if not text:
            continue

        # Strip PHP warnings that sometimes appear at the top of the response
        lines     = text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('Data;'):
                start_idx = i
                break

        clean_text = '\n'.join(lines[start_idx:])

        # Validate before saving
        chunk = pd.read_csv(io.StringIO(clean_text), sep=';')
        if chunk.empty:
            continue

        # Persist chunk to cache
        chunk.to_csv(cache_fp, index=False)
        n_downloaded += 1

    except Exception as exc:
        failed_months.append(label)
        pbar.write(f'  ✗ {label}: {exc}')

    time.sleep(REQUEST_DELAY)

print(f'\nDownload summary')
print(f'  Downloaded : {n_downloaded}')
print(f'  Cached     : {n_skipped}')
print(f'  Failed     : {len(failed_months)}')
if failed_months:
    print(f'  Failed months: {", ".join(failed_months)}')

# %% [markdown]
# ## Step 3 · Assemble from Cache
#
# Reads every CSV file present in `CACHE_DIR` and concatenates them into
# `df_raw`. This step is completely independent of Step 2 — you can re-run it
# (and all subsequent steps) without re-downloading anything.
#
# Only files matching the `{year}_{month:02d}.csv` pattern within the
# configured `START_YEAR`–`END_YEAR` range are included, so switching station
# or date range in Step 1 and re-running from Step 3 is safe.

# %%
chunk_files = sorted(
    CACHE_DIR.glob('*.csv'),
    key=lambda p: p.stem   # stem is '{year}_{month}' — lexicographic order is correct
)

# Filter to the configured date range only
def _in_range(fp: Path) -> bool:
    try:
        year, month = (int(x) for x in fp.stem.split('_'))
        return START_YEAR <= year <= END_YEAR
    except ValueError:
        return False

chunk_files = [fp for fp in chunk_files if _in_range(fp)]

assert chunk_files, (
    f'No cache files found in {CACHE_DIR} for {START_YEAR}–{END_YEAR}. '
    'Run Step 2 first to download the data.'
)

all_chunks = [pd.read_csv(fp) for fp in tqdm(chunk_files, desc='Loading cached chunks')]
df_raw = pd.concat(all_chunks, ignore_index=True)
df_raw = df_raw.drop_duplicates()

print(f'\nLoaded {len(chunk_files)} chunk files from cache.')
print(f'Raw shape: {df_raw.shape}')
print(f'\nRaw columns ({len(df_raw.columns)}):')
for i, col in enumerate(df_raw.columns):
    print(f'  [{i:2d}]  {repr(col)}')

display(df_raw.head(3))

# %% [markdown]
# ## Step 4 · Standardise — Datetime Index, Drop Categoricals, Coerce to Numeric
#
# ### Parameter Tuning Guidance
#
# | Parameter | Where | Description |
# |---|---|---|
# | `COLUMNS_TO_DROP` | Step 1 | Add any column names you want to exclude from the output. |
# | `TZ_SOURCE` | Step 1 | IANA timezone string for the raw timestamps. Meteosystem Italy serves local time (CET in winter, CEST in summer). Output is always timezone-naive UTC. |
#
# The `Data` and `Ora` columns are combined into a single `datetime` string
# before parsing. Both columns are then dropped. All remaining columns are
# coerced to numeric; non-numeric values become `NaN`.

# %%
df = df_raw.copy()

# --- Combine Data + Ora into a single datetime column ---
assert 'Data' in df.columns and 'Ora' in df.columns, (
    "Expected columns 'Data' and 'Ora' not found. "
    "Check Step 3 output for the actual date/time column names."
)

df['datetime'] = pd.to_datetime(
    df['Data'].astype(str) + ' ' + df['Ora'].astype(str),
    dayfirst=True,
    errors='coerce',
)
df = df.drop(columns=['Data', 'Ora'])

n_unparsed = df['datetime'].isnull().sum()
if n_unparsed:
    print(f'⚠ {n_unparsed} rows had unparseable datetime values — dropped.')
    df = df.dropna(subset=['datetime'])

# --- Timezone conversion → UTC naive ---
if TZ_SOURCE is not None:
    df['datetime'] = (
        df['datetime']
        .dt.tz_localize(TZ_SOURCE, ambiguous='NaT', nonexistent='shift_forward')
        .dt.tz_convert('UTC')
        .dt.tz_localize(None)   # strip tzinfo → timezone-naive UTC
    )
    df = df.dropna(subset=['datetime'])
    print(f'Timezone: {TZ_SOURCE} → UTC (timezone-naive)')
else:
    print('Timezone: no conversion applied (assumed UTC).')

# --- Set datetime as index ---
df = df.set_index('datetime')
df.index.name = 'datetime'
df = df.sort_index()

# --- Drop categorical / unwanted columns ---
cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f'Dropped columns    : {cols_to_drop}')

# --- Coerce all remaining columns to numeric ---
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

non_numeric_remaining = df.select_dtypes(exclude='number').columns.tolist()
if non_numeric_remaining:
    print(f'⚠ Non-numeric columns still present — add to COLUMNS_TO_DROP: {non_numeric_remaining}')

print(f'\nStandardised shape  : {df.shape}')
print(f'Index range         : {df.index.min()}  →  {df.index.max()}')
print(f'Index dtype         : {df.index.dtype}')
print(f'\nFinal columns ({len(df.columns)}):')
for col in df.columns:
    miss = df[col].isnull().mean() * 100
    print(f'  {col:<20}  {miss:5.1f}% missing')

display(df.head(3))

# %% [markdown]
# ## Step 5 · Validate, Clean Outliers, and Plot
#
# ### Parameter Tuning Guidance
#
# | Parameter | Type | Default | Description |
# |---|---|---|---|
# | `PLOT_COLS` | `list[str]` or `None` | `None` | Columns to include in the overview plot. Set to a list of column names to restrict the plot to those variables (e.g. `['Temp', 'Umid', 'Rad.Sol.']`). Set to `None` to plot all columns. |
# | `OUTLIER_IQR_FACTOR` | `float` | `5.0` | Multiplier on the IQR used to define outlier fences per column. Lower values are more aggressive (e.g. `3.0`); higher values are more conservative (e.g. `10.0`). Outliers are replaced with `NaN` **in `df` before saving** — they are not merely hidden in the plot. |
# | `APPLY_OUTLIER_FILTER` | `bool` | `True` | If `False`, the IQR filter is skipped entirely and outliers are left as-is. Set to `False` if you want to inspect the raw data before deciding on cleaning. |
#
# This step runs four sub-tasks in order:
# 1. **Structural validation** — sorted index, timezone-naive, no all-NaN columns, no non-numeric columns, date-range coverage.
# 2. **Internal-gap report** — for each column, counts NaN values that fall *between* the first and last valid observation (i.e. true internal gaps, not leading/trailing missing data).
# 3. **Outlier filter** — replaces values outside `[Q1 − k·IQR, Q3 + k·IQR]` with `NaN`. Applied to `df` directly so that cleaned data is what gets saved in Step 6.
# 4. **Overview plot** — plots the selected columns after cleaning so the chart reflects the final saved state.

# %%
# === USER INPUT ===
PLOT_COLS            = None   # e.g. ['Temp', 'Umid', 'Rad.Sol.'] or None for all columns
OUTLIER_IQR_FACTOR   = 5.0    # IQR fence multiplier — increase to keep more data
APPLY_OUTLIER_FILTER = True   # False = skip outlier replacement entirely
# ==================

import numpy as np

validation_ok = True

# ------------------------------------------------------------------
# 1. Structural validation
# ------------------------------------------------------------------
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

non_num = df.select_dtypes(exclude='number').columns.tolist()
if non_num:
    print(f'⚠ Non-numeric columns still present (add to COLUMNS_TO_DROP): {non_num}')
    validation_ok = False

yesterday_ts   = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
expected_start = pd.Timestamp(f'{START_YEAR}-01-01')
expected_end   = min(pd.Timestamp(f'{END_YEAR}-12-31'), yesterday_ts)

if df.index.min() > expected_start:
    print(f'⚠ Dataset starts {df.index.min()}, later than expected {expected_start}.')
    validation_ok = False
if df.index.max() < expected_end:
    print(f'⚠ Dataset ends {df.index.max()}, earlier than expected {expected_end}.')
    validation_ok = False

if validation_ok:
    print('✓ All structural validation checks passed.')
else:
    print('\n⚠ Some checks failed — review warnings above before saving.')

print(f'\nFinal dataset: {df.shape[0]:,} rows × {df.shape[1]} columns')

# ------------------------------------------------------------------
# 2. Internal-gap report
# ------------------------------------------------------------------
print('\n── Internal-gap report (NaNs between first and last valid observation) ──')
cols_to_check = PLOT_COLS if PLOT_COLS is not None else df.columns.tolist()
any_internal_gaps = False
for col in cols_to_check:
    if col not in df.columns:
        print(f'  {col:<20}  ⚠ column not found — check PLOT_COLS spelling')
        continue
    series = df[col]
    first_valid = series.first_valid_index()
    last_valid  = series.last_valid_index()
    if first_valid is None:
        print(f'  {col:<20}  entirely NaN — skipped')
        continue
    interior = series.loc[first_valid:last_valid]
    n_internal = interior.isnull().sum()
    pct_internal = n_internal / len(interior) * 100 if len(interior) > 0 else 0.0
    status = f'  {col:<20}  {n_internal:>6,} internal NaNs  ({pct_internal:.2f}%)'
    if n_internal > 0:
        print(f'⚠ {status}')
        any_internal_gaps = True
    else:
        print(f'✓ {status}')

if not any_internal_gaps:
    print('✓ No internal gaps detected in any plotted column.')

# ------------------------------------------------------------------
# 3. IQR outlier filter  (applied to df — affects saved output)
# ------------------------------------------------------------------
if APPLY_OUTLIER_FILTER:
    print(f'\n── Outlier filter (IQR × {OUTLIER_IQR_FACTOR}) ──')
    total_replaced = 0
    for col in df.columns:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue   # constant column — skip to avoid degenerate fence
        lower = q1 - OUTLIER_IQR_FACTOR * iqr
        upper = q3 + OUTLIER_IQR_FACTOR * iqr
        mask  = (df[col] < lower) | (df[col] > upper)
        n_out = mask.sum()
        if n_out > 0:
            df.loc[mask, col] = np.nan
            total_replaced += n_out
            print(f'  {col:<20}  {n_out:>6,} outliers → NaN  (fence: [{lower:.3g}, {upper:.3g}])')
    if total_replaced == 0:
        print('  ✓ No outliers detected in any column.')
    else:
        print(f'  Total replaced: {total_replaced:,} values')
else:
    print('\n── Outlier filter skipped (APPLY_OUTLIER_FILTER = False) ──')

# ------------------------------------------------------------------
# 4. Overview plot — reflects cleaned data
# ------------------------------------------------------------------
df_plot = df[PLOT_COLS] if PLOT_COLS is not None else df
plot_proxy_overview(df_plot, station_slug=STATION_SLUG, n_cols=4)

# %% [markdown]
# ## Step 6 · Save
#
# Saves the dataset to `data/raw/proxies/{OUTPUT_NAME}.csv`.
# The datetime index is written as the first column so Notebook 01 can load it
# with `pd.read_csv(PROXY_FILE, index_col=0, parse_dates=True)`.
#
# **In Notebook 01 Step 2, set:**
# ```python
# PROXY_FILE = 'data/raw/proxies/{OUTPUT_NAME}.csv'
# ```
# Then in Step 3, choose columns from the list printed in Step 4 above
# (e.g. `'Temp'`, `'Umid'`, `'Dew pt'`, `'Rad.Sol.'`, etc.).
#
# > To force-save despite validation warnings, replace `if validation_ok:` with `if True:`.

# %%
if validation_ok:
    df.to_csv(OUTPUT_PATH, index=True)
    print(f'✓ Proxy saved to : {OUTPUT_PATH}')
    print(f'  Shape          : {df.shape}')
    print(f'  Index range    : {df.index.min()}  →  {df.index.max()}')
    print(f'\nIn Notebook 01, set:')
    print(f"  PROXY_FILE = '{OUTPUT_PATH}'")
    print(f'\nAvailable columns for PROXY_COLS in Notebook 01 Step 3:')
    for col in df.columns:
        print(f"  '{col}'")
else:
    print('Save skipped — resolve validation warnings above before saving.')
    print('To force-save despite warnings, replace `if validation_ok:` with `if True:`.')
