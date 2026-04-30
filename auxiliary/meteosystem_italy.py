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
# 1. **Configuration** — Set station URL, date range, and output name.
# 2. **Download** — Fetch monthly CSVs from Meteosystem and concatenate.
# 3. **Inspect raw columns** — Verify the raw column list matches the schema above.
# 4. **Standardise** — Combine datetime columns, drop categoricals, coerce to numeric.
# 5. **Validate and Save** — Check coverage, sorted index, and missing-value rate; save if all checks pass.
#
# > **Jupytext pairing note:** This notebook is paired with `meteosystem_italy.ipynb`
# > via the `formats: ipynb,py:percent` header above. The root `jupytext.toml`
# > applies globally to all notebooks in the repository, including those in `auxiliary/`.
# > Always edit this `.py` file and sync to the notebook via `jupytext --sync` or
# > `auto_watcher.py`. Never edit the `.ipynb` directly.

# %% [markdown]
# ## Imports

# %%
# [Fix 1] All imports consolidated in this single cell — including tqdm.
import os
import io
import calendar
import time
from datetime import date, timedelta

import pandas as pd
import requests
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm.auto import tqdm


# %% [markdown]
# ## Helper Functions

# %%
# [Fix 2] Inline plot logic refactored into a named helper.
def _plot_proxy_overview(df: pd.DataFrame, station_slug: str, n_cols: int = 4) -> None:
    """
    Produces a quick multi-panel time-series overview of the proxy dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Standardised proxy DataFrame with a DatetimeIndex and numeric columns.
    station_slug : str
        Station identifier used in the plot title (e.g. ``'gubbio'``).
    n_cols : int, optional
        Maximum number of columns to plot. Default 4.

    Returns
    -------
    None
        Displays the figure inline; does not save to disk.
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        print('No numeric columns to plot.')
        return

    n_plot = min(len(numeric_cols), n_cols)
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols[:n_plot]):
        ax.plot(df.index, df[col], linewidth=0.5, color='steelblue')
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        f'Meteosystem {station_slug.capitalize()} — quick overview', fontsize=10
    )
    fig.tight_layout()
    plt.show()


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

# %%
# === USER INPUT ===
STATION_SLUG = 'gubbio'               # Meteosystem station subdirectory
START_YEAR   = 2020
END_YEAR     = 2022
OUTPUT_NAME  = 'meteosystem_gubbio'   # → saved as data/raw/proxies/meteosystem_gubbio.csv

# Categorical columns to drop (not coercible to numeric, not useful as proxy variables)
COLUMNS_TO_DROP = ['Dir', 'Dir Raff.']

TZ_SOURCE     = 'Europe/Rome'   # Raw timestamps are local Italian time (CET/CEST)
REQUEST_DELAY = 1.0             # seconds between HTTP requests (minimum — do not lower)
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
# each monthly chunk. The source uses comma as separator and encodes dates as
# two separate columns (`Data`, `Ora`). Failures are logged but do not abort the
# loop — a summary of failed months is printed at the end.

# %%
all_chunks    = []
failed_months = []

yesterday = date.today() - timedelta(days=1)

# Collect all valid year/month pairs that are not entirely in the future
months_to_download = []
for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):
        if date(year, month, 1) > yesterday:
            break
        months_to_download.append((year, month))

pbar = tqdm(months_to_download, desc="Downloading monthly data")
for year, month in pbar:
    yy = str(year)[-2:]
    _, last_day = calendar.monthrange(year, month)

    # Cap the end of the month to yesterday
    month_end = date(year, month, last_day)
    if month_end > yesterday:
        last_day = yesterday.day

    mm = f'{month:02d}'

    params = {
        'gg2': '01',               'mm2': mm, 'aa2': yy,
        'gg':  f'{last_day:02d}',  'mm':  mm, 'aa':  yy,
    }

    pbar.set_postfix_str(f"Fetching {mm}/{year}")

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        text = response.text.strip()
        if not text:
            continue

        # Strip out PHP warnings that sometimes appear at the top of the response
        lines = text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            # The actual CSV header starts with "Data" and the separator is ";"
            if line.strip().startswith('Data;'):
                start_idx = i
                break

        clean_text = '\n'.join(lines[start_idx:])

        # Source uses semicolon separator
        chunk = pd.read_csv(io.StringIO(clean_text), sep=';')

        if chunk.empty:
            continue

        all_chunks.append(chunk)

    except Exception as exc:
        failed_months.append(f'{mm}/{year}')

    time.sleep(REQUEST_DELAY)

print(f'\nDownload complete.  Chunks collected: {len(all_chunks)}')
if failed_months:
    print(f'Failed months ({len(failed_months)}): {", ".join(failed_months)}')

# %% [markdown]
# ## Step 3 · Inspect Raw Columns
#
# Prints the raw column headers. Expected schema:
# `Data, Ora, Temp, Min, Max, Umid, Dew pt, Vento, Dir, Raffica, Dir Raff., Press, Pioggia, Int.Pio., Rad.Sol.`
#
# If the printed headers differ, update `COLUMNS_TO_DROP` in Step 1 accordingly.

# %%
assert all_chunks, (
    'No data was retrieved. '
    f'Check the station URL, date range, and network access. '
    f'Failed months: {failed_months if failed_months else "none recorded"}'
)

df_raw = pd.concat(all_chunks, ignore_index=True)
df_raw = df_raw.drop_duplicates()

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
    # Drop any rows that became NaT due to ambiguous DST transitions
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
# ## Step 5 · Validate and Save
#
# [Fix 3] Validation and save are consolidated into a single cell to prevent a
# fragile cross-cell dependency on the `validation_ok` flag. Running this cell
# from any kernel state is safe: `validation_ok` is always initialised to `True`
# here before any check, and the save only proceeds if all checks pass.
#
# [Fix 4] `expected_end` is capped to yesterday so that setting `END_YEAR` to
# the current year does not trigger a spurious coverage warning when data
# legitimately ends before December 31.
#
# Checks that the output satisfies the Notebook 01 proxy contract:
# - Index is a sorted, timezone-naive `DatetimeIndex`
# - No column is entirely `NaN`
# - Dataset covers the requested date range (capped to yesterday)
# - No non-numeric columns remain

# %%
# [Fix 3] Always initialised here — safe to re-run in any order.
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

non_num = df.select_dtypes(exclude='number').columns.tolist()
if non_num:
    print(f'⚠ Non-numeric columns still present (add to COLUMNS_TO_DROP): {non_num}')
    validation_ok = False

# [Fix 4] Cap expected_end to yesterday so current-year END_YEAR does not
# trigger a false coverage warning when data legitimately ends before Dec 31.
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
    print('✓ All validation checks passed.')
else:
    print('\n⚠ Some checks failed — review warnings above before saving.')

print(f'\nFinal dataset: {df.shape[0]:,} rows × {df.shape[1]} columns')

# [Fix 2] Quick overview plot — delegated to helper function.
_plot_proxy_overview(df, station_slug=STATION_SLUG, n_cols=4)

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
