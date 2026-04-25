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
# # Oikolab Proxy Extraction
#
# Downloads hourly weather data from the Oikolab ERA5 API for a given location and date range,
# cleans the response, and saves the result in the format expected by **Notebook 01**.
#
# The Oikolab API enforces a 500-unit limit per request.
# With 10 parameters the limit is ~120 units per 6-month chunk, well within bounds.
#
# ---
#
# ## Output Format Contract
#
# | Requirement | Detail |
# |---|---|
# | **File** | `../data/raw/proxies/oikolab_weather.csv` (relative to `auxiliary/`) |
# | **Index** | `datetime` column (UTC timestamps), first CSV column |
# | **Read by NB01** | `pd.read_csv(..., index_col=0, parse_dates=True)` |
# | **Metadata dropped** | `coordinates (lat,lon)`, `model (name)`, `model elevation (surface)`, `utc_offset (hrs)` |
# | **Data columns** | 10 SHM-relevant ERA5 weather variables |
# | **Timezone** | UTC - align to local time in Notebook 01 if the sensor uses local time |
#
# ---
#
# ## Parameters downloaded
#
# | Variable | Role in SHM |
# |---|---|
# | `temperature` | Primary driver of thermal expansion/contraction |
# | `relative_humidity` | Governs masonry moisture content and hygric deformation |
# | `dewpoint_temperature` | Condensation threshold on surfaces |
# | `surface_solar_radiation` | Thermal forcing on sun-exposed facades |
# | `surface_thermal_radiation` | Longwave radiative cooling at night |
# | `total_precipitation` | Direct moisture input to the wall |
# | `wind_speed` | Wind-driven rain and evaporative cooling |
# | `wind_direction` | Driving rain orientation |
# | `skin_temperature` | Closest ERA5 proxy to surface-mounted sensor readings |
# | `snowfall` | Freeze-thaw loading; relevant for Apennine winters |
#
# ---
#
# ## Steps
# 1. **Parameters** - Location, date range, API key, output path.
# 2. **Download** - Fetch one 6-month window at a time, with progress reporting.
# 3. **Concatenate & clean** - Merge chunks, promote datetime index, drop metadata columns.
# 4. **Save** - Write CSV with `index=True`.

# %% [markdown]
# ## Step 1 - Parameters
#
# | Parameter | Description |
# |---|---|
# | `LAT`, `LON` | Decimal coordinates of the monitoring site |
# | `START`, `END` | Inclusive date range (`YYYY-MM-DD`). Should fully bracket the sensor window |
# | `API_KEY` | Oikolab API key |
# | `OUTPUT_PATH` | Destination CSV - must match `PROXY_FILE` in Notebook 01 |
# | `CHUNK_MONTHS` | Request window size in months. Reduce if you hit 500-unit errors |

# %%
import requests
import pandas as pd
import io
import os
import time
from datetime import date
from dateutil.relativedelta import relativedelta

# === USER INPUT ===
LAT          = 43.35343
LON          = 12.582047
START        = '2018-01-01'
END          = '2026-04-25'
API_KEY      = '8426a7e187b9481ab575814f707c8f8d'
OUTPUT_PATH  = '../data/raw/proxies/oikolab_weather.csv'
CHUNK_MONTHS = 6   # 6 months x 10 params ~ 60 units (limit is 500)
# ==================

META_COLS = [
    'coordinates (lat,lon)',
    'model (name)',
    'model elevation (surface)',
    'utc_offset (hrs)',
]

# %% [markdown]
# ## Step 2 - Download
#
# 10 SHM-relevant regressors. With 6-month chunks each request uses ~60 units.
# A 1-second pause between requests avoids rate-limiting.

# %%
PARAMS = [
    'temperature',
    'relative_humidity',
    'dewpoint_temperature',
    'surface_solar_radiation',
    'surface_thermal_radiation',
    'total_precipitation',
    'wind_speed',
    'wind_direction',
    'skin_temperature',
    'snowfall',
]

def build_chunk_windows(start_str, end_str, months):
    """Yield (chunk_start, chunk_end) pairs of `months` months each."""
    start = date.fromisoformat(start_str)
    end   = date.fromisoformat(end_str)
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + relativedelta(months=months) - relativedelta(days=1), end)
        yield cursor.isoformat(), chunk_end.isoformat()
        cursor = cursor + relativedelta(months=months)

windows = list(build_chunk_windows(START, END, CHUNK_MONTHS))
print(f'Fetching {len(windows)} chunks of {CHUNK_MONTHS} months ({len(PARAMS)} params each) ...')

chunks = []
for i, (chunk_start, chunk_end) in enumerate(windows):
    print(f'  [{i+1}/{len(windows)}] {chunk_start} -> {chunk_end} ...', end=' ', flush=True)
    r = requests.get(
        'https://api.oikolab.com/weather',
        params={
            'param': PARAMS,
            'lat': LAT,
            'lon': LON,
            'start': chunk_start,
            'end': chunk_end,
            'freq': 'H',
            'resample_method': 'mean',
            'format': 'csv',
        },
        headers={'api-key': API_KEY},
    )
    if not r.ok:
        print(f'FAILED ({r.status_code})')
        print(r.text)
        r.raise_for_status()
    chunk_df = pd.read_csv(io.StringIO(r.text))
    chunks.append(chunk_df)
    print(f'OK ({len(chunk_df)} rows)')
    time.sleep(1)

print(f'All {len(windows)} chunks downloaded.')

# %% [markdown]
# ## Step 3 - Concatenate and Clean
#
# - Chunks are concatenated and deduplicated on the datetime column.
# - `datetime (UTC)` is promoted to the index and renamed `datetime`.
# - Metadata columns are dropped.

# %%
df = pd.concat(chunks, ignore_index=True)
df = df.drop_duplicates(subset='datetime (UTC)')

df['datetime (UTC)'] = pd.to_datetime(df['datetime (UTC)'])
df = df.set_index('datetime (UTC)').sort_index()
df.index.name = 'datetime'

df = df.drop(columns=[c for c in META_COLS if c in df.columns])

print('Shape      :', df.shape)
print('Date range :', df.index.min(), '->', df.index.max())
print('Columns    :', list(df.columns))
print()

missing = df.isnull().sum()
if missing.any():
    print('WARNING - Missing values:')
    print(missing[missing > 0])
else:
    print('OK - No missing values.')

display(df.head())

# %% [markdown]
# ## Step 4 - Save
#
# Written with `index=True` so the datetime index is the first CSV column.
# Notebook 01 reads it with `pd.read_csv(..., index_col=0, parse_dates=True)`.

# %%
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=True)
print(f'Saved {df.shape[0]} rows x {df.shape[1]} cols -> {OUTPUT_PATH}')
