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
# Downloads hourly weather data from the Oikolab API for a given location and date range,
# cleans the response, and saves the result in the format expected by **Notebook 01**.
#
# ---
#
# ## Output Format Contract
#
# | Requirement | Detail |
# |---|---|
# | **File** | `data/raw/proxies/oikolab_weather.csv` |
# | **Index** | `datetime` column (UTC timestamps), written as CSV first column |
# | **Index written** | `index=True` — Notebook 01 reads with `index_col=0, parse_dates=True` |
# | **Metadata columns** | Dropped: `coordinates (lat,lon)`, `model (name)`, `model elevation (surface)`, `utc_offset (hrs)` |
# | **Data columns** | All weather variables returned by the API, unchanged |
# | **Timezone** | UTC (as returned by Oikolab) — align with sensor timezone in Notebook 01 if needed |
#
# ---
#
# ## Steps
# 1. **Parameters** — Set location, date range, API key, and output path.
# 2. **Download** — Call the Oikolab API.
# 3. **Parse & clean** — Set datetime index, drop metadata columns, inspect.
# 4. **Save** — Write CSV with index.

# %% [markdown]
# ## Step 1 · Parameters
#
# | Parameter | Description |
# |---|---|
# | `LAT`, `LON` | Decimal coordinates of the monitoring site |
# | `START`, `END` | Inclusive date range for the download (`YYYY-MM-DD`). Should fully bracket the sensor monitoring window |
# | `API_KEY` | Oikolab API key |
# | `OUTPUT_PATH` | Destination CSV — must match `PROXY_FILE` in Notebook 01 |

# %%
import requests
import pandas as pd
import io
import os

# === USER INPUT ===
LAT         = 43.35343
LON         = 12.582047
START       = '2018-01-01'
END         = '2026-04-16'
API_KEY     = '8426a7e187b9481ab575814f707c8f8d'
OUTPUT_PATH = '../data/raw/proxies/oikolab_weather.csv'
# ==================

# Oikolab metadata columns — always present in the response, never needed downstream
META_COLS = [
    'coordinates (lat,lon)',
    'model (name)',
    'model elevation (surface)',
    'utc_offset (hrs)',
]

# %% [markdown]
# ## Step 2 · Download
#
# All available Oikolab parameters are requested so the raw file is complete.
# Column selection for Notebook 01 is done there, not here.

# %%
r = requests.get(
    'https://api.oikolab.com/weather',
    params={
        'param': [
            'temperature',
            'dewpoint_temperature',
            'relative_humidity',
            'wetbulb_temperature',
            'skin_temperature',
            'urban_temperature',
            'wind_speed',
            'wind_direction',
            'total_cloud_cover',
            'total_precipitation',
            'surface_solar_radiation',
            'surface_thermal_radiation',
        ],
        'lat': LAT,
        'lon': LON,
        'start': START,
        'end': END,
        'freq': 'H',
        'resample_method': 'mean',
        'format': 'csv',
    },
    headers={'api-key': API_KEY},
)

r.raise_for_status()
print('Download complete. Status:', r.status_code)

# %% [markdown]
# ## Step 3 · Parse and Clean
#
# - `datetime (UTC)` is promoted to the index and renamed `datetime`.
# - Metadata columns (`coordinates`, `model`, `elevation`, `utc_offset`) are dropped.
# - A `.head()` and missingness summary are shown for quick inspection.

# %%
df = pd.read_csv(io.StringIO(r.text))

# Promote datetime column to index
df['datetime (UTC)'] = pd.to_datetime(df['datetime (UTC)'])
df = df.set_index('datetime (UTC)')
df.index.name = 'datetime'

# Drop metadata columns (keep only weather variables)
df = df.drop(columns=[c for c in META_COLS if c in df.columns])

print('Shape            :', df.shape)
print('Date range       :', df.index.min(), ' → ', df.index.max())
print('Columns          :', list(df.columns))
print()

# Missingness check
missing = df.isnull().sum()
if missing.any():
    print('⚠ Missing values detected:')
    print(missing[missing > 0])
else:
    print('✓ No missing values.')

display(df.head())

# %% [markdown]
# ## Step 4 · Save
#
# Written with `index=True` so the datetime index becomes the first CSV column.
# Notebook 01 reads it with `pd.read_csv(..., index_col=0, parse_dates=True)`.

# %%
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=True)
print(f'Saved {df.shape[0]} rows × {df.shape[1]} cols → {OUTPUT_PATH}')
