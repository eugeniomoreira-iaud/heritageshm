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
# One year of 63 parameters costs ~752 units, so the safe window is 6 months
# (~376 units). The download is split into 6-month chunks and concatenated.
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
# | **Data columns** | All ERA5 variables returned by the API, unchanged |
# | **Timezone** | UTC - align to local time in Notebook 01 if the sensor uses local time |
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
CHUNK_MONTHS = 6   # 6 months x 63 params ~ 376 units (limit is 500)
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
# Chunks are built by advancing a cursor in steps of `CHUNK_MONTHS`.
# A 1-second pause between requests avoids rate-limiting.

# %%
PARAMS = [
    'temperature', 'dewpoint_temperature', 'relative_humidity',
    'wetbulb_temperature', 'humidex_index', 'heating_degreeday',
    'cooling_degreeday', 'sea_surface_temperature', 'skin_temperature',
    'urban_temperature', 'wind_speed', 'wind_direction',
    '10m_wind_gust', '10m_u_component_of_wind', '10m_v_component_of_wind',
    '100m_wind_speed', '100m_wind_direction', '100m_u_component_of_wind',
    '100m_v_component_of_wind', 'total_cloud_cover', 'total_precipitation',
    'surface_pressure', 'mean_sea_level_pressure', 'surface_solar_radiation',
    'surface_thermal_radiation', 'direct_normal_solar_radiation',
    'surface_direct_solar_radiation', 'surface_diffuse_solar_radiation',
    'snowfall', 'snow_depth', 'soil_temperature_level_1',
    'soil_temperature_level_2', 'soil_temperature_level_3',
    'soil_temperature_level_4', 'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4', 'boundary_layer_height',
    'cloud_base_height', 'convective_inhibition',
    'convective_available_potential_energy', 'forecast_surface_roughness',
    'friction_velocity', 'high_vegetation_cover', 'low_vegetation_cover',
    'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation',
    'high_cloud_cover', 'medium_cloud_cover', 'low_cloud_cover',
    'evaporation', 'potential_evaporation', 'skin_reservoir_content',
    'surface_latent_heat_flux', 'surface_runoff', 'sub_surface_runoff',
    'total_column_rain_water', 'total_column_water_vapour',
    'downward_uv_radiation_at_the_surface', 'surface_net_solar_radiation',
    'surface_net_thermal_radiation', 'forecast_albedo',
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
print(f'Fetching {len(windows)} chunks of {CHUNK_MONTHS} months each ...')

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
