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
# # Notebook 01 · Data Quality, Proxies, and Gap Characterization
#
# **Pipeline position**: Phase A, Step 1 — runs after Notebook 00 (sensor)
# and any proxy extraction notebooks (e.g. 00b, 00c …).
#
# ---
#
# ## Expected Input Format
#
# All datasets consumed by this notebook must satisfy the following contract.
# Any preparation that does not meet it must be done *before* this notebook is run.
#
# | Requirement | Detail |
# |---|---|
# | **File format** | CSV |
# | **First column** | Datetime index — loaded with `index_col=0, parse_dates=True` |
# | **Index name** | Any name is accepted; normalised to `'datetime'` on load |
# | **Index regularity** | Near-regular is acceptable; gaps are handled here |
# | **Values** | Clean and compensated — spikes removed, power-outage zeros masked as `NaN` |
# | **Units** | Consistent within each file; no unit conversion is performed here |
# | **Timezone** | All files must share the same timezone (or all be timezone-naive) |
#
# Proxy files may have any column names — this notebook does not assume any
# specific source schema (Oikolab, ERA5, local station, etc.).
#
# ---
#
# ## Steps
# 1. **Load sensor** — Load the preprocessed sensor CSV; inspect head and sample rate.
# 2. **Load proxy** — Load any proxy CSV; inspect all available columns.
# 3. **Select proxy columns** — Choose which columns to carry forward.
# 4. **Coverage & rate audit** — Compare temporal windows and native sample rates.
# 5. **Alignment** — Resample proxy onto the sensor index.
# 6. **Gap characterization** — Diagnose missingness mechanism in the sensor target column.
# 7. **Save** — Export the aligned dataset.
#
# ## Outputs
#
# | Artifact | Path | Description |
# |---|---|---|
# | Aligned dataset | `data/interim/aligned/{station}_aligned_dataset.csv` | Sensor + proxy, resampled to sensor frequency |
# | Gap histogram   | `outputs/figures/01_01_{station}_gap_histogram.png`  | Distribution of sensor gap lengths |
# | Gap stats table | `outputs/tables/01_01_{station}_gap_stats.csv`       | Descriptive statistics of sensor gaps |

# %%
import os
import pandas as pd
import numpy as np
from IPython.display import display

from heritageshm.dataloader import load_preprocessed_sensor, save_interim_data
from heritageshm.preprocessing import align_multiple_proxies
from heritageshm.diagnostics import characterize_gaps
from heritageshm.viz import apply_theme

apply_theme(context='notebook')

# %% [markdown]
# ## Step 1 · Load Sensor Data
#
# ### Parameter Tuning Guidance
#
# | Parameter | Type | Description |
# |---|---|---|
# | `TARGET_STATION` | `str` | Station identifier — must match `{station}_preprocessed.csv` in `data/interim/sensor/` |
# | `TARGET_COL` | `str` | Primary structural response column used for gap characterization in Step 6 |
#
# **Prerequisite**: Notebook 00 must have been run to generate the preprocessed sensor CSV.

# %%
# === USER INPUT ===
TARGET_STATION = 'st02'
TARGET_COL     = 'absinc'   # Primary structural response column
# ==================

SENSOR_FILE = 'data/interim/sensor/%s_preprocessed.csv' % TARGET_STATION
df_sensor   = load_preprocessed_sensor(SENSOR_FILE)
df_sensor.index.name = 'datetime'

# --- Infer sensor native sample rate ---
sensor_diffs = df_sensor.index.to_series().diff().dropna()
sensor_rate  = sensor_diffs.median()

print('Sensor station   : %s'        % TARGET_STATION)
print('Shape            : %s'         % str(df_sensor.shape))
print('Date range       : %s  \u2192  %s' % (df_sensor.index.min(), df_sensor.index.max()))
print('Median sample \u0394t : %s'         % sensor_rate)
print('Columns          : %s'         % list(df_sensor.columns))
display(df_sensor.head())

# %% [markdown]
# ## Step 2 · Load Proxy Data
#
# ### Parameter Tuning Guidance
#
# | Parameter | Type | Description |
# |---|---|---|
# | `PROXY_FILE` | `str` | Path to any proxy CSV. Change this to point at a different source without modifying anything else in the notebook |
#
# The loader uses `index_col=0, parse_dates=True`.
# No specific column names are assumed — the full column list is printed below
# so you can select what to keep in Step 3.
#
# **All preparation** (unit conversion, timezone harmonisation, metadata-column
# removal) must have been done in the proxy extraction notebook before this step.

# %%
# === USER INPUT ===
PROXY_FILE = 'data/raw/proxies/oikolab_weather.csv'
# ==================

df_proxy = pd.read_csv(PROXY_FILE, index_col=0, parse_dates=True)
df_proxy.index.name = 'datetime'

# --- Infer proxy native sample rate ---
proxy_diffs = df_proxy.index.to_series().diff().dropna()
proxy_rate  = proxy_diffs.median()

print('Proxy file       : %s'         % PROXY_FILE)
print('Shape            : %s'         % str(df_proxy.shape))
print('Date range       : %s  \u2192  %s' % (df_proxy.index.min(), df_proxy.index.max()))
print('Median sample \u0394t : %s'         % proxy_rate)
print('\nAvailable columns:')
for i, c in enumerate(df_proxy.columns):
    print('  [%2d]  %s' % (i, c))

display(df_proxy.head())

# %% [markdown]
# ## Step 3 · Select Proxy Columns
#
# ### Parameter Tuning Guidance
#
# | Parameter | Type | Description |
# |---|---|---|
# | `PROXY_COLS` | `list[str]` | Column names to retain. Copy names exactly from the list printed in Step 2 |
# | `PROXY_LABEL` | `str` | Short label prepended to column names after alignment (e.g. `'oikolab'` → `'oikolab_temperature (degC)'`). Change per source to avoid collisions |
# | `MAX_PROXY_MISSING_FRAC` | `float` | Post-alignment warning threshold (0–1). Default `0.10` |
#
# **Selection guidance for masonry / historic stone structures**:
# - **Temperature** — primary driver of thermal expansion/contraction.
# - **Relative humidity / dewpoint** — co-primary for hygrothermal coupling in
#   porous stone; do not omit.
# - **Solar / thermal radiation** — important for structures with significant
#   thermal mass or direct sun exposure.
# - **Wind** — relevant for tall or slender elements.
#
# Use cointegration tests in Notebook 02 to validate this selection quantitatively.

# %%
# === USER INPUT ===
PROXY_COLS = [
    'temperature (degC)',
    'dewpoint_temperature (degC)',
    'relative_humidity (0-1)',
    'wetbulb_temperature (degC)',
    'skin_temperature (degC)',
    'urban_temperature (degC)',
    # 'wind_speed (m/s)',
    # 'wind_direction (deg)',
    # 'total_cloud_cover (0-1)',
    # 'total_precipitation (mm of water equivalent)',
    'surface_solar_radiation (W/m^2)',
    'surface_thermal_radiation (W/m^2)',
]

PROXY_LABEL            = 'oikolab'   # change per source (era5, station_a, etc.)
MAX_PROXY_MISSING_FRAC = 0.10        # warn if any column exceeds 10 % missing after alignment
# ==================

missing_requested = [c for c in PROXY_COLS if c not in df_proxy.columns]
if missing_requested:
    print('\u26a0 WARNING: The following requested columns were not found and will be skipped:')
    for c in missing_requested:
        print('    %s' % c)

df_proxy = df_proxy[[c for c in PROXY_COLS if c in df_proxy.columns]].copy()
print('Selected %d proxy column(s): %s' % (len(df_proxy.columns), list(df_proxy.columns)))

# %% [markdown]
# ## Step 4 · Coverage and Sample-Rate Audit
#
# Compares the sensor and proxy datasets on two axes before any data is merged:
#
# - **Temporal coverage** — does the proxy fully bracket the sensor window?
#   A silent mismatch produces NaN-filled proxy columns for the uncovered period
#   with no further downstream warning.
# - **Native sample rate** — are the two datasets at compatible frequencies?
#   A rate mismatch is not an error, but determines whether alignment will
#   downsample or upsample the proxy.

# %%
sensor_start, sensor_end = df_sensor.index.min(), df_sensor.index.max()
proxy_start,  proxy_end  = df_proxy.index.min(),  df_proxy.index.max()

print('\u2500' * 60)
print('%-22s  %-26s  %s' % ('', 'Start', 'End'))
print('\u2500' * 60)
print('%-22s  %-26s  %s' % ('Sensor', str(sensor_start), str(sensor_end)))
print('%-22s  %-26s  %s' % ('Proxy',  str(proxy_start),  str(proxy_end)))
print('\u2500' * 60)
print()
print('Sensor median \u0394t : %s' % sensor_rate)
print('Proxy  median \u0394t : %s' % proxy_rate)
print()

# Coverage
coverage_ok = (proxy_start <= sensor_start) and (proxy_end >= sensor_end)
if coverage_ok:
    print('\u2713 Proxy fully covers the sensor monitoring window.')
else:
    if proxy_start > sensor_start:
        print('\u26a0 WARNING: Proxy starts %s after sensor start — '
              'sensor rows before %s will have NaN proxies.'
              % (proxy_start - sensor_start, proxy_start))
    if proxy_end < sensor_end:
        print('\u26a0 WARNING: Proxy ends %s before sensor end — '
              'sensor rows after %s will have NaN proxies.'
              % (sensor_end - proxy_end, proxy_end))

# Rate comparison
if sensor_rate == proxy_rate:
    print('\u2713 Sensor and proxy share the same native sample rate.')
elif proxy_rate < sensor_rate:
    ratio = int(round(sensor_rate / proxy_rate))
    print('\u2139 Proxy is ~%dx finer than sensor '
          '(proxy \u0394t=%s, sensor \u0394t=%s). '
          'Alignment will downsample (aggregate) the proxy.'
          % (ratio, proxy_rate, sensor_rate))
else:
    ratio = int(round(proxy_rate / sensor_rate))
    print('\u2139 Proxy is ~%dx coarser than sensor '
          '(proxy \u0394t=%s, sensor \u0394t=%s). '
          'Alignment will upsample (interpolate) the proxy.'
          % (ratio, proxy_rate, sensor_rate))

assert coverage_ok, (
    'Proxy temporal coverage is insufficient. '
    'Re-run the proxy extraction notebook to extend the download window '
    'so it fully brackets the sensor monitoring period.'
)

# %% [markdown]
# ## Step 5 · Alignment
#
# ### Parameter Tuning Guidance
#
# | Parameter | Type | Default | Description |
# |---|---|---|---|
# | `TARGET_FREQ` | `str` | `'h'` | Resampling target frequency. `'h'` suits most SHM applications; `'30min'`/`'15min'` for fast dynamics; `'2h'`/`'3h'` for slow thermal responses |
#
# The proxy is resampled to `TARGET_FREQ` and joined onto the sensor index.
# `add_prefix=True` uses `PROXY_LABEL` to prefix each proxy column, preventing
# silent name collisions when a second proxy source is added later.

# %%
# === USER INPUT ===
TARGET_FREQ = 'h'   # Hourly — recommended default for masonry SHM
# ==================

proxies_dict = {PROXY_LABEL: df_proxy}

df_aligned = align_multiple_proxies(
    df_sensor,
    proxies_dict,
    resample_freq=TARGET_FREQ,
    add_prefix=True,
)

print('Aligned dataset shape: %s' % str(df_aligned.shape))
display(df_aligned.head())

# --- Post-alignment missingness audit ---
missing_frac           = df_aligned.isnull().mean().rename('missing_frac').to_frame()
missing_frac['pct']    = (missing_frac['missing_frac'] * 100).round(2)
missing_frac['n_miss'] = df_aligned.isnull().sum()

print('\n--- Post-Alignment Missingness Audit ---')
display(missing_frac.sort_values('missing_frac', ascending=False))

fully_missing = missing_frac[missing_frac['missing_frac'] == 1.0].index.tolist()
if fully_missing:
    raise ValueError(
        'The following columns are entirely NaN after alignment — '
        'check index compatibility and proxy coverage: %s' % fully_missing
    )

high_missing = missing_frac[
    (missing_frac['missing_frac'] > MAX_PROXY_MISSING_FRAC) &
    (missing_frac['missing_frac'] < 1.0)
].index.tolist()
if high_missing:
    print('\n\u26a0 WARNING: Columns exceeding the %.0f%% missing threshold:'
          % (MAX_PROXY_MISSING_FRAC * 100))
    for col in high_missing:
        print('  %s \u2014 %.1f%% missing' % (col, missing_frac.loc[col, 'pct']))
    print('  Re-run the proxy extraction notebook or remove these columns from PROXY_COLS.')
else:
    print('\n\u2713 All columns are within the %.0f%% missing threshold.'
          % (MAX_PROXY_MISSING_FRAC * 100))

# %% [markdown]
# ## Step 6 · Gap Characterization
#
# Analyses missing data in `TARGET_COL` (the primary structural response).
# The test classifies the missingness mechanism by correlating a binary
# missingness indicator with every other numeric column in the aligned dataset:
#
# | Mechanism | Meaning | Implication |
# |---|---|---|
# | **MCAR** | Gaps are random, unrelated to any observed variable | Standard interpolation may suffice |
# | **MAR** | Missingness correlates with an observed covariate | Regression-based imputation preferred |
# | **MNAR-power** | Missingness correlates with `charge` ≈ 0 | Power-outage gaps; proxy-based NeuralProphet imputation |
#
# ### Parameter Tuning Guidance
#
# | Parameter | Type | Default | Description |
# |---|---|---|---|
# | `HISTOGRAM_BINS` | `int` | `50` | Number of histogram bins (20–100 typical) |
# | `HISTOGRAM_COLOR` | `str` | `'black'` | Bar fill colour. Named (`'black'`, `'steelblue'`) or hex (`'#2c3e50'`) |
# | `max_impute_gap` | `int` | `0` | Max consecutive NaNs to fill by linear interpolation *before* classification. Keep at `0` to classify raw gaps |

# %%
# === USER INPUT ===
HISTOGRAM_BINS  = 50       # Number of histogram bins
HISTOGRAM_COLOR = 'black'  # Bar fill colour
# ==================

GAP_FIG_PATH = 'outputs/figures/01_01_%s_gap_histogram.png'  % TARGET_STATION
GAP_TAB_PATH = 'outputs/tables/01_01_%s_gap_stats.csv'       % TARGET_STATION
ALIGNED_PATH = 'data/interim/aligned/%s_aligned_dataset.csv' % TARGET_STATION

os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/tables',  exist_ok=True)
os.makedirs(os.path.dirname(ALIGNED_PATH), exist_ok=True)

# %%
gap_characterization_ok = False

try:
    df_aligned, gap_stats, gap_lengths = characterize_gaps(
        df_aligned,
        target_col=TARGET_COL,
        max_impute_gap=0,
        histogram_bins=HISTOGRAM_BINS,
        bar_color=HISTOGRAM_COLOR,
        save_plot_path=GAP_FIG_PATH,
    )
    gap_characterization_ok = True

except Exception as e:
    print('\u26a0 Gap characterization failed: %s' % e)
    print('The aligned dataset will NOT be saved. Resolve the error before proceeding.')

# %% [markdown]
# ### Gap Statistics Summary

# %%
if gap_characterization_ok:
    total_obs     = len(df_aligned)
    total_missing = df_aligned[TARGET_COL].isnull().sum()
    missing_pct   = total_missing / total_obs * 100

    print('\n--- Gap Statistics for \'%s\' ---' % TARGET_COL)
    print('Total observations : %d'           % total_obs)
    print('Total missing      : %d  (%.2f%%)' % (total_missing, missing_pct))
    display(gap_stats.rename('value').to_frame())

    gap_stats_export = gap_stats.rename('value').to_frame()
    gap_stats_export.loc['total_missing_pct'] = missing_pct
    gap_stats_export.to_csv(GAP_TAB_PATH)
    print('\nGap statistics saved to : ' + GAP_TAB_PATH)
    print('Gap histogram saved to  : ' + GAP_FIG_PATH)

# %% [markdown]
# ## Step 7 · Save Aligned Dataset
#
# Saves only if Step 6 completed without errors.

# %%
if gap_characterization_ok:
    save_interim_data(df_aligned, ALIGNED_PATH)
    print('Aligned dataset saved to: ' + ALIGNED_PATH)
else:
    print('Save skipped \u2014 gap characterization did not complete successfully.')
