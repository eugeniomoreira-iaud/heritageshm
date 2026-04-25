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
# # Notebook 02 · Proxy Validation and Lag Screening
#
# **Goal:** Rank candidate environmental regressors for the target variable (e.g., `absinc`) using Pearson correlation and lagged cross-correlation screening. Since the timeseries are already stationary, we focus solely on Pearson correlation to understand the best fit and determine optimal thermal inertia lags.
#
# **Steps:**
# 1. **Load Aligned Data** — Load the dataset from Notebook 01.
# 2. **Initial Pearson Correlation** — Check immediate linear relationships between the target and proxies.
# 3. **Lag Screening** — Sweep through time lags to find the maximum cross-correlation (thermal memory).
# 4. **Feature Matrix Generation** — Align proxies with their optimal lags and save as a Parquet feature matrix for modeling.

# %%
import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display

sys.path.insert(0, os.path.abspath('..'))
from heritageshm.diagnostics import get_longest_contiguous_block, screen_optimal_lags
from heritageshm.features import build_optimal_feature_matrix
from heritageshm.viz import apply_theme, plot_target_vs_proxies, plot_cross_correlation_lags, plot_correlation_heatmap

apply_theme(context='notebook')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/interim/aligned/st02_aligned_dataset.csv"
OUTPUT_PATH = "data/processed/feature_matrix.csv"
LAGS_PATH   = "data/processed/optimal_lags.csv"
FIG_PATH    = "outputs/figures/"

# %% [markdown]
# ### Parameter Tuning Guidance
#
# **TARGET**: The primary structural variable being analyzed (e.g., 'absinc'). 
# Must match the column name used as the target in Notebook 01.
#
# **MAX_LAG_H**: The maximum delay (in hours) to screen for thermal inertia.
# - Masonry structures: 24 to 72 hours.
# - Metal/slender structures: 6 to 12 hours.
#
# **LAG_STEP**: Resolution of the lag sweep.
# - Usually `1` (hourly), but can be increased to speed up screening.

# %%
# ── Config ────────────────────────────────────────────────────────────────────
TARGET      = "absinc"
MAX_LAG_H   = 12
LAG_STEP    = 1

# %% [markdown]
# ## Step 1 · Load Data

# %%
df = pd.read_csv(DATA_PATH)
df.index = pd.to_datetime(df.datetime)

# %% [markdown]
# ### Parameter Tuning Guidance
#
# **EXCLUDE**: A list of data columns to ignore during correlation screening.
# - Drop variables that are not environmental proxies (e.g., internal sensor metrics like `temp`, `hum`, `charge`).
# - Drop variables structurally irrelevant to the analysis.

# %%
# Drop non-proxy columns like charge or duplicate variables
EXCLUDE  = [
    "charge", "datetime", "temp", "hum", 
    "wind_speed (m/s)", "wind_direction (deg)", 
    "total_cloud_cover (0-1)", TARGET
]
proxies  = [c for c in df.columns if c not in EXCLUDE]

print(f"Loaded: {df.shape[0]} rows | {df.index[0]} → {df.index[-1]}")
print(f"Target : {TARGET}")
print(f"Proxies: {proxies}")
display(df[[TARGET] + proxies].describe().T)

# %% [markdown]
# ### Visualizing Target vs. Proxies
# Let's plot the target variable alongside each proxy to visually inspect their relationships over time.

# %%
plot_target_vs_proxies(df, TARGET, proxies, save_plot=True, save_path=FIG_PATH, filename='02_01_target_vs_proxies')

# %% [markdown]
# ## Step 1b · Reference Window and Differenced Series
#
# **Reference window.** The target signal contains gaps of varying length. The analysis is restricted to the longest contiguous non-missing block of the target variable, identified by detecting time jumps larger than 1.5× the expected sampling step. The boundaries of this window — `REF_START` and `REF_END` — are stored as constants for reuse in Notebook 04 as the model training period.
#
# **First-order differencing.** All variables within the reference window are first-differenced before Pearson correlation and lag screening. This conservative transformation guarantees stationarity regardless of the individual integration order of each series, eliminating the risk of spurious correlation. Differencing is applied exclusively for proxy ranking and lag identification; the level series
# `df_ref` is preserved and used for feature matrix assembly in Step 4.

# %%
# --- Identify longest contiguous non-missing block of the target ---
REF_START, REF_END = get_longest_contiguous_block(df, target_col=TARGET, expected_step='1h')

# Slice to reference window
df_ref = df.loc[REF_START:REF_END].copy()

# --- Build first-differenced DataFrame for correlation screening ---
df_diff = df_ref[[TARGET] + proxies].diff().dropna()

print(f"\nReference window shape   : {df_ref.shape}")
print(f"Differenced shape        : {df_diff.shape}")

# %% [markdown]
# ## Step 2 · Initial Pearson Correlation
# Since our timeseries are already stationary, we screen for immediate linear relationships.

# %%
# Calculate Pearson correlation for all proxies
initial_corrs = df_diff.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("--- Initial Pearson Correlation (Lag 0) ---")
print(initial_corrs)

# %% [markdown]
# ## Step 3 · Lag Screening (Thermal Inertia)
# Structural responses to environmental conditions (like temperature) are typically delayed due to thermal inertia. By screening shifted cross-correlations, we align the proxy optimally with the target.

# %%
optimal_lags, all_lags, corrs_dict = screen_optimal_lags(df_diff, TARGET, proxies, MAX_LAG_H, LAG_STEP)

plot_cross_correlation_lags(all_lags, corrs_dict, optimal_lags, TARGET, save_plot=True, save_path=FIG_PATH, filename='02_02_cross_correlation_lags')

# %% [markdown]
# ## Step 4 · Assemble Feature Matrix
# Shift the proxies by their optimal lags to construct the final dataset.

# %%
# %% [markdown]
# ### Parameter Tuning Guidance
#
# **EXCLUDE_FROM_FEATURES**: Environmental proxies to discard from the final dataset.
# - Used to prune highly collinear or redundant proxies (e.g., overlapping temperature scales) based on Step 2 and 3 findings.
# - Ensures downstream imputation models remain interpretable and numerically stable.

# %%
# Drop proxies excluded from the feature matrix
EXCLUDE_FROM_FEATURES = [
    'surface_thermal_radiation (W/m^2)',
    'urban_temperature (degC)',
    'wetbulb_temperature (degC)',
    'dewpoint_temperature (degC)',
    'temperature (degC)',
]

# Assemble feature matrix from level series using optimal lags
df_features = build_optimal_feature_matrix(df_ref, TARGET, optimal_lags, exclude_proxies=EXCLUDE_FROM_FEATURES)

# Plot correlation heatmap
plot_correlation_heatmap(df_features, title="Feature Matrix Correlation Heatmap", save_plot=True, save_path=FIG_PATH, filename='02_03_correlation_heatmap')

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_features.to_csv(OUTPUT_PATH)
print(f"\nSaved feature matrix: {df_features.shape}")
print(f"Path: {OUTPUT_PATH}")

final_lags = {p: l for p, l in optimal_lags.items() if p not in EXCLUDE_FROM_FEATURES}
pd.Series(final_lags, name="optimal_lag").to_csv(LAGS_PATH, index_label="proxy")
print(f"Saved optimal lags: {LAGS_PATH}")

# %%
