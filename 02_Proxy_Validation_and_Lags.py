# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath('..'))
from heritageshm.diagnostics import shift_and_correlate
from heritageshm.viz import apply_theme

apply_theme(context='notebook')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/interim/aligned/st02_aligned_dataset.csv"
OUTPUT_PATH = "data/processed/feature_matrix.parquet"
FIG_PATH    = "outputs/figures/"

# ── Config ────────────────────────────────────────────────────────────────────
TARGET      = "absinc"
MAX_LAG_H   = 72          # maximum lag to screen (hours); 72h = 3 days thermal memory
LAG_STEP    = 1           # step in hours

# %% [markdown]
# ## Step 1 · Load Data

# %%
df = pd.read_csv(DATA_PATH)
df.index = pd.to_datetime(df.datetime)

# Drop non-proxy columns like charge or duplicate variables
EXCLUDE  = ["charge", "datetime", "temp", "hum", "oikolab_wind_speed (m/s)", "oikolab_wind_direction (deg)", "oikolab_total_cloud_cover (0-1)", TARGET]
proxies  = [c for c in df.columns if c not in EXCLUDE]

print(f"Loaded: {df.shape[0]} rows | {df.index[0]} → {df.index[-1]}")
print(f"Target : {TARGET}")
print(f"Proxies: {proxies}")
display(df[[TARGET] + proxies].describe().T)

# %% [markdown]
# ### Visualizing Target vs. Proxies
# Let's plot the target variable alongside each proxy to visually inspect their relationships over time.

# %%
fig, axes = plt.subplots(len(proxies), 1, figsize=(15, 4 * len(proxies)), sharex=True)
if len(proxies) == 1:
    axes = [axes]

for ax, proxy in zip(axes, proxies):
    # Plot Target
    color_target = 'black'
    ax.plot(df.index, df[TARGET], label=TARGET, color=color_target, alpha=0.7, linewidth=1)
    ax.set_ylabel(TARGET, color=color_target)
    ax.tick_params(axis='y', labelcolor=color_target)
    
    # Plot Proxy on twin axis
    ax2 = ax.twinx()
    color_proxy = 'tab:blue'
    ax2.plot(df.index, df[proxy], label=proxy, color=color_proxy, alpha=0.7, linewidth=1)
    ax2.set_ylabel(proxy, color=color_proxy)
    ax2.tick_params(axis='y', labelcolor=color_proxy)
    
    ax.set_title(f"{TARGET} vs {proxy}")
    ax.grid(True, alpha=0.3)
    
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2 · Initial Pearson Correlation
# Since our timeseries are already stationary, we screen for immediate linear relationships.

# %%
# Calculate Pearson correlation for all proxies
initial_corrs = df[[TARGET] + proxies].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("--- Initial Pearson Correlation (Lag 0) ---")
print(initial_corrs)

# %% [markdown]
# ## Step 3 · Lag Screening (Thermal Inertia)
# Structural responses to environmental conditions (like temperature) are typically delayed due to thermal inertia. By screening shifted cross-correlations, we align the proxy optimally with the target.

# %%
optimal_lags = {}
plt.figure(figsize=(10, 5))

for proxy in proxies:
    lags, corrs = shift_and_correlate(df, TARGET, proxy, MAX_LAG_H, LAG_STEP)
    
    # Find lag with maximum absolute correlation
    max_idx = np.argmax(np.abs(corrs))
    best_lag = lags[max_idx]
    best_corr = corrs[max_idx]
    
    optimal_lags[proxy] = best_lag
    
    print(f"{proxy}: Optimal Lag = {best_lag}h, r = {best_corr:.3f}")
    plt.plot(lags, corrs, label=proxy.split('_')[1] if '_' in proxy else proxy)

plt.title(f"Cross-Correlation w/ {TARGET} across varying Lags")
plt.xlabel("Lag (Hours)")
plt.ylabel("Pearson Correlation (r)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 4 · Assemble Feature Matrix
# Shift the proxies by their optimal lags to construct the final dataset.

# %%
df_features = df[[TARGET]].copy()

for proxy, lag in optimal_lags.items():
    # Shift proxy forward by 'lag' hours 
    if lag == 0:
        df_features[f"{proxy}_lag0"] = df[proxy]
    else:
        df_features[f"{proxy}_lag{lag}"] = df[proxy].shift(lag)

# Plot correlation heatmap of the final features
plt.figure(figsize=(10, 8))
sns.heatmap(df_features.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Matrix Correlation Heatmap")
plt.tight_layout()
plt.show()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_features.to_parquet(OUTPUT_PATH)
print(f"\nSaved feature matrix: {df_features.shape}")
print(f"Path: {OUTPUT_PATH}")
