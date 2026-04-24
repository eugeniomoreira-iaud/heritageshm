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
#     display_name: heritageshm_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 04 · Grey-Box Decomposition and Residual-Based Monitoring
#
# **Goal:** Apply the NeuralProphet grey-box framework to the fully reconstructed
# inclinometer series and perform residual-based anomaly detection.
#
# **Phase B of the `heritageshm` pipeline:**
#
# 1. **Load Data** — Imputed series (Notebook 03) + feature matrix (Notebook 02).
# 2. **Residual Diagnostics on Input** — Verify that the imputed series preserves
#    the structural-environmental relationship established in the reference period.
# 3. **NeuralProphet Configuration and Training** — Grey-box decomposition with
#    trend, Fourier seasonalities, AR lags, and lagged environmental regressors.
# 4. **Component Extraction and Variance Attribution** — Interpret the decomposed
#    signal: trend, seasonality, AR memory, exogenous contribution, residual.
# 5. **Residual Diagnostics** — ADF stationarity and Ljung-Box whiteness tests
#    on NeuralProphet residuals.
# 6. **EWMA and CUSUM Control Charts** — Fit control limits on the reference
#    period; apply to the full monitoring horizon.
# 7. **Joint Alarm Detection and Reporting** — Identify and characterise
#    anomalous structural episodes.
# 8. **Save Outputs** — Residuals, component table, alarm flags.

# %%
import sys
import os
import warnings
import numpy as np
import pandas as pd
from IPython.display import display
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath('..'))

from heritageshm.diagnostics import (
    test_signal_stationarity,
    test_residual_stationarity,
    test_residual_whiteness,
)
from heritageshm.decomposition import (
    build_neuralprophet_df,
    configure_model,
    train_model,
    extract_components,
    compute_residuals,
    summarise_components,
)
from heritageshm.monitoring import (
    compute_reference_stats,
    ewma_chart,
    cusum_chart,
    joint_alarm,
    alarm_summary,
)
from heritageshm.viz import apply_theme

apply_theme(context='notebook')

# ── Paths ──────────────────────────────────────────────────────────────────────
IMPUTED_PATH      = "data/processed/absinc_imputed.csv"
FEATURE_PATH      = "data/processed/feature_matrix.csv"
LAGS_PATH         = "data/processed/optimal_lags.csv"
FIG_PATH          = "outputs/figures/"
TABLE_PATH        = "outputs/tables/"
MODEL_PATH        = "outputs/models/"

os.makedirs(FIG_PATH,   exist_ok=True)
os.makedirs(TABLE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# %% [markdown]
# ### Parameter Tuning Guidance
#
# **TARGET**: The structural response variable — must match Notebooks 00–03.
#
# **REGRESSOR_COLS**: Lagged proxy columns retained from the feature matrix
# (Notebook 02). These must be available at every time step because they come
# from gapless reanalysis data.
#
# **N_LAGS**: AR lags passed to NeuralProphet. For hourly masonry data, 24 h
# captures one full diurnal cycle of structural memory beyond the proxy signal.
# Increase to 48 h if the Ljung-Box test fails (residual autocorrelation remains).
#
# **REF_START / REF_END**: Reference (training) window. Must match the longest
# contiguous block identified in Notebook 02. Control chart limits are anchored
# to this period.
#
# **EWMA_LAMBDA**: Smoothing parameter λ ∈ (0, 1]. Lower = more smoothing,
# better detection of gradual drift. Recommended: 0.1–0.2 for masonry.
#
# **EWMA_L**: Control limit multiplier (in σ). Default 3.0 → ARL₀ ≈ 370.
#
# **CUSUM_K**: Allowance parameter (in σ units). k = 0.5 targets shifts ≥ 1σ.
#
# **CUSUM_H**: Decision interval. h = 5 → ARL₀ ≈ 465 for normally distributed residuals.
#
# **JOINT_WINDOW**: Coincidence window (hours). An alarm is raised only when
# both EWMA and CUSUM remain in violation for this many consecutive steps.
# Default 24 h suppresses single-spike false positives.

# %%
# ── Config ────────────────────────────────────────────────────────────────────
TARGET         = "absinc"
N_LAGS         = 24
VALID_FRACTION = 0.15
EPOCHS         = 150
LEARNING_RATE  = 0.001

# Control chart parameters
EWMA_LAMBDA    = 0.15
EWMA_L         = 3.0
CUSUM_K        = 0.5
CUSUM_H        = 5.0
JOINT_WINDOW   = 24   # hours

# Reference period — update to match Notebook 02 output
REF_START = "2018-08-01"   # ← replace with actual ref_start from Notebook 02
REF_END   = "2020-12-31"   # ← replace with actual ref_end from Notebook 02

# %% [markdown]
# ## Step 1 · Load Data

# %%
df_imputed = pd.read_csv(IMPUTED_PATH, parse_dates=["datetime"], index_col="datetime")
df_features = pd.read_csv(FEATURE_PATH, parse_dates=["datetime"], index_col="datetime")
optimal_lags = pd.read_csv(LAGS_PATH, index_col="proxy")["optimal_lag"].to_dict()

# Identify regressor columns: all feature matrix columns except the target
REGRESSOR_COLS = [c for c in df_features.columns if c != TARGET]

# Merge imputed target with lagged proxy features on a shared hourly index
df = df_imputed[[TARGET, f"{TARGET}_imputed_flag"]].join(df_features[REGRESSOR_COLS], how='left')

# Reindex onto a complete regular hourly grid to guarantee no implicit gaps
full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1h")
df = df.reindex(full_idx)

print(f"Merged dataset   : {df.shape}")
print(f"Date range       : {df.index[0]} → {df.index[-1]}")
print(f"Target NaN       : {df[TARGET].isna().sum()} rows after imputation")
print(f"Regressors       : {REGRESSOR_COLS}")
display(df.head(3))

# %% [markdown]
# ## Step 2 · Pre-Decomposition Diagnostics
#
# Before training NeuralProphet, we verify two conditions:
# (a) the imputed target series is not I(2) or higher — a NeuralProphet
#     requirement since it handles I(0)/I(1) series via internal detrending;
# (b) the structural-environmental relationship established during the reference
#     period is preserved in the imputed series (downstream preservation test).

# %%
print("=== ADF Integration Order — Imputed Target and Regressors ===")
cols_to_test = [TARGET] + REGRESSOR_COLS
adf_table = test_signal_stationarity(df.dropna(subset=[TARGET]), cols_to_test)
display(adf_table)

# %%
# Downstream preservation test: Pearson correlation between imputed target
# and the primary proxy, compared across reference vs. imputed-only periods.
primary_proxy = REGRESSOR_COLS[0]  # assumes ranked by Notebook 02 output

ref_mask  = (df.index >= REF_START) & (df.index <= REF_END)
imp_mask  = df[f"{TARGET}_imputed_flag"] == 1

r_ref = df.loc[ref_mask, [TARGET, primary_proxy]].dropna().corr().iloc[0, 1]
r_imp = df.loc[imp_mask, [TARGET, primary_proxy]].dropna().corr().iloc[0, 1]

print(f"\n--- Downstream Preservation Test (primary proxy: {primary_proxy}) ---")
print(f"Pearson r — Reference period  : {r_ref:.4f}")
print(f"Pearson r — Imputed segments  : {r_imp:.4f}")
print(f"Δr = {abs(r_ref - r_imp):.4f}  {'✓ Preserved' if abs(r_ref - r_imp) < 0.15 else '⚠ Degraded — review imputation'}")

# %% [markdown]
# ## Step 3 · NeuralProphet Configuration and Training
#
# ### Parameter Tuning Guidance
#
# **Training window**: NeuralProphet is trained exclusively on the reference period
# (longest contiguous block). This ensures that control chart limits derived from
# reference-period residuals are consistent with the model's in-distribution behaviour.
# The full dataset is used only for prediction (component extraction).
#
# **Validation split**: 15% of the training window is held out for loss monitoring.
# If validation loss diverges from training loss, reduce `EPOCHS` or increase
# `trend_reg` / `ar_reg` to strengthen regularisation.

# %%
# Restrict training to the reference period (observed data only, no imputed rows)
df_ref = df.loc[REF_START:REF_END].copy()
df_ref = df_ref[df_ref[f"{TARGET}_imputed_flag"] == 0]  # observed only
df_ref = df_ref.dropna(subset=[TARGET] + REGRESSOR_COLS)

print(f"Training window  : {df_ref.index[0]} → {df_ref.index[-1]}")
print(f"Training rows    : {len(df_ref)} (observed only, no imputed rows)")

# Build NeuralProphet input format
df_train_np = build_neuralprophet_df(df_ref, TARGET, REGRESSOR_COLS)

# Configure model
model = configure_model(
    regressor_cols=REGRESSOR_COLS,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    n_lags=N_LAGS,
    ar_reg=0.1,
    trend_reg=0.5,
    seasonality_reg=0.1,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    batch_size=64,
)

# Train
model, metrics, df_train_split, df_val_split = train_model(
    model, df_train_np, valid_fraction=VALID_FRACTION
)

# Plot training loss
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(metrics["Loss"],     label="Training Loss",   linewidth=1.5)
ax.plot(metrics["Loss_val"], label="Validation Loss", linewidth=1.5, linestyle='--')
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("NeuralProphet Training / Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "04_01_training_loss.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_PATH, "04_01_training_loss.svg"), format='svg', bbox_inches='tight')
plt.show()

# Save model
model.save(os.path.join(MODEL_PATH, "neuralprophet_greybox.np"))
print("Model saved.")

# %% [markdown]
# ## Step 4 · Full-Series Prediction and Component Extraction
#
# Prediction is run on the **full** dataset (reference + imputed periods).
# NeuralProphet uses the trained weights to decompose the entire series,
# including the reconstructed gap intervals.

# %%
# Build full-series NeuralProphet DataFrame (include imputed rows for prediction)
df_full_np = build_neuralprophet_df(
    df.dropna(subset=[TARGET] + REGRESSOR_COLS),
    TARGET,
    REGRESSOR_COLS
)

df_pred, component_cols = extract_components(model, df_full_np)
residuals = compute_residuals(df_pred, target_col='y', yhat_col='yhat1')

print(f"\nComponent columns identified: {component_cols}")
print(f"Residual series length      : {len(residuals)}")

# Variance attribution table
comp_summary = summarise_components(df_pred, component_cols)
comp_summary.to_csv(os.path.join(TABLE_PATH, "04_component_variance.csv"))
display(comp_summary)

# %% [markdown]
# ## Step 5 · Component Visualisation

# %%
# --- Plot 1: Observed vs. Fitted ---
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(df_pred.index, df_pred['y'],     color='black',     linewidth=0.6, label='Observed / Imputed', zorder=3)
ax.plot(df_pred.index, df_pred['yhat1'], color='steelblue', linewidth=1.0, label='NeuralProphet fit',  zorder=2)
ax.set_title("NeuralProphet Grey-Box Fit — Observed vs. Fitted")
ax.set_xlabel("Date"); ax.set_ylabel(f"{TARGET} (mdeg)")
ax.legend(loc='upper left'); ax.grid(True, alpha=0.3)
sns.despine(); plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "04_02_observed_vs_fitted.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_PATH, "04_02_observed_vs_fitted.svg"), format='svg', bbox_inches='tight')
plt.show()

# %%
# --- Plot 2: Stacked decomposition panel ---
# Show trend + seasonal + AR + regressor contributions + residual
panel_cols = [c for c in component_cols if c in df_pred.columns] + ['residual']
df_plot = df_pred.copy()
df_plot['residual'] = residuals

n_panels = len(panel_cols)
fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3 * n_panels), sharex=True)
if n_panels == 1:
    axes = [axes]

colors = sns.color_palette("tab10", n_colors=n_panels)
labels = {
    'trend':         'Structural Trend',
    'season_yearly': 'Yearly Seasonality',
    'season_weekly': 'Weekly Seasonality',
    'residual':      'Structural Residual',
}

for ax, col, color in zip(axes, panel_cols, colors):
    label = labels.get(col, col)
    if col == 'residual':
        ax.fill_between(df_plot.index, df_plot[col], 0, color=color, alpha=0.5)
    else:
        ax.plot(df_plot.index, df_plot[col], color=color, linewidth=0.8)
    ax.set_ylabel(label, fontsize=9)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.2)
    sns.despine(ax=ax)

axes[-1].set_xlabel("Date")
fig.suptitle("NeuralProphet Grey-Box Decomposition", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "04_03_decomposition_panel.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_PATH, "04_03_decomposition_panel.svg"), format='svg', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 6 · Residual Diagnostics
#
# Two tests are required before applying control charts:
# - **ADF stationarity**: residuals must be I(0) for EWMA/CUSUM to have valid ARL properties.
# - **Ljung-Box whiteness**: residuals should be approximately white noise (no unexplained
#   autocorrelation). Significant autocorrelation at this stage indicates that N_LAGS
#   is insufficient and should be increased.

# %%
print("=== Residual Diagnostics ===")
is_stationary, p_adf = test_residual_stationarity(residuals)
is_white,      p_lb  = test_residual_whiteness(residuals, lags=24)

diag_table = pd.DataFrame([{
    'Test':       'ADF (stationarity)',
    'Statistic':  'p-value',
    'Value':      round(p_adf, 6),
    'Pass':       '✓' if is_stationary else '✗',
    'Requirement': 'p < 0.05',
}, {
    'Test':       'Ljung-Box (whiteness, lag=24)',
    'Statistic':  'p-value',
    'Value':      round(p_lb, 6),
    'Pass':       '✓' if is_white else '✗',
    'Requirement': 'p > 0.05',
}])
display(diag_table)

if not is_stationary:
    print("\n⚠ WARNING: Residuals are non-stationary. "
          "Consider increasing trend_reg or adding a changepoint.")
if not is_white:
    print("\n⚠ WARNING: Residuals show autocorrelation. "
          "Consider increasing N_LAGS or adding weekly seasonality.")

# %%
# Residual distribution plot
from scipy.stats import gaussian_kde, norm

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left: histogram + KDE + normal overlay
ax = axes[0]
res_vals = residuals.dropna().values
ax.hist(res_vals, bins=60, density=True, color='steelblue', edgecolor='white', alpha=0.6)
xr = np.linspace(res_vals.min(), res_vals.max(), 300)
ax.plot(xr, gaussian_kde(res_vals)(xr), color='steelblue', linewidth=2, label='KDE')
ax.plot(xr, norm.pdf(xr, res_vals.mean(), res_vals.std()),
        color='crimson', linewidth=1.5, linestyle='--', label='Normal')
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.set_title("Residual Distribution"); ax.set_xlabel("Residual (mdeg)"); ax.set_ylabel("Density")
ax.legend(); ax.grid(True, alpha=0.3); sns.despine(ax=ax)

# Right: ACF of residuals (manual computation)
ax = axes[1]
max_lags_acf = 48
acf_vals = [pd.Series(res_vals).autocorr(lag=l) for l in range(1, max_lags_acf + 1)]
conf_bound = 1.96 / np.sqrt(len(res_vals))
ax.bar(range(1, max_lags_acf + 1), acf_vals, color='steelblue', alpha=0.7)
ax.axhline( conf_bound, color='crimson', linestyle='--', linewidth=1, label='95% CI')
ax.axhline(-conf_bound, color='crimson', linestyle='--', linewidth=1)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title("Residual Autocorrelation Function (ACF)")
ax.set_xlabel("Lag (hours)"); ax.set_ylabel("ACF")
ax.legend(); ax.grid(True, alpha=0.3); sns.despine(ax=ax)

plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "04_04_residual_diagnostics.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_PATH, "04_04_residual_diagnostics.svg"), format='svg', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 7 · EWMA and CUSUM Control Charts
#
# ### Parameter Tuning Guidance
#
# Control limits are computed **exclusively from the reference period** residuals.
# This anchors the "normal behaviour" baseline before applying charts to the
# full monitoring horizon, which includes imputed segments.
#
# If the reference period residuals are non-Gaussian, consider using
# empirical quantiles (e.g., 99.5th percentile) instead of μ ± L·σ for EWMA limits.

# %%
mu, sigma, ref_residuals = compute_reference_stats(residuals, REF_START, REF_END)

ewma_stat, ucl_ewma, lcl_ewma, alarm_ewma = ewma_chart(
    residuals, mu, sigma, lam=EWMA_LAMBDA, L=EWMA_L
)
cusum_pos, cusum_neg, h_cusum, alarm_cusum = cusum_chart(
    residuals, mu, sigma, k=CUSUM_K, h=CUSUM_H
)
alarm_joint_series = joint_alarm(alarm_ewma, alarm_cusum, window=JOINT_WINDOW)
alarm_df = alarm_summary(alarm_joint_series, residuals)

# Export alarm table
alarm_df.to_csv(os.path.join(TABLE_PATH, "04_alarm_episodes.csv"), index=False)
display(alarm_df)

# %% [markdown]
# ## Step 8 · Control Chart Visualisation

# %%
# --- Plot 3: EWMA control chart ---
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(ewma_stat.index, ewma_stat, color='steelblue', linewidth=0.8, label='EWMA statistic')
ax.axhline(ucl_ewma,  color='crimson', linestyle='--', linewidth=1.2, label=f'UCL / LCL (L={EWMA_L}σ)')
ax.axhline(lcl_ewma,  color='crimson', linestyle='--', linewidth=1.2)
ax.axhline(mu,        color='grey',    linestyle=':',  linewidth=0.8, label='μ (reference)')

# Shade alarm regions
alarm_idx = alarm_ewma[alarm_ewma].index
for ts in alarm_idx:
    ax.axvspan(ts, ts + pd.Timedelta(hours=1), color='crimson', alpha=0.15, linewidth=0)

# Shade reference period
ax.axvspan(pd.Timestamp(REF_START), pd.Timestamp(REF_END),
           color='green', alpha=0.06, label='Reference period')

ax.set_title(f"EWMA Control Chart (λ={EWMA_LAMBDA}, L={EWMA_L})")
ax.set_xlabel("Date"); ax.set_ylabel("EWMA statistic (mdeg)")
ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
sns.despine(); plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "04_05_ewma_chart.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_PATH, "04_05_ewma_chart.svg"), format='svg', bbox_inches='tight')
plt.show()

# %%
# --- Plot 4: CUSUM control chart (two-sided) ---
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(cusum_pos.index, cusum_pos, color='steelblue', linewidth=0.8, label='C⁺ (upper CUSUM)')
ax.plot(cusum_neg.index, cusum_neg, color='darkorange', linewidth=0.8, label='C⁻ (lower CUSUM)')
ax.axhline(h_cusum, color='crimson', linestyle='--', linewidth=1.2, label=f'Decision interval h={CUSUM_H}')

alarm_c_idx = alarm_cusum[alarm_cusum].index
for ts in alarm_c_idx:
    ax.axvspan(ts, ts + pd.Timedelta(hours=1), color='crimson', alpha=0.15, linewidth=0)

ax.axvspan(pd.Timestamp(REF_START), pd.Timestamp(REF_END),
           color='green', alpha=0.06, label='Reference period')

ax.set_title(f"CUSUM Control Chart (k={CUSUM_K}, h={CUSUM_H})")
ax.set_xlabel("Date"); ax.set_ylabel("CUSUM statistic (σ units)")
ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
sns.despine(); plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "04_06_cusum_chart.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_PATH, "04_06_cusum_chart.svg"), format='svg', bbox_inches='tight')
plt.show()

# %%
# --- Plot 5: Joint alarm overview on residual series ---
fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

# Upper panel: residual + EWMA alarms
ax = axes[0]
ax.plot(residuals.index, residuals, color='steelblue', linewidth=0.5, label='Residual', alpha=0.7)
ax.axhline(mu + EWMA_L * sigma * np.sqrt(EWMA_LAMBDA / (2 - EWMA_LAMBDA)),
           color='crimson', linestyle='--', linewidth=1, label='EWMA UCL/LCL')
ax.axhline(mu - EWMA_L * sigma * np.sqrt(EWMA_LAMBDA / (2 - EWMA_LAMBDA)),
           color='crimson', linestyle='--', linewidth=1)
for ts in alarm_joint_series[alarm_joint_series].index:
    ax.axvspan(ts, ts + pd.Timedelta(hours=1), color='crimson', alpha=0.3, linewidth=0)
ax.axvspan(pd.Timestamp(REF_START), pd.Timestamp(REF_END),
           color='green', alpha=0.06, label='Reference period')
ax.set_ylabel("Structural Residual (mdeg)"); ax.legend(loc='upper left', fontsize=8)
ax.set_title("Structural Residuals with Joint EWMA+CUSUM Alarm Periods (red)")
ax.grid(True, alpha=0.2); sns.despine(ax=ax)

# Lower panel: imputed flag overlay to distinguish imputed vs. observed alarms
ax = axes[1]
imp_flag = df[f"{TARGET}_imputed_flag"].reindex(residuals.index).fillna(0)
ax.fill_between(residuals.index, 0, imp_flag,
                color='sandybrown', alpha=0.5, label='Imputed segment')
ax.fill_between(residuals.index, 0, alarm_joint_series.reindex(residuals.index).fillna(False).astype(int),
                color='crimson', alpha=0.5, label='Joint alarm')
ax.set_ylabel("Flag (0/1)"); ax.set_xlabel("Date")
ax.legend(loc='upper left', fontsize=8)
ax.set_title("Alarm and Imputation Flags — Operational Context")
ax.grid(True, alpha=0.2); sns.despine(ax=ax)

plt.tight_layout()
fig.savefig(os.path.join(FIG_PATH, "04_07_joint_alarm_overview.png"), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIG_PATH, "04_07_joint_alarm_overview.svg"), format='svg', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 9 · Save Full Outputs

# %%
# Assemble and save the complete monitoring output table
out_monitoring = pd.DataFrame({
    TARGET:               df_pred['y'],
    'yhat1':              df_pred['yhat1'],
    'residual':           residuals,
    'ewma_stat':          ewma_stat,
    'cusum_pos':          cusum_pos,
    'cusum_neg':          cusum_neg,
    'alarm_ewma':         alarm_ewma.astype(int),
    'alarm_cusum':        alarm_cusum.astype(int),
    'alarm_joint':        alarm_joint_series.reindex(residuals.index).fillna(False).astype(int),
}, index=residuals.index)

# Append decomposition components
for col in component_cols:
    if col in df_pred.columns:
        out_monitoring[col] = df_pred[col].reindex(residuals.index)

out_monitoring.to_csv(os.path.join(TABLE_PATH, "04_monitoring_output.csv"))
print(f"Saved monitoring output : {out_monitoring.shape}")
display(out_monitoring.head(3))

# %%
# Print final diagnostics summary for the manuscript
print("\n========= MANUSCRIPT-READY SUMMARY =========")
print(f"NeuralProphet training period  : {df_ref.index[0].date()} → {df_ref.index[-1].date()}")
print(f"Training rows (observed only)  : {len(df_ref)}")
print(f"AR lags (n_lags)               : {N_LAGS} h")
print(f"Exogenous regressors           : {REGRESSOR_COLS}")
print(f"ADF p-value (residuals)        : {p_adf:.4e}  {'✓' if is_stationary else '✗'}")
print(f"Ljung-Box p-value (lag=24)     : {p_lb:.4e}  {'✓' if is_white else '✗'}")
print(f"EWMA (λ={EWMA_LAMBDA}, L={EWMA_L})  UCL={ucl_ewma:.4f}  LCL={lcl_ewma:.4f}")
print(f"CUSUM (k={CUSUM_K}, h={CUSUM_H})")
print(f"Joint alarm episodes           : {len(alarm_df)}")
print("=============================================")
