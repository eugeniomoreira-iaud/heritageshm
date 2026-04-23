"""
Module: imputation.py
Handles advanced gap-filling models and imputation utilities for the heritage SHM pipeline.
"""

import pandas as pd
import numpy as np


def get_gap_blocks(series):
    """
    Identifies contiguous gap blocks in a time series.

    Parameters:
    - series: pandas Series with datetime index

    Returns:
    - DataFrame with columns ['start', 'end', 'duration_h'] for each gap block
    """
    is_gap = series.isna()
    gap_start = series.index[is_gap & ~is_gap.shift(1, fill_value=False)]
    gap_end = series.index[is_gap & ~is_gap.shift(-1, fill_value=False)]
    blocks = pd.DataFrame({"start": gap_start, "end": gap_end})
    blocks["duration_h"] = (
        (blocks["end"] - blocks["start"]).dt.total_seconds() / 3600 + 1
    )
    return blocks.reset_index(drop=True)


def build_feature_row(ts, working_series, working_diff, df_source,
                      proxy_lags, ar_lags, target_diff_col):
    """
    Build a single-row feature dict for timestamp `ts`.

    - Proxy values    : always from df_source (original dataset, never imputed).
    - AR lags         : from working_diff (first differences of the running
                        reconstructed level — includes Δy of prior predictions
                        inside gaps).
    - Time encodings  : derived from ts directly.

    Note: working_series is no longer used for AR lags but is kept as an
    argument to allow callers to optionally inspect the level series.
    """
    row = {}

    # Proxy regressors at their validated optimal lags
    for proxy, lag in proxy_lags.items():
        lag_ts = ts - pd.Timedelta(hours=lag)
        row[f"{proxy}_lag{lag}"] = (
            df_source[proxy].get(lag_ts, np.nan)
            if lag_ts in df_source.index else np.nan
        )

    # Cyclic time encodings
    row["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    row["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365.25)
    row["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * ts.month / 12)

    # AR lags on the DIFFERENCED target
    for lag in ar_lags:
        lag_ts = ts - pd.Timedelta(hours=lag)
        row[f"{target_diff_col}_lag{lag}"] = (
            working_diff.get(lag_ts, np.nan)
            if lag_ts in working_diff.index else np.nan
        )

    return row


def build_training_matrix(df, proxy_lags, ar_lags, target_diff_col):
    """
    Build the vectorised training feature matrix.
    AR lags are computed on target_diff_col (Δy), not the level.
    Used only for training — iterative prediction uses build_feature_row.
    """
    X = pd.DataFrame(index=df.index)

    # Proxy regressors at optimal lags
    for proxy, lag in proxy_lags.items():
        X[f"{proxy}_lag{lag}"] = df[proxy].shift(lag)

    # Cyclic time encodings
    X["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    X["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    X["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # AR lags on the DIFFERENCED target
    for lag in ar_lags:
        X[f"{target_diff_col}_lag{lag}"] = df[target_diff_col].shift(lag)

    return X


def impute_gap_iterative(model, gap_idx, working_series, working_diff,
                         df_source, proxy_lags, ar_lags,
                         target_col, target_diff_col):
    """
    Iterative one-step-ahead imputation in the first-difference domain.

    The model predicts Δŷ_t = y_t − y_{t-1}.
    The level is reconstructed step by step as:
        ŷ_t = ŷ_{t-1} + Δŷ_t
    seeded from the last observed value immediately before the gap.

    This eliminates systematic level bias: the reconstruction is structurally
    anchored to the last real observation and drifts only as a zero-mean
    random walk (cumulative sum of prediction errors).

    Parameters
    ----------
    working_series : pd.Series
        Running level series (observed + prior imputed values).
        Updated in-place with each predicted level.
    working_diff : pd.Series
        Running first-difference series.
        Updated in-place with each predicted Δy.
        Used by build_feature_row for AR lags of Δy.
    df_source : pd.DataFrame
        Original dataset — used exclusively for proxy values.
    target_col : str
        Name of the level target column (for seed lookup).
    target_diff_col : str
        Name of the differenced target column (for AR feature names).

    Returns
    -------
    pd.Series of reconstructed level predictions indexed by gap_idx.
    Timestamps where proxy data is unavailable are returned as NaN.
    """
    predictions = {}

    # ── Seed: last observed level before the gap ─────────────────────────────
    prev_level = np.nan
    for lookback in range(1, 49):
        seed_ts = gap_idx[0] - pd.Timedelta(hours=lookback)
        seed_val = working_series.get(seed_ts, np.nan)
        if not np.isnan(seed_val):
            prev_level = seed_val
            break

    if np.isnan(prev_level):
        # No observed seed within 48 h — cannot anchor; return all NaN
        return pd.Series({ts: np.nan for ts in gap_idx})

    # ── Iterative prediction ──────────────────────────────────────────────────
    for ts in gap_idx:
        row = build_feature_row(ts, working_series, working_diff,
                                df_source, proxy_lags, ar_lags,
                                target_diff_col)
        Xrow = pd.DataFrame([row])

        if Xrow.isna().any(axis=1).values[0]:
            predictions[ts] = np.nan
            # Do not update prev_level — seed remains at last valid value
            continue

        delta_pred = float(model.predict(Xrow)[0])
        level_pred = prev_level + delta_pred

        # Write back to both working series
        working_series[ts] = level_pred
        working_diff[ts] = delta_pred

        predictions[ts] = level_pred
        prev_level = level_pred

    return pd.Series(predictions)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_synthetic_gap_window(df, target_col, gap_duration_h):
    """
    Finds the longest contiguous block of observed data and extracts a synthetic 
    gap window from its center.
    """
    obs_runs  = (~df[target_col].isna()).astype(int)
    run_ends  = df.index[(obs_runs == 1) & (obs_runs.shift(-1, fill_value=0) == 0)]
    run_starts= df.index[(obs_runs == 1) & (obs_runs.shift(1,  fill_value=0) == 0)]
    run_lengths = [(e - s).total_seconds()/3600 for s, e in zip(run_starts, run_ends)]
    
    longest_run_idx = int(np.argmax(run_lengths))
    longest_start   = run_starts[longest_run_idx]
    longest_end     = run_ends[longest_run_idx]
    
    run_idx     = pd.date_range(longest_start, longest_end, freq="1h")
    mid         = len(run_idx) // 2
    gap_idx_val = run_idx[mid - gap_duration_h//2 : mid + gap_duration_h//2]
    
    return longest_start, longest_end, run_lengths[longest_run_idx], gap_idx_val

def evaluate_synthetic_gap(y_true, y_pred):
    """
    Evaluates synthetic gap point predictions against the ground truth.
    """
    valid_mask = y_pred.notna() & y_true.notna()
    y_true_sc  = y_true[valid_mask]
    y_pred_sc  = y_pred[valid_mask]
    
    rmse = np.sqrt(mean_squared_error(y_true_sc, y_pred_sc))
    mae  = mean_absolute_error(y_true_sc, y_pred_sc)
    r2   = r2_score(y_true_sc, y_pred_sc)
    bias = float((y_pred_sc - y_true_sc).mean())
    maxe = float((y_pred_sc - y_true_sc).abs().max())
    
    return valid_mask, y_true_sc, y_pred_sc, rmse, mae, r2, bias, maxe

def get_bootstrap_uncertainty(n_bootstrap, random_seed, X_train, y_train, gap_idx_val, working_full, working_full_diff, df_full, proxy_lags, ar_lags, target, target_diff_col, xgb_params):
    """
    Runs iterative reconstruction over a gap using resampled bootstrap models 
    to estimate structural uncertainty in the predictions.
    """
    from xgboost import XGBRegressor
    boot_preds = np.full((n_bootstrap, len(gap_idx_val)), np.nan)

    for b in range(n_bootstrap):
        rng      = np.random.default_rng(random_seed + b)
        boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
        X_b      = X_train.iloc[boot_idx]
        y_b      = y_train.iloc[boot_idx]
        m_b      = XGBRegressor(**xgb_params)
        m_b.fit(X_b, y_b, verbose=False)

        working_b      = working_full.copy()
        working_b_diff = working_full_diff.copy()
        working_b.loc[gap_idx_val]      = np.nan
        working_b_diff.loc[gap_idx_val] = np.nan

        preds_b = impute_gap_iterative(
            m_b, gap_idx_val,
            working_b, working_b_diff,
            df_full, proxy_lags, ar_lags,
            target, target_diff_col
        )
        boot_preds[b, :] = preds_b.values
        if (b + 1) % 10 == 0:
            print(f"  Bootstrap {b+1}/{n_bootstrap} done")
            
    boot_mean = np.nanmean(boot_preds, axis=0)
    boot_std  = np.nanstd(boot_preds,  axis=0)
    return boot_mean, boot_std

def calibrate_uncertainty(boot_std, residual_std):
    """
    Calibrates the raw bootstrap std to accurately reflect true prediction errors using a conformal scale factor.
    """
    mean_boot_std   = np.nanmean(boot_std[boot_std > 0])
    conformal_scale = residual_std / mean_boot_std if mean_boot_std > 0 else 1.0
    boot_std_cal    = boot_std * conformal_scale
    return mean_boot_std, conformal_scale, boot_std_cal

def impute_all_gaps_with_uncertainty(model, gap_blocks, working_full, working_full_diff, df_full, proxy_lags, ar_lags, target, target_diff, n_bootstrap, random_seed, X_train, y_train, xgb_params, conformal_scale):
    """
    High-level wrapper that iteratively loops over all gap blocks, performing point imputation 
    and generating bootstrapped confidence bounds calibrated for each gap.
    """
    from xgboost import XGBRegressor
    imputed_flag = pd.Series(False, index=df_full.index)
    imputed_std  = pd.Series(np.nan, index=df_full.index)
    log_rows = []

    for gap_num, gap_row in gap_blocks.iterrows():
        gap_idx = pd.date_range(gap_row["start"], gap_row["end"], freq="1h")

        preds_gap = impute_gap_iterative(
            model, gap_idx,
            working_full, working_full_diff,
            df_full, proxy_lags, ar_lags,
            target, target_diff
        )

        valid_preds = preds_gap.dropna()
        imputed_flag.loc[valid_preds.index] = True

        boot_gap = np.full((n_bootstrap, len(gap_idx)), np.nan)
        for b in range(n_bootstrap):
            rng      = np.random.default_rng(random_seed + b)
            boot_idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            m_b      = XGBRegressor(**xgb_params)
            m_b.fit(X_train.iloc[boot_idx], y_train.iloc[boot_idx], verbose=False)
            
            working_b      = working_full.copy()
            working_b_diff = working_full_diff.copy()
            preds_b = impute_gap_iterative(
                m_b, gap_idx,
                working_b, working_b_diff,
                df_full, proxy_lags, ar_lags,
                target, target_diff
            )
            boot_gap[b, :] = preds_b.values

        gap_std_raw = np.nanstd(boot_gap, axis=0)
        gap_std_cal = gap_std_raw * conformal_scale
        imputed_std.loc[gap_idx] = gap_std_cal

        mean_sigma    = (
            round(float(np.nanmean(gap_std_cal)), 4)
            if not np.all(np.isnan(gap_std_cal)) else None
        )
        imputed_count = valid_preds.shape[0]

        log_rows.append({
            "Gap #":        gap_num + 1,
            "Start":        gap_row["start"].strftime("%Y-%m-%d %H:%M"),
            "End":          gap_row["end"].strftime("%Y-%m-%d %H:%M"),
            "Duration (h)": int(gap_row["duration_h"]),
            "Imputed (h)":  imputed_count,
            "Not imputed":  int(gap_row["duration_h"]) - imputed_count,
            "Mean σ (cal)": mean_sigma if mean_sigma is not None else "—",
        })

        if (gap_num + 1) % 50 == 0 or gap_num < 5:
            print(f"Gap {gap_num+1:3d}/{len(gap_blocks)} | "
                  f"{int(gap_row['duration_h']):5.0f} h | "
                  f"imputed {imputed_count:5d} h | "
                  f"σ_cal = {mean_sigma if mean_sigma else '—'}")
                  
    return working_full, imputed_flag, imputed_std, pd.DataFrame(log_rows)
