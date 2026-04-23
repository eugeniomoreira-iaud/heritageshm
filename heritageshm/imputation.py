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
