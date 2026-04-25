"""
Module: diagnostics.py
Handles gap taxonomy characterization, cointegration testing,
and residual diagnostics (ADF, Ljung-Box).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import pearsonr, chi2_contingency


def shift_and_correlate(df, target, proxy, max_lag, step=1):
    """
    Screens cross-correlation across varying lags to find the optimal thermal
    inertia between a structural signal and an environmental proxy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing both ``target`` and ``proxy`` columns.
    target : str
        Name of the structural response column.
    proxy : str
        Name of the environmental proxy column.
    max_lag : int
        Maximum lag to test (number of time steps, e.g. 72 for 72 hours).
    step : int, optional
        Step size between tested lags.  Default ``1``.

    Returns
    -------
    lags : np.ndarray
        Array of tested lag values.
    corrs : list of float
        Pearson correlation coefficient at each lag.
    """
    lags = np.arange(0, max_lag + 1, step)
    corrs = []

    df_clean = df[[target, proxy]].dropna()

    for lag in lags:
        shifted = df_clean[proxy] if lag == 0 else df_clean[proxy].shift(lag)
        valid = pd.concat([df_clean[target], shifted], axis=1).dropna()
        if len(valid) > 100:
            r, _ = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        else:
            r = 0
        corrs.append(r)

    return lags, corrs


def characterize_gaps(
    df,
    target_col,
    max_impute_gap=0,
    save_plot_path=None,
    histogram_bins=50,
    bar_color='black',
):
    """
    Characterizes missing data: gap duration, continuity, recurrence, and
    association with observed covariates (missingness-correlation test).

    Optionally applies linear interpolation for small gaps, then generates a
    clean histogram of gap lengths.

    Histogram style
    ---------------
    Bars are filled solid with ``bar_color`` (default black), zero gap between
    bars (``rwidth=1``), and no edge lines (``edgecolor='none'``).  The number
    of bins is controlled by ``histogram_bins``.

    Missingness-correlation test
    ----------------------------
    For every *other* numeric column in ``df`` a Pearson correlation is computed
    between a binary missingness indicator of ``target_col`` and the covariate
    values (NaN rows in either series are dropped pairwise).  Columns with
    |r| >= 0.10 and p < 0.05 are flagged as MAR candidates; if the strongest
    predictor is ``charge`` (the power-loss sentinel) the diagnosis is refined
    to MNAR-power.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    target_col : str
        Column to analyse for missing data.
    max_impute_gap : int, optional
        Maximum number of consecutive NaNs to fill by linear interpolation.
        Default ``0`` (classify only, no imputation).
    save_plot_path : str or None, optional
        File path to save the histogram figure.  ``None`` to skip.  Default ``None``.
    histogram_bins : int, optional
        Number of bins for the gap-length histogram.  Default ``50``.
    bar_color : str, optional
        Matplotlib colour for the histogram bars.  Default ``'black'``.

    Returns
    -------
    df_imputed : pd.DataFrame
        DataFrame with imputation applied (if ``max_impute_gap > 0``).
    gap_stats : pd.Series
        Descriptive statistics of all detected gap lengths.
    gap_lengths : pd.Series
        Exact length (in time steps) of every contiguous missing block.
    """
    if target_col not in df.columns:
        raise KeyError("Column '%s' not found in DataFrame." % target_col)

    df_imputed = df.copy()

    # ── Optional linear interpolation for short gaps ──────────────────────────
    if max_impute_gap > 0:
        print("Applying linear interpolation for gaps <= %d consecutive values..."
              % max_impute_gap)
        mask = df_imputed[target_col].isnull()
        if mask.any():
            blocks    = mask.astype(int).groupby(mask.astype(int).diff().ne(0).cumsum())
            gap_sizes = blocks.transform('sum')
            interp    = df_imputed[target_col].interpolate(method='linear')
            interp[mask & (gap_sizes > max_impute_gap)] = np.nan
            df_imputed[target_col] = interp

    # ── Gap detection ─────────────────────────────────────────────────────────
    missing_mask = df_imputed[target_col].isnull()

    if not missing_mask.any():
        print("No missing data found in '%s'." % target_col)
        return df_imputed, pd.Series(dtype=float), pd.Series(dtype=int)

    gap_blocks  = missing_mask.astype(int).groupby(
        missing_mask.astype(int).diff().ne(0).cumsum()
    )
    block_sums  = gap_blocks.sum()
    gap_lengths = block_sums[block_sums > 0]
    gap_stats   = gap_lengths.describe()

    print("\n--- Gap Taxonomy for '%s' ---" % target_col)
    print("Total Gaps Detected : %d"   % int(gap_stats['count']))
    print("Average Gap Length  : %.2f time steps" % gap_stats['mean'])
    print("Maximum Gap Length  : %d time steps"   % int(gap_stats['max']))
    print("Minimum Gap Length  : %d time steps"   % int(gap_stats['min']))
    print("Gap Length Std Dev  : %.2f time steps" % gap_stats['std'])

    # ── Missingness-correlation test ──────────────────────────────────────────
    # Build a binary indicator: 1 = missing in target_col, 0 = present
    miss_indicator = missing_mask.astype(int)
    numeric_cols   = [
        c for c in df_imputed.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    mar_candidates = []
    mnar_power     = False

    if numeric_cols:
        print("\n--- Missingness-Covariate Correlation Test ---")
        print("%-45s  %8s  %10s  %s" % ('Covariate', '|r|', 'p-value', 'Flag'))
        print("-" * 75)
        for col in numeric_cols:
            pair = pd.concat([miss_indicator, df_imputed[col]], axis=1).dropna()
            if len(pair) < 30:
                continue
            r, p = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
            flagged = abs(r) >= 0.10 and p < 0.05
            flag_str = 'MAR candidate' if flagged else ''
            print("%-45s  %8.4f  %10.4e  %s" % (col, abs(r), p, flag_str))
            if flagged:
                mar_candidates.append((col, abs(r), p))
                if col == 'charge':
                    mnar_power = True

    # ── Diagnostic verdict ────────────────────────────────────────────────────
    print("\n--- Missingness Mechanism Diagnosis ---")
    if mnar_power:
        print("Diagnosis : MNAR-power  (missingness strongly correlated with 'charge').")
        print("            Power-outage gaps are structurally caused, not random.")
        print("Imputation: Proxy-based regression is preferred (NeuralProphet with")
        print("            environmental regressors).  Sequence models (BiLSTM) are")
        print("            secondary.  Simple interpolation is inappropriate for")
        print("            long outage blocks.")
    elif mar_candidates:
        best = max(mar_candidates, key=lambda x: x[1])
        print("Diagnosis : MAR  (missingness correlated with observed covariate '%s', |r|=%.3f)."
              % (best[0], best[1]))
        print("Imputation: Regression-based methods using correlated covariates are appropriate.")
    elif gap_stats['mean'] > 10 and gap_stats['max'] > 100:
        print("Diagnosis : Likely MNAR/MAR (long structured outages; no observed correlate found).")
        print("            Consider checking for unlogged operational events.")
        print("Imputation: Advanced methods (proxy-based regression, XGBoost) recommended.")
    else:
        print("Diagnosis : Likely MCAR  (short, scattered gaps; no covariate correlation).")
        print("Imputation: Standard interpolation or basic models may suffice.")

    # ── Histogram ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        gap_lengths,
        bins=histogram_bins,
        color=bar_color,
        edgecolor='none',   # no bar border
        rwidth=1.0,         # bars flush — no gap between them
        linewidth=0,
    )

    ax.set_title(
        "Distribution of Gap Lengths — '%s'  (max_impute_gap=%d)"
        % (target_col, max_impute_gap)
    )
    ax.set_xlabel("Gap Length (Consecutive Missing Time Steps)")
    ax.set_ylabel("Frequency (Number of Occurrences)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    sns.despine(ax=ax)
    fig.tight_layout()

    if save_plot_path:
        import os
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        fig.savefig(save_plot_path, dpi=150)
        print("Saved histogram to " + save_plot_path)

    plt.show()

    return df_imputed, gap_stats, gap_lengths


def test_cointegration(df, target_col, proxy_col, alpha=0.05):
    """
    Performs the Engle-Granger two-step cointegration test to validate
    the physical long-run equilibrium between the structural response and
    an environmental proxy.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    proxy_col : str
    alpha : float, optional
        Significance level.  Default ``0.05``.

    Returns
    -------
    is_cointegrated : bool
    p_value : float
    """
    df_clean = df.dropna(subset=[target_col, proxy_col])

    if len(df_clean) < 100:
        raise ValueError(
            "Insufficient overlapping data points to run a reliable cointegration test."
        )

    score, p_value, _ = coint(df_clean[target_col], df_clean[proxy_col])
    is_cointegrated   = p_value < alpha

    print("\n--- Engle-Granger Cointegration Test ---")
    print("Target: %s | Proxy: %s" % (target_col, proxy_col))
    print("Test Statistic : %.4f" % score)
    print("P-value        : %.4e" % p_value)

    if is_cointegrated:
        print("Result: Reject H0 — variables share a stationary long-run equilibrium.")
    else:
        print("Result: Fail to reject H0 — proxy is NOT cointegrated with target.")

    return is_cointegrated, p_value


def test_residual_stationarity(residuals, alpha=0.05):
    """
    Tests structural residuals for stationarity using ADF.

    Parameters
    ----------
    residuals : pd.Series or array-like
    alpha : float, optional
        Significance level.  Default ``0.05``.

    Returns
    -------
    is_stationary : bool
    p_value : float
    """
    res_clean  = pd.Series(residuals).dropna()
    adf_result = adfuller(res_clean)
    p_value    = adf_result[1]
    is_stationary = p_value < alpha

    print("\n--- ADF Stationarity Test ---")
    print("Test Statistic : %.4f" % adf_result[0])
    print("P-value        : %.4e" % p_value)

    if is_stationary:
        print("Result: Residuals are stationary (I(0)). Suitable for control chart monitoring.")
    else:
        print("Result: Residuals are non-stationary. Decomposition may be incomplete.")

    return is_stationary, p_value


def test_residual_whiteness(residuals, lags=10, alpha=0.05):
    """
    Tests residuals for white noise using Ljung-Box.

    Parameters
    ----------
    residuals : pd.Series or array-like
    lags : int, optional
        Number of lags to test.  Default ``10``.
    alpha : float, optional
        Significance level.  Default ``0.05``.

    Returns
    -------
    is_white : bool
    p_value : float
    """
    res_clean = pd.Series(residuals).dropna()
    lb_result = acorr_ljungbox(res_clean, lags=[lags], return_df=True)
    p_value   = lb_result['lb_pvalue'].iloc[0]
    is_white  = p_value > alpha

    print("\n--- Ljung-Box Whiteness Test (Lags=%d) ---" % lags)
    print("Test Statistic : %.4f" % lb_result['lb_stat'].iloc[0])
    print("P-value        : %.4e" % p_value)

    if is_white:
        print("Result: Residuals are white noise (no significant autocorrelation).")
    else:
        print("Result: Residuals show significant autocorrelation.")

    return is_white, p_value


def test_signal_stationarity(df, cols, alpha=0.05):
    """
    Runs ADF on a list of columns to determine their integration order.

    Used to justify Pearson correlation (I(0)) vs. cointegration testing (I(1)).

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of str
    alpha : float, optional
        Significance level.  Default ``0.05``.

    Returns
    -------
    pd.DataFrame
        ADF statistic, p-value, and verdict per column.
    """
    records = []
    for col in cols:
        series = df[col].dropna()
        if len(series) < 20:
            records.append({
                'Variable': col,
                'ADF Statistic': None,
                'p-value': None,
                'Stationary (I(0))': None,
                'Decision': 'Insufficient data',
            })
            continue
        adf_stat, p_value, _, _, _, _ = adfuller(series, autolag='AIC')
        is_stationary = p_value < alpha
        records.append({
            'Variable': col,
            'ADF Statistic': round(adf_stat, 4),
            'p-value': round(p_value, 6),
            'Stationary (I(0))': is_stationary,
            'Decision': 'Proceed with Pearson' if is_stationary
                        else '\u26a0 Difference or test cointegration',
        })
        print("%-50s  ADF=%8.4f  p=%.4e  %s"
              % (col, adf_stat, p_value, '\u2713 I(0)' if is_stationary else '\u2717 I(1)'))

    return pd.DataFrame(records).set_index('Variable')


def get_longest_contiguous_block(df, target_col, expected_step='1h'):
    """
    Identifies the start and end timestamps of the longest contiguous
    non-missing block of ``target_col``.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    expected_step : str, optional
        Target frequency string.  Default ``'1h'``.

    Returns
    -------
    ref_start : pd.Timestamp
    ref_end : pd.Timestamp
    """
    target_series    = df[target_col].dropna()
    expected_step_td = pd.Timedelta(expected_step)

    time_diffs = target_series.index.to_series().diff()
    breaks     = time_diffs[time_diffs > expected_step_td * 1.5].index

    block_starts    = [target_series.index[0]] + list(breaks)
    block_ends      = list(breaks - expected_step_td) + [target_series.index[-1]]
    block_lengths_h = [(e - s).total_seconds() / 3600
                       for s, e in zip(block_starts, block_ends)]

    longest_idx = int(pd.Series(block_lengths_h).idxmax())
    ref_start   = block_starts[longest_idx]
    ref_end     = block_ends[longest_idx]

    print("Longest contiguous block : %s \u2192 %s" % (ref_start, ref_end))
    print("Duration                 : %.0f h (%.1f days)"
          % (block_lengths_h[longest_idx], block_lengths_h[longest_idx] / 24))

    return ref_start, ref_end


def screen_optimal_lags(df_diff, target, proxies, max_lag_h, lag_step):
    """
    Screens optimal thermal-inertia lags across multiple proxies.

    Parameters
    ----------
    df_diff : pd.DataFrame
    target : str
    proxies : list of str
    max_lag_h : int
    lag_step : int

    Returns
    -------
    optimal_lags : dict
    all_lags : list
    corrs_dict : dict
    """
    optimal_lags = {}
    corrs_dict   = {}
    all_lags     = None

    for proxy in proxies:
        lags, corrs = shift_and_correlate(df_diff, target, proxy, max_lag_h, lag_step)
        if all_lags is None:
            all_lags = lags

        max_idx  = np.argmax(np.abs(corrs))
        best_lag  = lags[max_idx]
        best_corr = corrs[max_idx]

        optimal_lags[proxy] = best_lag
        corrs_dict[proxy]   = corrs
        print("%s: Optimal Lag = %dh, r = %.3f" % (proxy, best_lag, best_corr))

    return optimal_lags, all_lags, corrs_dict
