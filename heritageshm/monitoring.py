"""
Module: monitoring.py
EWMA and CUSUM control charts for residual-based structural anomaly detection.

Design rationale:
- Control limits are estimated from a reference (baseline) period of stationary residuals.
- EWMA is sensitive to gradual, small-magnitude drifts.
- CUSUM (two-sided) is sensitive to sustained directional shifts.
- Both charts are applied jointly; an alarm requires both to trigger within a
  configurable coincidence window, reducing false-positive rate.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reference period statistics
# ---------------------------------------------------------------------------

def compute_reference_stats(residuals, ref_start, ref_end):
    """
    Computes mean and standard deviation of residuals over the reference period.
    These parameters anchor the control chart limits.

    Parameters:
    - residuals: pd.Series of structural residuals (full monitoring horizon).
    - ref_start: Start timestamp of the reference (baseline) period.
    - ref_end: End timestamp of the reference (baseline) period.

    Returns:
    - mu: Reference mean (should be near zero for well-decomposed residuals).
    - sigma: Reference standard deviation.
    - ref_residuals: Residuals slice within the reference window.
    """
    ref_residuals = residuals.loc[ref_start:ref_end].dropna()
    mu = ref_residuals.mean()
    sigma = ref_residuals.std()
    print(f"\n--- Reference Period Statistics ---")
    print(f"Period   : {ref_start} → {ref_end}")
    print(f"N points : {len(ref_residuals)}")
    print(f"Mean (μ) : {mu:.6f}")
    print(f"Std  (σ) : {sigma:.6f}")
    return mu, sigma, ref_residuals


# ---------------------------------------------------------------------------
# EWMA control chart
# ---------------------------------------------------------------------------

def ewma_chart(residuals, mu, sigma, lam=0.2, L=3.0):
    """
    Exponentially Weighted Moving Average (EWMA) control chart.

    The EWMA statistic at time t is:
        Z_t = λ * e_t + (1 − λ) * Z_{t-1},   Z_0 = μ

    Control limits (steady-state approximation):
        UCL = μ + L * σ * sqrt(λ / (2 − λ))
        LCL = μ − L * σ * sqrt(λ / (2 − λ))

    Parameters:
    - residuals: pd.Series of structural residuals.
    - mu: Reference mean from compute_reference_stats().
    - sigma: Reference standard deviation.
    - lam: EWMA smoothing parameter λ ∈ (0, 1]. Smaller = more smoothing.
            Recommended: 0.1–0.3 for gradual drift detection.
    - L: Control limit multiplier (in σ units). Default 3.0 (≈ 0.27% false alarm).

    Returns:
    - ewma_stat: pd.Series of EWMA statistic values.
    - ucl: Upper control limit (scalar).
    - lcl: Lower control limit (scalar).
    - alarm_ewma: pd.Series of booleans — True where a violation occurs.
    """
    z = np.zeros(len(residuals))
    e = residuals.values
    z[0] = mu
    for t in range(1, len(e)):
        if np.isnan(e[t]):
            z[t] = z[t - 1]
        else:
            z[t] = lam * e[t] + (1 - lam) * z[t - 1]

    ewma_stat = pd.Series(z, index=residuals.index, name='ewma')
    cl_width = L * sigma * np.sqrt(lam / (2 - lam))
    ucl = mu + cl_width
    lcl = mu - cl_width

    alarm_ewma = (ewma_stat > ucl) | (ewma_stat < lcl)
    alarm_ewma.name = 'alarm_ewma'

    print(f"\n--- EWMA Chart ---")
    print(f"λ={lam}, L={L}")
    print(f"UCL = {ucl:.6f}  |  LCL = {lcl:.6f}")
    print(f"Total EWMA alarms : {alarm_ewma.sum()}")

    return ewma_stat, ucl, lcl, alarm_ewma


# ---------------------------------------------------------------------------
# CUSUM control chart (two-sided)
# ---------------------------------------------------------------------------

def cusum_chart(residuals, mu, sigma, k=0.5, h=5.0):
    """
    Two-sided CUSUM (Cumulative Sum) control chart.

    Upper CUSUM:   C+_t = max(0, C+_{t-1} + (e_t − μ) / σ − k)
    Lower CUSUM:   C-_t = max(0, C-_{t-1} − (e_t − μ) / σ − k)
    Alarm when C+_t > h  or  C-_t > h.

    Parameters:
    - residuals: pd.Series of structural residuals.
    - mu: Reference mean.
    - sigma: Reference standard deviation.
    - k: Allowance (slack) parameter. Typically 0.5σ (detects ≥1σ shifts).
    - h: Decision interval. Typically h=4–5 for ARL₀ ≈ 370–500.

    Returns:
    - cusum_pos: pd.Series of upper CUSUM statistic.
    - cusum_neg: pd.Series of lower CUSUM statistic.
    - h: Decision interval (passed through for plotting).
    - alarm_cusum: pd.Series of booleans — True where a violation occurs.
    """
    e = ((residuals - mu) / sigma).values
    cp = np.zeros(len(e))
    cn = np.zeros(len(e))

    for t in range(1, len(e)):
        val = 0.0 if np.isnan(e[t]) else e[t]
        cp[t] = max(0.0, cp[t - 1] + val - k)
        cn[t] = max(0.0, cn[t - 1] - val - k)

    cusum_pos = pd.Series(cp, index=residuals.index, name='cusum_pos')
    cusum_neg = pd.Series(cn, index=residuals.index, name='cusum_neg')

    alarm_cusum = (cusum_pos > h) | (cusum_neg > h)
    alarm_cusum.name = 'alarm_cusum'

    print(f"\n--- CUSUM Chart ---")
    print(f"k={k}, h={h}")
    print(f"Total CUSUM alarms : {alarm_cusum.sum()}")

    return cusum_pos, cusum_neg, h, alarm_cusum


# ---------------------------------------------------------------------------
# Joint alarm logic
# ---------------------------------------------------------------------------

def joint_alarm(alarm_ewma, alarm_cusum, window=24):
    """
    Raises a joint alarm only when both EWMA and CUSUM are in alarm state
    within a sliding coincidence window of `window` time steps.

    This conservative logic reduces false positives from single-chart spurious triggers.

    Parameters:
    - alarm_ewma: pd.Series (bool) from ewma_chart().
    - alarm_cusum: pd.Series (bool) from cusum_chart().
    - window: Number of consecutive steps both must be in alarm (default 24 h).

    Returns:
    - alarm_joint: pd.Series (bool) of joint alarms.
    """
    both = (alarm_ewma & alarm_cusum).astype(int)
    rolling_sum = both.rolling(window=window, min_periods=1).sum()
    alarm_joint = rolling_sum >= window
    alarm_joint.name = 'alarm_joint'

    print(f"\n--- Joint Alarm Logic (coincidence window = {window} steps) ---")
    print(f"Total joint alarm periods : {alarm_joint.sum()}")

    return alarm_joint


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def alarm_summary(alarm_joint, residuals):
    """
    Returns a DataFrame listing each contiguous joint alarm period,
    its start, end, duration (hours), and mean residual magnitude.

    Parameters:
    - alarm_joint: pd.Series (bool) from joint_alarm().
    - residuals: pd.Series of structural residuals.

    Returns:
    - summary_df: DataFrame of alarm episodes.
    """
    alarm_int = alarm_joint.astype(int)
    starts = alarm_int[(alarm_int == 1) & (alarm_int.shift(1, fill_value=0) == 0)].index
    ends   = alarm_int[(alarm_int == 1) & (alarm_int.shift(-1, fill_value=0) == 0)].index

    records = []
    for s, e in zip(starts, ends):
        seg = residuals.loc[s:e]
        records.append({
            'alarm_start': s,
            'alarm_end': e,
            'duration_h': len(seg),
            'mean_residual': round(seg.mean(), 4),
            'max_abs_residual': round(seg.abs().max(), 4),
        })

    summary_df = pd.DataFrame(records)
    print(f"\n--- Alarm Episode Summary ({len(summary_df)} episodes) ---")
    print(summary_df.to_string(index=False) if not summary_df.empty else "No joint alarms detected.")
    return summary_df