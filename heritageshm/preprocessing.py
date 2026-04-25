"""
Module: preprocessing.py
Handles signal cleaning, outlier removal, timestamp alignment,
and user-defined physical compensation (e.g., thermal correction).

Physical pipeline order
-----------------------
1. Power-loss removal   — charge == 0 rows are hardware artefacts, not physics.
                          Remove them before any signal analysis so outage zeros
                          never enter the spike-detection window.
2. Spike removal        — rolling-median context filter on the clean signal.
                          Each sample is compared against the local median of its
                          nearest valid neighbours on both sides.  Post-outage
                          spike runs are handled correctly because all spike rows
                          deviate from the pre- AND post-recovery context.
3. Temp. compensation   — linear thermal correction applied to the clean signal.
4. Normalisation        — series shifted so it starts at 0.
"""
import pandas as pd
import numpy as np
import os

_SEP = '\u2500' * 50   # separator line used in print banners


# ──────────────────────────────────────────────────────────────────────
def clean_signal_robust(
    df,
    signal_col,
    spike_threshold,
    window=7,
    min_valid=3,
    valid_charge_col=None,
):
    """
    Clean a raw sensor signal using the correct physical pipeline order:

    **Step 1 — Power-loss removal (runs first)**
        Drops all rows where the charge column equals zero.  These rows carry
        a zero signal that is a hardware artefact, not a physical reading.  By
        removing them *before* spike detection, outage zeros never enter the
        rolling-median window and cannot distort spike classification.

    **Step 2 — Rolling-median context spike filter**
        For each valid (non-NaN) sample, computes the absolute deviation from
        the *local median* of the nearest ``window`` valid neighbours on each
        side (up to ``2 * window`` neighbours total, centred on the point).

        Why median instead of first-order difference:
          - A true isolated spike deviates from its neighbours on **both** sides.
          - A legitimate recovery reading after a post-outage spike run is close
            to its right-side (post-recovery) neighbours and is **not** flagged.
          - A difference-based filter cannot make this distinction; the median
            context filter can.

        Points with fewer than ``min_valid`` valid neighbours are not evaluated
        (not enough context to judge) and are always kept.

    Parameters
    ----------
    df : pd.DataFrame
        Station DataFrame with a DatetimeIndex.
    signal_col : str
        Name of the structural signal column (e.g. ``'absinc'``).
    spike_threshold : float
        Maximum allowed absolute deviation from the local median (signal units).
        A sample is flagged as a spike when
        ``|signal[i] - median(neighbours)| > spike_threshold``.
    window : int, optional
        Half-window size: the number of valid neighbours searched on each side
        of each sample when building the local median.  Default ``7``.
        Increase for noisy / fast-varying signals; decrease for very sparse data.
    min_valid : int, optional
        Minimum number of valid neighbours required to evaluate a sample.
        Samples with fewer neighbours are kept unconditionally.  Default ``3``.
    valid_charge_col : str or None, optional
        Column whose zero values indicate a power-loss event (e.g. ``'charge'``).
        Set to ``None`` to skip the power-loss filter.  Default ``None``.

    Returns
    -------
    df_clean : pd.DataFrame
        Retained rows only (no ``drop_reason`` column).
    df_dropped : pd.DataFrame
        All removed rows with an extra ``'drop_reason'`` column:
        ``'power_loss'`` or ``'spike'``.
    """
    df_clean = df.copy()
    n0 = len(df_clean)
    dropped_frames = []

    # ── Step 1: power-loss removal (MUST run before spike filter) ─────────────
    if valid_charge_col and valid_charge_col in df_clean.columns:
        power_mask = df_clean[valid_charge_col] == 0
        n_power = power_mask.sum()
        if n_power > 0:
            df_power = df_clean[power_mask].copy()
            df_power['drop_reason'] = 'power_loss'
            dropped_frames.append(df_power)
            df_clean = df_clean[~power_mask]
            print("  Dropped %7d power-loss rows (zero charge in '%s')"
                  % (n_power, valid_charge_col))

    # ── Step 2: rolling-median context spike filter ──────────────────────────
    if signal_col not in df_clean.columns:
        print("  WARNING: signal column '%s' not found — spike filter skipped."
              % signal_col)
    else:
        sig = pd.to_numeric(df_clean[signal_col], errors='coerce')

        # Work only on the valid (non-NaN) values; keep their original index
        # positions so we can map results back to df_clean.
        valid_mask = sig.notna()
        valid_vals = sig[valid_mask].values       # 1-D array of valid values
        valid_pos  = np.where(valid_mask)[0]      # integer positions in sig
        n_valid    = len(valid_vals)

        spike_flag = np.zeros(n_valid, dtype=bool)

        for k in range(n_valid):
            lo = max(0, k - window)
            hi = min(n_valid, k + window + 1)

            # Collect neighbours (all valid samples within the window, excl. self)
            neighbour_vals = np.concatenate([
                valid_vals[lo:k],
                valid_vals[k + 1:hi],
            ])

            if len(neighbour_vals) < min_valid:
                continue  # not enough context — keep the point

            local_median = np.median(neighbour_vals)
            if abs(valid_vals[k] - local_median) > spike_threshold:
                spike_flag[k] = True

        n_spikes = spike_flag.sum()
        if n_spikes > 0:
            spike_positions = valid_pos[spike_flag]
            spike_index     = df_clean.index[spike_positions]
            df_spikes = df_clean.loc[spike_index].copy()
            df_spikes['drop_reason'] = 'spike'
            dropped_frames.append(df_spikes)
            df_clean = df_clean.drop(index=spike_index)
            print("  Dropped %7d spike rows  "
                  "(|deviation from local median| > %s, window=%d, min_valid=%d)"
                  % (n_spikes, spike_threshold, window, min_valid))
        else:
            print("  No spikes detected  (threshold=%s, window=%d)"
                  % (spike_threshold, window))

    retained_pct = 100 * len(df_clean) / n0 if n0 else 0.0
    print("  Retained %7d / %d rows (%.1f %%)" % (len(df_clean), n0, retained_pct))

    df_dropped = (
        pd.concat(dropped_frames)
        if dropped_frames
        else pd.DataFrame(columns=df.columns.tolist() + ['drop_reason'])
    )
    return df_clean, df_dropped


# ──────────────────────────────────────────────────────────────────────
def process_station(
    st,
    df_st,
    signal_col,
    temp_col,
    comp_coeff,
    spike_threshold,
    output_dir,
    window=7,
    min_valid=3,
):
    """
    Full preprocessing pipeline for a single station.

    Executes the four-stage physical pipeline in the correct order:

    1. **Power-loss removal** — zero-charge rows removed first so outage zeros
       never enter the spike-detection window.
    2. **Spike removal** — rolling-median context filter; post-outage spike runs
       are correctly removed because each spike row deviates from its
       pre-outage AND post-recovery neighbours.
    3. **Temperature compensation** — linear thermal correction applied to the
       clean signal using the first reading as thermal baseline.
    4. **Normalisation** — series shifted so the first valid compensated value
       is 0.

    Saves two CSVs per station:
    - ``{st}_preprocessed.csv``  — cleaned + compensated data (raw preserved
      as ``{signal_col}_raw`` for visualisation).
    - ``{st}_dropped.csv``        — all removed rows with ``drop_reason``.

    Parameters
    ----------
    st : str
        Station identifier (e.g. ``'st01'``).
    df_st : pd.DataFrame
        Raw station DataFrame from ``organize_sensor_data()``.
    signal_col : str
        Name of the structural signal column (e.g. ``'absinc'``).
    temp_col : str
        Name of the on-board temperature column (e.g. ``'temp'``).
    comp_coeff : float
        Thermal compensation coefficient (mdeg \u00b7 \u00b0C\u207b\u00b9 \u00b7 10\u207b\u00b3).
    spike_threshold : float
        Maximum allowed absolute deviation from the local rolling median
        (signal units).  Passed to ``clean_signal_robust()``.
    output_dir : str
        Directory for the saved interim CSVs.
    window : int, optional
        Half-window size for the rolling-median spike filter.  Default ``7``.
    min_valid : int, optional
        Minimum valid neighbours required to evaluate a sample.  Default ``3``.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned and compensated DataFrame (``{signal_col}_raw`` column removed).
    output_path : str
        Full path of the saved preprocessed CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + _SEP)
    print("  Station : %s  |  raw rows: %d" % (st, len(df_st)))
    print("  Filter  : rolling-median  window=%d  min_valid=%d  threshold=%s"
          % (window, min_valid, spike_threshold))
    print(_SEP)

    # Steps 1 & 2: power-loss removal then spike filter
    df_clean, df_dropped = clean_signal_robust(
        df_st,
        signal_col=signal_col,
        spike_threshold=spike_threshold,
        window=window,
        min_valid=min_valid,
        valid_charge_col='charge',
    )

    # Steps 3 & 4: temperature compensation + normalisation
    if df_clean.empty:
        print("  WARNING: no rows remain after cleaning — "
              "compensation skipped. Review spike_threshold / window.")
    elif signal_col in df_clean.columns and temp_col in df_clean.columns:
        df_clean[signal_col + '_raw'] = df_clean[signal_col].copy()
        df_clean = apply_compensation(
            df=df_clean,
            target_col=signal_col,
            new_col_name=signal_col + '_comp',
            comp_func=temp_compensation,
            normalize=True,
            temp_col=temp_col,
            comp_coeff=comp_coeff,
        )
        df_clean = df_clean.drop(columns=[signal_col])
        df_clean = df_clean.rename(columns={signal_col + '_comp': signal_col})

    # Save main CSV (raw column kept for visualisation in Notebook 00)
    output_path = os.path.join(output_dir, st + '_preprocessed.csv')
    df_clean.to_csv(output_path)
    print("  Saved  \u2192 " + output_path)
    print("  Shape  : %s  |  %s \u2192 %s"
          % (str(df_clean.shape), df_clean.index.min(), df_clean.index.max()))

    # Save dropped rows companion CSV
    dropped_path = os.path.join(output_dir, st + '_dropped.csv')
    df_dropped.to_csv(dropped_path)
    print("  Dropped \u2192 %s  (%d rows)" % (dropped_path, len(df_dropped)))

    return df_clean.drop(columns=[signal_col + '_raw'], errors='ignore'), output_path


# ──────────────────────────────────────────────────────────────────────
def apply_compensation(df, target_col, new_col_name, comp_func,
                       normalize=True, **kwargs):
    """
    Applies a user-defined compensation formula to a specific column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        The column to compensate (e.g. ``'absinc'``).
    new_col_name : str
        Name for the compensated output column.
    comp_func : callable
        Function with signature ``(df, target_col, **kwargs) -> pd.Series``.
    normalize : bool, optional
        If ``True``, subtracts the first valid value so the series starts at
        0.  Default ``True``.
    **kwargs
        Extra arguments forwarded to ``comp_func``
        (e.g. ``temp_col='temp'``, ``comp_coeff=0.005``).

    Returns
    -------
    pd.DataFrame
        DataFrame with the new compensated column appended.
    """
    df_comp = df.copy()

    if target_col not in df_comp.columns:
        raise KeyError("Target column '%s' not found in DataFrame." % target_col)

    df_comp[new_col_name] = comp_func(df_comp, target_col, **kwargs)

    if normalize and len(df_comp[new_col_name].dropna()) > 0:
        first_valid = df_comp[new_col_name].dropna().iloc[0]
        df_comp[new_col_name] = df_comp[new_col_name] - first_valid

    return df_comp


# ──────────────────────────────────────────────────────────────────────
def temp_compensation(df, target_col, temp_col='temp', comp_coeff=0.005):
    """
    Calculates the thermal compensation correction and applies it to the signal.

    Uses the *first* reading as the thermal baseline (``df[temp_col].iloc[0]``).
    This is safe when the first record is trustworthy (i.e. the DataFrame has
    already had power-loss and spike rows removed by ``clean_signal_robust()``).

    Formula::

        corrected = signal - (temp - temp_ref) * comp_coeff * 1000

    where ``temp_ref = temp[0]``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the target signal and temperature columns.
        Must already be cleaned (power-loss and spike rows removed).
    target_col : str
        Structural signal column to compensate.
    temp_col : str, optional
        On-board temperature column.  Default ``'temp'``.
    comp_coeff : float, optional
        Thermal compensation coefficient (mdeg \u00b7 \u00b0C\u207b\u00b9 \u00b7 10\u207b\u00b3).
        Default ``0.005``.

    Returns
    -------
    pd.Series
        Compensated signal series (same index as ``df``).
    """
    temp_ref  = df[temp_col].iloc[0]
    temp_diff = (df[temp_col] - temp_ref) * comp_coeff * 1000
    return df[target_col] - temp_diff


# ──────────────────────────────────────────────────────────────────────
def filter_by_date_range(df, start_str=None, end_str=None, ignore_year=False):
    """
    Filters the DataFrame to a specific date range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    start_str : str or None
        ``'YYYY-MM-DD'`` or ``'MM-DD'`` if ``ignore_year`` is ``True``.
    end_str : str or None
        ``'YYYY-MM-DD'`` or ``'MM-DD'`` if ``ignore_year`` is ``True``.
    ignore_year : bool, optional
        If ``True``, filters strictly by month and day across all years.
        Default ``False``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if ignore_year:
        if not start_str or not end_str:
            raise ValueError(
                "Must provide both start_str and end_str when ignore_year=True."
            )
        df_filtered = df.copy()
        df_filtered['month_day'] = pd.to_datetime(
            '2000-' + df_filtered.index.strftime('%m-%d'),
            format='2000-%m-%d', errors='coerce',
        )
        start_date = pd.to_datetime('2000-' + start_str, format='2000-%m-%d', errors='coerce')
        end_date   = pd.to_datetime('2000-' + end_str,   format='2000-%m-%d', errors='coerce')
        mask = (
            (df_filtered['month_day'] >= start_date)
            & (df_filtered['month_day'] <= end_date)
        )
        return df_filtered[mask].drop(columns=['month_day'])
    else:
        if start_str and end_str:
            return df.loc[start_str:end_str]
        elif start_str:
            return df.loc[start_str:]
        elif end_str:
            return df.loc[:end_str]
        return df


# ──────────────────────────────────────────────────────────────────────
def align_and_resample(df_sensor, df_proxy, resample_freq='1H',
                       interpolation='time'):
    """
    Synchronizes the on-site sensor data with a single external proxy dataset.

    Parameters
    ----------
    df_sensor : pd.DataFrame
        High-frequency sensor data (e.g. 20-min resolution).
    df_proxy : pd.DataFrame
        Low-frequency proxy data (e.g. hourly ERA5).
    resample_freq : str, optional
        Target frequency string (e.g. ``'1H'``).  Default ``'1H'``.
    interpolation : str, optional
        Method for upsampling gaps (``'time'``, ``'spline'``, ``'linear'``).
        Default ``'time'``.

    Returns
    -------
    pd.DataFrame
        Merged, aligned DataFrame.
    """
    print("Resampling sensor data to %s..." % resample_freq)
    sensor_resampled = df_sensor.resample(resample_freq).mean()
    if interpolation == 'spline':
        sensor_resampled = sensor_resampled.interpolate(method='spline', order=3)
    else:
        sensor_resampled = sensor_resampled.interpolate(method=interpolation)
    proxy_resampled = df_proxy.resample(resample_freq).mean()
    print("Merging sensor and proxy datasets...")
    df_merged = pd.merge(
        sensor_resampled, proxy_resampled,
        left_index=True, right_index=True, how='inner',
    )
    print("Final aligned dataset contains %d rows." % len(df_merged))
    return df_merged


# ──────────────────────────────────────────────────────────────────────
def align_multiple_proxies(df_sensor, proxies_dict, resample_freq='h',
                           interpolation=None, add_prefix=True):
    """
    Synchronizes the on-site sensor data with multiple external proxy datasets.

    Creates a complete regular time index spanning the full sensor date range
    so that periods with no observations are preserved as NaN for gap analysis.

    Parameters
    ----------
    df_sensor : pd.DataFrame
        High-frequency sensor data.
    proxies_dict : dict
        Mapping of proxy names to DataFrames,
        e.g. ``{'era5': df_era, 'local': df_loc}``.
    resample_freq : str, optional
        Target frequency string (e.g. ``'h'``).  Default ``'h'``.
    interpolation : str or None, optional
        Method for upsampling (``'time'``, ``'spline'``, ``'linear'``).
        ``None`` (default) preserves all NaNs for gap characterisation.
    add_prefix : bool, optional
        If ``True``, prefixes proxy columns with their dictionary key.
        Default ``True``.

    Returns
    -------
    pd.DataFrame
        Merged aligned DataFrame with a complete regular index.
    """
    print("Resampling sensor data to %s..." % resample_freq)
    sensor_resampled = df_sensor.resample(resample_freq).mean()
    full_index = pd.date_range(
        start=sensor_resampled.index.min(),
        end=sensor_resampled.index.max(),
        freq=resample_freq,
    )
    sensor_resampled = sensor_resampled.reindex(full_index)
    sensor_resampled.index.name = 'datetime'
    print("Complete index: %d time steps (%s \u2192 %s)"
          % (len(full_index), sensor_resampled.index.min(), sensor_resampled.index.max()))
    print("Sensor NaN rows (gaps): %d"
          % sensor_resampled.isna().any(axis=1).sum())
    if interpolation:
        if interpolation == 'spline':
            sensor_resampled = sensor_resampled.interpolate(method='spline', order=3)
        else:
            sensor_resampled = sensor_resampled.interpolate(method=interpolation)
    df_merged = sensor_resampled
    for name, df_proxy in proxies_dict.items():
        print("Resampling proxy '%s' to %s..." % (name, resample_freq))
        proxy_resampled = df_proxy.resample(resample_freq).mean()
        if add_prefix:
            proxy_resampled.columns = [
                name + '_' + col for col in proxy_resampled.columns
            ]
        print("Merging '%s'..." % name)
        df_merged = pd.merge(
            df_merged, proxy_resampled,
            left_index=True, right_index=True, how='left',
        )
    print("Final aligned dataset contains %d rows." % len(df_merged))
    return df_merged
