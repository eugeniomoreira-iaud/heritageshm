"""
Module: preprocessing.py
Handles signal cleaning, outlier removal, timestamp alignment,
and user-defined physical compensation (e.g., thermal correction).
"""
import pandas as pd
import numpy as np
import os


def clean_signal(df, signal_col, spike_threshold, valid_charge_col=None):
    """
    Cleans a raw sensor signal in two sequential steps:

    1. Spike removal (gap-aware difference filter):
       Computes the absolute first-order difference on *valid consecutive pairs*
       only — i.e., pairs where neither the current nor the previous sample is
       NaN or zero (power-outage proxy).  This avoids false spikes at gap
       boundaries where the difference is inflated by a missing segment.

       For each valid pair whose |Δy| > spike_threshold the *later* sample is
       flagged as a spike and removed.

    2. Power-loss removal:
       Drops rows where the charge column equals zero.

    Parameters
    ----------
    df : pd.DataFrame
        Station DataFrame with a DatetimeIndex.
    signal_col : str
        Name of the structural signal column (e.g. 'absinc').
    spike_threshold : float
        Hard maximum allowed |Δy| between *valid consecutive* samples.
        Differences that cross NaN / zero-charge boundaries are excluded.
    valid_charge_col : str or None
        Column whose zero values indicate a power-loss event (e.g. 'charge').
        Set to None to skip this filter.

    Returns
    -------
    df_clean : pd.DataFrame
        Retained rows only.
    df_dropped : pd.DataFrame
        All dropped rows with an extra column 'drop_reason'
        ('spike' or 'power_loss') for downstream visualisation.
    """
    df_clean = df.copy()
    n0 = len(df_clean)
    dropped_frames = []

    # ── Step 1: gap-aware spike filter ───────────────────────────────────────
    if signal_col in df_clean.columns:
        numeric_signal = pd.to_numeric(df_clean[signal_col], errors='coerce')

        # A sample is a valid predecessor only when it is not NaN and not zero
        # (zero is used as a power-outage sentinel in these sensors).
        valid_for_diff = numeric_signal.notna() & (numeric_signal != 0)

        # Compute raw consecutive differences then mask out transitions that
        # cross a gap (NaN or zero on either side).
        raw_diff = numeric_signal.diff().abs()
        prev_valid = valid_for_diff.shift(1, fill_value=False)
        curr_valid = valid_for_diff
        meaningful_diff = raw_diff.where(prev_valid & curr_valid)

        spike_mask = meaningful_diff > spike_threshold
        n_spikes = spike_mask.sum()
        if n_spikes > 0:
            df_spikes = df_clean[spike_mask].copy()
            df_spikes['drop_reason'] = 'spike'
            dropped_frames.append(df_spikes)
            df_clean = df_clean[~spike_mask]
            print(f"  Dropped {n_spikes:>7} spike rows  "
                  f"(|\u0394 {signal_col}| > {spike_threshold}, gap-aware)")
    else:
        print(f"  WARNING: signal column '{signal_col}' not found — "
              f"spike filter skipped.")

    # ── Step 2: power-loss filter ─────────────────────────────────────────────
    if valid_charge_col and valid_charge_col in df_clean.columns:
        n_before = len(df_clean)
        power_mask = df_clean[valid_charge_col] == 0
        if power_mask.sum() > 0:
            df_power = df_clean[power_mask].copy()
            df_power['drop_reason'] = 'power_loss'
            dropped_frames.append(df_power)
            df_clean = df_clean[~power_mask]
            n_dropped = n_before - len(df_clean)
            print(f"  Dropped {n_dropped:>7} power-loss rows "
                  f"(zero charge in '{valid_charge_col}')")

    print(f"  Retained {len(df_clean):>7} / {n0} rows "
          f"({100 * len(df_clean) / n0:.1f} %)")

    df_dropped = (
        pd.concat(dropped_frames)
        if dropped_frames
        else pd.DataFrame(columns=df.columns.tolist() + ['drop_reason'])
    )
    return df_clean, df_dropped


def process_station(st, df_st, signal_col, temp_col, comp_coeff,
                    spike_threshold, output_dir):
    """
    Full preprocessing pipeline for a single station.

    1. Spike removal via gap-aware difference filter (excludes NaN / zero
       boundaries so gap edges never produce false spikes).
    2. Power-loss removal (zero charge rows).
    3. Temperature compensation + normalization (series starts at 0).
    4. Saves interim CSV with compensated column and raw column preserved.
    5. Saves a companion '_dropped.csv' with all removed rows and their reason.

    Parameters
    ----------
    st : str
        Station identifier (e.g. 'st01').
    df_st : pd.DataFrame
        Raw station DataFrame from organize_sensor_data().
    signal_col : str
        Name of the structural signal column (e.g. 'absinc').
    temp_col : str
        Name of the on-board temperature column (e.g. 'temp').
    comp_coeff : float
        Thermal compensation coefficient (mdeg \u00b7 \u00b0C\u207b\u00b9 \u00b7 10\u207b\u00b3).
    spike_threshold : float
        Maximum allowed absolute first-order difference on valid consecutive
        samples (signal units).  Differences that cross NaN / power-outage
        boundaries are automatically excluded.
    output_dir : str
        Directory for the saved interim CSV.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned and compensated DataFrame (raw column removed).
    output_path : str
        Full path of the saved CSV.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'\u2500'*50}")
    print(f"  Station: {st}  |  raw rows: {len(df_st)}")
    print(f"{'\u2500'*50}")

    # Steps 1 & 2: gap-aware spike filter then power-loss filter
    df_clean, df_dropped = clean_signal(
        df_st,
        signal_col=signal_col,
        spike_threshold=spike_threshold,
        valid_charge_col='charge',
    )

    # Step 3: temperature compensation + normalization
    if df_clean.empty:
        print(f"  WARNING: no rows remain after cleaning — "
              f"compensation skipped. Review spike_threshold.")
    elif signal_col in df_clean.columns and temp_col in df_clean.columns:
        df_clean[f'{signal_col}_raw'] = df_clean[signal_col].copy()
        df_clean = apply_compensation(
            df=df_clean,
            target_col=signal_col,
            new_col_name=f'{signal_col}_comp',
            comp_func=temp_compensation,
            normalize=True,
            temp_col=temp_col,
            comp_coeff=comp_coeff,
        )
        df_clean = df_clean.drop(columns=[signal_col])
        df_clean = df_clean.rename(columns={f'{signal_col}_comp': signal_col})

    # Step 4: save main CSV (raw preserved for visualisation)
    output_path = os.path.join(output_dir, f'{st}_preprocessed.csv')
    df_clean.to_csv(output_path)
    print(f"  Saved \u2192 {output_path}")
    print(f"  Shape  : {df_clean.shape}  |  "
          f"{df_clean.index.min()} \u2192 {df_clean.index.max()}")

    # Step 5: save dropped rows companion CSV
    dropped_path = os.path.join(output_dir, f'{st}_dropped.csv')
    df_dropped.to_csv(dropped_path)
    print(f"  Dropped \u2192 {dropped_path}  ({len(df_dropped)} rows)")

    return df_clean.drop(columns=[f'{signal_col}_raw'], errors='ignore'), output_path


def apply_compensation(df, target_col, new_col_name, comp_func, normalize=True, **kwargs):
    """
    Applies a user-defined compensation formula to a specific column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        The column to compensate (e.g. 'absinc').
    new_col_name : str
        Name for the compensated output column.
    comp_func : callable
        Function that takes (df, target_col, **kwargs) and returns a Series.
    normalize : bool
        If True, subtracts the first valid value so the series starts at 0.
    **kwargs
        Extra arguments forwarded to comp_func (e.g. comp_coeff=0.005).

    Returns
    -------
    pd.DataFrame
        DataFrame with the new compensated column appended.
    """
    df_comp = df.copy()

    if target_col not in df_comp.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    df_comp[new_col_name] = comp_func(df_comp, target_col, **kwargs)

    if normalize and len(df_comp[new_col_name].dropna()) > 0:
        first_valid = df_comp[new_col_name].dropna().iloc[0]
        df_comp[new_col_name] = df_comp[new_col_name] - first_valid

    return df_comp


def filter_by_date_range(df, start_str=None, end_str=None, ignore_year=False):
    """
    Filters the DataFrame to a specific date range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    start_str : str or None
        'YYYY-MM-DD' or 'MM-DD' if ignore_year is True.
    end_str : str or None
        'YYYY-MM-DD' or 'MM-DD' if ignore_year is True.
    ignore_year : bool
        If True, filters strictly by month and day across all years.

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
        mask = (df_filtered['month_day'] >= start_date) & (df_filtered['month_day'] <= end_date)
        return df_filtered[mask].drop(columns=['month_day'])
    else:
        if start_str and end_str:
            return df.loc[start_str:end_str]
        elif start_str:
            return df.loc[start_str:]
        elif end_str:
            return df.loc[:end_str]
        return df


def align_and_resample(df_sensor, df_proxy, resample_freq='1H', interpolation='time'):
    """
    Synchronizes the on-site sensor data with the external proxy data.

    Parameters
    ----------
    df_sensor : pd.DataFrame
        High-frequency sensor data (e.g. 20-min resolution).
    df_proxy : pd.DataFrame
        Low-frequency proxy data (e.g. hourly ERA5).
    resample_freq : str
        Target frequency string (e.g. '1H').
    interpolation : str
        Method for upsampling gaps ('time', 'spline', 'linear').

    Returns
    -------
    pd.DataFrame
        Merged, aligned DataFrame.
    """
    print(f"Resampling sensor data to {resample_freq}...")
    sensor_resampled = df_sensor.resample(resample_freq).mean()
    if interpolation == 'spline':
        sensor_resampled = sensor_resampled.interpolate(method='spline', order=3)
    else:
        sensor_resampled = sensor_resampled.interpolate(method=interpolation)
    proxy_resampled = df_proxy.resample(resample_freq).mean()
    print("Merging sensor and proxy datasets...")
    df_merged = pd.merge(sensor_resampled, proxy_resampled,
                         left_index=True, right_index=True, how='inner')
    print(f"Final aligned dataset contains {len(df_merged)} rows.")
    return df_merged


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
        Dictionary mapping proxy names to DataFrames, e.g.
        {'era5': df_era, 'local': df_loc}.
    resample_freq : str
        Target frequency string (e.g. 'h' for hourly).
    interpolation : str or None
        Method for upsampling ('time', 'spline', 'linear').
        None (default) preserves all NaNs for gap characterization.
    add_prefix : bool
        If True, prefixes proxy columns with their dictionary key.

    Returns
    -------
    pd.DataFrame
        Merged aligned DataFrame with complete regular index.
    """
    print(f"Resampling sensor data to {resample_freq}...")
    sensor_resampled = df_sensor.resample(resample_freq).mean()
    full_index = pd.date_range(
        start=sensor_resampled.index.min(),
        end=sensor_resampled.index.max(),
        freq=resample_freq,
    )
    sensor_resampled = sensor_resampled.reindex(full_index)
    sensor_resampled.index.name = 'datetime'
    print(f"Complete index: {len(full_index)} time steps "
          f"({sensor_resampled.index.min()} \u2192 {sensor_resampled.index.max()})")
    print(f"Sensor NaN rows (gaps): {sensor_resampled.isna().any(axis=1).sum()}")
    if interpolation:
        if interpolation == 'spline':
            sensor_resampled = sensor_resampled.interpolate(method='spline', order=3)
        else:
            sensor_resampled = sensor_resampled.interpolate(method=interpolation)
    df_merged = sensor_resampled
    for name, df_proxy in proxies_dict.items():
        print(f"Resampling proxy '{name}' to {resample_freq}...")
        proxy_resampled = df_proxy.resample(resample_freq).mean()
        if add_prefix:
            proxy_resampled.columns = [f"{name}_{col}" for col in proxy_resampled.columns]
        print(f"Merging '{name}'...")
        df_merged = pd.merge(df_merged, proxy_resampled,
                             left_index=True, right_index=True, how='left')
    print(f"Final aligned dataset contains {len(df_merged)} rows.")
    return df_merged


def temp_compensation(df, target_col, temp_col='temp', comp_coeff=0.005):
    """
    Calculates compensation value based on continuous baseline temperature variation.
    Intended to be passed into apply_compensation().

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing target and temperature columns.
    target_col : str
        Structural signal column to compensate.
    temp_col : str
        On-board temperature column.
    comp_coeff : float
        Thermal compensation coefficient (mdeg \u00b7 \u00b0C\u207b\u00b9 \u00b7 10\u207b\u00b3).

    Returns
    -------
    pd.Series
        Compensated signal series.
    """
    temp_diff = (df[temp_col] - df[temp_col].iloc[0]) * comp_coeff * 1000
    return df[target_col] - temp_diff
