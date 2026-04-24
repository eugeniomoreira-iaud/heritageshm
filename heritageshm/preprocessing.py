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

    1. **Spike removal (first-order difference filter):**
       Computes |Δy_t| = |y_t − y_{t−1}| for the signal column.
       Any row where this absolute difference exceeds `spike_threshold`
       is dropped. Applied *before* charge filtering so the difference
       is computed on the uninterrupted raw sequence.

    2. **Power-loss removal:**
       Drops rows where the charge column equals zero (sensor off or
       battery depleted). Applied after spike removal.

    Parameters
    ----------
    df : pd.DataFrame
        Raw station DataFrame with a DatetimeIndex.
    signal_col : str
        Name of the structural signal column (e.g. 'absinc').
    spike_threshold : float
        Maximum allowed absolute first-order difference (in signal units,
        e.g. mdeg). Rows exceeding this are treated as glitches and removed.
        A value of ~5–20 mdeg is typical for inclinometers; set based on
        the expected maximum physical rate of change between samples.
    valid_charge_col : str or None
        Column name for battery charge. Rows where this equals zero are
        dropped. Set to None to skip this filter.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with the same columns as the input.
    """
    df_clean = df.copy()
    n0 = len(df_clean)

    # ── Step 1: spike filter on first-order difference ───────────────────────
    if signal_col in df_clean.columns:
        numeric_signal = pd.to_numeric(df_clean[signal_col], errors='coerce')
        abs_diff = numeric_signal.diff().abs()
        spike_mask = abs_diff > spike_threshold
        n_spikes = spike_mask.sum()
        df_clean = df_clean[~spike_mask]
        if n_spikes > 0:
            print(f"  Dropped {n_spikes:>7} spike rows  "
                  f"(|Δ{signal_col}| > {spike_threshold})")
    else:
        print(f"  WARNING: signal column '{signal_col}' not found — "
              f"spike filter skipped.")

    # ── Step 2: power-loss filter ─────────────────────────────────────────────
    if valid_charge_col and valid_charge_col in df_clean.columns:
        n_before = len(df_clean)
        df_clean = df_clean[df_clean[valid_charge_col] != 0]
        n_dropped = n_before - len(df_clean)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped:>7} power-loss rows "
                  f"(zero charge in '{valid_charge_col}')")

    print(f"  Retained {len(df_clean):>7} / {n0} rows "
          f"({100 * len(df_clean) / n0:.1f} %)")

    return df_clean

def apply_compensation(df, target_col, new_col_name, comp_func, normalize=True, **kwargs):
    """
    Applies a user-defined compensation formula to a specific column.
    
    Parameters:
    - df: The input DataFrame.
    - target_col: The column to compensate (e.g., 'st01_absinc').
    - new_col_name: What to call the compensated column (e.g., 'st01_absinc_clean').
    - comp_func: A Python function that takes the DataFrame and returns a Series.
    - normalize: Boolean. If True, subtracts the first valid value so the series starts at 0.
    - kwargs: Extra arguments to pass to the comp_func (like comp_coeff=0.005).
              
    Returns:
    - DataFrame with the new compensated column appended.
    """
    df_comp = df.copy()
    
    if target_col not in df_comp.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    
    # Apply the external function
    df_comp[new_col_name] = comp_func(df_comp, target_col, **kwargs)
    
    # Normalize the compensated data so it starts at 0
    if normalize and len(df_comp[new_col_name].dropna()) > 0:
        first_valid = df_comp[new_col_name].dropna().iloc[0]
        df_comp[new_col_name] = df_comp[new_col_name] - first_valid
        
    return df_comp

def filter_by_date_range(df, start_str=None, end_str=None, ignore_year=False):
    """
    Filters the DataFrame to a specific date range.
    Can also filter by month/day while ignoring the year (e.g., to extract all August data).
    
    Parameters:
    - df: DataFrame with DatetimeIndex.
    - start_str: String format 'YYYY-MM-DD' or 'MM-DD' if ignore_year is True.
    - end_str: String format 'YYYY-MM-DD' or 'MM-DD' if ignore_year is True.
    - ignore_year: If True, filters strictly by month and day across all years.
    """
    if ignore_year:
        if not start_str or not end_str:
            raise ValueError("Must provide both start_str and end_str when ignore_year=True.")
            
        df_filtered = df.copy()
        # Create a temporary 'month_day' column using a fixed placeholder year (2000).
        df_filtered['month_day'] = pd.to_datetime('2000-' + df_filtered.index.strftime('%m-%d'), format='2000-%m-%d', errors='coerce')
        
        start_date = pd.to_datetime('2000-' + start_str, format='2000-%m-%d', errors='coerce')
        end_date = pd.to_datetime('2000-' + end_str, format='2000-%m-%d', errors='coerce')
        
        mask = (df_filtered['month_day'] >= start_date) & (df_filtered['month_day'] <= end_date)
        df_filtered = df_filtered[mask].drop(columns=['month_day'])
        return df_filtered
        
    else:
        # Standard Datetime filtering
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
    
    Parameters:
    - df_sensor: DataFrame of high-frequency sensor data (e.g., 20-min).
    - df_proxy: DataFrame of low-frequency proxy data (e.g., 1-hour).
    - resample_freq: The target frequency string (e.g., '1H' for hourly).
    - interpolation: Method to fill gaps during upsampling ('time', 'spline', 'linear').
    """
    print(f"Resampling sensor data to {resample_freq}...")
    
    # Resample sensor data (mean aggregation for downsampling)
    sensor_resampled = df_sensor.resample(resample_freq).mean()
    
    # Use spline interpolation to preserve smooth thermal cycles if requested
    if interpolation == 'spline':
        # Limit the spline order if data is extremely sparse
        sensor_resampled = sensor_resampled.interpolate(method='spline', order=3)
    else:
        sensor_resampled = sensor_resampled.interpolate(method=interpolation)
        
    # Resample proxy data to ensure perfect temporal alignment
    proxy_resampled = df_proxy.resample(resample_freq).mean()
    
    print("Merging sensor and proxy datasets...")
    # Inner join to keep only the overlapping monitoring period
    df_merged = pd.merge(sensor_resampled, proxy_resampled, left_index=True, right_index=True, how='inner')
    
    print(f"Final aligned dataset contains {len(df_merged)} rows.")
    return df_merged

def align_multiple_proxies(df_sensor, proxies_dict, resample_freq='h', interpolation=None, add_prefix=True):
    """
    Synchronizes the on-site sensor data with multiple external proxy datasets.
    
    Creates a complete regular time index spanning the full sensor date range,
    so that periods with no observations are preserved as NaN for gap analysis.
    
    Parameters:
    - df_sensor: DataFrame of high-frequency sensor data.
    - proxies_dict: Dictionary of proxy DataFrames, e.g. {'era5': df_era, 'local': df_loc}.
    - resample_freq: The target frequency string (e.g., 'h' for hourly).
    - interpolation: Method to fill small gaps during upsampling ('time', 'spline', 'linear').
                     Set to None (default) to preserve all NaNs for gap characterization.
    - add_prefix: Boolean. If True, prefixes proxy columns with their dictionary key.
    """
    print(f"Resampling sensor data to {resample_freq}...")
    
    # Resample sensor to target frequency (aggregates sub-intervals via mean)
    sensor_resampled = df_sensor.resample(resample_freq).mean()
    
    # Create a COMPLETE regular index from first to last timestamp
    full_index = pd.date_range(
        start=sensor_resampled.index.min(),
        end=sensor_resampled.index.max(),
        freq=resample_freq
    )
    
    # Reindex onto the complete grid — missing periods become NaN
    sensor_resampled = sensor_resampled.reindex(full_index)
    sensor_resampled.index.name = 'datetime'
    
    print(f"Complete index: {len(full_index)} time steps "
          f"({sensor_resampled.index.min()} → {sensor_resampled.index.max()})")
    print(f"Sensor NaN rows (gaps): {sensor_resampled.isna().any(axis=1).sum()}")
    
    # Only interpolate if explicitly requested (preserves NaN gaps by default)
    if interpolation:
        if interpolation == 'spline':
            sensor_resampled = sensor_resampled.interpolate(method='spline', order=3)
        else:
            sensor_resampled = sensor_resampled.interpolate(method=interpolation)
        
    df_merged = sensor_resampled
    
    for name, df_proxy in proxies_dict.items():
        print(f"Resampling proxy '{name}' to {resample_freq}...")
        proxy_resampled = df_proxy.resample(resample_freq).mean()
        
        # Rename columns to avoid collisions
        if add_prefix:
            proxy_resampled.columns = [f"{name}_{col}" for col in proxy_resampled.columns]
        
        print(f"Merging '{name}'...")
        # Left join to keep the full sensor index (proxies fill where available)
        df_merged = pd.merge(df_merged, proxy_resampled, left_index=True, right_index=True, how='left')
        
    print(f"Final aligned dataset contains {len(df_merged)} rows.")
    return df_merged

def temp_compensation(df, target_col, temp_col='temp', comp_coeff=0.005):
    """
    Calculates compensation value based on continuous baseline temperature variation.
    Intended to be passed into apply_compensation().
    """
    temp_diff = (df[temp_col] - df[temp_col].iloc[0]) * comp_coeff * 1000
    return df[target_col] - temp_diff

def process_station(st, df_st, signal_col, temp_col, comp_coeff,
                    spike_threshold, output_dir):
    """
    Full preprocessing pipeline for a single station.

    1. Spike removal via first-order difference filter.
    2. Power-loss removal (zero charge rows).
    3. Temperature compensation.
    4. Saves interim CSV with both raw and compensated columns.

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
        Thermal compensation coefficient (mdeg · °C⁻¹ · 10⁻³).
    spike_threshold : float
        Maximum allowed absolute first-order difference (signal units).
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

    print(f"\n{'─'*50}")
    print(f"  Station: {st}  |  raw rows: {len(df_st)}")
    print(f"{'─'*50}")

    # Step 1 & 2: spike filter then power-loss filter
    df_clean = clean_signal(
        df_st,
        signal_col=signal_col,
        spike_threshold=spike_threshold,
        valid_charge_col='charge',
    )

    # Step 3: temperature compensation
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
            normalize=False,
            temp_col=temp_col,
            comp_coeff=comp_coeff,
        )
        df_clean = df_clean.drop(columns=[signal_col])
        df_clean = df_clean.rename(columns={f'{signal_col}_comp': signal_col})

    # Step 4: save (raw column preserved for visualisation)
    output_path = os.path.join(output_dir, f'{st}_preprocessed.csv')
    df_clean.to_csv(output_path)
    print(f"  Saved → {output_path}")
    print(f"  Shape  : {df_clean.shape}  |  "
          f"{df_clean.index.min()} → {df_clean.index.max()}")

    return df_clean.drop(columns=[f'{signal_col}_raw'], errors='ignore'), output_path