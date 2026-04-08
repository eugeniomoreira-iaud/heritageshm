"""
Module: preprocessing.py
Handles signal cleaning, outlier removal, timestamp alignment, 
and user-defined physical compensation (e.g., thermal correction).
"""
import pandas as pd
import numpy as np

def clean_signal(df, valid_charge_col=None, outlier_thresholds=None):
    """
    Removes rows where the sensor lost power and drops extreme outliers.
    
    Parameters:
    - df: The input DataFrame.
    - valid_charge_col: String name of the battery/charge column. Drops rows where this is 0.
    - outlier_thresholds: Dictionary of {column_name: min_valid_value}. 
                          e.g., {'st01_absinc': -500}
                          
    Returns:
    - Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # 1. Filter out power loss (charge == 0)
    if valid_charge_col and valid_charge_col in df_clean.columns:
        initial_len = len(df_clean)
        df_clean = df_clean[df_clean[valid_charge_col] != 0]
        dropped = initial_len - len(df_clean)
        if dropped > 0:
            print(f"Dropped {dropped} rows due to zero charge in '{valid_charge_col}'.")
        
    # 2. Filter out extreme physical outliers
    if outlier_thresholds:
        for col, min_val in outlier_thresholds.items():
            if col in df_clean.columns:
                initial_len = len(df_clean)
                df_clean = df_clean[df_clean[col] > min_val]
                dropped = initial_len - len(df_clean)
                if dropped > 0:
                    print(f"Dropped {dropped} outliers below {min_val} in '{col}'.")
                
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