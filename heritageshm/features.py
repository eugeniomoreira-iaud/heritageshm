"""
Module: features.py
Handles the generation of physically motivated lagged regressors to capture 
thermal inertia (heat diffusion memory) in masonry structures.
"""
import pandas as pd
import numpy as np

def generate_lagged_features(df, target_col, proxy_col, max_lag=24, lag_step=1):
    """
    Generates lagged versions of the environmental proxy to embed the 
    thermal memory (heat diffusion dynamics) into the feature matrix.
    
    Parameters:
    - df: The input DataFrame.
    - target_col: The structural response column (kept unchanged).
    - proxy_col: The environmental proxy column to lag (e.g., skin temperature).
    - max_lag: The maximum number of backward time steps to generate (e.g., 24 hours).
    - lag_step: The interval between lags. E.g., if step=2, it generates lag 2, 4, 6...
    
    Returns:
    - A new DataFrame containing the original columns plus the newly generated lag features.
    """
    if proxy_col not in df.columns:
        raise KeyError(f"Proxy column '{proxy_col}' not found in DataFrame.")
        
    df_lagged = df.copy()
    
    print(f"Generating thermal lag features for '{proxy_col}' up to {max_lag} steps (step={lag_step})...")
    
    # Generate the lagged columns
    lag_cols = []
    for lag in range(lag_step, max_lag + 1, lag_step):
        col_name = f"{proxy_col}_lag_{lag}"
        df_lagged[col_name] = df_lagged[proxy_col].shift(lag)
        lag_cols.append(col_name)
        
    # Print a summary of the generated features
    print(f"Generated {len(lag_cols)} new features: {', '.join(lag_cols[:3])}... to {lag_cols[-1]}")
    
    return df_lagged

def rank_features_by_correlation(df, target_col, feature_cols=None):
    """
    Ranks the generated lagged features based on their absolute Pearson correlation 
    with the target structural response. This helps identify the dominant thermal lag.
    
    Parameters:
    - df: The input DataFrame containing both target and features.
    - target_col: The structural response column to correlate against.
    - feature_cols: List of column names to rank. If None, it ranks all columns except the target.
    
    Returns:
    - A pandas Series of absolute correlation values, sorted descending.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
        
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]
        
    # Ensure all requested feature columns actually exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        raise ValueError("No valid feature columns provided to rank.")
        
    # Calculate correlations
    print(f"\n--- Ranking Thermal Inertia Lags against '{target_col}' ---")
    corr_matrix = df[[target_col] + feature_cols].corr()
    
    # Extract correlation with target, take absolute value, drop the target itself, and sort
    target_corr = corr_matrix[target_col].drop(labels=[target_col]).abs().sort_values(ascending=False)
    
    # Print the top 5 most correlated lags
    print("Top 5 dominant thermal lags (Absolute Pearson R):")
    print(target_corr.head(5).to_string())
    
    return target_corr

def calculate_moving_averages(df, proxy_col, windows=[6, 12, 24]):
    """
    Alternative/complementary feature engineering: calculates moving averages 
    of the environmental proxy to represent accumulated thermal loading.
    
    Parameters:
    - df: The input DataFrame.
    - proxy_col: The environmental proxy column.
    - windows: List of integers representing rolling window sizes.
    
    Returns:
    - DataFrame with new moving average columns appended.
    """
    if proxy_col not in df.columns:
        raise KeyError(f"Proxy column '{proxy_col}' not found in DataFrame.")
        
    df_ma = df.copy()
    
    print(f"Generating moving averages for '{proxy_col}' (Windows: {windows})...")
    for window in windows:
        col_name = f"{proxy_col}_ma_{window}"
        # Use min_periods=1 to avoid creating excessive NaNs at the beginning
        df_ma[col_name] = df_ma[proxy_col].rolling(window=window, min_periods=1).mean()
        
    return df_ma