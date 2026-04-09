"""
Module: diagnostics.py
Handles gap taxonomy characterization, cointegration testing, 
and residual diagnostics (ADF, Ljung-Box).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import pearsonr

def shift_and_correlate(df, target, proxy, max_lag, step=1):
    """
    Screens cross-correlation across varying lags to find the optimal thermal inertia.
    
    Parameters:
    - df: DataFrame containing target and proxy.
    - target: Name of the target column.
    - proxy: Name of the proxy column.
    - max_lag: Maximum lag to test (e.g., 72 hours).
    - step: Step size for lags.
    
    Returns:
    - lags: Array of tested lag values.
    - corrs: Pearson correlation at each lag.
    """
    lags = np.arange(0, max_lag + 1, step)
    corrs = []
    
    # Drop NaNs just for the correlation window
    df_clean = df[[target, proxy]].dropna()
    
    for lag in lags:
        if lag == 0:
            shifted = df_clean[proxy]
        else:
            shifted = df_clean[proxy].shift(lag)
            
        # Re-drop NaNs after shift
        valid = pd.concat([df_clean[target], shifted], axis=1).dropna()
        if len(valid) > 100:
            r, _ = pearsonr(valid.iloc[:,0], valid.iloc[:,1])
        else:
            r = 0
            
        corrs.append(r)
        
    return lags, corrs

def characterize_gaps(df, target_col, max_impute_gap=0, save_plot_path=None):
    """
    Characterizes missing data: gap duration, continuity, and recurrence.
    Optionally applies linear interpolation for small gaps.
    Generates a histogram plot of the gap lengths and returns the updated dataframe, 
    statistical summary, and a series of exact gap lengths.
    
    Parameters:
    - df: DataFrame with a DatetimeIndex.
    - target_col: The column to analyze for missing data.
    - max_impute_gap: Maximum number of consecutive missing values to interpolate linearly. Default 0 (no imputation).
    - save_plot_path: Path to save the generated histogram figure. Optional.
    
    Returns:
    - df_imputed: DataFrame with simple imputation applied (if max_impute_gap > 0).
    - gap_stats: A pandas Series containing descriptive statistics of the gaps.
    - gap_lengths: A Series listing the exact length of every continuous gap sequence.
    """
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in DataFrame.")
        
    df_imputed = df.copy()
    
    if max_impute_gap > 0:
        print(f"Applying linear interpolation strictly for complete gaps <= {max_impute_gap} consecutive values...")
        mask = df_imputed[target_col].isnull()
        if mask.any():
            # Calculate length of each gap block
            blocks = mask.astype(int).groupby(mask.astype(int).diff().ne(0).cumsum())
            gap_sizes = blocks.transform('sum')
            
            # Interpolate everything linearly
            interp = df_imputed[target_col].interpolate(method='linear')
            
            # Restore NaNs for blocks that were strictly larger than max_impute_gap
            too_large_mask = mask & (gap_sizes > max_impute_gap)
            interp[too_large_mask] = np.nan
            
            df_imputed[target_col] = interp
        
    # Create a boolean mask where True = missing data
    missing_mask = df_imputed[target_col].isnull()
    
    # If there is no missing data at all
    if not missing_mask.any():
        print(f"No missing data found in '{target_col}'.")
        return df_imputed, pd.Series(dtype=float), pd.Series(dtype=int)
        
    # Group contiguous True values
    # diff().ne(0).cumsum() creates unique IDs for contiguous blocks
    gap_blocks = missing_mask.astype(int).groupby(missing_mask.astype(int).diff().ne(0).cumsum())
    
    # Sum the block lengths, but only keep the blocks that are actually missing (value > 0)
    gap_lengths = gap_blocks.sum()[gap_blocks.sum() > 0]
    
    gap_stats = gap_lengths.describe()
    
    print(f"\n--- Gap Taxonomy for '{target_col}' ---")
    print(f"Total Gaps Detected: {int(gap_stats['count'])}")
    print(f"Average Gap Length:  {gap_stats['mean']:.2f} time steps")
    print(f"Maximum Gap Length:  {int(gap_stats['max'])} time steps")
    
    # Simple heuristic to flag gap type based on SHM missing-data literature
    # Short scattered gaps = MCAR, Long contiguous outages = MNAR/MAR
    if gap_stats['mean'] > 10 and gap_stats['max'] > 100:
        print("Diagnosis: High likelihood of structured, contiguous outages (MNAR/MAR).")
        print("Recommendation: Advanced imputation (e.g., BiLSTM or XGBoost) required.")
    else:
        print("Diagnosis: Gaps appear scattered and random (MCAR).")
        print("Recommendation: Standard interpolation or basic models may suffice.")
        
    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    plt.hist(gap_lengths, bins=50, color='tab:red', alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of Consecutive Gap Sizes (max_impute_gap={max_impute_gap})')
    plt.xlabel('Gap Length (Consecutive Missing Time Steps)')
    plt.ylabel('Frequency (Number of Occurrences)')
    plt.yscale('log') # Log scale because there are many short gaps and few long gaps
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_plot_path:
        plt.savefig(save_plot_path)
        print(f"Saved histogram to {save_plot_path}")
        
    plt.show()
        
    return df_imputed, gap_stats, gap_lengths

def test_cointegration(df, target_col, proxy_col, alpha=0.05):
    """
    Performs the Engle-Granger two-step cointegration test to validate
    the physical long-run equilibrium between the structural response and the proxy.
    
    Parameters:
    - df: DataFrame containing both variables.
    - target_col: The structural sensor column (e.g., inclination).
    - proxy_col: The environmental proxy column (e.g., skin temperature).
    - alpha: Significance level for the test (default 0.05).
    
    Returns:
    - is_cointegrated (bool), p_value (float)
    """
    # Drop rows where either column is missing
    df_clean = df.dropna(subset=[target_col, proxy_col])
    
    if len(df_clean) < 100:
        raise ValueError("Insufficient overlapping data points to run a reliable cointegration test.")
        
    # Perform Engle-Granger test
    score, p_value, _ = coint(df_clean[target_col], df_clean[proxy_col])
    is_cointegrated = p_value < alpha
    
    print(f"\n--- Engle-Granger Cointegration Test ---")
    print(f"Target: {target_col} | Proxy: {proxy_col}")
    print(f"Test Statistic: {score:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    if is_cointegrated:
        print(f"Result: Reject Null Hypothesis. The variables share a stationary long-run equilibrium.")
    else:
        print(f"Result: Fail to reject Null Hypothesis. The proxy is NOT cointegrated with the target.")
        
    return is_cointegrated, p_value

def test_residual_stationarity(residuals, alpha=0.05):
    """
    Tests structural residuals for stationarity using Augmented Dickey-Fuller (ADF).
    This confirms that environmental noise has been adequately extracted.
    
    Parameters:
    - residuals: A pandas Series or array of model residuals.
    - alpha: Significance level (default 0.05).
    
    Returns:
    - is_stationary (bool), p_value (float)
    """
    res_clean = pd.Series(residuals).dropna()
    
    adf_result = adfuller(res_clean)
    p_value = adf_result[1]
    is_stationary = p_value < alpha
    
    print(f"\n--- ADF Stationarity Test ---")
    print(f"Test Statistic: {adf_result[0]:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    if is_stationary:
        print(f"Result: Residuals are stationary (I(0)). Suitable for control chart monitoring.")
    else:
        print(f"Result: Residuals are non-stationary. Decomposition may be incomplete.")
        
    return is_stationary, p_value

def test_residual_whiteness(residuals, lags=10, alpha=0.05):
    """
    Tests residuals for white noise (absence of autocorrelation) using Ljung-Box.
    
    Parameters:
    - residuals: A pandas Series or array of model residuals.
    - lags: Number of lags to test (default 10).
    - alpha: Significance level (default 0.05).
    
    Returns:
    - is_white (bool), p_value (float)
    """
    res_clean = pd.Series(residuals).dropna()
    
    lb_result = acorr_ljungbox(res_clean, lags=[lags], return_df=True)
    p_value = lb_result['lb_pvalue'].iloc[0]
    is_white = p_value > alpha
    
    print(f"\n--- Ljung-Box Whiteness Test (Lags={lags}) ---")
    print(f"Test Statistic: {lb_result['lb_stat'].iloc[0]:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    if is_white:
        print(f"Result: Residuals are white noise (No significant autocorrelation).")
    else:
        print(f"Result: Residuals show significant autocorrelation.")
        
    return is_white, p_value