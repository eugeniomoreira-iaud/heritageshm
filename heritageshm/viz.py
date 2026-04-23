"""
Module: viz.py
Handles all data visualizations for the heritage SHM pipeline. 
Relies on Seaborn's native contexts ('notebook', 'paper') for publication-ready formatting.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def apply_theme(context='notebook', style='ticks', palette='colorblind', custom_rc=None):
    """
    Applies the global seaborn theme.
    
    Parameters:
    - context: 'notebook' (default, larger fonts), 'paper' (smaller, dense fonts), 'talk', or 'poster'.
    - style: 'whitegrid', 'darkgrid', 'ticks', 'white', or 'dark'.
    - palette: Default color palette (e.g., 'colorblind', 'viridis', 'jet').
    - custom_rc: Optional dictionary of specific matplotlib rcParams to override defaults.
    """
    # Merge user's custom_rc with some basic sensible defaults (like enabling grids)
    base_rc = {"axes.axisbelow": True}
    if custom_rc:
        base_rc.update(custom_rc)
        
    sns.set_theme(context=context, style=style, palette=palette, rc=base_rc)

def _save_figure(fig, save_path, filename):
    """Internal helper to save figures as high-res PNG and vector SVG."""
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    png_path = os.path.join(save_path, f'{filename}.png')
    svg_path = os.path.join(save_path, f'{filename}.svg')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Plot saved successfully to {save_path} as {filename}")

def plot_annual_overlay(data, y_var, plot_type='scatter', cmap='viridis', 
                        alpha=0.6, width=4, num_xticks=10, 
                        title=None, xlabel='Date', ylabel=None, 
                        save_plot=False, save_path='outputs/figures', filename='annual_overlay',
                        theme_kwargs=None):
    """
    Overlays multiple years of data onto a single 12-month calendar axis.
    """
    # Apply theme only if explicitly provided
    if theme_kwargs is not None:
        apply_theme(**theme_kwargs)
    
    df = data.copy()
    if y_var not in df.columns:
        raise KeyError(f"Column '{y_var}' not found in data.")
        
    # Create a unified placeholder year (2000) to map all data onto a single 12-month axis
    df['month_day_sort'] = pd.to_datetime('2000-' + df.index.strftime('%m-%d %H:%M'),
                                          format='2000-%m-%d %H:%M', errors='coerce')
    
    years = df.index.year.unique()
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(years)))
    
    fig, ax = plt.subplots()
    
    for i, year in enumerate(years):
        year_data = df[df.index.year == year]
        if plot_type == 'scatter':
            ax.scatter(year_data['month_day_sort'], year_data[y_var], 
                       s=width**2, color=colors[i], edgecolors='None', 
                       alpha=alpha, label=str(year))
        elif plot_type == 'line':
            ax.plot(year_data['month_day_sort'], year_data[y_var], 
                    linewidth=width, color=colors[i], alpha=alpha, label=str(year))
            
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else y_var)
    ax.set_title(title if title else f'Annual Overlay: {y_var}')
    ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis to show only month-day names
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    
    # Restrict number of xticks to prevent crowding
    xtick_start, xtick_end = ax.get_xlim()
    new_ticks = pd.date_range(start=mdates.num2date(xtick_start), 
                              end=mdates.num2date(xtick_end), periods=num_xticks)
    ax.set_xticks(new_ticks)
    
    sns.despine()
    plt.tight_layout()
    
    if save_plot:
        _save_figure(fig, save_path, filename)
        
    plt.show()

def plot_time_series_comparison(df, cols, colors=None, plot_type='line',
                                title=None, xlabel='Date', ylabel=None,
                                alpha=0.8, width=2,
                                save_plot=False, save_path='outputs/figures', filename='ts_comparison',
                                theme_kwargs=None):
    """
    Standard time series plot for comparing multiple columns over chronological time.
    """
    if theme_kwargs is not None:
        apply_theme(**theme_kwargs)
    
    if colors is None:
        colors = sns.color_palette(n_colors=len(cols))
        
    fig, ax = plt.subplots()
    
    for i, col in enumerate(cols):
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue
            
        if plot_type == 'line':
            ax.plot(df.index, df[col], color=colors[i], linewidth=width, alpha=alpha, label=col)
        elif plot_type == 'scatter':
            ax.scatter(df.index, df[col], color=colors[i], s=width**2, alpha=alpha, label=col)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else 'Value')
    ax.set_title(title if title else 'Time Series Comparison')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    sns.despine()
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_plot:
        _save_figure(fig, save_path, filename)
        
    plt.show()

def plot_gap_availability(df, target_col, freq='D',
                          save_plot=False, save_path='outputs/figures', filename='gap_availability',
                          theme_kwargs=None):
    """
    Visualizes the daily data availability (missing vs. recorded points).
    """
    if theme_kwargs is not None:
        apply_theme(**theme_kwargs)
    
    counts = df[target_col].resample(freq).count()
    max_expected = df.resample(freq).size().max() 
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.bar(counts.index, counts, width=1, color='teal', alpha=0.7)
    ax.axhline(max_expected, color='red', linestyle='--', linewidth=2, label=f'Max Expected ({freq})')
    
    ax.set_title(f'Data Availability over Time: {target_col}')
    ax.set_ylabel('Valid Records Count')
    ax.set_xlabel('Date')
    ax.legend()
    
    sns.despine()
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_plot:
        _save_figure(fig, save_path, filename)
        
    plt.show()

def plot_target_vs_proxies(df, target, proxies, save_plot=False, save_path='outputs/figures', filename='ts_target_vs_proxies', theme_kwargs=None):
    if theme_kwargs is not None:
        apply_theme(**theme_kwargs)
        
    fig, axes = plt.subplots(len(proxies), 1, figsize=(15, 4 * len(proxies)), sharex=True)
    if len(proxies) == 1:
        axes = [axes]

    for ax, proxy in zip(axes, proxies):
        # Plot proxy first on the bottom axis
        color_proxy = 'tab:blue'
        ax.plot(df.index, df[proxy], label=proxy, color=color_proxy, alpha=0.7, linewidth=1)
        ax.set_ylabel(proxy, color=color_proxy)
        ax.tick_params(axis='y', labelcolor=color_proxy)
        
        # Plot target second on the twin axis so it sits on top
        ax2 = ax.twinx()
        color_target = 'black'
        ax2.plot(df.index, df[target], label=target, color=color_target, alpha=0.7, linewidth=1)
        ax2.set_ylabel(target, color=color_target)
        ax2.tick_params(axis='y', labelcolor=color_target)
        
        ax.set_title(f"{target} vs {proxy}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plot:
        _save_figure(fig, save_path, filename)
    plt.show()

def plot_cross_correlation_lags(lags, corrs_dict, optimal_lags_dict, target, save_plot=False, save_path='outputs/figures', filename='cross_correlation_lags', theme_kwargs=None):
    if theme_kwargs is not None:
        apply_theme(**theme_kwargs)
        
    fig, ax = plt.subplots(figsize=(10, 5))
    for proxy, corrs in corrs_dict.items():
        best_lag = optimal_lags_dict.get(proxy)
        label = f"{proxy} ({best_lag}h lag)" if best_lag is not None else proxy
        line, = ax.plot(lags, corrs, label=label)
        
        if best_lag is not None:
            idx = list(lags).index(best_lag)
            best_corr = corrs[idx]
            ax.scatter([best_lag], [best_corr], color=line.get_color(), zorder=5)
            
    ax.set_title(f"Cross-Correlation w/ {target} across varying Lags")
    ax.set_xlabel("Lag (Hours)")
    ax.set_ylabel("Pearson Correlation (r)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot:
        _save_figure(fig, save_path, filename)
    plt.show()

def plot_correlation_heatmap(df_features, title="Feature Matrix Correlation Heatmap", save_plot=False, save_path='outputs/figures', filename='correlation_heatmap', theme_kwargs=None):
    if theme_kwargs is not None:
        apply_theme(**theme_kwargs)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_features.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    if save_plot:
        _save_figure(fig, save_path, filename)
    plt.show()

def plot_gap_overview(df_full, target, gap_blocks, save_plot=False, save_path='outputs/figures', filename='gap_overview', theme_kwargs=None):
    if theme_kwargs: apply_theme(**theme_kwargs)
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(df_full.index, df_full[target], color="black", linewidth=0.6, label="Observed")
    for _, row in gap_blocks.iterrows():
        ax.axvspan(row["start"], row["end"], color="crimson", alpha=0.25, linewidth=0)
    obs_patch = mpatches.Patch(color="black",  label="Observed")
    gap_patch = mpatches.Patch(color="crimson", alpha=0.4, label="Gap (missing)")
    ax.legend(handles=[obs_patch, gap_patch], loc="upper left")
    ax.set_title("Inclinometer Series — Observed Data and Gap Regions")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} (µrad)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot: _save_figure(fig, save_path, filename)
    plt.show()

def plot_feature_importance(model, save_plot=False, save_path='outputs/figures', filename='feature_importance', theme_kwargs=None):
    if theme_kwargs: apply_theme(**theme_kwargs)
    importance = pd.Series(
        model.get_booster().get_score(importance_type="gain"),
        name="Gain"
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.38)))
    importance.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("XGBoost Feature Importance (Gain)")
    ax.set_xlabel("Gain")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if save_plot: _save_figure(fig, save_path, filename)
    plt.show()

def plot_synthetic_validation(df_full, target, gap_idx_val, y_pred_sc, rmse, valid_mask, gap_duration_h, save_plot=False, save_path='outputs/figures', filename='synthetic_gap_validation', theme_kwargs=None):
    if theme_kwargs: apply_theme(**theme_kwargs)
    context_start = gap_idx_val[0]  - pd.Timedelta(days=7)
    context_end   = gap_idx_val[-1] + pd.Timedelta(days=7)
    context_mask  = (df_full.index >= context_start) & (df_full.index <= context_end)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(gap_idx_val[valid_mask],
                    y_pred_sc - rmse, y_pred_sc + rmse,
                    color="steelblue", alpha=0.25, label=f"±1 RMSE band (±{rmse:.2f})")
    ax.axvspan(gap_idx_val[0], gap_idx_val[-1], color="salmon", alpha=0.12, linewidth=0)
    ax.plot(df_full.index[context_mask], df_full[target][context_mask],
            color="black", linewidth=0.9, label="Observed")
    ax.plot(gap_idx_val[valid_mask], y_pred_sc,
            color="steelblue", linewidth=1.4, label="XGBoost imputed (iterative, Δy)")
    ax.set_title(f"Synthetic Gap Validation — Observed vs. Imputed ({gap_duration_h//24}-day gap)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} (µrad)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot: _save_figure(fig, save_path, filename)
    plt.show()

def plot_residual_distribution(residuals_val, bias, save_plot=False, save_path='outputs/figures', filename='residual_distribution', theme_kwargs=None):
    from scipy.stats import gaussian_kde
    if theme_kwargs: apply_theme(**theme_kwargs)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals_val, bins=40, density=True, color="steelblue", edgecolor="white", alpha=0.7)
    xr = np.linspace(residuals_val.min(), residuals_val.max(), 300)
    ax.plot(xr, gaussian_kde(residuals_val)(xr), color="steelblue", linewidth=2)
    ax.axvline(0,    color="black",   linestyle="--", linewidth=1.2, label="Zero bias")
    ax.axvline(bias, color="crimson", linestyle="--", linewidth=1.2, label=f"Mean bias = {bias:.3f}")
    ax.set_title("Residual Distribution — Synthetic Gap (imputed − observed)")
    ax.set_xlabel("Residual (µrad)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot: _save_figure(fig, save_path, filename)
    plt.show()

def plot_bootstrap_uncertainty(df_full, target, gap_idx_val, boot_mean, boot_std_cal, y_true_val, n_bootstrap, save_plot=False, save_path='outputs/figures', filename='bootstrap_uncertainty', theme_kwargs=None):
    if theme_kwargs: apply_theme(**theme_kwargs)
    context_start = gap_idx_val[0]  - pd.Timedelta(days=7)
    context_end   = gap_idx_val[-1] + pd.Timedelta(days=7)
    context_mask  = (df_full.index >= context_start) & (df_full.index <= context_end)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axvspan(gap_idx_val[0], gap_idx_val[-1], color="salmon", alpha=0.12, linewidth=0)
    ax.fill_between(gap_idx_val, boot_mean - 2*boot_std_cal, boot_mean + 2*boot_std_cal,
                    color="steelblue", alpha=0.20, label="±2σ (calibrated)")
    ax.fill_between(gap_idx_val, boot_mean - boot_std_cal,   boot_mean + boot_std_cal,
                    color="steelblue", alpha=0.35, label="±1σ (calibrated)")
    ax.plot(df_full.index[context_mask], df_full[target][context_mask],
            color="black", linewidth=0.9, label="Observed")
    ax.plot(gap_idx_val, y_true_val, color="black", linewidth=0.8,
            linestyle="--", alpha=0.6, label="Ground truth (masked)")
    ax.plot(gap_idx_val, boot_mean, color="steelblue", linewidth=1.4,
            label="Bootstrap mean prediction")
    ax.set_title(f"Bootstrap Uncertainty Envelope — {n_bootstrap} resamples (conformal calibration)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} (µrad)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot: _save_figure(fig, save_path, filename)
    plt.show()

def plot_full_reconstruction(df_full, target, working_full, imputed_flag, imputed_std, save_plot=False, save_path='outputs/figures', filename='full_reconstruction', theme_kwargs=None):
    if theme_kwargs: apply_theme(**theme_kwargs)
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(df_full.index, df_full[target],
            color="black", linewidth=0.6, label="Observed", zorder=3)
    imp_idx = imputed_flag[imputed_flag].index
    ax.fill_between(
        imp_idx,
        (working_full.loc[imp_idx] - imputed_std.loc[imp_idx]).values,
        (working_full.loc[imp_idx] + imputed_std.loc[imp_idx]).values,
        color="steelblue", alpha=0.3, label="±1σ uncertainty"
    )
    ax.scatter(imp_idx, working_full.loc[imp_idx],
               color="steelblue", s=1.2, zorder=2, label="XGBoost imputed")
    ax.set_title("Full Reconstructed Inclinometer Series — Observed + Imputed")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} (µrad)")
    ax.legend(loc="upper left", markerscale=4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot: _save_figure(fig, save_path, filename)
    plt.show()

def plot_uncertainty_profile(df_full, imputed_std, save_plot=False, save_path='outputs/figures', filename='uncertainty_profile', theme_kwargs=None):
    if theme_kwargs: apply_theme(**theme_kwargs)
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.fill_between(df_full.index, 0, imputed_std.fillna(0),
                    color="steelblue", alpha=0.6)
    ax.set_title("Calibrated Uncertainty Profile — σ per Imputed Hour")
    ax.set_xlabel("Date")
    ax.set_ylabel("σ (µrad)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot: _save_figure(fig, save_path, filename)
    plt.show()