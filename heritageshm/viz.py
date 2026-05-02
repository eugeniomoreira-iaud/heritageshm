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

    Parameters
    ----------
    context : str
        'notebook' (default, larger fonts), 'paper', 'talk', or 'poster'.
    style : str
        'whitegrid', 'darkgrid', 'ticks', 'white', or 'dark'.
    palette : str
        Default color palette (e.g., 'colorblind', 'viridis').
    custom_rc : dict or None
        Optional matplotlib rcParams to override defaults.
    """
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


def _compute_internal_gap_spans(series):
    """
    Returns a list of (start, end) Timestamp tuples for every contiguous run of
    NaN values that lies strictly between the first and last valid observation
    (i.e. internal gaps only — leading/trailing NaNs are excluded).

    Parameters
    ----------
    series : pd.Series
        Numeric time series with a DatetimeIndex.

    Returns
    -------
    spans : list of tuple(pd.Timestamp, pd.Timestamp)
        Each tuple is (gap_start, gap_end) inclusive of the bounding NaN rows.
        Returns an empty list if no internal gaps are found.
    """
    first_valid = series.first_valid_index()
    last_valid  = series.last_valid_index()
    if first_valid is None or last_valid is None:
        return []

    interior = series.loc[first_valid:last_valid]
    is_nan   = interior.isnull()
    if not is_nan.any():
        return []

    spans     = []
    in_gap    = False
    gap_start = None

    for ts, null in zip(interior.index, is_nan):
        if null and not in_gap:
            gap_start = ts
            in_gap    = True
        elif not null and in_gap:
            spans.append((gap_start, interior.index[interior.index.get_loc(ts) - 1]))
            in_gap = False

    if in_gap:
        spans.append((gap_start, interior.index[-1]))

    return spans


def plot_annual_overlay(data, y_var, plot_type='scatter', cmap='viridis',
                        alpha=0.6, width=4, num_xticks=10,
                        title=None, xlabel='Date', ylabel=None,
                        save_plot=False, save_path='outputs/figures', filename='annual_overlay',
                        theme_kwargs=None):
    """
    Overlays multiple years of data onto a single 12-month calendar axis.
    """
    if theme_kwargs is not None:
        apply_theme(**theme_kwargs)
    df = data.copy()
    if y_var not in df.columns:
        raise KeyError(f"Column '{y_var}' not found in data.")
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
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
        color_proxy = 'tab:blue'
        ax.plot(df.index, df[proxy], label=proxy, color=color_proxy, alpha=0.7, linewidth=1)
        ax.set_ylabel(proxy, color=color_proxy)
        ax.tick_params(axis='y', labelcolor=color_proxy)
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
    ax.set_title("Inclinometer Series \u2014 Observed Data and Gap Regions")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} (mdeg)")
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
                    color="steelblue", alpha=0.25, label=f"\u00b11 RMSE band (\u00b1{rmse:.2f})")
    ax.axvspan(gap_idx_val[0], gap_idx_val[-1], color="salmon", alpha=0.12, linewidth=0)
    ax.plot(df_full.index[context_mask], df_full[target][context_mask],
            color="black", linewidth=0.9, label="Observed")
    ax.plot(gap_idx_val[valid_mask], y_pred_sc,
            color="steelblue", linewidth=1.4, label="XGBoost imputed (iterative, \u0394y)")
    ax.set_title(f"Synthetic Gap Validation \u2014 Observed vs. Imputed ({gap_duration_h//24}-day gap)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} (mdeg)")
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
    ax.set_title("Residual Distribution \u2014 Synthetic Gap (imputed \u2212 observed)")
    ax.set_xlabel("Residual (mdeg)")
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
                    color="steelblue", alpha=0.20, label="\u00b12\u03c3 (calibrated)")
    ax.fill_between(gap_idx_val, boot_mean - boot_std_cal,   boot_mean + boot_std_cal,
                    color="steelblue", alpha=0.35, label="\u00b11\u03c3 (calibrated)")
    ax.plot(df_full.index[context_mask], df_full[target][context_mask],
            color="black", linewidth=0.9, label="Observed")
    ax.plot(gap_idx_val, y_true_val, color="black", linewidth=0.8,
            linestyle="--", alpha=0.6, label="Ground truth (masked)")
    ax.plot(gap_idx_val, boot_mean, color="steelblue", linewidth=1.4,
            label="Bootstrap mean prediction")
    ax.set_title(f"Bootstrap Uncertainty Envelope \u2014 {n_bootstrap} resamples (conformal calibration)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} (mdeg)")
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
        color="steelblue", alpha=0.3, label="\u00b11\u03c3 uncertainty"
    )
    ax.scatter(imp_idx, working_full.loc[imp_idx],
               color="steelblue", s=1.2, zorder=2, label="XGBoost imputed")
    ax.set_title("Full Reconstructed Inclinometer Series \u2014 Observed + Imputed")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} (mdeg)")
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
    ax.set_title("Calibrated Uncertainty Profile \u2014 \u03c3 per Imputed Hour")
    ax.set_xlabel("Date")
    ax.set_ylabel("\u03c3 (mdeg)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot: _save_figure(fig, save_path, filename)
    plt.show()

def plot_compensation_comparison(
    file_path,
    signal_col,
    date_start=None,
    date_end=None,
    dropped_path=None,
    dot_size=2,
    dropped_dot_size=None,
    save_plot=False,
    save_path='outputs/figures',
    filename='00_compensation_comparison',
    theme_kwargs=None,
):
    """
    Loads a preprocessed station CSV and produces a two-panel figure:
      - Top panel : scatter of raw signal and compensated+normalized signal.
                    Dropped rows (spikes and power-loss) overlaid as coloured
                    markers if a companion '_dropped.csv' path is supplied.
      - Bottom panel: compensation difference (compensated - raw) as filled area.

    The CSV is the single source of truth for normalization.  This function
    never re-normalizes; date_start / date_end are zoom filters only.

    Parameters
    ----------
    file_path : str
        Path to the interim CSV for one station (output of Notebook 00).
    signal_col : str
        Base name of the structural signal (e.g. 'absinc').
    date_start : str or None
        ISO date 'YYYY-MM-DD' for zoom start.  None = full series.
    date_end : str or None
        ISO date 'YYYY-MM-DD' for zoom end.  None = full series.
    dropped_path : str or None
        Path to the companion '_dropped.csv'.  If supplied, dropped rows are
        overlaid on the top panel: spikes in crimson, power-loss in darkviolet.
    dot_size : float
        Marker size for the main signal scatter plots (default 2).
    dropped_dot_size : float or None
        Marker size for the dropped-value overlay.  When None (default),
        automatically set to dot_size * 4 so dropped points are always
        clearly distinguishable from the main signal cloud.
    save_plot : bool
        If True, saves PNG + SVG via _save_figure().
    save_path : str
        Directory for saved figures.
    filename : str
        Base filename (without extension).
    theme_kwargs : dict or None
        Optional keyword arguments forwarded to apply_theme().
    """
    if theme_kwargs is not None:
        apply_theme(**theme_kwargs)

    if dropped_dot_size is None:
        dropped_dot_size = dot_size * 4

    df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
    df.sort_index(inplace=True)

    raw_col  = f'{signal_col}_raw'
    comp_col = signal_col
    has_raw  = raw_col in df.columns

    df_dropped = None
    if dropped_path and os.path.exists(dropped_path):
        df_dropped = pd.read_csv(dropped_path, parse_dates=['datetime'], index_col='datetime')
        df_dropped.sort_index(inplace=True)

    if has_raw:
        raw_ref   = df[raw_col].dropna().iloc[0]
        raw_full  = df[raw_col] - raw_ref
        comp_full = df[comp_col]
        diff_full = comp_full - raw_full

    def _slice(s, s0, s1):
        if s0: s = s.loc[s0:]
        if s1: s = s.loc[:s1]
        return s

    if has_raw:
        raw_plot  = _slice(raw_full,  date_start, date_end)
        comp_plot = _slice(comp_full, date_start, date_end)
        diff_plot = _slice(diff_full, date_start, date_end)
    else:
        df = _slice(df, date_start, date_end)

    if df_dropped is not None and not df_dropped.empty:
        df_dropped = _slice(df_dropped, date_start, date_end)

    if has_raw:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        ax = axes[0]
        ax.scatter(raw_plot.index,  raw_plot,  s=dot_size, color='steelblue',
                   alpha=0.7, linewidths=0, label='Raw')
        ax.scatter(comp_plot.index, comp_plot, s=dot_size, color='darkorange',
                   alpha=0.7, linewidths=0, label='Compensated')

        if df_dropped is not None and not df_dropped.empty and signal_col in df_dropped.columns:
            raw_ref_val = df[raw_col].dropna().iloc[0] if has_raw else 0
            dropped_vals = df_dropped[signal_col] - raw_ref_val

            if 'drop_reason' in df_dropped.columns:
                for reason, color, label in [
                    ('spike',      'crimson',    'Dropped (spike)'),
                    ('power_loss', 'darkviolet', 'Dropped (power loss)'),
                ]:
                    subset_idx = df_dropped.index[df_dropped['drop_reason'] == reason]
                    if len(subset_idx):
                        ax.scatter(
                            dropped_vals.loc[subset_idx].index,
                            dropped_vals.loc[subset_idx],
                            s=dropped_dot_size,
                            color=color,
                            alpha=0.9,
                            linewidths=0,
                            zorder=5,
                            label=label,
                        )
            else:
                ax.scatter(dropped_vals.index, dropped_vals,
                           s=dropped_dot_size, color='crimson',
                           alpha=0.9, linewidths=0, zorder=5, label='Dropped')

        ax.set_ylabel(f'{signal_col} (mdeg)')
        ax.set_title(f'Signal Comparison \u2014 {os.path.basename(file_path)}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, frameon=True,
                  markerscale=max(1, 6 / max(dot_size, 0.1)))
        sns.despine(ax=ax)

        ax = axes[1]
        ax.fill_between(diff_plot.index, diff_plot, 0, where=(diff_plot >= 0),
                        color='darkorange', alpha=0.55, linewidth=0,
                        label='Positive correction')
        ax.fill_between(diff_plot.index, diff_plot, 0, where=(diff_plot < 0),
                        color='steelblue',  alpha=0.55, linewidth=0,
                        label='Negative correction')
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
        ax.set_ylabel('Difference (mdeg)')
        ax.set_xlabel('Date')
        ax.set_title('Compensation Applied (Compensated - Raw)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, frameon=True)
        sns.despine(ax=ax)

    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.scatter(df.index, df[comp_col], s=dot_size, color='darkorange',
                   alpha=0.7, linewidths=0, label='Compensated')
        ax.set_ylabel(f'{signal_col} (mdeg)')
        ax.set_xlabel('Date')
        ax.set_title(f'Compensated Signal \u2014 {os.path.basename(file_path)} '
                     f'(no raw column "{raw_col}" found)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, frameon=True,
                  markerscale=max(1, 6 / max(dot_size, 0.1)))
        sns.despine(ax=ax)

    plt.tight_layout()
    if save_plot:
        _save_figure(fig, save_path, filename)
    plt.show()


def plot_proxy_overview(
    df,
    station_slug,
    n_cols=4,
    highlight_gaps=True,
    gap_color='crimson',
    gap_alpha=0.25,
    save_plot=False,
    save_path='outputs/figures',
    filename='proxy_overview',
    theme_kwargs=None,
):
    """
    Multi-panel time-series overview of all numeric columns in a proxy DataFrame,
    with optional visual highlighting of internal gaps.

    Intended for use in auxiliary proxy-download notebooks (e.g.
    ``meteosystem_italy``) after data standardisation and outlier filtering,
    to visually confirm that the series are complete and plausible before saving.

    When ``highlight_gaps=True``, each subplot overlays semi-transparent
    crimson bands over every contiguous run of NaN values that lies strictly
    *between* the first and last valid observation of that column (internal gaps
    only). Leading or trailing missing data at the series edges is not shaded.
    The gap-span computation reuses ``_compute_internal_gap_spans``.

    Parameters
    ----------
    df : pd.DataFrame
        Standardised proxy DataFrame with a ``DatetimeIndex`` and numeric
        columns (output of the standardisation step).
    station_slug : str
        Station identifier used in the plot title (e.g. ``'gubbio'``).
    n_cols : int, optional
        Maximum number of columns to plot. Default 4.
    highlight_gaps : bool, optional
        If ``True`` (default), shade internal NaN runs in each subplot using
        ``_compute_internal_gap_spans``. Set to ``False`` to suppress shading.
    gap_color : str, optional
        Matplotlib colour for the gap highlight bands. Default ``'crimson'``.
    gap_alpha : float, optional
        Opacity for the gap highlight bands (0–1). Default ``0.25``.
    save_plot : bool, optional
        If ``True``, saves the figure as PNG and SVG via ``_save_figure()``.
        Default ``False``.
    save_path : str, optional
        Directory for saved figures. Default ``'outputs/figures'``.
    filename : str, optional
        Base filename without extension. Default ``'proxy_overview'``.
    theme_kwargs : dict or None, optional
        Optional keyword arguments forwarded to ``apply_theme()``.

    Returns
    -------
    None
        Displays the figure inline. Saves to disk only if ``save_plot=True``.
    """
    import matplotlib.patches as mpatches

    if theme_kwargs is not None:
        apply_theme(**theme_kwargs)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        print('plot_proxy_overview: no numeric columns found — nothing to plot.')
        return

    cols_to_plot = numeric_cols
    n_plot = len(cols_to_plot)
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    legend_gap_added = False

    for ax, col in zip(axes, cols_to_plot):
        ax.plot(df.index, df[col], linewidth=0.5, color='steelblue')
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

        if highlight_gaps:
            spans = _compute_internal_gap_spans(df[col])
            for gap_start, gap_end in spans:
                ax.axvspan(gap_start, gap_end,
                           color=gap_color, alpha=gap_alpha, linewidth=0)
            if spans and not legend_gap_added:
                gap_patch = mpatches.Patch(
                    color=gap_color, alpha=gap_alpha * 2,
                    label='Internal gap (missing data)'
                )
                ax.legend(handles=[gap_patch], loc='upper right',
                          fontsize=7, frameon=True)
                legend_gap_added = True

    axes[0].set_title(
        f'Meteosystem {station_slug.capitalize()} \u2014 quick overview',
        fontsize=10,
    )
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_plot:
        _save_figure(fig, save_path, filename)
    plt.show()
