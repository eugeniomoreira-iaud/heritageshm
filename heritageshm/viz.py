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

def apply_theme(context='notebook', style='whitegrid', palette='colorblind', custom_rc=None):
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
    # Apply theme (use empty dict if None provided)
    apply_theme(**(theme_kwargs or {}))
    
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
    apply_theme(**(theme_kwargs or {}))
    
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
    apply_theme(**(theme_kwargs or {}))
    
    counts = df[target_col].resample(freq).count()
    max_expected = df.resample(freq).size().max() 
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.bar(counts.index, counts, width=1, color='teal', alpha=0.7)
    ax.axhline(max_expected, color='red', linestyle='--', linewidth=2, label=f'Max Expected ({freq})')
    
    ax.set_title(f'Data Availability over Time: {target_col}')
    ax.set_ylabel('Valid Records Count')
    ax.set_xlabel('Date')
    ax.legend()
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_plot:
        _save_figure(fig, save_path, filename)
        
    plt.show()