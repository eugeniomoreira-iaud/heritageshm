#!/usr/bin/env python
# coding: utf-8

# ![Logo_DICA2.jpg](assets/Logo_DICA2.jpg)

# # Data exploration

# This is a code to explore the time series of the processed SHM data. A series of plots will be drawn in order to visualize the time series. Note that it will take into account the provided csv file have a column called "date" (in format YYYY-MM-DD) and another column called "time" (in format HH:MM:SS). This will be taken into account in the custom plot functions.

# ## Import libraries

# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import os
import numpy as np
import seaborn as sns


# ## Custom functions

# ### Read and parse csv

# In[4]:


def read_and_parse_data(file_path):
    data = pd.read_csv(file_path)
    data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
    data.set_index('datetime', inplace=True)
    return data


# ### Compare Year statistics

# For a DataFrame with a datetime index (recorded every 20 minutes from 2018 to 2024) and for variables in var_list, this function groups data by the "instant" (the same month-day and time-of-day) and computes the standard deviation, maximum, minimum, and mean across the different years. No filling is performed if an instant is missing in some years.
# 
# In addition, for each aggregated statistic, a normalized value (between 0 and 1) is computed using min–max normalization:
#     
#     norm_value = (x - min(x)) / (max(x) - min(x))
#     
# If a column has constant values (max == min), the normalized value is set to 0.0.
# 
# The function also creates a full datetime column for plotting by converting the instant (e.g., "06-15 10:20") into a datetime with a placeholder year (2000).
# 
# **Parameters**
# 
#     data : pd.DataFrame
#         A DataFrame with a datetime index.
#     var_list : list of str
#         List of column names (variables) for which statistics are to be computed.
#     
# **Returns**
# 
# 
#     stats : pd.DataFrame
#         DataFrame with index equal to the instant string and columns for each variable's
#         standard deviation, maximum, minimum, and mean plus corresponding normalized columns 
#         (suffixed with '_norm'). Also includes a 'month_day_sort'
#         column containing a full datetime (with year=2000) for sorting/plotting.

# In[7]:


def compare_instant_stats(data, var_list):
    # Work on a copy so as not to modify the original DataFrame.
    df = data.copy()

    # Create a grouping key by extracting month-day and time from the datetime index.
    # This groups together data points that occur at the same calendar instant (ignoring the year).
    df['instant'] = df.index.strftime('%m-%d %H:%M')

    # Group by the 'instant' and compute the desired statistics for each variable.
    agg_dict = {var: ['std', 'max', 'min', 'mean'] for var in var_list}
    stats = df.groupby('instant').agg(agg_dict)

    # Flatten the MultiIndex in the columns (e.g., ('st02_temp', 'std') -> 'st02_temp_std')
    stats.columns = [f'{var}_{stat}' for var, stat in stats.columns]

    # Create a full datetime column from the instant strings using a placeholder year (2000).
    # Any invalid conversions become NaT.
    stats['datetime'] = pd.to_datetime('2000-' + stats.index,
                                               format='2000-%m-%d %H:%M',
                                               errors='coerce')

    # Drop rows where conversion failed and sort by the full datetime.
    stats = stats.dropna(subset=['datetime']).sort_values('datetime')

    # For each aggregated column, compute the normalized value using min–max scaling.
    for col in stats.columns:
        if col != 'datetime':
            col_min = stats[col].min()
            col_max = stats[col].max()
            norm_col = col + "_norm"
            # Handle constant columns to avoid division by zero.
            if col_max == col_min:
                stats[norm_col] = 0.0
            else:
                stats[norm_col] = (stats[col] - col_min) / (col_max - col_min)

    return stats


# ### Compare weekly or daily statistics

# Groups the data by day or week and computes statistical metrics for specified variables.
# 
# **Parameters:**
# 
#     - data (pd.DataFrame): The input DataFrame with a DatetimeIndex.
#     - var_list (list): List of variables to compute statistics for.
#     - group_by (str): The grouping frequency, either 'day' or 'week'.
# 
# **Returns:**
# 
#     - pd.DataFrame: DataFrame containing the computed statistics.

# In[13]:


def compare_grouped_stats(data, var_list, group_by='day'):
    df = data.copy()

    # Determine the resampling frequency
    if group_by == 'day':
        group_freq = 'D'
    elif group_by == 'week':
        group_freq = 'W'
    elif group_by == 'hour':
        group_freq = 'h'
    else:
        raise ValueError("group_by must be either 'day', 'week', or 'hour'")

    # Group by the specified frequency and compute the desired statistics for each variable
    agg_dict = {var: ['std', 'max', 'min', 'mean'] for var in var_list}
    stats = df.resample(group_freq).agg(agg_dict)

    # Flatten the MultiIndex in the columns
    stats.columns = [f'{var}_{stat}' for var, stat in stats.columns]

    # Include the original variables (e.g., their mean values) in the output
    for var in var_list:
        stats[var] = df[var].resample(group_freq).mean()

    # For each aggregated column, compute the normalized value using min–max scaling
    for col in stats.columns:
        if col not in var_list:  # Skip columns that already hold the original variable values
            col_min = stats[col].min()
            col_max = stats[col].max()
            norm_col = col + "_norm"
            # Avoid division by zero if all values are the same
            if col_max == col_min:
                stats[norm_col] = 0.0
            else:
                stats[norm_col] = (stats[col] - col_min) / (col_max - col_min)

    return stats


# ### Filter by month and day range

# In[16]:


def filter_by_month_day_range(df, start_str, end_str):

    # Determine where the datetime information is stored.
    if 'datetime' in df.columns:
        dt_series = pd.to_datetime(df['datetime'])
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_series = df.index
    else:
        raise KeyError("No 'datetime' column found and the index is not a DatetimeIndex.")

    # Work on a copy to avoid modifying the original DataFrame.
    df = df.copy()

    # Create a temporary 'month_day' column using a fixed placeholder year (2000).
    # Using errors='coerce' ensures any invalid strings become NaT.
    df['month_day'] = pd.to_datetime('2000-' + dt_series.strftime('%m-%d'),
                                     format='2000-%m-%d',
                                     errors='coerce')

    # Convert start_str and end_str to datetime objects (with year=2000).
    start_date = pd.to_datetime('2000-' + start_str, format='2000-%m-%d', errors='coerce')
    end_date   = pd.to_datetime('2000-' + end_str, format='2000-%m-%d', errors='coerce')

    # Filter the DataFrame: keep only rows where 'month_day' is between start_date and end_date.
    filtered = df[(df['month_day'] >= start_date) & (df['month_day'] <= end_date)]

    # Optionally, drop the temporary column.
    filtered = filtered.drop(columns=['month_day'])

    return filtered


# ### Compare dataframes

# In[19]:


def compare_dataframes_by_date(df1, df2, col1, col2, baseline_date):
    """
    Compare two DataFrames by aligning them on their datetime index and calculating the difference 
    relative to a baseline date.

    Parameters:
    -----------
    df1 : pd.DataFrame
        First DataFrame with a datetime index and a column named col1.
    df2 : pd.DataFrame
        Second DataFrame with a datetime index and a column named col2.
    col1 : str
        The column name from df1 to compare.
    col2 : str
        The column name from df2 to compare.
    baseline_date : str or pd.Timestamp
        The date to use as the baseline. The value corresponding to this date in each DataFrame 
        will be subtracted from all entries so that at the baseline date the value becomes 0.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the datetime index and two columns (col1 and col2) showing the difference
        from the baseline value.
    """
    # Convert baseline_date to a Timestamp if necessary.
    baseline_date = pd.to_datetime(baseline_date)

    # Extract the desired columns and ensure a copy is made
    df1_col = df1[[col1]].copy()
    df2_col = df2[[col2]].copy()

    # Merge the two DataFrames on the datetime index using an inner join so that only matching dates remain
    merged = pd.concat([df1_col, df2_col], axis=1, join='inner')

    # Check that the baseline_date exists in the merged DataFrame index
    if baseline_date not in merged.index:
        raise ValueError("The baseline_date is not present in the aligned data index.")

    # Get the baseline values for each column.
    baseline_val1 = merged.loc[baseline_date, col1]
    baseline_val2 = merged.loc[baseline_date, col2]

    # Subtract the baseline values so that the baseline date becomes 0.
    merged[col1] = merged[col1] - baseline_val1
    merged[col2] = merged[col2] - baseline_val2

    return merged


# ### RMSE

# In[22]:


def grouped_rmse_single(data, col1, col2, group_by='day'):
    """
    Groups the DataFrame by day or week and computes the RMSE between two specified columns.

    Parameters:
    -----------
    data : pd.DataFrame
        A DataFrame with a datetime index.
    col1 : str
        The name of the first column.
    col2 : str
        The name of the second column.
    group_by : str, optional
        Grouping frequency. Must be either 'day' or 'week'. Default is 'day'.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the aggregated datetime index and a single column "RMSE",
        which holds the RMSE computed for each group.

    Example:
    --------
    rmse_df = grouped_rmse_single(df, 'value1', 'value2', group_by='week')
    print(rmse_df.head())
    """
    # Determine the frequency alias for grouping.
    if group_by == 'day':
        freq = 'D'
    elif group_by == 'week':
        freq = 'W'
    else:
        raise ValueError("group_by must be either 'day' or 'week'")

    # Group the DataFrame by the given frequency using the datetime index.
    grouped = data.groupby(pd.Grouper(freq=freq))

    # Compute the RMSE for each group.
    rmse_series = grouped.apply(
        lambda group: np.sqrt(np.mean((group[col1] - group[col2]) ** 2))
        if len(group) > 0 else np.nan
    )

    # Create a new DataFrame with the aggregated datetime index and RMSE column.
    rmse_df = pd.DataFrame(rmse_series, columns=["RMSE"])
    return rmse_df


# ### Custom plot variable by date

# In[136]:


def plot_time_series(data, y_var, plot_type='scatter', num_xticks=10, cmap='viridis', width=5, alpha=0.7,
                     save_plot=False, save_path='.', filename='chart', years=None, hlines=True, vlines=True,
                     title=None, xlabel='date', ylabel=None):
    # Default years if none provided
    if years is None:
        years = data.index.year.unique()

    # Default ylabel if none provided
    if ylabel is None:
        ylabel = y_var

    # Create a colormap for the distinct years
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(years)))

    # Compute x coordinates using date and time information.
    # Here, we create a full datetime using a placeholder year (2000) while preserving time-of-day.
    data['month_day_sort'] = pd.to_datetime('2000-' + data.index.strftime('%m-%d %H:%M'),
                                              format='2000-%m-%d %H:%M',
                                              errors='coerce')

    # Plotting
    plt.figure()
    for i, year in enumerate(years):
        year_data = data[data.index.year == year]
        if plot_type == 'scatter':
            plt.scatter(year_data['month_day_sort'], year_data[y_var], s=width**2, color=colors[i],edgecolors='None', 
                        alpha=alpha, label=str(year))  # Use 'color' instead of 'c'
        elif plot_type == 'line':
            plt.plot(year_data['month_day_sort'], year_data[y_var], linewidth=width, color=colors[i], 
                     alpha=alpha, label=str(year))  # Use 'color' instead of 'c'
        else:
            raise ValueError("plot_type must be either 'scatter' or 'line'")

    # Customize the plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else '')
    plt.legend(title='Year')

    # Get current axis and set the tick formatter to show only month-day.
    ax = plt.gca()
    date_formatter = mdates.DateFormatter('%b')
    ax.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=0)

    # Limit the number of x-ticks:
    # Get the current x-axis limits as float numbers (matplotlib date numbers)
    xtick_start, xtick_end = ax.get_xlim()
    # Convert these limits to datetime objects
    start_dt = mdates.num2date(xtick_start)
    end_dt = mdates.num2date(xtick_end)
    # Create a new date range with num_xticks equally spaced ticks.
    new_ticks = pd.date_range(start=start_dt, end=end_dt, periods=num_xticks)
    ax.set_xticks(new_ticks)

    # Customize grid lines
    if hlines:
        plt.grid(axis='y', linestyle=':', color='lightgray')
    if vlines:
        plt.grid(axis='x', linestyle=':', color='lightgray')

    # Customize spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Save plot if required
    if save_plot:
        base_filename = f'{filename}_comparison'
        save_path_png = os.path.join(save_path, f'{base_filename}.png')
        save_path_svg = os.path.join(save_path, f'{base_filename}.svg')
        plt.savefig(save_path_png, dpi=300)
        plt.savefig(save_path_svg)
        print(f'Plot saved as {save_path_png} and {save_path_svg}')

    plt.tight_layout()
    plt.show()


# ### Custom plot variable stats by date

# In[139]:


def plot_stats_values(data, y_vars, plot_type='scatter', num_xticks=10, colors=None, width=None, alpha=None, markers=None,
                      save_plot=False, save_path='.', filename='chart', hlines=True, vlines=True,
                      title=None, xlabel='date', ylabel=None):

    # Default colors if none provided.
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(y_vars)))

    if ylabel is None:
        ylabel = 'Change Value'

    # Determine which column to use for x-axis.
    if 'datetime' in data.columns:
        x_col = 'datetime'
    elif 'month_day_sort' in data.columns:
        x_col = 'month_day_sort'
    else:
        data = data.copy()
        try:
            dt_index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("The DataFrame index cannot be converted to datetime.") from e
        data['month_day_sort'] = pd.to_datetime('2000-' + dt_index.strftime('%m-%d %H:%M'),
                                                  format='2000-%m-%d %H:%M',
                                                  errors='coerce')
        x_col = 'month_day_sort'

    # Process the 'width' parameter: if None, default to 5 for each variable.
    if width is None:
        width = [5] * len(y_vars)
    elif np.isscalar(width):
        width = [width] * len(y_vars)

    # Process the 'alpha' parameter: if None, default to 0.7 for each variable.
    if alpha is None:
        alpha = [0.7] * len(y_vars)
    elif np.isscalar(alpha):
        alpha = [alpha] * len(y_vars)

    # Process markers only for scatter plots.
    if markers is None:
        markers = ['o'] * len(y_vars)
    elif np.isscalar(markers):
        markers = [markers] * len(y_vars)

    # Plotting
    plt.figure()
    for i, y_var in enumerate(y_vars):
        if plot_type == 'scatter':
            # For scatter plots, we square the width value to mimic the area-based sizing from function #1.
            plt.scatter(data[x_col], data[y_var],
                        s=width[i]**2,
                        c=[colors[i]],
                        edgecolors=None,
                        alpha=alpha[i],
                        marker=markers[i],
                        label=y_var)
        elif plot_type == 'line':
            # For line plots, 'width' controls the line width. Markers are ignored.
            plt.plot(data[x_col], data[y_var],
                     linewidth=width[i],
                     color=colors[i],
                     alpha=alpha[i],
                     label=y_var)
        else:
            raise ValueError("plot_type must be either 'scatter' or 'line'")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title if title else '')
    plt.legend(title='Variable')

    # Set x-ticks: calculate new tick positions using the current x-axis limits.
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b '))
    plt.xticks(rotation=0)

    xtick_start, xtick_end = ax.get_xlim()
    start_dt = mdates.num2date(xtick_start)
    end_dt = mdates.num2date(xtick_end)
    new_ticks = pd.date_range(start=start_dt, end=end_dt, periods=num_xticks)
    ax.set_xticks(new_ticks)

    # Customize grid lines and spines.
    if hlines:
        plt.grid(axis='y', linestyle=':', color='lightgray')
    if vlines:
        plt.grid(axis='x', linestyle=':', color='lightgray')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Save plot if required.
    if save_plot:
        base_filename = filename + '_stats'
        save_path_png = os.path.join(save_path, f'{base_filename}.png')
        save_path_svg = os.path.join(save_path, f'{base_filename}.svg')
        plt.savefig(save_path_png, dpi=300)
        plt.savefig(save_path_svg)
        print(f'Plot saved as {save_path_png} and {save_path_svg}')

    plt.tight_layout()
    plt.show()


# ## Processing experimental data

# ### Load data

# In[76]:


# Define the folder paths
file_path = r"data\processed_data\GUBBIO_ST02_PROCESSED.csv"

chart_path = 'img_aid'
save_plot = True

data = read_and_parse_data(file_path)

data.head()


# ### Plotting temperature trends

# #### Whole year

# In[141]:


cs = {"axes.spines.right": False, 
      "axes.spines.top": False,
      "axes.linewidth": 3,
      "axes.grid": True,
      "grid.color": 'lightgray',
      "grid.linestyle": "--",
      "grid.linewidth": 0.8,
      "axes.axisbelow": True,
      'axes.titlesize': 20,      # Title font size
      'axes.labelsize': 15,      # X and Y label font size
      'xtick.labelsize': 15,     # X tick label size
      'ytick.labelsize': 15,     # Y tick label size
      'legend.fontsize': 12,     # Legend label font size
      'legend.title_fontsize': 15, # Legend title font size
      'legend.markerscale': 2,
      'figure.figsize': (15, 6)

}

sns.set_theme(style='ticks', context='notebook', font_scale=1, rc=cs)


# In[104]:


plot_time_series(data, y_var='st02_temp',  num_xticks=7, cmap='jet', plot_type='scatter', width= 4, alpha=0.5,
                 save_plot=save_plot, save_path= chart_path, filename = 'temp_year', hlines=True, vlines=True,
                 title='Temperature Time Series', xlabel='Date', ylabel='Temperature (°C)')


# In[106]:


grouped_temp=compare_grouped_stats(data, var_list = ['st02_temp'], group_by='day')
grouped_temp.head()


# In[108]:


plot_time_series(grouped_temp, y_var='st02_temp_mean', num_xticks=13, cmap='jet', plot_type='line', width=4, alpha=0.8,
                 save_plot=save_plot, save_path= chart_path, filename = 'temp_year_avgday', hlines=True, vlines=True,
                 title='Temperature Time Series \n [average by day]', xlabel='Date', ylabel='Temperature (°C)')


# In[110]:


stats_temp_df = compare_instant_stats(data, ['st02_temp'])
stats_temp_df.head()


# In[111]:


plot_stats_values(stats_temp_df, y_vars=[ 'st02_temp_max', 'st02_temp_min','st02_temp_mean'], num_xticks=14,
                   colors=['blue', 'red','darkorange'], plot_type = 'line', width=[0.5,0.5,2], alpha=[0.3, 0.3, 0.5], markers =['x', 'x','o'] , 
                   save_plot=save_plot, save_path= chart_path, filename = 'temp_year', hlines=True, vlines=True,
                   title='Statistics for temperature values', xlabel='Date', ylabel='Temperature (°C)')


# #### Month

# In[114]:


month_data = filter_by_month_day_range(data, "08-01", "08-31")
month_data.sort_values(by=['datetime'], inplace=True)
month_data.head()


# In[116]:


plot_time_series(month_data, y_var='st02_temp', num_xticks=30, cmap='jet', plot_type = 'line', width=3, alpha=0.6,
                 save_plot=save_plot, save_path=chart_path, filename = 'temp_month', hlines=True, vlines=True,
                 title='Temperature Time Series', xlabel='Date', ylabel='Temperature (°C)')


# In[118]:


stats_month_temp_df = compare_instant_stats(month_data, ['st02_temp'])
stats_month_temp_df.head()


# In[119]:


plot_stats_values(stats_month_temp_df, y_vars=[ 'st02_temp_max', 'st02_temp_min','st02_temp_mean'], plot_type = 'line', num_xticks=14,
                   colors=['blue', 'red','darkorange'], width=[1,1,3], alpha=[0.3, 0.3, 1], markers =['x', 'x','o'] , 
                   save_plot=save_plot, save_path= chart_path, filename = 'temp_month', hlines=True, vlines=True,
                   title='Statistics for temperature values', xlabel='Date', ylabel='Temperature (°C)')


# #### Some days

# In[124]:


daily_data = filter_by_month_day_range(data, "08-01", "08-10")
daily_data.sort_values(by=['datetime'], inplace=True)
daily_data.head()


# In[125]:


plot_time_series(daily_data, y_var='st02_temp', num_xticks=12, cmap='jet', plot_type='line', width=5, alpha=0.6,
                 save_plot=save_plot, save_path=chart_path, filename = 'temp_days', hlines=True, vlines=True,
                 title='Temperature Time Series', xlabel='Date', ylabel='Temperature (°C)')


# In[127]:


stats_days_temp_df = compare_instant_stats(daily_data, ['st02_temp'])
stats_days_temp_df.head()


# In[128]:


plot_stats_values(stats_days_temp_df, y_vars=[ 'st02_temp_max', 'st02_temp_min','st02_temp_mean'], plot_type = 'line', num_xticks=14,
                   colors=['blue', 'red','darkorange'], width=[1,1,5], alpha=[0.3, 0.3, 1], markers =['x', 'x','o'] , 
                   save_plot=save_plot, save_path= chart_path, filename = 'temp_days', hlines=True, vlines=True,
                   title='Statistics for temperature values', xlabel='Date', ylabel='Temperature (°C)')


# ### Plotting tilt trends

# #### Whole year

# In[143]:


plot_time_series(data, y_var='st02_absinc_clean',  num_xticks=7, cmap='jet', plot_type='scatter', width= 4, alpha=0.5,
                 save_plot=save_plot, save_path= chart_path, filename = 'tilt_year', hlines=True, vlines=True,
                 title='Inclination Time Series', xlabel='Date', ylabel='Inclination (milidegrees)')


# In[48]:


grouped_tilt=compare_grouped_stats(data, var_list = ['st02_absinc_clean'], group_by='day')
grouped_tilt.head()


# In[49]:


plot_time_series(grouped_tilt, y_var='st02_absinc_clean_mean', num_xticks=13, cmap='jet', plot_type='line', width=5, alpha=0.8,
                 save_plot=save_plot, save_path= chart_path, filename = 'tilt_year_avgday', hlines=True, vlines=True,
                 title='Inclination Time Series \n [average by day]', xlabel='Date', ylabel='Inclination (milidegrees)')


# In[50]:


stats_tilt_df = compare_instant_stats(data, ['st02_absinc_clean'])
stats_tilt_df.head()


# In[51]:


plot_stats_values(stats_tilt_df, y_vars=[ 'st02_absinc_clean_max', 'st02_absinc_clean_min','st02_absinc_clean_mean'], plot_type='line', num_xticks=14,
                   colors=['blue', 'red','darkorange'], width=[0.5,0.5,2], alpha=[0.3, 0.3, 0.5], markers =['x', 'x','o'] , 
                   save_plot=save_plot, save_path= chart_path, filename = 'tilt_year', hlines=True, vlines=True,
                   title='Statistics for inclination values', xlabel='Date', ylabel='Inclination (milidegrees)')


# #### Month

# In[53]:


month_data = filter_by_month_day_range(data, "08-01", "08-31")
month_data.sort_values(by=['datetime'], inplace=True)
month_data.head()


# In[54]:


plot_time_series(month_data, y_var='st02_absinc_clean', num_xticks=30, cmap='jet', plot_type = 'line', width=3, alpha=0.8,
                 save_plot=save_plot, save_path=chart_path, filename = 'tilt_month', hlines=True, vlines=True,
                 title='Inclination Time Series', xlabel='Date', ylabel='Inclination (milidegrees)')


# In[55]:


stats_month_tilt_df = compare_instant_stats(month_data, ['st02_absinc_clean'])
stats_month_tilt_df.head()


# In[56]:


plot_stats_values(stats_month_tilt_df, y_vars=[ 'st02_absinc_clean_max', 'st02_absinc_clean_min','st02_absinc_clean_mean'],plot_type = 'line',  num_xticks=14,
                   colors=['blue', 'red','darkorange'], width=[1,1,3], alpha=[0.3, 0.3, 1], markers =['x', 'x','o'] , 
                   save_plot=save_plot, save_path= chart_path, filename = 'tilt_month', hlines=True, vlines=True,
                   title='Statistics for inclination values', xlabel='Date', ylabel='Inclination (milidegrees)')


# #### Some days

# In[145]:


daily_data = filter_by_month_day_range(data, "08-01", "08-10")
daily_data.sort_values(by=['datetime'], inplace=True)
daily_data.head()


# In[149]:


plot_time_series(daily_data, y_var='st02_absinc_clean', num_xticks=12, plot_type='line', cmap='jet', width=3, alpha=0.6,
                 save_plot=save_plot, save_path=chart_path, filename = 'tilt_days', hlines=True, vlines=True,
                 title='Inclination Time Series', xlabel='Date', ylabel='Inclination (milidegrees)')


# In[60]:


stats_days_tilt_df = compare_instant_stats(daily_data, ['st02_absinc_clean'])
stats_days_temp_df.head()


# In[61]:


plot_stats_values(stats_days_tilt_df, y_vars=[ 'st02_absinc_clean_max', 'st02_absinc_clean_min','st02_absinc_clean_mean'], plot_type='line', num_xticks=17,
                   colors=['blue', 'red','darkorange'], width=[1,1,5], alpha=[0.3, 0.3, 1], markers =['x', 'x','o'] , 
                   save_plot=save_plot, save_path= chart_path, filename = 'tilt_days', hlines=True, vlines=True,
                   title='Statistics for inclination values', xlabel='Date', ylabel='Inclination (milidegrees)')


# ## Comparing simulation data

# ### Loading data

# In[64]:


# Load SRA data.
sra = pd.read_csv('data/processed_data/sra_jul-aug-sep.csv')

# Ensure that both 'date' and 'time' are strings (in case they are not).
sra['date'] = sra['date'].astype(str)
sra['time'] = sra['time'].astype(str)

# Create the 'datetime' column by concatenating 'date' and 'time'.
sra['datetime'] = pd.to_datetime(sra['date'] + ' ' + sra['time'])
sra.set_index('datetime', inplace=True)

# Optionally, view the first few rows to verify.
sra.head()


# In[65]:


cmap = plt.get_cmap('jet')
colors = []
num_colors = 5

for i in range(num_colors):
    # Handle edge case for single color request
    if num_colors == 1:
        position = 0.5  # Middle of colormap
    else:
        position = i / (num_colors - 1)  # Evenly spaced positions [0, 1]

    colors.append(cmap(position))

print(colors)


# In[66]:


sra_aug = filter_by_month_day_range(sra, "08-01", "08-15")
plot_stats_values(sra_aug, y_vars=[ 'tb_rad','tf_rad','wb_rad','wt_tad', 'wf_rad'], plot_type='line', num_xticks=17,
                   colors=[colors[4],colors[3],colors[2],colors[1],colors[0]], width=[3,3,3,3,3], alpha=[0.7, 0.7, 0.7, 0.7, 0.7], markers =['x','x','o','o','o'] , 
                   save_plot=save_plot, save_path= chart_path, filename = 'sra_days', hlines=True, vlines=True,
                   title='Solar radiation analysis', xlabel='Date', ylabel='Radiation (Kw/m²)')


# In[67]:


# Load FEM data.
fem_06 = pd.read_csv('data/processed_data/fem_sim_rev09.csv', sep=';')

# Ensure that both 'date' and 'time' are strings (in case they are not).
fem_06['date'] = fem_06['date'].astype(str)
fem_06['time'] = fem_06['time'].astype(str)

# Create the 'datetime' column by concatenating 'date' and 'time'.
fem_06['datetime'] = pd.to_datetime(fem_06['date'] + ' ' + fem_06['time'])
fem_06.set_index('datetime', inplace=True)

# Optionally, view the first few rows to verify.
fem_06.head()


# In[68]:


data.head()


# In[312]:


data_hour= compare_grouped_stats(data, var_list = ['st02_absinc_clean'], group_by='hour')
comparison_df = compare_dataframes_by_date(data, fem_06, 'st02_absinc_clean', 'tilt', '2021-09-08')
comparison_df=filter_by_month_day_range(comparison_df, "09-01", "09-15")
comparison_df.sort_values(by=['datetime'], inplace=True)
comparison_df.head()


# In[314]:


comparison_df['diff'] = comparison_df['tilt'].sub(comparison_df['st02_absinc_clean'], axis = 0)
comparison_df.head()


# In[316]:


plot_stats_values(comparison_df, y_vars=[ 'st02_absinc_clean', 'tilt'], plot_type='line', num_xticks=17,
                   colors=['blue', 'red'], width=[4,4], alpha=[0.8, 0.8], markers =['x','o'] , 
                   save_plot=True, save_path= chart_path, filename = '-shm_fem_sep', hlines=True, vlines=True,
                   title='Comparison \n Experimental x Simulated', xlabel='Date', ylabel='Inclination (milidegrees)')


# In[318]:


plot_stats_values(comparison_df, y_vars=[ 'diff'], plot_type='scatter', num_xticks=17,
                   colors=['darkorange'], width=[6], alpha=[0.8], markers =['o'] , 
                   save_plot=True, save_path= chart_path, filename = '-error_shm_fem_sep', hlines=True, vlines=True,
                   title='Error \n Experimental x Simulated', xlabel='Date', ylabel='milidegrees')


# In[176]:


rsme = grouped_rmse_single(data = comparison_df, col1 = 'st02_absinc_clean', col2 = 'tilt', group_by='day')
rsme.head()


# In[72]:


plot_stats_values(rsme, y_vars=[ 'RMSE'], plot_type='line', num_xticks=17,
                   colors=['darkorange'], width=[3], alpha=[1], markers =['o'] , 
                   save_plot=save_plot, save_path= chart_path, filename = 'shm_fem_rmse', hlines=True, vlines=True,
                   title='Root Mean Squared Error per week \n Experimental x Simulated', xlabel='Date', ylabel='Inclination (milidegrees)')


# In[ ]:




