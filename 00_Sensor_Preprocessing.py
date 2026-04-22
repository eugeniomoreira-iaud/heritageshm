# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: neuralprophet_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook 00 · Sensor Preprocessing
#
# This notebook handles raw data extraction specifically for on-site sensors:
#
# 1. **File Inspection** — Examine an unknown raw sensor file before loading
# 2. **Data Loading** — Read all raw sensor files from `/data/raw/sensor/`
# 3. **Signal Cleaning** — Remove power-loss rows and physical outliers (optional)
# 4. **Save** — Export the cleaned, standardized dataset to `/data/interim/sensor/`
#

# %%
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath('..'))

from heritageshm.dataloader import inspect_raw_file, load_sensor_directory, organize_sensor_data
from heritageshm.preprocessing import clean_signal, apply_compensation, temp_compensation
from heritageshm.viz import apply_theme

apply_theme(context='notebook')

# %% [markdown]
# ## Step 1 · File Inspection
# Before loading all files, inspect one representative raw file to identify the delimiter, 
# column count, and date format.

# %%
# === USER INPUT ===
# Replace with the path to any one raw file in your data folder
sample_file = r"./data/raw/sensor/GUBBIO_20180726.adc"
# ==================

inspect_raw_file(sample_file)

# %% [markdown]
# ## Step 2 · Load and Merge Raw Sensor Files
# Load all pieces of the sensor data.

# %%
# === USER INPUT ===
RAW_FOLDER    = 'data/raw/sensor'
FILE_EXT      = '.adc'
SEPARATOR     = '\t'
HEADER        = None

# Map each station's sequential fields to the standard pipeline schema.
# Use None if a sensor reading is missing in your raw data.
STATIONS = {
    'st01': ['charge', 'temp', 'hum', 'absinc'],
    'st02': ['charge', 'temp', 'hum', 'absinc'],
    'st03': ['charge', 'temp', 'hum', 'absinc']
}
# ==================

df_raw = load_sensor_directory(
    folder_path=RAW_FOLDER,
    extension=FILE_EXT,
    sep=SEPARATOR,
    header=HEADER,
    column_names=None,
    date_col=0,  # Map 1st column to date
    time_col=1,  # Map 2nd column to time
    save_combined=False
)

print(f"\nLoaded dataset shape: {df_raw.shape}")
print(f"Date range: {df_raw.index.min()} → {df_raw.index.max()}")

# Organize data into independent station datasets using the dataloader module
stations_dict = organize_sensor_data(df_raw, STATIONS)
print(f"Organized {len(stations_dict)} independent stations from raw data.")

# %% [markdown]
# ## Step 3 · Signal Cleaning and Saving
# Clean the data and save to the interim format as a standardized CSV series.

# %%
# Iterate through all the datasets created for the clean process
from IPython.display import display
import os

os.makedirs('data/interim/sensor', exist_ok=True)

for st, df_st in stations_dict.items():
    print(f"--- Processing station {st} ---")
    
    # 1. Clean signal: power loss and outliers
    outlier_thresholds = {'absinc': -500}
    df_clean = clean_signal(df_st, valid_charge_col='charge', outlier_thresholds=outlier_thresholds)
    
    # 2. Apply temperature calibration/compensation
    if len(df_clean) > 0 and 'temp' in df_clean.columns and 'absinc' in df_clean.columns:
        df_clean = apply_compensation(
            df=df_clean, 
            target_col='absinc', 
            new_col_name='absinc_clean', 
            comp_func=temp_compensation, 
            normalize=True, 
            temp_col='temp', 
            comp_coeff=0.005
        )
        # Keep only the compensated absinc (drop raw, rename compensated)
        df_clean = df_clean.drop(columns=['absinc'])
        df_clean = df_clean.rename(columns={'absinc_clean': 'absinc'})
    
    # 3. Save to a specific file
    output_path = f"data/interim/sensor/{st}_preprocessed.csv"
    df_clean.to_csv(output_path)
    print(f"Preprocessed {st} data saved to: {output_path}")

    # 4. Preview the cleaned dataset
    print(f"Shape: {df_clean.shape} | Date range: {df_clean.index.min()} → {df_clean.index.max()}")
    display(df_clean.head())
    print()

