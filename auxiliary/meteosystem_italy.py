# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Weather Data Collection: Gubint, Italy
#
# This notebook automates the process of downloading hourly weather data from the Meteosystem website for the Gubbio station. It iterates through a specified date range, fetches monthly CSV files, and compiles them into a single dataset for further analysis.

# %%
import pandas as pd
import requests
import calendar
import time
import io
import matplotlib.pyplot as plt

# Configuration
base_url = "https://www.meteosystem.com/dati/gubbio/csv.php"
start_year = 2020
end_year = 2022  # Reduced range for faster execution during testing
output_file = "Gubbio_Hourly_Weather_Dataset.csv"

# %% [markdown]
# ## 1. Data Extraction
#
# We iterate through each month of the selected years. For each month, we construct a URL with the appropriate start and end dates and download the CSV content.

# %%
all_data = []

for year in range(start_year, end_year + 1):
    yy = str(year)[-2:] 
    for month in range(1, 13):
        _, last_day = calendar.monthrange(year, month)
        mm = f"{month:02d}"
        gg_start = "01"
        gg_end = f"{last_day:02d}"
        
        params = {
            'gg2': gg_start, 'mm2': mm, 'aa2': yy, # Start date
            'gg': gg_end,    'mm': mm,  'aa': yy   # End date
        }
        
        print(f"Downloading data for {mm}/{year}...")
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status() 
            
            if not response.text.strip():
                print(f" -> No data available for this month. Skipping.")
                continue
                
            # Note: Using semicolon as separator as per original code
            df = pd.read_csv(io.StringIO(response.text), sep=";")
            all_data.append(df)
            
            # Polite delay to avoid overwhelming the server
            time.sleep(1)
            
        except Exception as e:
            print(f" -> Failed to download {mm}/{year}: {e}")

print("\nData collection complete.")

# %% [markdown]
# ## 2. Data Aggregation and Storage
#
# Once all monthly chunks are collected, we merge them into a single DataFrame and save it to a CSV file for permanent storage.

# %%
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Basic cleaning: Ensure date column is parsed if it exists
    # Note: We don't know the exact column names, so we'll just inspect first.
    
    final_df.to_csv(output_file, index=False)
    print(f"Success! Dataset saved as '{output_file}'.")
    print(f"Total rows: {len(final_df)}")
    display(final_df.head())
else:
    print("No data was retrieved.")

# %% [markdown]
# ## 3. Visualization
#
# Below, we attempt to visualize the trends in the dataset. We will look for numeric columns that might represent temperature or other weather metrics.

# %%
if 'final_df' in locals():
    # Find numeric columns
    numeric_cols = final_df.select_dtypes(include=['number']).columns.tolist()
    print(f"Numeric columns found: {numeric_cols}")

    if numeric_cols:
        # We need a time/date column for meaningful plotting.
        # Let's try to find a column that looks like a date or index.
        # For this demo, we will plot the first numeric column found.
        target_col = numeric_cols[0]
        print(f"Plotting {target_col} over the dataset index.")
        
        plt.figure(figsize=(15, 6))
        plt.plot(final_df.index, final_df[target_col], label=target_col, color='tab:blue')
        plt.title(f'Trend of {target_col} in Gubbio')
        plt.xlabel('Record Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("No numeric columns found to plot.")
else:
    print("No data available to visualize. Run previous cells first.")
