"""
Module: dataloader.py
Handles reading, parsing, inspecting, and combining raw sensor data and environmental proxies.
"""
import os
import glob
import csv
import pandas as pd
from tqdm import tqdm

def inspect_raw_file(file_path):
    """
    Inspects an unknown raw sensor file to help the user determine the 
    correct delimiter, headers, and column count before loading.
    """
    print(f"--- Inspecting File: {os.path.basename(file_path)} ---")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read the first few lines
            head = [next(f) for _ in range(5)]
    except StopIteration:
        print("File is empty or has fewer than 5 lines.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
        
    # Attempt to sniff the dialect (delimiter, quote char)
    try:
        dialect = csv.Sniffer().sniff(head[0])
        print(f"Detected Delimiter: '{dialect.delimiter}'")
    except csv.Error:
        print("Could not automatically detect delimiter. Please inspect the raw lines below.")
        
    # Count columns based on the first line
    if 'dialect' in locals():
        col_count = len(head[0].split(dialect.delimiter))
        print(f"Detected Columns: {col_count}")
        
    print("\nFirst 5 lines of the file:")
    print("-" * 40)
    for line in head:
        print(line.strip())
    print("-" * 40)
    print("Use this information to set 'sep' and 'column_names' in load_sensor_directory().\n")

def read_sensor_file(file_path, sep='\t', header=None, column_names=None, 
                     date_col='date', time_col='time', decimal_comma=True):
    """
    Reads a single raw sensor file, handles European decimal formats, 
    and resolves duplicated timestamps by prioritizing non-zero values.
    """
    # Read the file
    df = pd.read_csv(file_path, sep=sep, header=header)
    
    # Handle European decimal commas if requested
    if decimal_comma:
        df = df.replace({',': '.'}, regex=True)
        
    # Assign column names if provided
    if column_names:
        if len(column_names) != len(df.columns):
            raise ValueError(f"Provided {len(column_names)} column names, but data has {len(df.columns)} columns.")
        df.columns = column_names

    # Parse Datetime and set as index
    if date_col in df.columns and time_col in df.columns:
        # Assuming DD/MM/YY or YYYY-MM-DD, pd.to_datetime usually handles it.
        # dayfirst=True helps with DD/MM/YY formats common in European datasets.
        df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), dayfirst=True, errors='coerce')
        
        # Drop the old strings
        df = df.drop(columns=[date_col, time_col])
    elif date_col in df.columns:
        df['datetime'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.drop(columns=[date_col])
    else:
        raise KeyError(f"Could not find the specified date/time columns ('{date_col}', '{time_col}') to create a DatetimeIndex.")
    
    # Convert all remaining columns to numeric
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')

    # Resolve Duplicates (Based on original Gubbio logic)
    duplicates = df[df.duplicated(subset=['datetime'], keep=False)]
    if not duplicates.empty:
        unique_rows = []
        grouped = duplicates.groupby('datetime')
        
        for dt, group in grouped:
            rep_row = group.iloc[0].copy()
            for col in group.columns:
                if col == 'datetime':
                    continue
                # If values differ, pick the first non-zero value
                if len(set(group[col].dropna())) > 1:
                    non_zeros = group[col][group[col] != 0].dropna().unique()
                    rep_row[col] = non_zeros[0] if len(non_zeros) > 0 else 0
            unique_rows.append(rep_row)
            
        processed_df = pd.DataFrame(unique_rows)
        non_duplicates = df.drop_duplicates(subset=['datetime'], keep=False)
        df = pd.concat([non_duplicates, processed_df], ignore_index=True)

    # Clean up index
    df = df.dropna(subset=['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    return df

def load_sensor_directory(folder_path, extension='.adc', sep='\t', header=None, 
                          column_names=None, date_col='date', time_col='time', 
                          save_combined=False, output_path=None):
    """
    Reads all specified files in a directory, concatenates them, 
    and returns a single chronological DataFrame.
    """
    search_pattern = os.path.join(folder_path, f'*{extension}')
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        raise FileNotFoundError(f"No files ending in '{extension}' found in {folder_path}.")
        
    all_data = []
    print(f"Found {len(file_list)} files matching '{extension}'. Processing...")
    
    for file_path in tqdm(file_list, desc="Loading sensor data"):
        df = read_sensor_file(file_path, sep=sep, header=header, 
                              column_names=column_names, date_col=date_col, time_col=time_col)
        all_data.append(df)
        
    combined_df = pd.concat(all_data)
    
    # Final safety check for cross-file duplicates
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df.sort_index(inplace=True)
    
    if save_combined and output_path:
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path)
        print(f"Combined data saved to: {output_path}")
        
    return combined_df

def load_proxy_data(file_path, datetime_col='datetime'):
    """
    A simpler loader for pre-formatted CSVs like ERA5-Land proxy data.
    """
    df = pd.read_csv(file_path, parse_dates=[datetime_col], index_col=datetime_col)
    df.sort_index(inplace=True)
    return df

def load_preprocessed_sensor(file_path, datetime_col='datetime'):
    """
    Loads a preprocessed, clean sensor CSV file generated by Notebook 00.
    """
    df = pd.read_csv(file_path, parse_dates=[datetime_col], index_col=datetime_col)
    df.sort_index(inplace=True)
    return df

def save_interim_data(df, file_path):
    """
    Saves processed DataFrames to the interim data folder for use in the next pipeline stage.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path)
    print(f"Interim data successfully saved to {file_path}")

def organize_sensor_data(df_raw, stations_config):
    """
    Organizes raw sensor data into independent dataframes per station.
    
    Parameters:
    - df_raw: DataFrame from load_sensor_directory with date/time as index.
    - stations_config: Dictionary mapping station names to lists of field names 
                       (use None for missing sequential columns).
    """
    import pandas as pd
    stations_dict = {}
    col_idx = 0

    for st, fields in stations_config.items():
        df_st = pd.DataFrame(index=df_raw.index)
        for field in fields:
            # Avoid indexing out of bounds if there are fewer columns than expected
            if col_idx < len(df_raw.columns):
                if field is not None:
                    df_st[field] = df_raw.iloc[:, col_idx]
                col_idx += 1
        stations_dict[st] = df_st
        
    return stations_dict
