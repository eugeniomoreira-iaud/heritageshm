"""
Module: dataloader.py
Handles reading, parsing, inspecting, and combining raw sensor data and environmental proxies.
"""
import os
import glob
import csv
import pandas as pd
from tqdm import tqdm

def inspect_raw_file(file_path, n_preview=5):
    """
    Inspects a raw sensor file and robustly detects its delimiter,
    decimal symbol, column count, and likely date/time format.

    Handles European decimal commas correctly by excluding ',' from
    delimiter candidates when it is detected as the decimal separator.

    Parameters
    ----------
    file_path : str
        Path to any one raw file in the sensor directory.
    n_preview : int
        Number of lines to print as a preview (default 5).

    Returns
    -------
    dict with keys: 'delimiter', 'decimal', 'n_columns', 'header'
        Ready to copy-paste into the Configuration block of Notebook 00.
    """
    import re

    print(f"{'─'*60}")
    print(f"  FILE INSPECTION: {os.path.basename(file_path)}")
    print(f"{'─'*60}")

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]
            lines = lines[:max(n_preview, 10)]  # read enough for robust detection
    except Exception as e:
        print(f"  ERROR: Could not read file — {e}")
        return {}

    if not lines:
        print("  WARNING: File appears empty.")
        return {}

    # ── 1. Decimal symbol detection ──────────────────────────────────────────
    # Look for patterns like  "3,500"  (European) vs  "3.500"  (Anglo)
    euro_pattern = re.compile(r'\b\d+,\d+\b')
    anglo_pattern = re.compile(r'\b\d+\.\d+\b')
    euro_hits  = sum(bool(euro_pattern.search(l))  for l in lines)
    anglo_hits = sum(bool(anglo_pattern.search(l)) for l in lines)

    if euro_hits > anglo_hits:
        decimal_sym = ','
        decimal_label = "comma  ','  (European format — decimal_comma=True)"
    else:
        decimal_sym = '.'
        decimal_label = "period '.'  (Anglo format — decimal_comma=False)"

    # ── 2. Delimiter detection ───────────────────────────────────────────────
    # Candidates: tab, semicolon, pipe, space.
    # Comma is added only when it is NOT the decimal separator.
    candidates = [('\t', 'TAB'), (';', 'semicolon'), ('|', 'pipe')]
    if decimal_sym != ',':
        candidates.append((',', 'comma'))

    best_delim = None
    best_label = 'unknown'
    best_score = -1

    for delim, label in candidates:
        counts = []
        for line in lines[:10]:
            # Temporarily neutralise decimal symbol to avoid false splits
            test_line = line.replace(decimal_sym + '', '') if decimal_sym == delim else line
            counts.append(test_line.count(delim))
        # Score = minimum count (consistency) × mean count (density)
        if min(counts) > 0:
            score = min(counts) * (sum(counts) / len(counts))
            if score > best_score:
                best_score = score
                best_delim = delim
                best_label = label

    if best_delim is None:
        best_delim = '\t'
        best_label = 'TAB (fallback — no clear delimiter detected)'

    # ── 3. Column count ──────────────────────────────────────────────────────
    col_counts = []
    for line in lines[:10]:
        parts = line.split(best_delim)
        col_counts.append(len(parts))
    n_cols = max(set(col_counts), key=col_counts.count)   # mode

    # ── 4. Header detection ──────────────────────────────────────────────────
    first_field = lines[0].split(best_delim)[0].strip()
    has_header = not bool(re.match(r'^\d', first_field))
    header_val = 0 if has_header else None

    # ── 5. Date format hint ──────────────────────────────────────────────────
    date_hint = 'unknown'
    date_patterns = [
        (r'^\d{2}/\d{2}/\d{2}',       'DD/MM/YY  (dayfirst=True)'),
        (r'^\d{2}/\d{2}/\d{4}',       'DD/MM/YYYY (dayfirst=True)'),
        (r'^\d{4}-\d{2}-\d{2}',       'YYYY-MM-DD (dayfirst=False)'),
        (r'^\d{2}-\d{2}-\d{4}',       'DD-MM-YYYY (dayfirst=True)'),
        (r'^\d{2}\.\d{2}\.\d{4}',     'DD.MM.YYYY (dayfirst=True)'),
    ]
    for pattern, label in date_patterns:
        if re.match(pattern, lines[0].split(best_delim)[0].strip()):
            date_hint = label
            break

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print(f"\n  DETECTED STRUCTURE")
    print(f"  {'Delimiter':<22}: {best_label!r}")
    print(f"  {'Decimal symbol':<22}: {decimal_label}")
    print(f"  {'Column count':<22}: {n_cols}")
    print(f"  {'Header row':<22}: {'Yes (row 0)' if has_header else 'None (header=None)'}")
    print(f"  {'Date format hint':<22}: {date_hint}")

    print(f"\n  SUGGESTED CONFIGURATION")
    delim_repr = '\\t' if best_delim == '\t' else best_delim
    print(f"  SEPARATOR           = '{delim_repr}'")
    print(f"  DECIMAL_COMMA       = {decimal_sym == ','}")
    print(f"  HEADER              = {header_val}")
    print(f"  # Expected columns  : {n_cols}")

    print(f"\n  FIRST {n_preview} LINES (raw)")
    print(f"  {'─'*56}")
    for line in lines[:n_preview]:
        print(f"  {line}")
    print(f"  {'─'*56}\n")

    return {
        'delimiter':  best_delim,
        'decimal':    decimal_sym,
        'n_columns':  n_cols,
        'header':     header_val,
    }

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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Assuming DD/MM/YY or YYYY-MM-DD, pd.to_datetime usually handles it.
            # dayfirst=True helps with DD/MM/YY formats common in European datasets.
            df['datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), dayfirst=True, format='mixed', errors='coerce')
        
        # Drop the old strings
        df = df.drop(columns=[date_col, time_col])
    elif date_col in df.columns:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df['datetime'] = pd.to_datetime(df[date_col], dayfirst=True, format='mixed', errors='coerce')
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
