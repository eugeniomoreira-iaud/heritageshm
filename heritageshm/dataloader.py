"""
Module: dataloader.py
Handles reading, parsing, inspecting, and combining raw sensor data and
environmental proxies.
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
    dict
        Keys: ``'delimiter'``, ``'decimal'``, ``'n_columns'``, ``'header'``.
        Ready to copy-paste into the Configuration block of Notebook 00.
    """
    import re

    print("\n" + "\u2500" * 60)
    print("  FILE INSPECTION: " + os.path.basename(file_path))
    print("\u2500" * 60)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]
            lines = lines[:max(n_preview, 10)]
    except Exception as e:
        print("  ERROR: Could not read file \u2014 " + str(e))
        return {}

    if not lines:
        print("  WARNING: File appears empty.")
        return {}

    # ── 1. Decimal symbol detection ──────────────────────────────────────────
    euro_pattern  = re.compile(r'\b\d+,\d+\b')
    anglo_pattern = re.compile(r'\b\d+\.\d+\b')
    euro_hits  = sum(bool(euro_pattern.search(l))  for l in lines)
    anglo_hits = sum(bool(anglo_pattern.search(l)) for l in lines)

    if euro_hits > anglo_hits:
        decimal_sym   = ','
        decimal_label = "comma  ','  (European format \u2014 decimal_comma=True)"
    else:
        decimal_sym   = '.'
        decimal_label = "period '.'  (Anglo format \u2014 decimal_comma=False)"

    # ── 2. Delimiter detection ───────────────────────────────────────────────
    candidates = [('\t', 'TAB'), (';', 'semicolon'), ('|', 'pipe')]
    if decimal_sym != ',':
        candidates.append((',', 'comma'))

    best_delim = None
    best_label = 'unknown'
    best_score = -1

    for delim, label in candidates:
        counts = []
        for line in lines[:10]:
            test_line = line.replace(decimal_sym + '', '') if decimal_sym == delim else line
            counts.append(test_line.count(delim))
        if min(counts) > 0:
            score = min(counts) * (sum(counts) / len(counts))
            if score > best_score:
                best_score = score
                best_delim = delim
                best_label = label

    if best_delim is None:
        best_delim = '\t'
        best_label = 'TAB (fallback \u2014 no clear delimiter detected)'

    # ── 3. Column count ──────────────────────────────────────────────────────
    col_counts = []
    for line in lines[:10]:
        parts = line.split(best_delim)
        col_counts.append(len(parts))
    n_cols = max(set(col_counts), key=col_counts.count)

    # ── 4. Header detection ──────────────────────────────────────────────────
    first_field = lines[0].split(best_delim)[0].strip()
    has_header  = not bool(re.match(r'^\d', first_field))
    header_val  = 0 if has_header else None

    # ── 5. Date format hint ──────────────────────────────────────────────────
    date_hint = 'unknown'
    date_patterns = [
        (r'^\d{2}/\d{2}/\d{2}',   'DD/MM/YY  (dayfirst=True)'),
        (r'^\d{2}/\d{2}/\d{4}',   'DD/MM/YYYY (dayfirst=True)'),
        (r'^\d{4}-\d{2}-\d{2}',   'YYYY-MM-DD (dayfirst=False)'),
        (r'^\d{2}-\d{2}-\d{4}',   'DD-MM-YYYY (dayfirst=True)'),
        (r'^\d{2}\.\d{2}\.\d{4}', 'DD.MM.YYYY (dayfirst=True)'),
    ]
    for pattern, label in date_patterns:
        if re.match(pattern, lines[0].split(best_delim)[0].strip()):
            date_hint = label
            break

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print("\n  DETECTED STRUCTURE")
    print("  %-22s: %r" % ('Delimiter',      best_label))
    print("  %-22s: %s" % ('Decimal symbol', decimal_label))
    print("  %-22s: %d" % ('Column count',   n_cols))
    print("  %-22s: %s" % ('Header row',     'Yes (row 0)' if has_header else 'None (header=None)'))
    print("  %-22s: %s" % ('Date format hint', date_hint))

    print("\n  SUGGESTED CONFIGURATION")
    delim_repr = '\\t' if best_delim == '\t' else best_delim
    print("  SEPARATOR           = '%s'" % delim_repr)
    print("  DECIMAL_COMMA       = %s" % (decimal_sym == ','))
    print("  HEADER              = %s" % header_val)
    print("  # Expected columns  : %d" % n_cols)

    print("\n  FIRST %d LINES (raw)" % n_preview)
    print("  " + "\u2500" * 56)
    for line in lines[:n_preview]:
        print("  " + line)
    print("  " + "\u2500" * 56 + "\n")

    return {
        'delimiter': best_delim,
        'decimal':   decimal_sym,
        'n_columns': n_cols,
        'header':    header_val,
    }


def read_sensor_file(
    file_path,
    sep='\t',
    header=None,
    column_names=None,
    date_col=0,
    time_col=1,
    decimal_comma=True,
):
    """
    Reads a single raw sensor file, handles European decimal formats,
    and resolves duplicated timestamps by prioritising non-zero values.

    Parameters
    ----------
    file_path : str
        Path to the raw file.
    sep : str, optional
        Column delimiter passed to ``pd.read_csv``.  Default ``'\\t'``.
    header : int or None, optional
        Row number to use as column names, or ``None`` when the file has no
        header row.  Default ``None``.
    column_names : list of str or None, optional
        Names to assign to every column **after** reading.  Must match the
        total number of columns in the file.  ``None`` to leave them as
        integer indices.  Default ``None``.
    date_col : int or str, optional
        Column index (int) or name (str) that contains the date string.
        Default ``0``.
    time_col : int or str, optional
        Column index (int) or name (str) that contains the time string.
        Default ``1``.
    decimal_comma : bool, optional
        When ``True``, replaces all commas inside cell values with periods
        before numeric conversion \u2014 required for European locale files.
        Default ``True``.

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame with a ``DatetimeIndex`` named ``'datetime'``.
    """
    import warnings

    df = pd.read_csv(file_path, sep=sep, header=header)

    # European decimal commas: replace before any numeric coercion
    if decimal_comma:
        df = df.replace({',': '.'}, regex=True)

    # Assign column names when provided
    if column_names is not None:
        if len(column_names) != len(df.columns):
            raise ValueError(
                "Provided %d column names but data has %d columns."
                % (len(column_names), len(df.columns))
            )
        df.columns = column_names

    # Resolve date_col / time_col: accept both integer positions and names
    def _resolve_col(df, ref):
        """Return the column label for an int position or str name."""
        if isinstance(ref, int):
            return df.columns[ref]
        return ref  # already a name

    date_label = _resolve_col(df, date_col)
    time_label = _resolve_col(df, time_col)

    # Build datetime index
    if date_label in df.columns and time_label in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df['datetime'] = pd.to_datetime(
                df[date_label].astype(str) + ' ' + df[time_label].astype(str),
                dayfirst=True, format='mixed', errors='coerce',
            )
        df = df.drop(columns=[date_label, time_label])
    elif date_label in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df['datetime'] = pd.to_datetime(
                df[date_label], dayfirst=True, format='mixed', errors='coerce',
            )
        df = df.drop(columns=[date_label])
    else:
        raise KeyError(
            "Could not find date column %r (resolved to %r) in the file."
            % (date_col, date_label)
        )

    # Convert remaining columns to numeric
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')

    # Resolve cross-row duplicates: keep first non-zero value per timestamp
    duplicates = df[df.duplicated(subset=['datetime'], keep=False)]
    if not duplicates.empty:
        unique_rows = []
        for dt, group in duplicates.groupby('datetime'):
            rep_row = group.iloc[0].copy()
            for col in group.columns:
                if col == 'datetime':
                    continue
                if len(set(group[col].dropna())) > 1:
                    non_zeros = group[col][group[col] != 0].dropna().unique()
                    rep_row[col] = non_zeros[0] if len(non_zeros) > 0 else 0
            unique_rows.append(rep_row)
        processed_df   = pd.DataFrame(unique_rows)
        non_duplicates = df.drop_duplicates(subset=['datetime'], keep=False)
        df = pd.concat([non_duplicates, processed_df], ignore_index=True)

    df = df.dropna(subset=['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df


def load_sensor_directory(
    folder_path,
    extension='.adc',
    sep='\t',
    header=None,
    column_names=None,
    date_col=0,
    time_col=1,
    decimal_comma=True,
    save_combined=False,
    output_path=None,
):
    """
    Reads every file matching *extension* in *folder_path*, concatenates them
    into a single chronological DataFrame, and optionally saves the result.

    Parameters
    ----------
    folder_path : str
        Directory that contains the raw sensor files.
    extension : str, optional
        File extension filter (e.g. ``'.adc'``, ``'.csv'``).  Default ``'.adc'``.
    sep : str, optional
        Column delimiter.  Default ``'\\t'``.
    header : int or None, optional
        Row number of the header, or ``None`` for header-less files.  Default ``None``.
    column_names : list of str or None, optional
        Column names to assign after reading.  Default ``None``.
    date_col : int or str, optional
        Column index or name that holds the date string.  Default ``0``.
    time_col : int or str, optional
        Column index or name that holds the time string.  Default ``1``.
    decimal_comma : bool, optional
        Replace commas with periods inside cell values before numeric conversion
        (required for European locale files).  Default ``True``.
    save_combined : bool, optional
        When ``True``, saves the combined DataFrame to *output_path*.  Default ``False``.
    output_path : str or None, optional
        Destination path for the combined CSV.  Required when *save_combined* is ``True``.

    Returns
    -------
    pd.DataFrame
        Combined, sorted DataFrame with a ``DatetimeIndex``.
    """
    search_pattern = os.path.join(folder_path, '*' + extension)
    file_list = glob.glob(search_pattern)

    if not file_list:
        raise FileNotFoundError(
            "No files ending in '%s' found in %s." % (extension, folder_path)
        )

    all_data = []
    print("Found %d files matching '%s'. Processing..." % (len(file_list), extension))

    for file_path in tqdm(file_list, desc="Loading sensor data"):
        df = read_sensor_file(
            file_path,
            sep=sep,
            header=header,
            column_names=column_names,
            date_col=date_col,
            time_col=time_col,
            decimal_comma=decimal_comma,
        )
        all_data.append(df)

    combined_df = pd.concat(all_data)

    # Remove any cross-file duplicate timestamps
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df.sort_index(inplace=True)

    if save_combined and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path)
        print("Combined data saved to: " + output_path)

    return combined_df


def load_proxy_data(file_path, datetime_col='datetime'):
    """
    Loads a pre-formatted proxy CSV (e.g. ERA5-Land) with a datetime index.

    Parameters
    ----------
    file_path : str
        Path to the proxy CSV.
    datetime_col : str, optional
        Name of the datetime column.  Default ``'datetime'``.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(file_path, parse_dates=[datetime_col], index_col=datetime_col)
    df.sort_index(inplace=True)
    return df


def load_preprocessed_sensor(file_path, datetime_col='datetime'):
    """
    Loads a preprocessed sensor CSV produced by Notebook 00.

    Parameters
    ----------
    file_path : str
        Path to the preprocessed CSV.
    datetime_col : str, optional
        Name of the datetime column.  Default ``'datetime'``.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(file_path, parse_dates=[datetime_col], index_col=datetime_col)
    df.sort_index(inplace=True)
    return df


def save_interim_data(df, file_path):
    """
    Saves a processed DataFrame to the interim data folder.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    file_path : str
        Destination path (directory is created if it does not exist).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path)
    print("Interim data successfully saved to " + file_path)


def organize_sensor_data(df_raw, stations_config):
    """
    Splits a wide combined DataFrame into per-station DataFrames.

    Columns are assigned sequentially: the first ``len(fields)`` columns of
    *df_raw* go to the first station, the next block to the second station,
    and so on.  Set a field name to ``None`` to skip that column position.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Combined DataFrame from ``load_sensor_directory``.
    stations_config : dict
        Mapping of station identifiers to lists of field names,
        e.g. ``{'st01': ['charge', 'temp', 'hum', 'absinc'], ...}``.

    Returns
    -------
    dict
        Mapping of station identifiers to individual DataFrames.
    """
    stations_dict = {}
    col_idx = 0

    for st, fields in stations_config.items():
        df_st = pd.DataFrame(index=df_raw.index)
        for field in fields:
            if col_idx < len(df_raw.columns):
                if field is not None:
                    df_st[field] = df_raw.iloc[:, col_idx]
                col_idx += 1
        stations_dict[st] = df_st

    return stations_dict
