import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

pixel_folder = Path("")  # Enter your file path/folder path in this place
tauc_file = Path("")  # Enter your file path/folder path in this place
temperature_file = Path("")  # Enter your file path/folder path in this place
urbach_file = Path("")  # Enter your file path/folder path in this place
result_folder = Path("")  # Enter your file path/folder path in this place

# Load the core log files needed for the analysis


def _ensure_time_columns(dataframe: pd.DataFrame, keyword: str = 'timestamp') -> pd.DataFrame:
    """Add 'timestamp' and 'time_hours' columns, accepting either datetimes or hour floats."""

    time_col = next((col for col in dataframe.columns if keyword in col.lower()), None)
    if time_col is None:
        raise KeyError(
            f"No column containing '{keyword}' found in dataframe columns: {list(dataframe.columns)}"
        )

    parsed_dt = pd.to_datetime(dataframe[time_col], errors='coerce')
    if parsed_dt.notna().any():
        dataframe['timestamp'] = parsed_dt
        origin = dataframe['timestamp'].min()
        dataframe['time_hours'] = (
            (dataframe['timestamp'] - origin).dt.total_seconds() / 3600.0
        )
    else:
        numeric_hours = pd.to_numeric(dataframe[time_col], errors='coerce')
        if not numeric_hours.notna().any():
            raise ValueError(
                f"Column '{time_col}' cannot be parsed as datetime or numeric hours."
            )
        baseline = numeric_hours.min(skipna=True)
        dataframe['time_hours'] = numeric_hours - baseline
        dataframe['timestamp'] = pd.Timestamp('1970-01-01') + pd.to_timedelta(
            dataframe['time_hours'], unit='h'
        )

    return dataframe


tauc_df = pd.read_csv(tauc_file)
_ensure_time_columns(tauc_df)
tauc_df = tauc_df.sort_values('timestamp').reset_index(drop=True)

temperature_df = pd.read_csv(temperature_file, sep=None, engine='python')
temp_timestamp_col = next(
    (col for col in temperature_df.columns if 'timestamp' in col.lower()),
    None,
)
if temp_timestamp_col is not None:
    temperature_df['timestamp'] = pd.to_datetime(
        temperature_df[temp_timestamp_col], errors='coerce'
    )
    temp0 = temperature_df['timestamp'].min()
    temperature_df['time_hours'] = (
        (temperature_df['timestamp'] - temp0).dt.total_seconds() / 3600.0
    )
else:
    elapsed_col = next(
        (col for col in temperature_df.columns if 'elapsed' in col.lower()),
        None,
    )
    if elapsed_col is None:
        raise KeyError(
            'No timestamp/elapsed column found in temperature log for time conversion.'
        )
    temperature_df['time_hours'] = pd.to_numeric(
        temperature_df[elapsed_col], errors='coerce'
    ) / 3600.0

urbach_df = pd.read_csv(urbach_file)
_ensure_time_columns(urbach_df)
urbach_df = urbach_df.sort_values('timestamp').reset_index(drop=True)

# Read every pixel-level log into a dataframe keyed by filename stem
pixel_dataframes = {}
for csv_path in sorted(pixel_folder.glob('*.csv')):
    df = pd.read_csv(csv_path)
    # Normalise pixel timestamps to hours from the earliest recorded point
    timestamp_col = next((col for col in df.columns if 'timestamp' in col.lower()), None)
    if timestamp_col is not None:
        numeric_ts = pd.to_numeric(df[timestamp_col], errors='coerce')
        if numeric_ts.notna().any():
            baseline = numeric_ts.min()
            df['time_hours'] = (numeric_ts - baseline) / 3600.0
        else:
            datetimes = pd.to_datetime(df[timestamp_col], errors='coerce')
            if datetimes.notna().any():
                baseline = datetimes.min()
                df['time_hours'] = (
                    (datetimes - baseline).dt.total_seconds() / 3600.0
                )
            else:
                df['time_hours'] = np.nan
    pixel_dataframes[csv_path.stem] = df

# Persist the processed datasets for downstream analysis steps
result_folder.mkdir(parents=True, exist_ok=True)
tauc_df.to_csv(result_folder / 'tauc_metrics.csv', index=False)
temperature_df.to_csv(result_folder / 'temperature_log.csv', index=False)
urbach_df.to_csv(result_folder / 'urbach_metrics.csv', index=False)
for name, df in pixel_dataframes.items():
    df.to_csv(result_folder / f'{name}.csv', index=False)
