import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#This is the analysis and plot fo TC at 19,09,2025
#The IV-curve and the Spectra results are stored in the same folder

PIXEL_OUTPUT_FOLDER = Path(
    ""  # Enter your file path/folder path in this place
    'Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/pixel'
)
SPECTRA_FILE = Path(
    ""  # Enter your file path/folder path in this place
    'Pervoskite Space(Master)/Data/ThermalCycling/MA_Data/Spectra.csv'
)
cycle_range = (1, 5)
result_folder_name = f"Result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
result_path = PIXEL_OUTPUT_FOLDER / result_folder_name
result_path.mkdir(parents=True, exist_ok=True)

PIXEL_TIMESTAMP_COLUMNS = {
    'timestamp_h': (
        'Timestamp[h]',
        'timestamp_h',
        'Time [h]',
        'time_hours',
    ),
    'timestamp_s': (
        'Timestamp[s]',
        'timestamp_s',
        'Time [s]',
        'Elapsed_s',
    ),
    'cycle_number': (
        'Cycle Number',
        'cycle_number',
        'cycle',
    ),
}

PIXEL_VALUE_COLUMNS = {
    'temperature_c': (
        'Temperature [C]',
        'temperature_c',
        'Temp_A_C',
        'Temp [C]',
    ),
    'isc': (
        'Isc[mA/cm2]',
        'isc',
        'isc_mean',
    ),
    'voc': (
        'Voc [V]',
        'voc',
        'voc_mean',
    ),
    'i_mpp': (
        'I_MPP [mA/cm2]',
        'i_mpp',
    ),
    'v_mpp': (
        'V_MPP [V]',
        'v_mpp',
    ),
    'ff': (
        'FF',
        'ff',
    ),
    'pce': (
        'PCE [%]',
        'pce',
    ),
    'rs': (
        'Rs',
        'rs',
    ),
    'rp': (
        'Rp',
        'rp',
    ),
}

PIXEL_METRIC_TRANSFORMS = {
    'voc': np.abs,
    'v_mpp': np.abs,
}

SPECTRA_COLUMN_MAP = {
    'timestamp_hours': 'timestamp(in Hour):',
    'band_gap': 'band_gap',
    'tauc_slope': 'tauc_slope',
    'tauc_slope_b': 'tauc_slope_b',
    'tauc_r2': 'tauc_r2',
    'tauc_rmse': 'tauc_rmse',
    'window_start': 'window_start',
    'window_end': 'window_end',
    'cycle_number_avg': 'cycle_number_avg',
    'temperature': 'temperature',
}

TEMPERATURE_LOG_FILE = Path(
    ""  # Enter your file path/folder path in this place
    'Pervoskite Space(Master)/Data/ThermalCycling/MA_Data/temperature_log.csv'
)


def _resolve_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _load_pixel_timeseries(file_path: Path):
    df = pd.read_csv(file_path)

    timestamp_h_col = _resolve_column(df, PIXEL_TIMESTAMP_COLUMNS['timestamp_h'])
    if timestamp_h_col is not None:
        df['timestamp_h'] = pd.to_numeric(df[timestamp_h_col], errors='coerce')
    else:
        timestamp_s_col = _resolve_column(df, PIXEL_TIMESTAMP_COLUMNS['timestamp_s'])
        if timestamp_s_col is None:
            raise ValueError(f'Missing timestamp column in {file_path}')
        df['timestamp_h'] = pd.to_numeric(df[timestamp_s_col], errors='coerce') / 3600.0

    cycle_col = _resolve_column(df, PIXEL_TIMESTAMP_COLUMNS['cycle_number'])
    if cycle_col is None:
        raise ValueError(f'Missing cycle number column in {file_path}')
    df['cycle_number'] = pd.to_numeric(df[cycle_col], errors='coerce')

    df = df.dropna(subset=['timestamp_h', 'cycle_number']).copy()
    if df.empty:
        raise ValueError(f'No valid timestamp/cycle data in {file_path}')
    df['cycle_number'] = np.rint(df['cycle_number']).astype(int)
    df.sort_values('timestamp_h', inplace=True)
    df.reset_index(drop=True, inplace=True)

    data = {
        'timestamp_h': df['timestamp_h'].to_numpy(dtype=float),
        'cycle_number': df['cycle_number'].to_numpy(dtype=int),
    }

    for alias, candidates in PIXEL_VALUE_COLUMNS.items():
        column = _resolve_column(df, candidates)
        if column is None:
            continue
        values = pd.to_numeric(df[column], errors='coerce').to_numpy(dtype=float)
        transform = PIXEL_METRIC_TRANSFORMS.get(alias)
        if transform is not None:
            values = transform(values)
        data[alias] = values
    return data


def _discover_pixel_files(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f'Pixel folder not found: {folder}')
    mapping = {}
    for csv_path in sorted(folder.glob('px*_result.csv')):
        pixel_id = csv_path.stem.split('_')[0]
        mapping[pixel_id] = csv_path
    if not mapping:
        raise FileNotFoundError(f'No pixel CSV files found in {folder}')
    return mapping


def _load_mapped_columns(file_path, column_map):
    df = pd.read_csv(file_path)
    data = {}
    for alias, column in column_map.items():
        values = df[column].to_numpy()
        if alias == 'timestamp_h':
            values = values / 3600
        elif alias == 'voc':
            values = np.abs(values.astype(float))
        elif alias == 'pce':
            values = values.astype(float) / 100.0
        data[alias] = values
    return data


pixel_files = _discover_pixel_files(PIXEL_OUTPUT_FOLDER)
px_data = {
    pixel: _load_pixel_timeseries(path)
    for pixel, path in pixel_files.items()
}
spectra_data = _load_mapped_columns(SPECTRA_FILE, SPECTRA_COLUMN_MAP)

def load_temperature_log(file_path):
    df = (
        pd.read_csv(file_path, usecols=['Elapsed_s', 'Temp_A_C'])
        .dropna()
        .sort_values('Elapsed_s')
    )
    timestamps_h = df['Elapsed_s'].to_numpy(dtype=float) / 3600
    temperatures_c = df['Temp_A_C'].to_numpy(dtype=float)
    return timestamps_h, temperatures_c

temperature_timestamp_h, temperature_values_c = load_temperature_log(TEMPERATURE_LOG_FILE)

def _select_temperature_window(time_min, time_max):
    mask = (temperature_timestamp_h >= time_min) & (temperature_timestamp_h <= time_max)
    if not mask.any():
        return temperature_timestamp_h, temperature_values_c
    return temperature_timestamp_h[mask], temperature_values_c[mask]

for alias, values in spectra_data.items():
    globals()[f'spectra_{alias}'] = values

PX_COLOR_CYCLE = (
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
)

for name, data in px_data.items():
    for alias, values in data.items():
        globals()[f'{name}_{alias}'] = values

def _build_cycle_mask(cycles, cycle_selection):
    """
    Return a boolean mask that selects the desired cycles from ``cycles``.

    Parameters
    ----------
    cycles : array-like
        One-dimensional array containing the cycle numbers associated with each
        measurement.
    cycle_selection : object
        Selector that determines which cycles to keep. Supports the following:

        * ``None`` – keep all cycles.
        * Tuple of two values ``(lower, upper)`` – inclusive numeric range.
        * Slice or ``range`` – treated as inclusive range/membership.
        * Iterable of numeric values (list, set, ndarray, tuple with length != 2)
          – keep cycles matching any of the provided values.
        * Scalar – keep cycles equal to the provided value.

    Returns
    -------
    np.ndarray
        Boolean mask aligned with ``cycles`` that is ``True`` for rows that
        satisfy the selection criteria.
    """
    cycles_array = np.asarray(cycles)
    if cycles_array.ndim != 1:
        raise ValueError('cycles must be a one-dimensional array')

    if cycles_array.dtype.kind in 'OUS':
        numeric_cycles = pd.to_numeric(cycles_array, errors='coerce')
    else:
        numeric_cycles = cycles_array.astype(float, copy=False)

    mask = np.isfinite(numeric_cycles)

    if cycle_selection is None:
        return mask

    def _to_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if isinstance(cycle_selection, slice):
        lower = _to_float(cycle_selection.start)
        upper = _to_float(cycle_selection.stop)
        if lower is not None:
            mask &= numeric_cycles >= lower
        if upper is not None:
            mask &= numeric_cycles <= upper
        step = cycle_selection.step
        if step not in (None, 1, 0):
            if lower is None:
                lower = np.nanmin(numeric_cycles[mask])
            if upper is None:
                upper = np.nanmax(numeric_cycles[mask])
            valid_values = np.arange(lower, upper + (step / 2.0), step)
            mask &= np.isin(numeric_cycles, valid_values)
        return mask

    if isinstance(cycle_selection, tuple) and len(cycle_selection) == 2:
        lower = _to_float(cycle_selection[0])
        upper = _to_float(cycle_selection[1])
        if lower is not None:
            mask &= numeric_cycles >= lower
        if upper is not None:
            mask &= numeric_cycles <= upper
        return mask

    if isinstance(cycle_selection, range):
        selection_values = np.fromiter(cycle_selection, dtype=float)
        if selection_values.size == 0:
            return np.zeros_like(mask, dtype=bool)
        return mask & np.isin(numeric_cycles, selection_values)

    if isinstance(cycle_selection, (list, set, np.ndarray)) or (
        isinstance(cycle_selection, tuple) and len(cycle_selection) != 2
    ):
        selection_values = np.array(list(cycle_selection), dtype=float, copy=False)
        if selection_values.size == 0:
            return np.zeros_like(mask, dtype=bool)
        return mask & np.isin(numeric_cycles, selection_values)

    target = _to_float(cycle_selection)
    if target is None:
        return np.zeros_like(mask, dtype=bool)
    return mask & np.isclose(numeric_cycles, target, equal_nan=False)

def plot_px_parameter(px_name, cycle_selection=None):
    data = px_data[px_name]
    timestamp_all = data['timestamp_h']
    cycles = data['cycle_number']
    y_aliases = ('isc', 'voc', 'ff', 'pce', 'rs', 'rp')
    y_labels = {
        'voc': 'VOC [V]',
        'isc': 'ISC [mA/cm^2]',
        'ff': 'Fill factor',
        'pce': 'PCE [%]',
        'rp': 'Rp [Ohm]',
        'rs': 'Rs [Ohm]',
    }

    for y_alias in y_aliases:
        if y_alias not in data:
            continue
        values_all = data[y_alias]
        errors_all = data.get(f'{y_alias}_error')

        mask = _build_cycle_mask(cycles, cycle_selection)

        timestamp = timestamp_all[mask]
        values = values_all[mask]
        errors = errors_all[mask] if errors_all is not None else None
        if errors is not None and np.isnan(errors).all():
            errors = None

        if timestamp.size:
            temp_time, temp_values = _select_temperature_window(timestamp.min(), timestamp.max())
        else:
            temp_time = np.array([])
            temp_values = np.array([])

        fig, ax1 = plt.subplots(figsize=(5, 3))
        label = y_alias.upper()
        if errors is not None:
            ax1.errorbar(
                timestamp,
                values,
                yerr=errors,
                fmt='o',
                markersize=4,
                capsize=3,
                label=label,
            )
        else:
            ax1.scatter(timestamp, values, s=3, label=label)
        ax1.set_xlabel('Timestamp [h]')
        ax1.set_ylabel(y_labels.get(y_alias, y_alias.upper()))
        if y_alias == 'rp':
            ax1.set_ylim(0, 10)

        ax2 = ax1.twinx()
        if temp_time.size:
            ax2.plot(temp_time, temp_values, '--', color='tab:red', alpha=0.6, label='Temperature [C]')
        ax2.set_ylabel('Temperature [C]')

        handles_1, labels_1 = ax1.get_legend_handles_labels()
        if handles_1:
            ax1.legend(handles_1, labels_1, loc='lower right')
        handles_2, labels_2 = ax2.get_legend_handles_labels()
        if handles_2:
            ax2.legend(handles_2, labels_2, loc='upper right')
        fig.tight_layout()

        filename = f"{px_name}_{y_alias.upper()}.png"
        fig.savefig(result_path / filename, dpi=300)
        #plt.show()
        plt.close(fig)

def plot_multi_px_parameter(cycle_selection=None):
    px_names = tuple(px_data.keys())
    y_aliases = ('isc', 'voc', 'ff', 'pce', 'rs', 'rp')
    y_labels = {
        'voc': 'V$_{OC}$ [V]',
        'isc': 'I$_{SC}$ [mA/cm$^2$]',
        'ff': 'FF',
        'pce': 'PCE [%]',
        'rp': 'R$_p$ [$\Omega$]',
        'rs': 'R$_s$ [$\Omega$]',
    }

    for y_alias in y_aliases:
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax2 = ax1.twinx()

        primary_handles = []
        primary_labels = []
        temperature_line = None
        time_windows = []

        for idx, px_name in enumerate(px_names):
            color = PX_COLOR_CYCLE[idx % len(PX_COLOR_CYCLE)]
            data = px_data[px_name]
            if y_alias not in data:
                continue
            cycles = data['cycle_number']
            mask = _build_cycle_mask(cycles, cycle_selection)

            timestamp = data['timestamp_h'][mask]
            values = data[y_alias][mask]

            if timestamp.size == 0:
                continue

            scatter = ax1.scatter(timestamp, values, s=3, color=color, label=px_name.upper())
            primary_handles.append(scatter)
            primary_labels.append(px_name.upper())
            time_windows.append((timestamp.min(), timestamp.max()))

        if time_windows:
            window_min = min(start for start, _ in time_windows)
            window_max = max(end for _, end in time_windows)
            temp_time, temp_values = _select_temperature_window(window_min, window_max)
            if temp_time.size:
                (temperature_line,) = ax2.plot(
                    temp_time,
                    temp_values,
                    '--',
                    alpha=0.6,
                    color='tab:red',
                    label='Temperature [C]',
                )

        ax1.set_xlabel('Timestamp [h]')
        ax1.set_ylabel(y_labels.get(y_alias, y_alias.upper()))
        ax2.set_ylabel('Temperature [C]')

        if primary_handles:
            ax1.legend(primary_handles, primary_labels, loc='lower right')
        if temperature_line is not None:
            ax2.legend(loc='upper right')

        fig.tight_layout()

        filename = f"multi_{y_alias.upper()}.png"
        fig.savefig(result_path / filename, dpi=300)
        #plt.show()
        plt.close(fig)

for px_name in px_data:
    plot_px_parameter(px_name, cycle_selection=cycle_range)
plot_multi_px_parameter(cycle_selection=cycle_range)

def plot_spectra_band_gap(cycle_selection=None):
    cycles = spectra_data['cycle_number_avg']
    mask = _build_cycle_mask(cycles, cycle_selection)

    timestamp = spectra_data['timestamp_hours'][mask]
    band_gap = spectra_data['band_gap'][mask]
    if timestamp.size == 0:
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(timestamp, band_gap, s=3, label='Band Gap')
    ax1.set_xlabel('Timestamp [h]')
    ax1.set_ylim(1.5, 1.6)
    ax1.set_ylabel('Band Gap [eV]')

    ax2 = ax1.twinx()
    temp_time, temp_values = _select_temperature_window(timestamp.min(), timestamp.max())
    if temp_time.size:
        ax2.plot(temp_time, temp_values, '--', color='tab:red', alpha=0.6, label='Temperature [C]')
    ax2.set_ylabel('Temperature [C]')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    if lines_2:
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    elif lines_1:
        ax1.legend(loc='best')

    fig.tight_layout()

    filename = 'spectra_band_gap.png'
    fig.savefig(result_path / filename, dpi=300)
    #plt.show()
    plt.close(fig)

plot_spectra_band_gap(cycle_selection=cycle_range)
