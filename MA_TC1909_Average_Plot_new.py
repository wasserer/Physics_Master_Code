
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
# Try to make the error bar as the shadow
SHOW_ERROR_BARS = True
FIGURE_DPI = 300
RESULT_DIR = Path(
    ""  # Enter your file path/folder path in this place
    'Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log'
)

tauc_file = Path(
    ""  # Enter your file path/folder path in this place
    'ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/tauc_metrics.csv'
)
temperature_file = Path(
    ""  # Enter your file path/folder path in this place
    'ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/temperature_log.csv'
)
urbach_file = Path(
    ""  # Enter your file path/folder path in this place
    'ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/urbach_metrics.csv'
)
pixel_folder = Path(
    ""  # Enter your file path/folder path in this place
    'ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/pixel'
)

required_paths = {
    'tauc metrics': tauc_file,
    'urbach metrics': urbach_file,
    'temperature log': temperature_file,
}

for label, filepath in required_paths.items():
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing {label} CSV: {filepath}")


def _read_csv(path: Path, *, usecols=None):
    try:
        return pd.read_csv(path, usecols=usecols)
    except Exception as exc:
        raise ValueError(f'Failed to load CSV at {path}: {exc}') from exc


tauc_df = _read_csv(tauc_file)
urbach_df = _read_csv(urbach_file)
temperature_df = _read_csv(temperature_file, usecols=['Elapsed_s'])
temperature_hours = (
    pd.to_numeric(temperature_df['Elapsed_s'], errors='coerce') / 3600.0
).dropna().to_numpy(dtype=float)
if not pixel_folder.is_dir():
    raise FileNotFoundError(f'Missing pixel data directory: {pixel_folder}')

pixel_dataframes = {
    csv_path.stem: _read_csv(csv_path)
    for csv_path in sorted(pixel_folder.glob('*.csv'))
}


PIXEL_COLUMN_MAP = {
    'Timestamp[s]': 'timestamp_s',
    'Temperature [C]': 'temperature_c',
    'Cycle Number': 'cycle_number',
    'Isc[mA/cm2]': 'isc',
    'Voc [V]': 'voc',
    'I_MPP [mA/cm2]': 'i_mpp',
    'V_MPP [V]': 'v_mpp',
    'FF': 'ff',
    'PCE [%]': 'pce',
    'Rs': 'rs',
    'Rp': 'rp',
}

PIXEL_METRIC_KEYS = (
    'temperature_c',
    'isc',
    'voc',
    'i_mpp',
    'v_mpp',
    'ff',
    'pce',
    'rs',
    'rp',
)

PIXEL_CYCLE_COLUMNS = ('timestamp_h',) + PIXEL_METRIC_KEYS

METRIC_LABELS = {
    'timestamp_h': 'Time [h]',
    'temperature_c': r'Temperature [$^\circ$C]',
    'isc': r'$I_{sc}$',
    'voc': r'$V_{oc}$',
    'ff': 'FF',
    'rs': r'$R_s$',
    'rp': r'$R_p$',
    'pce': 'PCE',
    'i_mpp': r'$I_{\mathrm{MPP}}$',
    'v_mpp': r'$V_{\mathrm{MPP}}$',
    'tauc_timestamp_h': 'Time [h]',
    'tauc_temperature': r'Temperature [$^\circ$C]',
    'temperature': r'Temperature [$^\circ$C]',
    'urbach_timestamp_h': 'Time [h]',
    'urbach_temperature': r'Temperature [$^\circ$C]',
}


def _prepare_pixel_frame(raw: pd.DataFrame) -> pd.DataFrame:
    renamed = raw.rename(columns=PIXEL_COLUMN_MAP)
    renamed = renamed[list(PIXEL_COLUMN_MAP.values())].copy()
    for column in renamed.columns:
        renamed[column] = pd.to_numeric(renamed[column], errors='coerce')
    renamed.dropna(subset=['cycle_number'], inplace=True)
    renamed['cycle_number'] = renamed['cycle_number'].astype('Int64')
    renamed.sort_values('cycle_number', inplace=True)
    renamed['timestamp_h'] = renamed['timestamp_s'] / 3600.0
    return renamed


pixel_tables: Dict[str, pd.DataFrame] = {
    name: _prepare_pixel_frame(df)
    for name, df in pixel_dataframes.items()
}

pixel_cycle_stats: Dict[str, pd.DataFrame] = {
    name: table.groupby('cycle_number')[list(PIXEL_CYCLE_COLUMNS)]
    .agg(['mean', 'sem'])
    .sort_index()
    for name, table in pixel_tables.items()
}

pixel_cycle_mean_tables: Dict[str, pd.DataFrame] = {
    name: stats.xs('mean', axis=1, level=1)
    for name, stats in pixel_cycle_stats.items()
}

pixel_cycle_sem_tables: Dict[str, pd.DataFrame] = {
    name: stats.xs('sem', axis=1, level=1)
    for name, stats in pixel_cycle_stats.items()
}

pixel_metric_arrays: Dict[str, Dict[str, np.ndarray]] = {
    metric: {
        name: table[metric].to_numpy(dtype=float)
        for name, table in pixel_cycle_mean_tables.items()
    }
    for metric in PIXEL_CYCLE_COLUMNS
}

pixel_metric_sem_arrays: Dict[str, Dict[str, np.ndarray]] = {
    metric: {
        name: table[metric].to_numpy(dtype=float)
        for name, table in pixel_cycle_sem_tables.items()
    }
    for metric in PIXEL_CYCLE_COLUMNS
}

pixel_cycle_numbers: Dict[str, np.ndarray] = {
    name: table.index.to_numpy(dtype=int, copy=False)
    for name, table in pixel_cycle_mean_tables.items()
}

_pixel_concat = pd.concat(
    [
        table.reset_index().assign(pixel=name)
        for name, table in pixel_cycle_mean_tables.items()
    ],
    ignore_index=True,
)

PIXEL_AVERAGE_COLUMNS = [
    'timestamp_h',
    'temperature_c',
    'isc',
    'voc',
    'i_mpp',
    'v_mpp',
    'ff',
    'pce',
    'rs',
    'rp',
]

pixel_cycle_average = (
    _pixel_concat.groupby('cycle_number')[PIXEL_AVERAGE_COLUMNS]
    .mean()
    .sort_index()
)
pixel_cycle_sem = (
    _pixel_concat.groupby('cycle_number')[PIXEL_AVERAGE_COLUMNS]
    .sem()
    .sort_index()
)


TAUC_COLUMN_MAP = {
    'timestamp(in Hour):': 'tauc_timestamp_h',
    'band_gap': 'tauc_band_gap',
    'tauc_slope': 'tauc_slope',
    'tauc_slope_b': 'tauc_slope_b',
    'tauc_r2': 'tauc_r2',
    'tauc_rmse': 'tauc_rmse',
    'window_start': 'tauc_window_start',
    'window_end': 'tauc_window_end',
    'cycle_number_avg': 'cycle_number',
    'temperature': 'tauc_temperature',
}

tauc_processed = tauc_df.rename(columns=TAUC_COLUMN_MAP)
tauc_processed = tauc_processed[list(TAUC_COLUMN_MAP.values())].copy()
for column in tauc_processed.columns:
    tauc_processed[column] = pd.to_numeric(tauc_processed[column], errors='coerce')
tauc_processed.dropna(subset=['cycle_number'], inplace=True)
tauc_processed['cycle_number'] = tauc_processed['cycle_number'].astype('Int64')
tauc_grouped = tauc_processed.groupby('cycle_number').agg(['mean', 'sem']).sort_index()
tauc_cycle_average = tauc_grouped.xs('mean', axis=1, level=1)
tauc_cycle_sem = tauc_grouped.xs('sem', axis=1, level=1)


URBACH_COLUMN_MAP = {
    'timestamp(in Hour):': 'urbach_timestamp_h',
    'cycle': 'cycle_number',
    'urbach_energy': 'urbach_energy',
    'urbach_slope': 'urbach_slope',
    'urbach_intercept': 'urbach_intercept',
    'urbach_r2': 'urbach_r2',
    'urbach_rmse': 'urbach_rmse',
    'window_start': 'urbach_window_start',
    'window_end': 'urbach_window_end',
}

urbach_processed = urbach_df.rename(columns=URBACH_COLUMN_MAP)
urbach_processed = urbach_processed[list(URBACH_COLUMN_MAP.values())].copy()
for column in urbach_processed.columns:
    urbach_processed[column] = pd.to_numeric(urbach_processed[column], errors='coerce')
urbach_processed.dropna(subset=['cycle_number'], inplace=True)
urbach_processed['cycle_number'] = urbach_processed['cycle_number'].astype('Int64')
urbach_grouped = urbach_processed.groupby('cycle_number').agg(['mean', 'sem']).sort_index()
urbach_cycle_average = urbach_grouped.xs('mean', axis=1, level=1)
urbach_cycle_sem = urbach_grouped.xs('sem', axis=1, level=1)


cycle_averages = (
    pixel_cycle_average.join(tauc_cycle_average, how='outer')
    .join(urbach_cycle_average, how='outer')
    .sort_index()
)

cycle_sem = (
    pixel_cycle_sem.join(tauc_cycle_sem, how='outer')
    .join(urbach_cycle_sem, how='outer')
    .sort_index()
)

cycle_numbers = cycle_averages.index.astype(int, copy=False).to_numpy()
average_arrays = {
    column: cycle_averages[column].to_numpy(dtype=float)
    for column in cycle_averages.columns
}
cycle_error_arrays = {
    column: cycle_sem[column].to_numpy(dtype=float)
    for column in cycle_sem.columns
}

PIXEL_COLOR_CYCLE = (
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
)


def _label_from_metric(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name.replace('_', ' ').title())


def _pixel_display_name(pixel_name: str) -> str:
    base = pixel_name.split('_', 1)[0]
    return base.lower()


def _plot_cycle_metric(metric_name: str, values: np.ndarray, output_dir: Path) -> None:
    mask = np.isfinite(cycle_numbers) & np.isfinite(values)
    if not np.any(mask):
        return

    cycle_color = '#042940'
    errors = cycle_error_arrays.get(metric_name)
    if SHOW_ERROR_BARS and errors is not None:
        errors = np.asarray(errors, dtype=float)
        mask = mask & np.isfinite(errors)
        if not np.any(mask):
            return

    plot_cycles = cycle_numbers[mask]
    plot_values = values[mask]
    plot_errors = None
    if SHOW_ERROR_BARS and errors is not None:
        plot_errors = errors[mask]

    fig, ax = plt.subplots(figsize=(7, 4))

    # Draw main line
    ax.plot(
        plot_cycles,
        plot_values,
        marker='o',
        linestyle='-',
        markersize=3,
        linewidth=1.2,
        alpha=0.9,
        color=cycle_color,
    )

    # Draw error band (fill_between) if enabled
    if SHOW_ERROR_BARS and plot_errors is not None:
        ax.fill_between(
            plot_cycles,
            plot_values - plot_errors,
            plot_values + plot_errors,
            color=cycle_color,
            alpha=0.25,
            linewidth=0,
        )

    ax.set_xlabel('Cycle Number')
    ax.set_ylabel(_label_from_metric(metric_name))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    metric_lower = metric_name.lower()
    if metric_lower == 'rs':
        ax.set_ylim(0, 0.15)
    elif metric_lower == 'rp':
        ax.set_ylim(0, 30)

    safe_name = metric_name.lower().replace(' ', '_')
    fig.savefig(output_dir / f'{safe_name}_vs_cycle.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    log_path = output_dir / f'{safe_name}_vs_cycle.csv'
    log_frame = pd.DataFrame(
        {
            'cycle_number': plot_cycles,
            metric_name: plot_values,
        }
    )
    if plot_errors is not None:
        log_frame[f'{metric_name}_error'] = plot_errors
    log_frame.to_csv(log_path, index=False)


def _plot_pixel_metric(metric_name: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    metric_lower = metric_name.lower()
    plotted_any = False
    log_rows = []

    for idx, (pixel_name, table) in enumerate(pixel_cycle_mean_tables.items()):
        if metric_name not in table.columns:
            continue

        cycles = table.index.to_numpy(dtype=float, copy=False)
        values = table[metric_name].to_numpy(dtype=float)
        mask = np.isfinite(cycles) & np.isfinite(values)
        if not np.any(mask):
            continue

        pixel_display = _pixel_display_name(pixel_name)
        raw_errors = None
        if SHOW_ERROR_BARS:
            raw_errors = pixel_metric_sem_arrays.get(metric_name, {}).get(pixel_name)
            if raw_errors is not None:
                raw_errors = np.asarray(raw_errors, dtype=float)
                mask = mask & np.isfinite(raw_errors)
                if not np.any(mask):
                    continue
        plot_errors = raw_errors[mask] if (SHOW_ERROR_BARS and raw_errors is not None) else None

        plot_cycles = cycles[mask]
        plot_values = values[mask]
        color = PIXEL_COLOR_CYCLE[idx % len(PIXEL_COLOR_CYCLE)]

        # Draw main line
        ax.plot(
            plot_cycles,
            plot_values,
            marker='o',
            linestyle='-',
            color=color,
            label=pixel_display,
            markersize=3,
            linewidth=1.2,
            alpha=0.9,
        )

        # Draw error band
        if SHOW_ERROR_BARS and plot_errors is not None:
            ax.fill_between(
                plot_cycles,
                plot_values - plot_errors,
                plot_values + plot_errors,
                color=color,
                alpha=0.25,
                linewidth=0,
            )

        plotted_any = True
        for idx_point, (cyc, val) in enumerate(zip(plot_cycles, plot_values)):
            row = {
                'cycle_number': cyc,
                'pixel': pixel_display,
                metric_name: val,
            }
            if plot_errors is not None:
                row[f'{metric_name}_error'] = plot_errors[idx_point]
            log_rows.append(row)

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel('Cycle Number')
    ax.set_ylabel(_label_from_metric(metric_name))
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
    if metric_lower == 'rs':
        ax.set_ylim(0, 0.15)
    elif metric_lower == 'rp':
        ax.set_ylim(0, 20)
    ax.legend()

    safe_name = metric_name.lower().replace(' ', '_')
    fig.savefig(output_dir / f'pixels_{safe_name}_vs_cycle.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    log_path = output_dir / f'pixels_{safe_name}_vs_cycle.csv'
    if log_rows:
        log_frame = pd.DataFrame(log_rows)
        log_frame.sort_values(['pixel', 'cycle_number'], inplace=True)
        log_frame.to_csv(log_path, index=False)


if __name__ == '__main__':
    result_dir = RESULT_DIR / 'Result_Average'
    result_dir.mkdir(parents=True, exist_ok=True)

    for metric in cycle_averages.columns:
        _plot_cycle_metric(metric, average_arrays[metric], result_dir)

    for metric in PIXEL_METRIC_KEYS:
        _plot_pixel_metric(metric, result_dir)
