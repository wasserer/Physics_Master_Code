import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

TEMPERATURE_RANGES = [
    ('lt20', (None, 20)),
    ('20_40', (20, 40)),
    ('40_50', (40, 50)),
    ('50_60', (50, 60)),
    ('60_70', (60, 70)),
    ('70_80', (70, 80)),
]

PLOT_TEMPERATURE_RANGE_LABELS = [
    #'20_40',
    #'40_50',
    '50_60',
    '60_70',
    '70_80',
]

def select_temperature_ranges(temperature_ranges, selection):
    if not selection:
        return temperature_ranges
    label_to_range = dict(temperature_ranges)
    missing_labels = [label for label in selection if label not in label_to_range]
    if missing_labels:
        raise ValueError(f'Requested temperature ranges not defined: {missing_labels}')
    return [(label, label_to_range[label]) for label in selection]

px1_file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/pixel/px1_result.csv'
px2_file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/pixel/px2_result.csv'
px3_file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/pixel/px3_result.csv'
px4_file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/pixel/px4_result.csv'
px5_file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/pixel/px5_result.csv'
px6_file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/pixel/px6_result.csv'

px1_df = pd.read_csv(px1_file)
px2_df = pd.read_csv(px2_file)
px3_df = pd.read_csv(px3_file)
px4_df = pd.read_csv(px4_file)
px5_df = pd.read_csv(px5_file)
px6_df = pd.read_csv(px6_file)

cycle_metrics = {
    'Voc [V]': 'Voc_avg',
    'Isc[mA/cm2]': 'Isc_avg',
    'FF': 'FF_avg',
    'Rs': 'Rs_avg',
    'Rp': 'Rp_avg',
    'PCE [%]': 'PCE_avg',
}

METRIC_YLIMS = {
    'Voc [V]': (0, 1),
    'Isc[mA/cm2]': (0, 6),
    'FF': (0, 0.8),
    'Rs': (0, 0.2),
    'PCE [%]': (0, 18),
    'MPP': (-0.7, 0),
}

def calculate_cycle_metrics_by_temperature_ranges(df, temperature_ranges, metrics_map):
    mean_column_order = [
        f'{alias}_{label}'
        for label, _ in temperature_ranges
        for alias in metrics_map.values()
    ]
    error_column_order = [
        f'{alias}_err_{label}'
        for label, _ in temperature_ranges
        for alias in metrics_map.values()
    ]

    combined = None
    combined_error = None
    for label, (lower, upper) in temperature_ranges:
        temp_series = df['Temperature [C]']
        mask = pd.Series(True, index=df.index)
        if lower is not None:
            mask &= temp_series >= lower
        if upper is not None:
            mask &= temp_series < upper

        range_df = df.loc[mask]
        if range_df.empty:
            range_avg = pd.DataFrame(columns=list(metrics_map.values()))
            range_error = pd.DataFrame(columns=list(metrics_map.values()))
            range_avg.index.name = 'Cycle Number'
            range_error.index.name = 'Cycle Number'
        else:
            cycle_group = range_df.groupby('Cycle Number')
            metric_keys = list(metrics_map.keys())
            range_avg = cycle_group[metric_keys].mean().rename(columns=metrics_map)
            range_avg = range_avg.reindex(columns=list(metrics_map.values()))
            range_std = cycle_group[metric_keys].std().rename(columns=metrics_map)
            range_std = range_std.reindex(columns=list(metrics_map.values()))
            range_count = cycle_group[metric_keys].count().rename(columns=metrics_map)
            with np.errstate(divide='ignore', invalid='ignore'):
                range_error = range_std.divide(np.sqrt(range_count.replace(0, np.nan)))
            range_error = range_error.reindex(columns=list(metrics_map.values()))

        range_avg = range_avg.add_suffix(f'_{label}')
        range_error = range_error.add_suffix(f'_err_{label}')

        if combined is None:
            combined = range_avg
            combined_error = range_error
        else:
            combined = combined.join(range_avg, how='outer')
            combined_error = combined_error.join(range_error, how='outer')
    if combined is None:
        combined = pd.DataFrame(columns=mean_column_order)
        combined_error = pd.DataFrame(columns=error_column_order)
        combined.index = pd.Index([], name='Cycle Number')
        combined_error.index = pd.Index([], name='Cycle Number')
    else:
        combined = combined.reindex(columns=mean_column_order)
        combined_error = combined_error.reindex(columns=error_column_order)

    combined_df = combined.sort_index().reset_index()
    combined_error_df = combined_error.sort_index().reset_index()
    metrics_array = combined_df.to_numpy()
    error_array = combined_error_df.to_numpy()
    return combined_df, combined_error_df, metrics_array, error_array

px_frames = {
    'px1': px1_df,
    'px2': px2_df,
    'px3': px3_df,
    'px4': px4_df,
    'px5': px5_df,
    'px6': px6_df,
}

px_cycle_avg = {}
for name, df in px_frames.items():
    cycle_group = df.groupby('Cycle Number')
    mean_df = (
        cycle_group[list(cycle_metrics.keys())]
        .mean()
        .rename(columns=cycle_metrics)
    )
    pce_std = cycle_group['PCE [%]'].std()
    pce_count = cycle_group['PCE [%]'].count()
    mean_df['PCE_err'] = pce_std.divide(np.sqrt(pce_count.replace(0, np.nan)))
    px_cycle_avg[name] = (
        mean_df.reset_index()
        .sort_values('Cycle Number')
    )

px_cycle_avg_by_temp = {}
px_cycle_err_by_temp = {}
px_cycle_avg_arrays = {}
px_cycle_err_arrays = {}
for name, df in px_frames.items():
    avg_df, err_df, avg_array, err_array = calculate_cycle_metrics_by_temperature_ranges(
        df,
        TEMPERATURE_RANGES,
        cycle_metrics,
    )
    px_cycle_avg_by_temp[name] = avg_df
    px_cycle_err_by_temp[name] = err_df
    px_cycle_avg_arrays[name] = avg_array
    px_cycle_err_arrays[name] = err_array

EXPORT_ROOT = Path('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/Average_pixel_different_temperature')
metric_filenames = {
    'Voc [V]': 'Voc_result.csv',
    'Isc[mA/cm2]': 'Isc_result.csv',
    'FF': 'FF_result.csv',
    'Rs': 'Rs_result.csv',
    'Rp': 'Rp_result.csv',
    'PCE [%]': 'PCE_result.csv',
}

EXPORT_ROOT.mkdir(exist_ok=True)
for name, avg_df in px_cycle_avg_by_temp.items():
    pixel_dir = EXPORT_ROOT / name
    pixel_dir.mkdir(parents=True, exist_ok=True)
    err_df = px_cycle_err_by_temp[name]

    for metric, alias in cycle_metrics.items():
        output_columns = {'Cycle Number': avg_df['Cycle Number']}
        for label, _ in TEMPERATURE_RANGES:
            mean_col = f'{alias}_{label}'
            err_col = f'{alias}_err_{label}'
            output_columns[f'{label}_avg'] = avg_df.get(
                mean_col,
                pd.Series(np.nan, index=avg_df.index),
            )
            output_columns[f'{label}_err'] = err_df.get(
                err_col,
                pd.Series(np.nan, index=avg_df.index),
            )

        metric_df = pd.DataFrame(output_columns)
        metric_path = pixel_dir / metric_filenames[metric]
        metric_df.to_csv(metric_path, index=False)

def adjust_brightness(color, brightness):
    rgb = np.array(mcolors.to_rgb(color))
    adjusted = np.clip(rgb * brightness + (1 - brightness), 0, 1)
    return mcolors.to_hex(adjusted)

def format_temp_range(lower, upper):
    lower_str = f'{int(lower)}' if lower is not None else ''
    upper_str = f'{int(upper)}' if upper is not None else ''
    if lower is None and upper is not None:
        return f'T < {upper_str} °C'
    if lower is not None and upper is None:
        return f'T ≥ {lower_str} °C'
    return f'{lower_str} °C ≤ T < {upper_str} °C'

PLOT_ROOT = EXPORT_ROOT / 'plot'
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

PIXEL_COLOR_MAP = {
    'px1': '#1f77b4',
    'px2': '#ff7f0e',
    'px3': '#2ca02c',
    'px4': '#d62728',
    'px5': '#9467bd',
    'px6': '#8c564b',
}

plot_temperature_ranges = select_temperature_ranges(
    TEMPERATURE_RANGES,
    PLOT_TEMPERATURE_RANGE_LABELS,
)
if not plot_temperature_ranges:
    raise ValueError('No temperature ranges selected for plotting.')

brightness_levels = np.linspace(1.0, 0.5, len(plot_temperature_ranges))
HIGHLIGHT_RANGE = (36.5, 44.5)
HIGHLIGHT_COLOR = '#ffcccc'
HIGHLIGHT_ALPHA = 0.35

for pixel_name, avg_df in px_cycle_avg_by_temp.items():
    err_df = px_cycle_err_by_temp[pixel_name]
    cycle_numbers = avg_df['Cycle Number']
    for metric, alias in cycle_metrics.items():
        base_color = PIXEL_COLOR_MAP.get(pixel_name, '#000000')
        fig, ax = plt.subplots()
        has_data = False
        for brightness, (label, (lower, upper)) in zip(brightness_levels, plot_temperature_ranges):
            mean_col = f'{alias}_{label}'
            err_col = f'{alias}_err_{label}'
            if mean_col not in avg_df.columns:
                continue
            y_values = avg_df[mean_col]
            if y_values.isna().all():
                continue
            y_errors = err_df.get(err_col, pd.Series(np.nan, index=avg_df.index)).fillna(0)
            color = adjust_brightness(base_color, brightness)
            ax.errorbar(
                cycle_numbers,
                y_values,
                yerr=y_errors,
                marker='o',
                markersize=3,
                linestyle='-',
                color=color,
                capsize=3,
                label=format_temp_range(lower, upper)
            )
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.axvspan(
            HIGHLIGHT_RANGE[0],
            HIGHLIGHT_RANGE[1],
            color=HIGHLIGHT_COLOR,
            alpha=HIGHLIGHT_ALPHA,
            zorder=0,
        )

        metric_label = metric.replace('[', '').replace(']', '').replace('/', '_').replace(' ', '_')
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel(metric)
        ax.set_title(f'{pixel_name.upper()} {metric} vs Cycle Number by Temperature Range')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(title='Temperature Range')
        ylim = METRIC_YLIMS.get(metric)
        if ylim is not None:
            ax.set_ylim(*ylim)
        plt.tight_layout()

        if metric == 'PCE [%]':
            figure_name = f'{pixel_name}_PCE.png'
        else:
            figure_name = f'{pixel_name}_{metric_label}.png'
        figure_path = PLOT_ROOT / figure_name
        #plt.show()
        fig.savefig(figure_path, dpi=300)
        plt.close(fig)
