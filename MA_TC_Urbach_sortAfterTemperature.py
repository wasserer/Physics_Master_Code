import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

TEMPERATURE_RANGES = [
    ('lt20', (None, 20)),
    ('20_40', (20, 40)),
    ('40_50', (40, 50)),
    ('50_55', (50, 55)),
    ('55_60', (55, 60)),
    ('60_65', (60, 65)),
    ('65_70', (65, 70)),
    ('70_80', (70, 80)),
]

URBACH_FILE = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/urbach_metrics_test.csv'
RESULT_FOLDER = Path('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/Average_urbach_different_temperature')
CYCLE_COLUMN = 'cycle'
BASE_COLOR = '#2ca02c'
HIGHLIGHT_RANGE = (36.5, 44.5)
HIGHLIGHT_COLOR = '#ffcccc'
HIGHLIGHT_ALPHA = 0.35
DISPLAY_LABELS = {'70_80'}
DISPLAY_RANGES = [item for item in TEMPERATURE_RANGES if item[0] in DISPLAY_LABELS]
BRIGHTNESS_LEVELS = np.linspace(1.0, 0.5, len(DISPLAY_RANGES))

def calculate_urbach_stats_by_temperature(df, temperature_ranges, cycle_col):
    mean_columns = []
    err_columns = []
    combined_mean = None
    combined_err = None

    for label, (lower, upper) in temperature_ranges:
        temp_series = df['temperature']
        mask = pd.Series(True, index=df.index)
        if lower is not None:
            mask &= temp_series >= lower
        if upper is not None:
            mask &= temp_series < upper

        range_df = df.loc[mask]
        if range_df.empty:
            range_mean = pd.DataFrame(columns=['urbach_energy'])
            range_err = pd.DataFrame(columns=['urbach_energy'])
            range_mean.index.name = cycle_col
            range_err.index.name = cycle_col
        else:
            stats_mean = []
            stats_err = []
            for cycle_value, cycle_df in range_df.groupby(cycle_col):
                values = cycle_df['urbach_energy'].dropna()
                if values.empty:
                    continue
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                filtered = values[(values >= lower_bound) & (values <= upper_bound)]
                if filtered.empty:
                    filtered = values
                mean_val = filtered.mean()
                count_val = filtered.count()
                std_val = filtered.std(ddof=1)
                if count_val <= 1:
                    err_val = 0.0
                elif std_val is not None:
                    err_val = std_val / np.sqrt(count_val)
                else:
                    err_val = np.nan
                stats_mean.append((cycle_value, mean_val))
                stats_err.append((cycle_value, err_val))

            if stats_mean:
                range_mean = pd.DataFrame(stats_mean, columns=[cycle_col, 'urbach_energy']).set_index(cycle_col)
                range_err = pd.DataFrame(stats_err, columns=[cycle_col, 'urbach_energy']).set_index(cycle_col)
            else:
                range_mean = pd.DataFrame(columns=['urbach_energy'])
                range_err = pd.DataFrame(columns=['urbach_energy'])
                range_mean.index.name = cycle_col
                range_err.index.name = cycle_col

        mean_col_name = f'urbach_energy_{label}'
        err_col_name = f'urbach_energy_err_{label}'
        mean_columns.append(mean_col_name)
        err_columns.append(err_col_name)
        range_mean = range_mean.rename(columns={'urbach_energy': mean_col_name})
        range_err = range_err.rename(columns={'urbach_energy': err_col_name})

        if combined_mean is None:
            combined_mean = range_mean
            combined_err = range_err
        else:
            combined_mean = combined_mean.join(range_mean, how='outer')
            combined_err = combined_err.join(range_err, how='outer')

    if combined_mean is None:
        combined_mean = pd.DataFrame(columns=mean_columns)
        combined_err = pd.DataFrame(columns=err_columns)
        combined_mean.index = pd.Index([], name=cycle_col)
        combined_err.index = pd.Index([], name=cycle_col)
    else:
        combined_mean = combined_mean.reindex(columns=mean_columns)
        combined_err = combined_err.reindex(columns=err_columns)

    combined_mean = combined_mean.sort_index().reset_index()
    combined_err = combined_err.sort_index().reset_index()
    combined_mean = combined_mean.rename(columns={cycle_col: 'Cycle Number'})
    combined_err = combined_err.rename(columns={cycle_col: 'Cycle Number'})
    return combined_mean, combined_err

def adjust_brightness(color, brightness):
    rgb = np.array(mcolors.to_rgb(color))
    adjusted = np.clip(rgb * brightness + (1 - brightness), 0, 1)
    return mcolors.to_hex(adjusted)

def format_temperature_label(lower, upper):
    lower_str = f'{int(lower)}' if lower is not None else ''
    upper_str = f'{int(upper)}' if upper is not None else ''
    if lower is None and upper is not None:
        return f'T < {upper_str} °C'
    if lower is not None and upper is None:
        return f'T ≥ {lower_str} °C'
    return f'{lower_str} °C ≤ T < {upper_str} °C'

RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

urbach_df = pd.read_csv(URBACH_FILE)
urbach_mean_df, urbach_err_df = calculate_urbach_stats_by_temperature(
    urbach_df,
    TEMPERATURE_RANGES,
    CYCLE_COLUMN,
)

export_columns = {'Cycle Number': urbach_mean_df['Cycle Number']}
for label, _ in TEMPERATURE_RANGES:
    export_columns[f'{label}_avg'] = urbach_mean_df[f'urbach_energy_{label}']
    export_columns[f'{label}_err'] = urbach_err_df[f'urbach_energy_err_{label}']

export_df = pd.DataFrame(export_columns)
csv_path = RESULT_FOLDER / 'urbach_energy_temperature_ranges.csv'
export_df.to_csv(csv_path, index=False)

fig, ax = plt.subplots(figsize = (7, 4), dpi = 300)

for brightness, (label, (lower, upper)) in zip(BRIGHTNESS_LEVELS, DISPLAY_RANGES):
    avg_col = f'urbach_energy_{label}'
    err_col = f'urbach_energy_err_{label}'
    y_values = urbach_mean_df[avg_col]*1000
    if y_values.isna().all():
        continue
    y_errors = urbach_err_df[err_col].fillna(0)*1000*3
    color = adjust_brightness(BASE_COLOR, brightness)
    ax.errorbar(
        urbach_mean_df['Cycle Number'],
        y_values,
        yerr=y_errors,#Adjust the error intensity
        marker='o',
        markersize=3,
        linestyle='-',
        color=color,
        capsize=3,
        label=format_temperature_label(lower, upper),
    )

ax.axvspan(
    HIGHLIGHT_RANGE[0],
    HIGHLIGHT_RANGE[1],
    color=HIGHLIGHT_COLOR,
    alpha=HIGHLIGHT_ALPHA,
    zorder=0,
)

ax.set_xlabel('Cycle Number')
ax.set_ylabel('Urbach Energy [meV]')
#ax.set_title('Urbach Energy vs Cycle Number by Temperature Range')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(title='Temperature Range')
plt.tight_layout()

plot_path = RESULT_FOLDER / 'urbach_energy_temperature_ranges.png'
fig.savefig(plot_path, dpi=300)
plt.close(fig)
