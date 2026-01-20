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

BANDGAP_FILE = ""  # Enter your file path/folder path in this place
RESULT_FOLDER = Path("")  # Enter your file path/folder path in this place
CYCLE_COLUMN = 'cycle'
BASE_COLOR = '#d62728'
BRIGHTNESS_LEVELS = np.linspace(1.0, 0.5, len(TEMPERATURE_RANGES))
HIGHLIGHT_RANGE = (36.5, 44.5)
HIGHLIGHT_COLOR = '#ffcccc'
HIGHLIGHT_ALPHA = 0.35

def calculate_bandgap_stats_by_temperature(df, temperature_ranges, cycle_col):
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
            range_mean = pd.DataFrame(columns=['band_gap'])
            range_err = pd.DataFrame(columns=['band_gap'])
            range_mean.index.name = cycle_col
            range_err.index.name = cycle_col
        else:
            stats_mean = []
            stats_err = []
            for cycle_value, cycle_df in range_df.groupby(cycle_col):
                values = cycle_df['band_gap'].dropna()
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
                range_mean = pd.DataFrame(stats_mean, columns=[cycle_col, 'band_gap']).set_index(cycle_col)
                range_err = pd.DataFrame(stats_err, columns=[cycle_col, 'band_gap']).set_index(cycle_col)
            else:
                range_mean = pd.DataFrame(columns=['band_gap'])
                range_err = pd.DataFrame(columns=['band_gap'])
                range_mean.index.name = cycle_col
                range_err.index.name = cycle_col

        mean_col_name = f'band_gap_{label}'
        err_col_name = f'band_gap_err_{label}'
        mean_columns.append(mean_col_name)
        err_columns.append(err_col_name)
        range_mean = range_mean.rename(columns={'band_gap': mean_col_name})
        range_err = range_err.rename(columns={'band_gap': err_col_name})

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

bandgap_df = pd.read_csv(BANDGAP_FILE)
bandgap_mean_df, bandgap_err_df = calculate_bandgap_stats_by_temperature(
    bandgap_df,
    TEMPERATURE_RANGES,
    CYCLE_COLUMN,
)

export_columns = {'Cycle Number': bandgap_mean_df['Cycle Number']}
for label, _ in TEMPERATURE_RANGES:
    export_columns[f'{label}_avg'] = bandgap_mean_df[f'band_gap_{label}']
    export_columns[f'{label}_err'] = bandgap_err_df[f'band_gap_err_{label}']

export_df = pd.DataFrame(export_columns)
csv_path = RESULT_FOLDER / 'band_gap_temperature_ranges.csv'
export_df.to_csv(csv_path, index=False)

fig, ax = plt.subplots(figsize= (7, 4), dpi = 300)

for brightness, (label, (lower, upper)) in zip(BRIGHTNESS_LEVELS, TEMPERATURE_RANGES):
    if label in {'lt20', '20_40', '40_50', '50_55', '55_60', '60_65', '65_70'}:
        continue
    avg_col = f'band_gap_{label}'
    err_col = f'band_gap_err_{label}'
    y_values = bandgap_mean_df[avg_col]
    if y_values.isna().all():
        continue
    y_errors = bandgap_err_df[err_col].fillna(0)
    color = adjust_brightness(BASE_COLOR, brightness)
    ax.errorbar(
        bandgap_mean_df['Cycle Number'],
        y_values,
        yerr=y_errors*1.5*1.5, #Adjust the error intensity
        marker='o',
        markersize=3,
        linestyle='-',
        color=color,
        capsize=3,
        label=format_temperature_label(lower, upper),
    )

highlight_start, highlight_end = HIGHLIGHT_RANGE
# Soften the cycles of interest to make the anomaly band easy to spot.

ax.axvspan(
    highlight_start,
    highlight_end,
    color=HIGHLIGHT_COLOR,
    alpha=HIGHLIGHT_ALPHA,
    zorder=0,
)

ax.set_xlabel('Cycle Number')
ax.set_ylabel('Band Gap [eV]')
#ax.set_title('Band Gap vs Cycle Number at high Range')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(title='Temperature Range')
plt.tight_layout()

plot_path = RESULT_FOLDER / 'band_gap_temperature_ranges.png'
fig.savefig(plot_path, dpi=300)
plt.close(fig)
