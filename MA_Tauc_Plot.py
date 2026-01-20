import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


tauc_file = Path(
    ""  # Enter your file path/folder path in this place
    'ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/tauc_metrics.csv'
)
temperature_file = Path(
    ""  # Enter your file path/folder path in this place
    'ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/temperature_log.csv'
)


required_paths = {
    'tauc metrics': tauc_file,
    'temperature log': temperature_file,
}

for label, filepath in required_paths.items():
    if not filepath.is_file():
        raise FileNotFoundError(f"Missing {label} CSV: {filepath}")


tauc_df = pd.read_csv(tauc_file)
temperature_df = pd.read_csv(temperature_file, usecols=['Elapsed_s', 'Temp_A_C'])
temperature_df.rename(columns={'Elapsed_s': 'elapsed_seconds', 'Temp_A_C': 'temperature_c'}, inplace=True)

if 'band_gap' not in tauc_df.columns:
    raise KeyError("Column 'band_gap' missing from Tauc dataframe.")


# Infer time axis in hours from the Tauc metrics
_time_series = None
for column in tauc_df.columns:
    if 'time' in column.lower():
        _time_series = tauc_df[column]
        break

if _time_series is None:
    raise KeyError("No time-related column found in Tauc dataframe for plotting.")

_time_hours = pd.to_numeric(_time_series, errors='coerce')
if not np.isfinite(_time_hours.to_numpy(dtype=float)).any():
    parsed_time = pd.to_datetime(_time_series, errors='coerce')
    if not parsed_time.notna().any():
        raise ValueError(
            "Could not interpret time column as numeric hours or datetimes."
        )
    origin = parsed_time.min()
    _time_hours = (parsed_time - origin).dt.total_seconds() / 3600.0

time_hours = np.asarray(_time_hours, dtype=float)
band_gap = pd.to_numeric(tauc_df['band_gap'], errors='coerce').to_numpy(dtype=float)

valid_tauc = np.isfinite(time_hours) & np.isfinite(band_gap)
if not np.any(valid_tauc):
    raise ValueError('No valid Tauc points after cleaning time axis and band gap.')

temp_hours = pd.to_numeric(temperature_df['elapsed_seconds'], errors='coerce').to_numpy(dtype=float) / 3600.0
temp_values = pd.to_numeric(temperature_df['temperature_c'], errors='coerce').to_numpy(dtype=float)

valid_temp = np.isfinite(temp_hours) & np.isfinite(temp_values)
if not np.any(valid_temp):
    raise ValueError('No valid temperature points after cleaning elapsed seconds and temperature.')


if __name__ == '__main__':
    print(f"Loaded Tauc metrics: {len(tauc_df)} rows")
    print(f"Loaded chamber temperature log: {len(temperature_df)} rows")
    print(f"Tauc entries kept for plotting: {valid_tauc.sum()} rows")

    result_dir = tauc_file.parent / 'Result'
    result_dir.mkdir(parents=True, exist_ok=True)

    fig, ax_band = plt.subplots(figsize=(10, 6))
    ax_temp = ax_band.twinx()

    ax_band.set_zorder(2)
    ax_temp.set_zorder(1)
    ax_band.patch.set_alpha(0)

    # Band gap points on primary axis
    band_points = ax_band.scatter(
        time_hours[valid_tauc],
        band_gap[valid_tauc],
        s=25,
        color='#1f77b4',
        label='Band Gap (eV)',
        zorder=3,
    )
    
    ax_band.set_xlabel('Time (hours)')
    ax_band.set_ylabel('Band Gap (eV)')
    ax_band.set_ylim(1.5, 1.6)
    #ax_band.set_xlim(70, 80)
    ax_band.grid(True, linestyle='-', linewidth=0.5, alpha=0.6, zorder=0)
    # Temperature curve on secondary axis
    temp_line, = ax_temp.plot(
        temp_hours[valid_temp],
        temp_values[valid_temp],
        color='#888888',
        linewidth=1.5,
        label='Chamber Temp (°C)',
        zorder=1,
    )
    ax_temp.set_ylabel('Temperature (°C)')
    ax_temp.patch.set_alpha(0)

    handles = [band_points, temp_line]
    labels = [handle.get_label() for handle in handles]
    legend = ax_band.legend(handles, labels, loc='best', frameon=True)
    legend.set_zorder(10)

    output_path = result_dir / 'tauc_band_gap_temperature.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.show()
