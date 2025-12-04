import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folderPath = Path('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/pixel')
result_Path = Path('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/Result_FirstCycles')
temperature = Path('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/temperature_log.csv')

PIXEL_COLOR_MAP = {
    'px1': '#1f77b4',
    'px2': '#ff7f0e',
    'px3': '#2ca02c',
    'px4': '#d62728',
    'px5': '#9467bd',
    'px6': '#8c564b',
}

PIXEL_FILES = {
    f'px{i}': folderPath / f'px{i}_result.csv'
    for i in range(1, 7)
}

PARAMETERS_TO_PLOT = [
    'Isc[mA/cm2]',
    'Voc [V]',
    'I_MPP [mA/cm2]',
    'V_MPP [V]',
    'FF',
    'PCE [%]',
    'Rs',
    'Rp'
]

PARAMETER_YLIMS = {
    'FF': (0.4, 1.0),
    'Isc[mA/cm2]': (3, 7),
    'Voc [V]': (0.5, 1.1),
    'PCE [%]': (5, 20),
}


def sanitize_filename(parameter_name: str) -> str:
    return (
        parameter_name.lower()
        .replace('[', '')
        .replace(']', '')
        .replace('%', 'pct')
        .replace('/', '_')
        .replace(' ', '_')
    )


def load_pixel_dataframes() -> dict:
    pixel_dataframes = {}
    for pixel, csv_path in PIXEL_FILES.items():
        df = pd.read_csv(csv_path)
        if 'Cycle Number' in df.columns:
            df = df[df['Cycle Number'] <= 4].copy()
        else:
            raise KeyError(f"'Cycle Number' column missing in {csv_path}")
        if 'Timestamp[s]' in df.columns:
            df['time_hours'] = df['Timestamp[s]'] / 3600.0
        elif 'Timestamp' in df.columns:
            timestamps = pd.to_datetime(df['Timestamp'])
            df['time_hours'] = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0
        else:
            raise KeyError(f"'Timestamp[s]' column missing in {csv_path}")
        pixel_dataframes[pixel] = df
    return pixel_dataframes


def get_time_bounds(pixel_dataframes: dict) -> tuple[float, float]:
    time_min = np.inf
    time_max = -np.inf
    for df in pixel_dataframes.values():
        if df.empty or 'time_hours' not in df.columns:
            continue
        time_min = min(time_min, df['time_hours'].min())
        time_max = max(time_max, df['time_hours'].max())

    if not np.isfinite(time_min):
        time_min = 0.0
    if not np.isfinite(time_max):
        time_max = time_min + 1.0
    if time_min == time_max:
        time_max = time_min + 1e-3
    return time_min, time_max


def load_temperature_log() -> pd.DataFrame:
    df = pd.read_csv(temperature, parse_dates=['Timestamp'])
    if 'Elapsed_s' in df.columns:
        df['elapsed_hours'] = df['Elapsed_s'] / 3600.0
    else:
        # fallback using Timestamp origin
        df['elapsed_hours'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600.0
    return df


def plot_parameters(pixel_dataframes: dict, temperature_df: pd.DataFrame) -> None:
    result_Path.mkdir(parents=True, exist_ok=True)
    temp_hours = temperature_df['elapsed_hours']
    temp_values = temperature_df['Temp_A_C']
    time_min, time_max = get_time_bounds(pixel_dataframes)

    for parameter in PARAMETERS_TO_PLOT:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        for pixel, df in pixel_dataframes.items():
            if parameter not in df.columns:
                continue
            ax.scatter(
                df['time_hours'],
                df[parameter],
                label=pixel,
                color=PIXEL_COLOR_MAP.get(pixel, '#333333'),
                s=15,
                edgecolors='none',
            )

        ax.set_xlabel('Time [h]')
        ax.set_ylabel(parameter)
        pad = 0.02 * (time_max - time_min)
        ax.set_xlim(time_min - pad, time_max + pad)
        if parameter in PARAMETER_YLIMS:
            ax.set_ylim(PARAMETER_YLIMS[parameter])
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(temp_hours, temp_values, color='black', alpha=0.5, label='Temperature')
        ax2.set_ylabel('Temperature [°C]')
        ax2.set_ylim(-80, 115)

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles2:
            handles = handles1 + handles2
            labels = labels1 + labels2
        else:
            handles, labels = handles1, labels1
        if handles:
            ax.legend(handles, labels, loc='upper right', fontsize='small', ncol=2)

        fig.tight_layout()
        output_file = result_Path / f"{sanitize_filename(parameter)}_vs_time.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)


def main():
    pixel_dfs = load_pixel_dataframes()
    temperature_df = load_temperature_log()
    plot_parameters(pixel_dfs, temperature_df)


if __name__ == '__main__':
    main()
