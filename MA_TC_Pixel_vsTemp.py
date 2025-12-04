import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PIXEL_DIR = Path(
    "/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/"
    "Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/"
    "TC_1909_IV/TC_log/pixel"
)

PIXEL_FILES = {f"px{idx}": PIXEL_DIR / f"px{idx}_result.csv" for idx in range(1, 7)}

SELECTED_CYCLES = [1, 10, 20, 30, 40, 50]
COLORS = ["firebrick", "darkorange", "gold", "mediumseagreen", "deepskyblue", "midnightblue"]
METRICS = {
    "PCE [%]": {"ylabel": "PCE [%]", "title": "PCE vs Temperature", "fname": "pce"},
    "Voc [V]": {"ylabel": "Voc [V]", "title": "Voc vs Temperature", "fname": "voc"},
    "Isc[mA/cm2]": {"ylabel": "Isc [mA/cm2]", "title": "Isc vs Temperature", "fname": "isc"},
    "FF": {"ylabel": "Fill Factor", "title": "FF vs Temperature", "fname": "ff"},
}

CYCLE_COLORS = dict(zip(SELECTED_CYCLES, COLORS))
SAVE_DIR = Path(
    "/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/"
    "Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/"
    "TC_1909_IV/TC_log/Result_Average_vs_Temperature"
)


def load_pixel_data(pixel_files, selected_cycles):
    """Read pixel CSV logs and keep only the requested thermal cycles."""
    pixel_data = {}
    for pixel, file_path in pixel_files.items():
        df = pd.read_csv(file_path)
        filtered_df = df[df["Cycle Number"].isin(selected_cycles)].copy()
        filtered_df["Cycle Number"] = filtered_df["Cycle Number"].astype(int)
        pixel_data[pixel] = filtered_df
    return pixel_data


def plot_metric(pixel_data, metric, metric_meta):
    """Scatter plot a metric versus temperature, color-coded by thermal cycle."""
    for pixel, df in pixel_data.items():
        plt.figure(figsize=(7, 5))
        for cycle, cycle_df in df.groupby("Cycle Number"):
            cycle_label = int(cycle)
            plt.scatter(
                cycle_df["Temperature [C]"],
                cycle_df[metric],
                color=CYCLE_COLORS.get(cycle_label, "gray"),
                label=f"Cycle {cycle_label}",
                alpha=0.8,
            )
        plt.title(f"{pixel.upper()} {metric_meta['title']}")
        plt.xlabel("Temperature [C]")
        plt.ylabel(metric_meta["ylabel"])
        plt.legend(title="Cycle")
        plt.tight_layout()
        filename = f"{pixel}_{metric_meta['fname']}_vs_temp.png"
        plt.savefig(SAVE_DIR / filename, dpi=300)
        plt.close()


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    pixel_data = load_pixel_data(PIXEL_FILES, SELECTED_CYCLES)
    for metric, meta in METRICS.items():
        plot_metric(pixel_data, metric, meta)
