import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

spectra_data = "/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/tauc_metrics_test.csv"
urbach_data = "/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/TC_log/urbach_metrics_test.csv"
output_dir = Path("/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/image2latex")
output_dir.mkdir(parents=True, exist_ok=True)

selected_cycles = [1, 10, 20, 30, 40, 50]

spectra_df = pd.read_csv(spectra_data)
spectra_df = spectra_df[spectra_df["cycle"].isin(selected_cycles)]
urbach_df = pd.read_csv(urbach_data)
urbach_df = urbach_df[urbach_df["cycle"].isin(selected_cycles)]

color = ["firebrick", "darkorange", "gold", "mediumseagreen", "deepskyblue", "midnightblue"]

cycle_colors = dict(zip(selected_cycles, color))

plt.figure(figsize = (7, 5))
for cycle, cycle_df in spectra_df.groupby("cycle"):
    plt.scatter(
        cycle_df["temperature"],
        cycle_df["band_gap"],
        color=cycle_colors.get(cycle, "gray"),
        label=f"Cycle {cycle}",
        alpha = 0.8,
    )

plt.xlabel("Temperature (C)")
plt.ylabel("Band gap (eV)")
plt.legend(title="Cycle", loc="best")
plt.grid()
plt.tight_layout()
#Turn this on if zoom in
plt.xlim(20, 80)
plt.savefig(output_dir / "TC_Spectra_vs_temp.png", dpi=300)
#plt.show()

plt.figure(figsize=(7, 5))
for cycle, cycle_df in urbach_df.groupby("cycle"):
    plt.scatter(
        cycle_df["temperature"],
        cycle_df["urbach_energy"],
        color=cycle_colors.get(cycle, "gray"),
        label=f"Cycle {cycle}",
        alpha=0.8,
    )

plt.xlabel("Temperature (C)")
plt.ylabel("Urbach energy (eV)")
#Turn this on if zoom in
plt.xlim(20, 80)
plt.legend(title="Cycle", loc="best")
plt.grid()
plt.tight_layout()
plt.savefig(output_dir / "TC_Urbach_vs_temp.png", dpi=300)
#plt.show()
