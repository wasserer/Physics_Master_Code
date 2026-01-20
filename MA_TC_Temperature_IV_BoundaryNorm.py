import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

from plot_module.solsim_analyzer import solarSimulator

# MA_TC_Temperature_IV.py
folderPath_px6 = Path("")  # Enter your file path/folder path in this place
#folderPath_px6 = Path("")  # Enter your file path/folder path in this place
analyzer = solarSimulator(folderPath=folderPath_px6)
analyzer.loadFolderData_Cycling()

currents_smoothed = []
window = 9
for i in range(0, len(analyzer.PCE)):
    currents_smoothed.append(np.convolve(analyzer.currents[i], np.ones(window) / window, mode='same'))

temperatures = np.asarray(analyzer.temperature, dtype=float)
threshold_temp = 50
mask = temperatures > threshold_temp
filtered_indices = np.where(mask)[0]

if not np.any(mask):
    raise ValueError(f"No temperatures found above {threshold_temp} °C")

# Use BoundaryNorm to keep distinct bins for sparsely sampled temperatures
temperature_boundaries = [-30, -5, 10, 30, 45, 52, 58, 64, 70, 80]
cmap = plt.cm.get_cmap("jet")
color_norm = BoundaryNorm(temperature_boundaries, ncolors=cmap.N, extend="both")
colors = cmap(color_norm(temperatures))

# Persist IV metrics for the measurements used in the plot.
log_rows = []
for idx in filtered_indices:
    log_rows.append({
        "Label": analyzer.labels[idx],
        "Temperature [°C]": analyzer.temperature[idx],
        "Isc [mA/cm^2]": analyzer.Isc[idx],
        "Voc [V]": analyzer.Voc[idx],
        "FF": analyzer.FF[idx],
        "PCE [%]": analyzer.PCE[idx],
        "Rs [Ohm]": analyzer.Rs[idx],
        "Rp [Ohm]": analyzer.Rp[idx],
    })

if log_rows:
    log_path = folderPath_px6 / "iv_metrics_over_50C.csv"
    with log_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Logged filtered IV metrics to {log_path}")

fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

# Plot every IV curve with colors defined by the boundary-based bins.
for idx, color in enumerate(colors):
    ax.plot(
        -analyzer.voltages[idx],
        currents_smoothed[idx],
        color=color,
        alpha=0.85
    )
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 6)
ax.set_xlabel("Voltage [V]")
ax.set_ylabel("Current Density [mA/cm$^2$]")
cbar_norm = plt.Normalize(vmin=temperature_boundaries[0], vmax=temperature_boundaries[-1])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=cbar_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="Temperature [°C]")
pretty_ticks = np.arange(-20, 100, 20)
cbar.set_ticks(pretty_ticks[(pretty_ticks >= temperatures.min()) & (pretty_ticks <= temperatures.max())])
save_path = ""  # Enter your file path/folder path in this place
fig.savefig(save_path, dpi=300, bbox_inches="tight")
