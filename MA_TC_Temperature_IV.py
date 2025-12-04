import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_module.solsim_analyzer import solarSimulator

# MA_TC_Temperature_IV.py
folderPath_px6 = Path('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/Pixel_Temperature_3rd')
#folderPath_px6 = Path('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV/Pixel_Temperature_2nd')
analyzer = solarSimulator(folderPath=folderPath_px6)
analyzer.loadFolderData_Cycling()

currents_smoothed = []
window = 9
for i in range (0, len(analyzer.PCE)):
    currents_smoothed.append(np.convolve(analyzer.currents[i], np.ones(window)/window, mode='same'))

temperatures = np.asarray(analyzer.temperature, dtype=float)
threshold_temp = 50
mask = temperatures > threshold_temp
if not np.any(mask):
    raise ValueError(f"No temperatures found above {threshold_temp} °C")

filtered_temperatures = temperatures[mask]
temp_norm = plt.Normalize(vmin=filtered_temperatures.min(), vmax=filtered_temperatures.max())
cmap = plt.cm.get_cmap("jet")
colors = cmap(temp_norm(filtered_temperatures))
filtered_indices = np.where(mask)[0]

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

# Plot all unselected curves in the background for additional context.
unselected_indices = np.where(~mask)[0]
if len(unselected_indices) > 0:
    alpha_values = np.linspace(0.5, 0.9, len(unselected_indices))
else:
    alpha_values = []
for idx, alpha in zip(unselected_indices, alpha_values):
    temperature = analyzer.temperature[idx]
    ax.plot(
        -analyzer.voltages[idx],
        currents_smoothed[idx],
        color="black",
        alpha=alpha,
        label=f"{temperature}$^\circ$C"
    )

for color, idx in zip(colors, filtered_indices):
    ax.plot(
        -analyzer.voltages[idx],
        #analyzer.currents[i],
        currents_smoothed[idx],
        color=color,
        alpha=0.7
        #s=4,   # smaller circles (default ~36),
    )
ax.set_xlim(0, 0.95)
ax.set_ylim(0, 6)
ax.set_xlabel("Voltage [V]")
ax.set_ylabel("Current Density [mA/cm$^2$]")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=temp_norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Temperature [°C]")
ax.legend()
save_path = "/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/image2latex/IV_3rd_cycle.png"
fig.savefig(save_path, dpi=300, bbox_inches="tight")
