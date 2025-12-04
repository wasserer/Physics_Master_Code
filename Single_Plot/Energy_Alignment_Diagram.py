import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Define layers and their energy levels (HOMO/VBM and LUMO/CBM or work function)
# Each tuple: (name, lower_level_eV, upper_level_eV)
layers = [
    ("ITO", -4.7, -4.7),  # Work function only
    ("PTAA", -5.2, -2.4),
    ("FAPbI3", -5.5, -3.9),
    ("C60", -6.1, -4.0),
    ("Ag", -4.7, -4.7)   # Work function only
]

# Nice, consistent styling
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("seaborn-whitegrid")

fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True, dpi = 300)

# Colors by role
colors = {
    "ITO": "#c0c0c0",      # electrode (silver-like)
    "PTAA": "#e74c3c",     # HTL (red-like)
    "FAPbI3": "#7f8c8d",   # absorber (gray)
    "C60": "#1f77b4",      # ETL (blue)
    "Ag": "#c0c0c0"        # electrode (silver-like)
}

bar_width = 0.6

# Compute dynamic y-limits with margin
vals = [v for _, lo, hi in layers for v in (lo, hi)]
ymin = min(vals) - 0.4
# Ensure 0 eV fits within the visible range when using normal axis order
ymax = max(0, max(vals)) + 0.4

# Draw bars and WF markers
for i, (name, low, high) in enumerate(layers):
    facecolor = colors.get(name, "#87ceeb")
    if low == high:
        # Electrode work function: draw black edge underlay then silver overlay for visible edge
        if name in ("ITO", "Ag"):
            ax.hlines(low, i - bar_width/2, i + bar_width/2,
                      colors="black", linewidth=5, zorder=3)
            ax.hlines(low, i - bar_width/2, i + bar_width/2,
                      colors=facecolor, linewidth=3, zorder=4)
            ax.plot(
                i,
                low,
                marker="o",
                markersize=5,
                markerfacecolor=facecolor,
                markeredgecolor="black",
                markeredgewidth=1.0,
                zorder=5,
            )
        else:
            # fallback: colored line only
            ax.hlines(low, i - bar_width/2, i + bar_width/2,
                      colors=facecolor, linewidth=4, zorder=3)
            ax.plot(i, low, marker="o", color=facecolor, markersize=5, zorder=4)
        # Normal axis: subtract offset to place label below the line (outside)
        ax.text(
            i,
            low - 0.22,
            f"{low:.1f} eV",
            ha="center",
            va="top",
            fontsize=10,
            color="#2c3e50",
            fontweight="bold",
            zorder=6,
        )
    else:
        # Band gap as a rounded rectangle bar
        ax.bar(
            i,
            high - low,
            bottom=low,
            width=bar_width,
            color=facecolor,
            edgecolor="#2c3e50",
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )
        # Numeric labels outside the bar: lower label below (visually), upper label above (visually)
        ax.text(
            i,
            low - 0.22,
            f"{low:.1f} eV",
            ha="center",
            va="top",
            fontsize=10,
            color="#2c3e50",
            fontweight="bold",
            zorder=4,
        )
        ax.text(
            i,
            high + 0.16,
            f"{high:.1f} eV",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2c3e50",
            fontweight="bold",
            zorder=4,
        )

# X tick labels for layer names at the bottom
ax.set_xticks(range(len(layers)))
ax.set_xticklabels([name for name, *_ in layers], fontsize=11)
ax.tick_params(axis='x', pad=6)

"""Arrows removed as requested"""

"""Removed BCP note text per request"""

# Formatting: fixed y-limits and vacuum reference
ax.set_ylim(-7, -1.8)
ax.set_xlim(-0.6, len(layers) - 0.4)
ax.set_ylabel("Energy (eV)", fontsize=12)
ax.tick_params(axis="y", labelleft=False)

# Vacuum level reference
ax.axhline(0, color="#95a5a6", linewidth=1.2, linestyle="--", zorder=1)

# Normal y-axis order (0 above -1): do not invert

# Frame: show all spines for a clear border
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.2)
    ax.spines[side].set_color("black")
ax.grid(axis="y", which="major", linestyle=":", color="#bdc3c7")
ax.grid(axis="x", which="major", alpha=0.0)

# No title per request

# Save high-resolution outputs
plt.savefig("energy_alignment.png", dpi=300, bbox_inches="tight")
plt.savefig("energy_alignment.svg", bbox_inches="tight")
