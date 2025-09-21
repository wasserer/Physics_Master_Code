from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from plot_module.Spectra import Spectroscopy
from plot_module.solsim_analyzer import solarSimulator
from plot_module.tc_analyzer import ThermalCycling


# Absolute path to the thermal cycling data for this analysis.
TC_FOLDER = Path(
    "/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/"
    "Pervoskite Space(Master)/Data/ThermalCycling/TEMP_TC_1809"
)

METRIC_STYLES = {
    "Voc": {"color": "#d62728", "label": "Voc"},
    "Isc": {"color": "#1f77b4", "label": "Isc"},
    "PCE": {"color": "#2ca02c", "label": "PCE"},
    "FF": {"color": "#9467bd", "label": "FF"},
    "Rs": {"color": "#8c564b", "label": "Rs (Ohm cm^2)"},
    "Rp": {"color": "#bcbd22", "label": "Rp (Ohm cm^2)"},
}


def resolve_cycle_mask(cycle_numbers, cycle_selector=None):
    """Return a boolean mask selecting cycles according to cycle_selector.

    cycle_selector options:
        - None: no filtering (returns None)
        - int: a single cycle number
        - tuple(start, end): inclusive range; use None for open bounds
        - range / iterable of ints: explicit cycle list
    """

    if cycle_selector is None or cycle_numbers is None:
        return None

    def _normalize_cycle(value):
        if value is None:
            return np.nan
        try:
            if isinstance(value, float) and np.isnan(value):
                return np.nan
            return int(value)
        except (TypeError, ValueError):
            return np.nan

    cycles = np.asarray([_normalize_cycle(val) for val in cycle_numbers], dtype=float)
    if cycles.size == 0:
        return None

    valid_entries = ~np.isnan(cycles)
    if not np.any(valid_entries):
        return None

    selector = cycle_selector

    if isinstance(selector, int):
        return cycles == selector

    if isinstance(selector, range):
        selector = set(selector)

    if isinstance(selector, tuple):
        if len(selector) == 2 and all(
            item is None or isinstance(item, (int, float)) for item in selector
        ):
            start, end = selector
            start = int(start) if start is not None else int(np.nanmin(cycles))
            end = int(end) if end is not None else int(np.nanmax(cycles))
            if start > end:
                start, end = end, start
            return (cycles >= start) & (cycles <= end)
        selector = list(selector)

    if isinstance(selector, (set, list, tuple)):
        if len(selector) == 0:
            return np.zeros_like(cycles, dtype=bool)
        return np.isin(cycles, [int(item) for item in selector])

    raise ValueError(
        "Unsupported cycle_selector type. Provide an int, range, iterable of ints, "
        "or a (start, end) tuple."
    )


def ensure_result_dir(base_dir: Path) -> Path:
    result_dir = base_dir / "Result"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def create_spectra_plots(
    spectra: Spectroscopy,
    result_dir: Path,
    cycle_selector=None,
) -> None:
    times = np.asarray(getattr(spectra, "timestampAbsHN", []), dtype=float)
    bandgap = np.asarray(getattr(spectra, "bandGap", []), dtype=float)
    cycle_numbers = getattr(spectra, "cycleNum_avg", None)
    if cycle_numbers is None:
        cycle_numbers = getattr(spectra, "cycleNum", None)

    target_len = min(times.shape[0], bandgap.shape[0])
    if target_len == 0:
        return
    times = times[:target_len]
    bandgap = bandgap[:target_len]

    if cycle_numbers is not None:
        cycle_numbers = np.asarray(cycle_numbers, dtype=float)
        if cycle_numbers.shape[0] < target_len:
            target_len = cycle_numbers.shape[0]
            times = times[:target_len]
            bandgap = bandgap[:target_len]
            cycle_numbers = cycle_numbers[:target_len]
        elif cycle_numbers.shape[0] > target_len:
            cycle_numbers = cycle_numbers[:target_len]

    mask = resolve_cycle_mask(cycle_numbers, cycle_selector)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != bandgap.shape[0]:
            raise ValueError("Cycle mask length must match spectra band gap data length.")
        times = times[mask]
        bandgap = bandgap[mask]
        if cycle_numbers is not None:
            cycle_numbers = cycle_numbers[mask]

    valid_mask = np.isfinite(times) & np.isfinite(bandgap)
    if cycle_numbers is not None:
        cycle_numbers = np.asarray(cycle_numbers, dtype=float)
        valid_mask &= np.isfinite(cycle_numbers)

    times = times[valid_mask]
    bandgap = bandgap[valid_mask]
    if cycle_numbers is not None:
        cycle_numbers = cycle_numbers[valid_mask]

    if times.size == 0 or bandgap.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    if cycle_numbers is None or cycle_numbers.size == 0:
        rel_times = times - times.min()
        ax.scatter(rel_times, bandgap, color="#1f77b4", s=35)
    else:
        cycle_indices = np.rint(cycle_numbers).astype(int)
        unique_cycles = np.unique(cycle_indices)
        cmap = cm.get_cmap("rainbow")
        if unique_cycles.size == 1:
            color_sequence = [cmap(0.0)]
        else:
            color_sequence = cmap(np.linspace(0.0, 1.0, unique_cycles.size))
        color_sequence = color_sequence[::-1]

        for idx, cycle_id in enumerate(unique_cycles):
            cycle_mask = cycle_indices == cycle_id
            if not np.any(cycle_mask):
                continue
            cycle_times = times[cycle_mask]
            cycle_bandgap = bandgap[cycle_mask]
            sort_order = np.argsort(cycle_times)
            cycle_times = cycle_times[sort_order]
            cycle_bandgap = cycle_bandgap[sort_order]
            rel_times = cycle_times - cycle_times.min()
            cycle_color = color_sequence[idx]
            ax.scatter(rel_times, cycle_bandgap, color=cycle_color, s=30)

    ax.set_xlabel("Time within cycle (h)")
    ax.set_ylabel("Band Gap (eV)")
    ax.set_title("Band Gap Evolution")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(result_dir / "band_gap_vs_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_tauc_plots(
    spectra: Spectroscopy,
    result_dir: Path,
    max_plots: int = 100,
) -> None:
    """Write Tauc plot PNGs for at most max_plots spectra."""
    number = min(max_plots, len(spectra.valueTaucs))
    for idx in range(number):
        timestamp = getattr(spectra, "timestamps", [None])
        stamp = timestamp[idx] if idx < len(timestamp) else None
        stamp_str = stamp.strftime("%Y-%m-%d %H:%M:%S") if stamp is not None else "Unknown time"
        fig, ax = plt.subplots()
        ax.plot(
            spectra.eV,
            spectra.valueTaucs[idx],
            label=f"Tauc Plot at temperature {spectra.temperatures[idx]:.1f} °C",
            color="#1f77b4",
        )
        ax.plot(
            spectra.eV,
            spectra.tauc_slope_b[idx] + spectra.tauc_slope[idx] * spectra.eV,
            label=f"Fit Line, band gap = {spectra.bandGap[idx]:.3f} eV",
            color="#ff7f0e",
        )
        ax.set_xlim(1.45, 2)
        ax.set_ylim(0, 2)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Tauc Plot (Absorbance * photon energy)^0.5")
        ax.set_title(f"Tauc Plot at {stamp_str}")
        ax.legend()
        ax.grid(True)
        fig.savefig(result_dir / f"tauc_plot_{idx:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_pixel_metric(
    pixel,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    result_dir: Path,
    color: str,
    cycle_selector=None,
) -> None:
    """Plot all cycle traces in one panel with time reset per cycle and rainbow colors."""

    times = np.asarray(getattr(pixel, "timestampAbsH", []), dtype=float)
    values = np.asarray(getattr(pixel, metric, []), dtype=float)
    cycle_numbers = getattr(pixel, "cycleNum", None)
    if cycle_numbers is not None:
        cycle_numbers = np.asarray(cycle_numbers, dtype=float)
        if cycle_numbers.size == 0:
            cycle_numbers = None

    if times.size == 0 or values.size == 0:
        return

    arrays = [times, values]
    if cycle_numbers is not None:
        arrays.append(cycle_numbers)

    target_len = min(arr.size for arr in arrays if arr.size > 0)
    times = times[:target_len]
    values = values[:target_len]

    if cycle_numbers is not None:
        cycle_numbers = cycle_numbers[:target_len]

    if cycle_numbers is not None:
        cycle_mask = resolve_cycle_mask(cycle_numbers, cycle_selector)
        if cycle_mask is not None:
            mask = np.asarray(cycle_mask, dtype=bool)
            if mask.shape[0] != times.shape[0]:
                raise ValueError("Cycle mask length must match metric data length.")
            times = times[mask]
            values = values[mask]
            cycle_numbers = cycle_numbers[mask]

    valid_mask = np.isfinite(times) & np.isfinite(values)
    if cycle_numbers is not None:
        valid_mask &= np.isfinite(cycle_numbers)

    times = times[valid_mask]
    values = values[valid_mask]
    if cycle_numbers is not None:
        cycle_numbers = cycle_numbers[valid_mask]

    if times.size == 0 or values.size == 0:
        return

    if metric == "Voc":
        values = -values

    fig, ax_metric = plt.subplots(figsize=(7, 5), dpi=300)

    if cycle_numbers is None or cycle_numbers.size == 0:
        rel_times = times - times.min()
        ax_metric.scatter(rel_times, values, color=color, s=35)
        ax_metric.set_xlabel("Relative Time (h)")
        ax_metric.set_ylabel(ylabel)
        ax_metric.grid(True, alpha=0.3)
        ax_metric.set_xlim(0, 1)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(result_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    cycle_indices = np.rint(cycle_numbers).astype(int)
    unique_cycles = np.unique(cycle_indices)
    if unique_cycles.size == 0:
        return

    cmap = cm.get_cmap("rainbow")
    if unique_cycles.size == 1:
        color_sequence = [cmap(0.0)]
    else:
        color_sequence = cmap(np.linspace(0.0, 1.0, unique_cycles.size))
    color_sequence = color_sequence[::-1]

    for idx, cycle_id in enumerate(unique_cycles):
        cycle_mask = cycle_indices == cycle_id
        if not np.any(cycle_mask):
            continue

        cycle_times = times[cycle_mask]
        cycle_values = values[cycle_mask]
        sort_order = np.argsort(cycle_times)
        cycle_times = cycle_times[sort_order]
        cycle_values = cycle_values[sort_order]
        rel_times = cycle_times - cycle_times.min()

        cycle_color = color_sequence[idx]
        ax_metric.scatter(rel_times, cycle_values, color=cycle_color, s=30)

    ax_metric.set_xlabel("Time within cycle (h)")
    ax_metric.set_ylabel(ylabel)
    ax_metric.grid(True, alpha=0.3)
    ax_metric.set_xlim(0, 1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(result_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_pixel_plots(
    pixels: dict[str, object],
    result_dir: Path,
    cycle_selector=None,
) -> None:
    for pixel_name, pixel in pixels.items():
        pixel_id = pixel_name[2:]
        cycle_mask = resolve_cycle_mask(getattr(pixel, "cycleNum", None), cycle_selector)
        if cycle_mask is not None and cycle_mask.size and not cycle_mask.any():
            continue
        for metric, style in METRIC_STYLES.items():
            if not hasattr(pixel, metric):
                continue
            title = f"{style['label']} of pixel {pixel_id} during thermal cycling"
            filename = f"{pixel_name.lower()}_{metric.lower()}_vs_temperature.png"
            plot_pixel_metric(
                pixel,
                metric=metric,
                ylabel=style["label"],
                title=title,
                filename=filename,
                result_dir=result_dir,
                color=style["color"],
                cycle_selector=cycle_selector,
            )


def main() -> None:
    result_dir = ensure_result_dir(TC_FOLDER)

    tc = ThermalCycling(folderPath=str(TC_FOLDER))
    tc.sortData()

    # Adjust cycle_selector to focus on specific cycles if needed.
    # Examples: None (all cycles), 3 (only cycle 3), (1, 5) (cycles 1 through 5),
    # {2, 4, 6} (explicit set of cycles).
    cycle_selector = None

    spectra_folder = TC_FOLDER / "Spectra"
    spectra = Spectroscopy(folderPath=str(spectra_folder))
    spectra.Pipeline(
        darkFolder=str(
            spectra_folder
            / "Dark"
            / "20250915_155012_Thermal_Cycling_Spectrum_dark_Dark.dat"
        )
    )

    create_spectra_plots(spectra, result_dir, cycle_selector=cycle_selector)
    # create_tauc_plots(spectra, result_dir)  # Enable if you need Tauc plot PNGs

    pixels = {
        f"px{idx}": solarSimulator(folderPath=str(TC_FOLDER / f"px{idx}"))
        for idx in range(1, 7)
    }
    for pixel in pixels.values():
        pixel.loadFolderData_Cycling()

    create_pixel_plots(pixels, result_dir, cycle_selector=cycle_selector)


if __name__ == "__main__":
    main()
