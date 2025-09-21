from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_module.Spectra import Spectroscopy
from plot_module.solsim_analyzer import solarSimulator
from plot_module.tc_analyzer import ThermalCycling


# Absolute path to the thermal cycling data for this analysis.
TC_FOLDER = Path(
    '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TC_1709_50Cycle/TC_1909_IV'
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

    if times.size == 0 or bandgap.size == 0:
        return

    fig, ax = plt.subplots()
    ax.scatter(times, bandgap)
    ax.set_ylim(1.4, 1.6)
    ax.set_xlabel("Absolute Time (h)")
    ax.set_ylabel("Band Gap (eV)")
    ax.set_title("Band Gap Evolution")
    fig.savefig(result_dir / "band_gap_vs_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_tauc_plots(
    spectra: Spectroscopy,
    result_dir: Path,
    max_plots: int = 1000,
) -> None:
    """Write Tauc plot PNGs for at most max_plots spectra."""
    number = max(max_plots, len(spectra.valueTaucs))
    energy_axis = getattr(spectra, "eV", None)
    if energy_axis is None:
        if getattr(spectra, "wavelengths_interp", None):
            energy_axis = 1240 / np.asarray(spectra.wavelengths_interp[0], dtype=float)
            setattr(spectra, "eV", energy_axis)
        else:
            raise AttributeError("Spectroscopy object has no 'eV' axis; run Tauc calculation first.")
    energy_axis = np.asarray(energy_axis, dtype=float)
    window_bounds = getattr(spectra, "tauc_window_used", None)
    qc_flags = getattr(spectra, "tauc_qc_pass", None)
    window_label_added = False
    for idx in range(number):
        timestamp = getattr(spectra, "timestamps", [None])
        stamp = timestamp[idx] if idx < len(timestamp) else None
        stamp_str = stamp.strftime("%Y-%m-%d %H:%M:%S") if stamp is not None else "Unknown time"
        fig, ax = plt.subplots()
        ax.plot(
            energy_axis,
            spectra.valueTaucs[idx],
            label=f"Tauc Plot at temperature {spectra.temperatures[idx]:.1f} °C",
            color="#1f77b4",
        )
        ax.plot(
            energy_axis,
            spectra.tauc_slope_b[idx] + spectra.tauc_slope[idx] * energy_axis,
            label=f"Fit Line, band gap = {spectra.bandGap[idx]:.3f} eV",
            color="#ff7f0e",
        )

        if window_bounds and idx < len(window_bounds):
            use_bounds = True
            if qc_flags and idx < len(qc_flags):
                use_bounds = bool(qc_flags[idx])
            if use_bounds:
                window_start, window_end = window_bounds[idx]
                if np.isfinite(window_start) and np.isfinite(window_end):
                    label = None if window_label_added else "Selected window"
                    ax.axvline(window_start, color="#2ca02c", linestyle="--", linewidth=1.2, label=label)
                    ax.axvline(window_end, color="#2ca02c", linestyle="--", linewidth=1.2)
                    if label is not None:
                        window_label_added = True
        ax.set_xlim(1.45, 1.8)
        ax.set_ylim(0, 2.5) 
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Tauc Plot (Absorbance * photon energy)^0.5")
        ax.set_title(f"Tauc Plot at {stamp_str}")
        ax.legend()
        ax.grid(True)
        fig.savefig(result_dir / f"tauc_plot_{idx:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def create_absorbance_plots(
    spectra: Spectroscopy,
    result_dir: Path,
    max_plots: int = 100,
) -> None:
    """Write absorbance-vs-wavelength PNGs for at most max_plots spectra."""

    wavelength_sets = getattr(spectra, "wavelengths_interp", None)
    if not wavelength_sets:
        return

    wavelengths = np.asarray(wavelength_sets[0], dtype=float)
    if wavelengths.size == 0:
        return

    light_ref = np.asarray(getattr(spectra, "lightValue_interp", []), dtype=float)
    dark_ref = np.asarray(getattr(spectra, "darkValue_interp", []), dtype=float)
    if light_ref.shape != wavelengths.shape or dark_ref.shape != wavelengths.shape:
        return

    light_calibrated = light_ref - dark_ref
    epsilon = 1e-10
    safe_light = np.where(light_calibrated <= 0, epsilon, light_calibrated)

    averaged_values = getattr(spectra, "values_interpAvr", None)
    if averaged_values:
        spectra_sets = averaged_values
        filenames = getattr(spectra, "fileNameAveraged", [])
        timestamps = np.asarray(getattr(spectra, "timestampAbsHN", []), dtype=float)
        cycles = np.asarray(getattr(spectra, "cycleNum_avg", []), dtype=float)
    else:
        spectra_sets = getattr(spectra, "values_interp", [])
        filenames = getattr(spectra, "fileNameRaw", [])
        timestamps = np.asarray(getattr(spectra, "timestampAbsH", []), dtype=float)
        cycles = np.asarray(getattr(spectra, "cycleNum", []), dtype=float)

    if not spectra_sets:
        return

    limit = min(max_plots, len(spectra_sets))
    for idx in range(limit):
        value_arr = np.asarray(spectra_sets[idx], dtype=float)
        if value_arr.shape != wavelengths.shape:
            continue

        calibrated = value_arr - dark_ref
        transmission = calibrated / safe_light
        absorbance = -np.log10(np.maximum(transmission, epsilon))

        label_parts = []
        if idx < len(filenames):
            label_parts.append(filenames[idx])
        if idx < timestamps.shape[0] and not np.isnan(timestamps[idx]):
            label_parts.append(f"t = {timestamps[idx]:.2f} h")
        if idx < cycles.shape[0] and not np.isnan(cycles[idx]):
            label_parts.append(f"cycle {int(cycles[idx])}")

        title = "Absorbance vs Wavelength"
        if label_parts:
            title = f"{title} ({', '.join(label_parts)})"

        fig, ax = plt.subplots()
        ax.plot(wavelengths, absorbance, color="#2ca02c")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance (a.u.)")
        ax.set_title(title)
        ax.grid(True)
        fig.savefig(
            result_dir / f"absorbance_plot_{idx:03d}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_pixel_metric(
    pixel,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    result_dir: Path,
    color: str,
    cycle_mask=None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
    times = np.asarray(getattr(pixel, "timestampAbsH", []), dtype=float)
    temps = np.asarray(getattr(pixel, "temperature", []), dtype=float)
    values = np.asarray(getattr(pixel, metric), dtype=float)

    if cycle_mask is not None:
        mask = np.asarray(cycle_mask, dtype=bool)
        if mask.shape[0] != values.shape[0]:
            raise ValueError("Cycle mask length must match metric data length.")
        times = times[mask]
        temps = temps[mask]
        values = values[mask]

    if times.size == 0:
        plt.close(fig)
        return

    if metric == "Voc":
        values = -values

    ax1.scatter(times, values, color=color)
    ax1.set_ylabel(ylabel, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xlabel("Absolute Time (h)")
    if metric == "Rp":
        ax1.set_ylim(0, 20)
    else:
        ax1.set_ylim(0, None)

    ax2 = ax1.twinx()
    ax2.plot(times, temps, "#17becf", linestyle="--")
    ax2.set_ylabel("Temperature", color="#17becf")
    ax2.tick_params(axis="y", labelcolor="#17becf")
    ax2.set_ylim(0, None)

    ax1.set_title(title)

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
            title = (
                f"The temperature and the {style['label']} of the pixel {pixel_id} "
                "during the thermal cycling"
            )
            filename = f"{pixel_name.lower()}_{metric.lower()}_vs_temperature.png"
            plot_pixel_metric(
                pixel,
                metric=metric,
                ylabel=style["label"],
                title=title,
                filename=filename,
                result_dir=result_dir,
                color=style["color"],
                cycle_mask=cycle_mask,
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
    create_tauc_plots(spectra, result_dir)  # Enable if you need Tauc plot PNGs
    #create_absorbance_plots(spectra, result_dir)  # Enable if you need absorbance PNGs

    pixels = {
        f"px{idx}": solarSimulator(folderPath=str(TC_FOLDER / f"px{idx}"))
        for idx in range(1, 7)
    }
    for pixel in pixels.values():
        pixel.loadFolderData_Cycling()

    #create_pixel_plots(pixels, result_dir, cycle_selector=cycle_selector)


if __name__ == "__main__":
    main()
