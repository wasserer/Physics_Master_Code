from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_module.Spectra import Spectroscopy
from plot_module.solsim_analyzer_V2 import solarSimulator
from plot_module.tc_analyzer import ThermalCycling


# Absolute path to the thermal cycling data for this analysis.
TC_FOLDER = Path(
    ""  # Enter your file path/folder path in this place
)

TEMPERATURE_LOG_PATH = Path(
    ""  # Enter your file path/folder path in this place
)

PIXEL_RESULT_DIR = Path(
    ""  # Enter your file path/folder path in this place
)

PIXEL_COLOR_CYCLE = ("#d62728", "#1f77b4", "#ff7f0e")

METRIC_STYLES = {
    "Voc": {"color": PIXEL_COLOR_CYCLE[0], "label": "Voc"},
    "Isc": {"color": PIXEL_COLOR_CYCLE[1], "label": "Isc"},
    "PCE": {"color": PIXEL_COLOR_CYCLE[2], "label": "PCE"},
    "FF": {"color": PIXEL_COLOR_CYCLE[0], "label": "FF"},
    "Rs": {"color": PIXEL_COLOR_CYCLE[1], "label": "Rs (Ohm cm^2)"},
    "Rp": {"color": PIXEL_COLOR_CYCLE[2], "label": "Rp (Ohm cm^2)"},
}


class TemperatureLog:
    """Load and interpolate the chamber temperature log."""

    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path, sep=';', usecols=["Timestamp", "Temp_A_C"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["Temp_A_C"] = pd.to_numeric(df["Temp_A_C"], errors="coerce")
        df = df.dropna(subset=["Timestamp", "Temp_A_C"])
        if df.empty:
            raise ValueError("Temperature log file contains no valid Timestamp/Temp_A_C entries.")
        df = df.sort_values("Timestamp")

        self.start_timestamp = df["Timestamp"].iloc[0]
        rel_seconds = (df["Timestamp"] - self.start_timestamp).dt.total_seconds()
        self.absolute_hours = (rel_seconds / 3600.0).to_numpy(dtype=float)
        self.temperatures = df["Temp_A_C"].to_numpy(dtype=float)
        self.timestamps = df["Timestamp"].to_numpy()

    def as_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        return self.absolute_hours, self.temperatures

    def to_absolute_hours(self, datetimes) -> np.ndarray:
        if datetimes is None:
            return np.asarray([], dtype=float)
        result = []
        for dt in datetimes:
            if dt is None or (isinstance(dt, float) and np.isnan(dt)):
                result.append(np.nan)
                continue
            ts = pd.Timestamp(dt)
            delta = (ts - self.start_timestamp).total_seconds()
            result.append(delta / 3600.0)
        return np.asarray(result, dtype=float)

    def sample(self, hours) -> np.ndarray:
        hours = np.asarray(hours, dtype=float)
        if hours.size == 0 or self.absolute_hours.size == 0:
            return np.asarray([], dtype=float)
        result = np.full(hours.shape, np.nan, dtype=float)
        valid = np.isfinite(hours)
        if np.any(valid):
            result[valid] = np.interp(
                hours[valid],
                self.absolute_hours,
                self.temperatures,
                left=self.temperatures[0],
                right=self.temperatures[-1],
            )
        return result


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
    ax.set_xlim(0, 3)
    ax.set_xlabel("Absolute Time (h)")
    ax.set_ylabel("Band Gap (eV)")
    ax.set_title("Band Gap Evolution")
    fig.savefig(result_dir / "band_gap_vs_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_tauc_plots(
    spectra: Spectroscopy,
    result_dir: Path,
    temperature_log: TemperatureLog | None = None,
) -> None:
    """Write Tauc plot PNGs for every available spectrum."""
    value_taucs = getattr(spectra, "valueTaucs", [])
    if not value_taucs:
        return

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
    fallback_temperatures = np.asarray(getattr(spectra, "temperatures", []), dtype=float)
    if temperature_log is not None and getattr(spectra, "timestamps", None):
        time_hours = temperature_log.to_absolute_hours(getattr(spectra, "timestamps", []))
    else:
        time_hours = np.asarray(
            getattr(spectra, "timestampAbsHN", getattr(spectra, "timestampAbsH", [])),
            dtype=float,
        )
    window_label_added = False
    limit = len(value_taucs)
    timestamps = getattr(spectra, "timestamps", [])
    band_gaps = getattr(spectra, "bandGap", [])
    slopes = getattr(spectra, "tauc_slope", [])
    intercepts = getattr(spectra, "tauc_slope_b", [])

    for idx in range(limit):
        stamp = timestamps[idx] if idx < len(timestamps) else None
        stamp_str = stamp.strftime("%Y-%m-%d %H:%M:%S") if stamp is not None else "Unknown time"
        temp_value = np.nan
        if temperature_log is not None and time_hours.size > idx and np.isfinite(time_hours[idx]):
            temp_value = temperature_log.sample([time_hours[idx]])[0]
        elif fallback_temperatures.size > idx:
            temp_value = fallback_temperatures[idx]
        temp_label = f"{temp_value:.1f} °C" if np.isfinite(temp_value) else "unknown temperature"
        fig, ax = plt.subplots()
        ax.plot(
            energy_axis,
            value_taucs[idx],
            label=f"Tauc Plot at temperature {temp_label}",
            color="#1f77b4",
        )
        if idx < len(slopes) and idx < len(intercepts):
            slope = slopes[idx]
            intercept = intercepts[idx]
            if np.isfinite(slope) and np.isfinite(intercept):
                fit_line = intercept + slope * energy_axis
                label = "Fit Line"
                if idx < len(band_gaps) and np.isfinite(band_gaps[idx]):
                    label = f"Fit Line, band gap = {band_gaps[idx]:.3f} eV"
                ax.plot(
                    energy_axis,
                    fit_line,
                    label=label,
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
        ax.set_xlim(left=None, right=1.8)
        ax.set_ylim(0, 10)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Tauc Plot (Absorbance * photon energy)^0.5")
        ax.set_title(f"Tauc Plot at {stamp_str}")
        ax.legend()
        ax.grid(True)
        fig.savefig(result_dir / f"tauc_plot_{idx:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def create_urbach_plots(
    spectra: Spectroscopy,
    result_dir: Path,
    temperature_log: TemperatureLog | None = None,
) -> None:
    """Write ln(α) Urbach plots with fitted slopes."""

    urbach_sets = getattr(spectra, "urbach_y", None)
    if not urbach_sets:
        return

    energy_axis = getattr(spectra, "urbach_x", getattr(spectra, "eV", None))
    if energy_axis is None:
        raise AttributeError("Spectroscopy object has no Urbach energy axis; run calc_urbach first.")
    energy_axis = np.asarray(energy_axis, dtype=float)
    if energy_axis.size == 0:
        return

    slopes = np.asarray(getattr(spectra, "urbach_slope", []), dtype=float)
    intercepts = np.asarray(getattr(spectra, "urbach_slope_b", []), dtype=float)
    urbach_energy = np.asarray(getattr(spectra, "urbach_energy", []), dtype=float)
    energy_diff = np.asarray(getattr(spectra, "energyDifference", []), dtype=float)
    r2_scores = np.asarray(getattr(spectra, "urbach_fit_r2", []), dtype=float)
    rmse_scores = np.asarray(getattr(spectra, "urbach_fit_residual", []), dtype=float)
    window_bounds = getattr(spectra, "urbach_window_used", None)

    fallback_temperatures = np.asarray(getattr(spectra, "temperatures", []), dtype=float)
    timestamps = getattr(spectra, "timestamps", [])
    if temperature_log is not None and timestamps:
        time_hours = temperature_log.to_absolute_hours(timestamps)
    else:
        time_hours = np.asarray(
            getattr(spectra, "timestampAbsHN", getattr(spectra, "timestampAbsH", [])),
            dtype=float,
        )

    limit = len(urbach_sets)

    for idx in range(limit):
        y_values = np.asarray(urbach_sets[idx], dtype=float)
        if y_values.size != energy_axis.size:
            continue

        stamp = timestamps[idx] if idx < len(timestamps) else None
        stamp_str = stamp.strftime("%Y-%m-%d %H:%M:%S") if stamp is not None else "Unknown time"

        temp_value = np.nan
        if temperature_log is not None and idx < np.size(time_hours) and np.isfinite(time_hours[idx]):
            temp_value = temperature_log.sample([time_hours[idx]])[0]
        elif fallback_temperatures.size > idx:
            temp_value = fallback_temperatures[idx]
        temp_label = f"{temp_value:.1f} °C" if np.isfinite(temp_value) else "unknown temperature"

        fig, ax = plt.subplots()
        ax.plot(energy_axis, y_values, color="#1f77b4", label=f"ln(α), {temp_label}")

        slope = slopes[idx] if idx < slopes.size else np.nan
        intercept = intercepts[idx] if idx < intercepts.size else np.nan
        eu = urbach_energy[idx] if idx < urbach_energy.size else np.nan
        delta_e = energy_diff[idx] if idx < energy_diff.size else np.nan

        if np.isfinite(slope) and np.isfinite(intercept):
            fit_line = intercept + slope * energy_axis
            if np.isfinite(eu) and np.isfinite(delta_e):
                fit_label = f"Fit (E₀={eu:.3f} eV)"
            elif np.isfinite(eu):
                fit_label = f"Fit (E₀={eu:.3f} eV)"
            else:
                fit_label = "Fit"
            ax.plot(energy_axis, fit_line, color="#ff7f0e", linestyle="--", label=fit_label)

        if window_bounds and idx < len(window_bounds):
            win_lo, win_hi = window_bounds[idx]
            if np.isfinite(win_lo):
                ax.axvline(win_lo, color="#2ca02c", linestyle="--", linewidth=1.2)
            if np.isfinite(win_hi):
                ax.axvline(win_hi, color="#2ca02c", linestyle="--", linewidth=1.2)

        info_lines = []
        if np.isfinite(slope):
            info_lines.append(f"slope = {slope:.3e}")
        if idx < r2_scores.size and np.isfinite(r2_scores[idx]):
            info_lines.append(f"R² = {r2_scores[idx]:.3f}")
        if idx < rmse_scores.size and np.isfinite(rmse_scores[idx]):
            info_lines.append(f"RMSE = {rmse_scores[idx]:.3f}")

        if info_lines:
            ax.text(
                0.02,
                0.02,
                "\n".join(info_lines),
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("ln(α)")
        ax.set_title(f"Urbach Tail at {stamp_str}")
        ax.set_xlim(left=None, right=1.8)
        ax.set_ylim(bottom=None, top=7)
        ax.legend()
        ax.grid(True)
        fig.savefig(result_dir / f"urbach_plot_{idx:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_energy_difference(
    spectra: Spectroscopy,
    result_dir: Path,
    temperature_log: TemperatureLog | None = None,
) -> None:
    energy_diff = np.asarray(getattr(spectra, "energyDifference", []), dtype=float)
    if energy_diff.size == 0:
        return

    timestamps = getattr(spectra, "timestamps", None)
    if timestamps:
        time_axis = np.asarray(timestamps, dtype="datetime64[ns]")
    else:
        time_axis = np.asarray(getattr(spectra, "timestampAbsHN", []), dtype=float)

    limit = min(time_axis.size, energy_diff.size)
    if limit == 0:
        return

    # Convert to hours if timestamps are datetime
    if np.issubdtype(time_axis.dtype, np.datetime64):
        if temperature_log is not None:
            time_hours = temperature_log.to_absolute_hours(time_axis.tolist())[:limit]
        else:
            base_time = time_axis[0]
            delta = (time_axis[:limit] - base_time).astype('timedelta64[s]').astype(float)
            time_hours = delta / 3600.0
    else:
        time_hours = np.asarray(time_axis[:limit], dtype=float)

    y_values = energy_diff[:limit]

    fig, ax = plt.subplots()
    ax.plot(time_hours, y_values, marker="o", color="#9467bd")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("ΔE = Eg - E₀ (eV)")
    ax.set_title("Energy Difference vs Time")
    ax.set_ylim(0, 0.05)
    ax.grid(True)
    fig.savefig(result_dir / "energy_difference_vs_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def log_urbach_metrics(
    spectra: Spectroscopy,
    result_dir: Path,
    temperature_log: TemperatureLog | None = None,
) -> None:
    urbach_energy = np.asarray(getattr(spectra, "urbach_energy", []), dtype=float)
    if urbach_energy.size == 0:
        return

    count = urbach_energy.size

    mode = getattr(spectra, "tauc_last_mode", None)
    if mode == "mean":
        cycles = _align_to_count(getattr(spectra, "cycleNum_avg", []), count, np.nan)
        sources = _align_to_count(getattr(spectra, "fileNameAveraged", []), count, "")
    else:
        cycles = _align_to_count(getattr(spectra, "cycleNum", []), count, np.nan)
        sources = _align_to_count(getattr(spectra, "fileNameRaw", []), count, "")

    timestamp_candidates = _align_to_count(getattr(spectra, "timestamps", []), count, pd.NaT)
    timestamp_series = pd.to_datetime(pd.Series(timestamp_candidates), errors="coerce")
    timestamps = [ts.to_pydatetime() if pd.notna(ts) else None for ts in timestamp_series]

    time_hours_fallback = _align_to_count(
        getattr(spectra, "timestampAbsHN", getattr(spectra, "timestampAbsH", [])),
        count,
        np.nan,
    )
    absolute_hours = _resolve_absolute_hours(timestamps, time_hours_fallback, temperature_log)

    fallback_temperatures = _align_to_count(getattr(spectra, "temperatures", []), count, np.nan)
    temperatures = _resolve_temperatures(absolute_hours, fallback_temperatures, temperature_log)

    slopes = _align_to_count(getattr(spectra, "urbach_slope", []), count, np.nan)
    intercepts = _align_to_count(getattr(spectra, "urbach_slope_b", []), count, np.nan)
    r2_values = _align_to_count(getattr(spectra, "urbach_fit_r2", []), count, np.nan)
    rmse_values = _align_to_count(getattr(spectra, "urbach_fit_residual", []), count, np.nan)
    windows = _align_to_count(getattr(spectra, "urbach_window_used", []), count, (np.nan, np.nan))
    energy_diff = _align_to_count(getattr(spectra, "energyDifference", []), count, np.nan)

    window_start = [w[0] if isinstance(w, (list, tuple)) and len(w) >= 1 else np.nan for w in windows]
    window_end = [w[1] if isinstance(w, (list, tuple)) and len(w) >= 2 else np.nan for w in windows]

    df = pd.DataFrame(
        {
            "source": sources,
            "timestamp": pd.to_datetime(timestamps),
            "time_hours": absolute_hours,
            "temperature_c": temperatures,
            "cycle": cycles,
            "urbach_energy": urbach_energy,
            "urbach_slope": slopes,
            "urbach_intercept": intercepts,
            "urbach_r2": r2_values,
            "urbach_rmse": rmse_values,
            "window_start": window_start,
            "window_end": window_end,
            "energy_difference": energy_diff,
        }
    )

    df.to_csv(result_dir / "urbach_metrics.csv", index=False)


def create_absorbance_plots(
    spectra: Spectroscopy,
    result_dir: Path,
    max_plots: int = 10000,
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


def _align_to_count(seq, count, fill_value):
    """Return a list trimmed/padded to count elements."""
    items = list(seq) if seq is not None else []
    if len(items) >= count:
        return items[:count]
    return items + [fill_value] * (count - len(items))


def _resolve_absolute_hours(
    timestamps: list[pd.Timestamp | None],
    fallback_hours: list[float],
    temperature_log: TemperatureLog | None,
) -> np.ndarray:
    """Combine timestamp-based hours with fallback hour values."""

    hours_from_timestamp = np.full(len(fallback_hours), np.nan, dtype=float)
    if temperature_log is not None and timestamps:
        timestamp_inputs = [ts if ts is not None else None for ts in timestamps]
        hours_from_timestamp = temperature_log.to_absolute_hours(timestamp_inputs)

    fallback_array = np.asarray(fallback_hours, dtype=float)
    hours_from_timestamp = np.asarray(hours_from_timestamp, dtype=float)
    use_timestamp = np.isfinite(hours_from_timestamp)
    combined = np.where(use_timestamp, hours_from_timestamp, fallback_array)
    return combined


def _resolve_temperatures(
    absolute_hours: np.ndarray,
    fallback_temperatures: list[float],
    temperature_log: TemperatureLog | None,
) -> np.ndarray:
    """Sample chamber temperatures or fall back to provided values."""

    temps_from_log = np.full_like(absolute_hours, np.nan, dtype=float)
    if temperature_log is not None and absolute_hours.size:
        temps_from_log = temperature_log.sample(absolute_hours)

    fallback_array = np.asarray(fallback_temperatures, dtype=float)
    if fallback_array.size != temps_from_log.size:
        fallback_array = _align_to_count(fallback_array, temps_from_log.size, np.nan)
        fallback_array = np.asarray(fallback_array, dtype=float)

    use_log = np.isfinite(temps_from_log)
    combined = np.where(use_log, temps_from_log, fallback_array)
    return combined


def log_tauc_metrics(
    spectra: Spectroscopy,
    result_dir: Path,
    temperature_log: TemperatureLog | None = None,
) -> None:
    band_gaps = np.asarray(getattr(spectra, "bandGap", []), dtype=float)
    if band_gaps.size == 0:
        return

    count = band_gaps.size

    mode = getattr(spectra, "tauc_last_mode", None)
    if mode == "mean":
        cycles = _align_to_count(getattr(spectra, "cycleNum_avg", []), count, np.nan)
        sources = _align_to_count(getattr(spectra, "fileNameAveraged", []), count, "")
    else:
        cycles = _align_to_count(getattr(spectra, "cycleNum", []), count, np.nan)
        sources = _align_to_count(getattr(spectra, "fileNameRaw", []), count, "")

    timestamp_candidates = _align_to_count(getattr(spectra, "timestamps", []), count, pd.NaT)
    timestamp_series = pd.to_datetime(pd.Series(timestamp_candidates), errors="coerce")
    timestamps = [ts.to_pydatetime() if pd.notna(ts) else None for ts in timestamp_series]

    time_hours_fallback = _align_to_count(
        getattr(spectra, "timestampAbsHN", getattr(spectra, "timestampAbsH", [])),
        count,
        np.nan,
    )
    absolute_hours = _resolve_absolute_hours(timestamps, time_hours_fallback, temperature_log)

    fallback_temperatures = _align_to_count(getattr(spectra, "temperatures", []), count, np.nan)
    temperatures = _resolve_temperatures(absolute_hours, fallback_temperatures, temperature_log)

    slopes = _align_to_count(getattr(spectra, "tauc_slope", []), count, np.nan)
    intercepts = _align_to_count(getattr(spectra, "tauc_slope_b", []), count, np.nan)
    r2_values = _align_to_count(getattr(spectra, "tauc_fit_r2", []), count, np.nan)
    rmse_values = _align_to_count(getattr(spectra, "tauc_fit_residual", []), count, np.nan)
    windows = _align_to_count(getattr(spectra, "tauc_window_used", []), count, (np.nan, np.nan))

    window_start = [w[0] if isinstance(w, (list, tuple)) and len(w) >= 1 else np.nan for w in windows]
    window_end = [w[1] if isinstance(w, (list, tuple)) and len(w) >= 2 else np.nan for w in windows]

    df = pd.DataFrame(
        {
            "source": sources,
            "timestamp": pd.to_datetime(timestamps),
            "time_hours": absolute_hours,
            "temperature_c": temperatures,
            "cycle": cycles,
            "band_gap": band_gaps,
            "tauc_slope": slopes,
            "tauc_intercept": intercepts,
            "tauc_r2": r2_values,
            "tauc_rmse": rmse_values,
            "window_start": window_start,
            "window_end": window_end,
        }
    )
    df.to_csv(result_dir / "tauc_metrics.csv", index=False)


def plot_pixel_metric(
    pixel,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    result_dir: Path,
    color: str,
    cycle_mask=None,
    temperature_log: TemperatureLog | None = None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
    values = np.asarray(getattr(pixel, metric), dtype=float)

    pixel_datetimes = getattr(pixel, "timestamp", None)
    if temperature_log is not None and pixel_datetimes:
        base_times = temperature_log.to_absolute_hours(pixel_datetimes)
    else:
        base_times = np.asarray(getattr(pixel, "timestampAbsH", []), dtype=float)

    if temperature_log is not None:
        base_temps = temperature_log.sample(base_times)
    else:
        base_temps = np.asarray(getattr(pixel, "temperature", []), dtype=float)

    target_len = min(
        base_times.shape[0],
        values.shape[0],
        base_temps.shape[0] if base_temps.size else base_times.shape[0],
    )
    times = base_times[:target_len]
    values = values[:target_len]
    if base_temps.size:
        temps = base_temps[:target_len]
    else:
        temps = np.full(target_len, np.nan, dtype=float)

    if cycle_mask is not None:
        mask = np.asarray(cycle_mask, dtype=bool)
        if mask.shape[0] > values.shape[0]:
            mask = mask[: values.shape[0]]
        if mask.shape[0] != values.shape[0]:
            raise ValueError("Cycle mask length must match metric data length.")
        times = times[mask]
        temps = temps[mask]
        values = values[mask]

    valid = np.isfinite(times)
    times = times[valid]
    temps = temps[valid]
    values = values[valid]

    if times.size == 0:
        plt.close(fig)
        return

    #if metric == "Voc":
    #    values = -values

    ax1.scatter(times, values, color=color, s = 5)
    ax1.set_ylabel(ylabel, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_xlabel("Absolute Time (h)")
    if metric == "Rp":
        ax1.set_ylim(0, 20)
    else:
        ax1.set_ylim(0, None)

    if metric == "Voc":
        ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    temp_color = "#bfbfbf"
    if temperature_log is not None:
        log_hours, log_temps = temperature_log.as_pairs()
        if log_hours.size and log_temps.size:
            ax2.plot(log_hours, log_temps, temp_color, linestyle="--")
    elif np.any(np.isfinite(temps)):
        ax2.plot(times, temps, temp_color, linestyle="--")
    ax2.set_ylabel("Temperature", color=temp_color)
    ax2.tick_params(axis="y", labelcolor=temp_color)
    ax2.set_ylim(-80, 80)

    ax1.set_title(title)

    fig.savefig(result_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_pixel_plots(
    pixels: dict[str, object],
    result_dir: Path,
    cycle_selector=None,
    temperature_log: TemperatureLog | None = None,
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
                temperature_log=temperature_log,
            )

def _coerce_sequence(value):
    if value is None:
        return []
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def save_pixel_results(pixels: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    column_map = [
        ("Timestamp[s]", "timestampAbsS"),
        ("Labels", "labels"),
        ("Temperature [C]", "temperature"),
        ("Cycle Number", "cycleNum"),
        ("Isc[mA/cm2]", "Isc"),
        ("Voc [V]", "Voc"),
        ("I_MPP [mA/cm2]", "I_MPP"),
        ("V_MPP [V]", "V_MPP"),
        ("FF", "FF"),
        ("PCE [%]", "PCE"),
        ("Rs", "Rs"),
        ("Rp", "Rp"),
    ]
    for pixel_name, pixel in pixels.items():
        data = {}
        has_data = False
        for column, attr in column_map:
            values = _coerce_sequence(getattr(pixel, attr, None))
            if values:
                has_data = True
            data[column] = pd.Series(values)
        if not has_data:
            continue
        df = pd.DataFrame(data)
        output_path = output_dir / f"{pixel_name}_result.csv"
        df.to_csv(output_path, index=False)

def main() -> None:
    result_dir = ensure_result_dir(TC_FOLDER)

    #tc = ThermalCycling(folderPath=str(TC_FOLDER))
    #tc.sortData()

    temperature_log = TemperatureLog(TEMPERATURE_LOG_PATH)

    # Adjust cycle_selector to focus on specific cycles if needed.
    # Examples: None (all cycles), 3 (only cycle 3), (1, 5) (cycles 1 through 5),
    # {2, 4, 6} (explicit set of cycles).
    cycle_selector = None
    '''
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
    create_tauc_plots(
        spectra,
        result_dir,
        temperature_log=temperature_log,
    )  # Enable if you need Tauc plot PNGs
    create_urbach_plots(
        spectra,
        result_dir,
        temperature_log=temperature_log,
    )
    plot_energy_difference(spectra, result_dir, temperature_log=temperature_log)
    log_tauc_metrics(spectra, result_dir, temperature_log=temperature_log)
    log_urbach_metrics(spectra, result_dir, temperature_log=temperature_log)
    #create_absorbance_plots(spectra, result_dir)  # Enable if you need absorbance PNGs
    '''

    pixels = {
        f"px{idx}": solarSimulator(folderPath=str(TC_FOLDER / f"px{idx}"), pixel_area=0.143)
        for idx in range(1, 7)
    }
    for pixel in pixels.values():
        pixel.loadFolderData_Cycling()

    save_pixel_results(pixels, PIXEL_RESULT_DIR)

    create_pixel_plots(
        pixels,
        result_dir,
        cycle_selector=cycle_selector,
        temperature_log=temperature_log,
    )


if __name__ == "__main__":
    main()
