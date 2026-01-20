from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt

DEFAULT_DATA_DIR = Path(
    #""  # Enter your file path/folder path in this place
    #""  # Enter your file path/folder path in this place
    ""  # Enter your file path/folder path in this place
)

GROUP_PALETTE: dict[str, str] = {
    'R3-20s-with-Cs': '#EB1615',
    'R3-25s-with-Cs': '#EBBD15',
    'R5-25s-with-Cs': '#35B85B',
    'R5-30s-with-Cs': '#3C4CFF',
    'R3-25s-without-Cs': '#B835EB', 
}

STACK_ORDER_TOP_TO_BOTTOM: tuple[str, ...] = (
    "R3-25s-without-Cs",
    "R3-25s-with-Cs",
    "R3-20s-with-Cs",
    "R5-25s-with-Cs",
    "R5-30s-with-Cs",
)

CALIBRATE_WITH_ITO: bool = False  # shift peaks to the ITO reference angle when True
NORMALIZE_TO_PEAK: bool = True  # scale intensities relative to the FAPbI3 (100) peak
SMOOTHING_WINDOW: int = 5  # moving-average size for initial trace smoothing
ITO_CALIBRATION_WINDOW: Tuple[float, float] = (30.0, 31.0)
ITO_REFERENCE_ANGLE: float = 30.6
FAPBI3_PEAK_WINDOW: Tuple[float, float] = (13.0, 14.5)
FAPBI3_REFERENCE_ANGLE: float = 13.9
STACK_SPACING_FACTOR: float = 1.1
ZOOM_WINDOWS: tuple[tuple[float, float], ...] = (FAPBI3_PEAK_WINDOW,)
ZOOM_FIGURE_SIZE: tuple[float, float] = (5.0, 3.2)
ZOOM_OUTPUT_DIRNAME: str = "Zoom"


@dataclass(frozen=True)
class PeakMeasurement:
    angle: float
    intensity: float
    fwhm: float
    angle_error: float
    intensity_error: float
    fwhm_error: float


def load_xrd_pattern(file_path: Path) -> Tuple[list[float], list[float]]:
    """Extract angle/intensity pairs from an XRD export file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"XRD file not found: {file_path}")

    angles: list[float] = []
    intensities: list[float] = []
    data_section = False

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not data_section:
                if line.startswith("[Data]"):
                    data_section = True
                continue

            if not line or line.startswith("Angle"):
                continue

            parts = [chunk.strip() for chunk in raw_line.split(",") if chunk.strip()]
            if len(parts) < 2:
                continue

            try:
                angles.append(float(parts[0]))
                intensities.append(float(parts[1]))
            except ValueError:
                continue

    if not angles:
        raise ValueError(f"No numeric data found in {file_path}")

    return angles, intensities


def normalize_to_peak(
    angles: list[float],
    intensities: list[float],
    window: Tuple[float, float] = (30.0, 31.0),
    reference_angle: float = 30.6,
    *,
    align_to_reference: bool = True,
    scale_intensity: bool = True,
) -> Tuple[list[float], list[float]]:
    """Optionally align and/or scale traces using the dominant ITO feature."""
    min_angle, max_angle = window
    if min_angle >= max_angle:
        raise ValueError("Normalization window must have min < max")

    candidates = [i for i, angle in enumerate(angles) if min_angle <= angle <= max_angle]
    if not candidates:
        raise ValueError(f"No data points in normalization window {window}")

    peak_index = max(candidates, key=lambda idx: intensities[idx])
    peak_intensity = intensities[peak_index]
    if peak_intensity == 0:
        raise ValueError(f"Peak intensity is zero in window {window}")

    if align_to_reference:
        delta = reference_angle - angles[peak_index]
        shifted_angles = [angle + delta for angle in angles]
    else:
        shifted_angles = angles[:]

    if scale_intensity:
        normalized_intensities = [value / peak_intensity for value in intensities]
    else:
        normalized_intensities = intensities[:]

    return shifted_angles, normalized_intensities


def iter_xrd_files(directory: Path) -> Iterable[Path]:
    """Yield XRD text files within the provided directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Data directory missing: {directory}")

    for path in sorted(directory.glob("*.txt")):
        yield path


def prepare_normalized_patterns(
    directory: Path,
    window: Tuple[float, float] = ITO_CALIBRATION_WINDOW,
    align_to_reference: bool = CALIBRATE_WITH_ITO,
    scale_to_peak: bool = NORMALIZE_TO_PEAK,
    reference_angle: float = ITO_REFERENCE_ANGLE,
    smoothing_window: int = SMOOTHING_WINDOW,
    intensity_window: Tuple[float, float] = FAPBI3_PEAK_WINDOW,
    intensity_reference: float = FAPBI3_REFERENCE_ANGLE,
) -> list[tuple[Path, list[float], list[float]]]:
    """Load XRD files within `directory` and optionally normalize them."""
    patterns: list[tuple[Path, list[float], list[float]]] = []

    for file_path in iter_xrd_files(directory):
        try:
            angles, intensities = load_xrd_pattern(file_path)
        except ValueError as exc:
            print(f"Skipping {file_path.name}: {exc}")
            continue

        smoothed_intensities = smooth_series(intensities, window_size=smoothing_window)

        processed_angles = angles[:]
        processed_intensities = smoothed_intensities

        if align_to_reference:
            try:
                processed_angles, processed_intensities = normalize_to_peak(
                    processed_angles,
                    processed_intensities,
                    window,
                    reference_angle,
                    align_to_reference=True,
                    scale_intensity=False,
                )
            except ValueError as exc:
                print(f"Skipping alignment for {file_path.name}: {exc}")
                processed_angles = angles[:]
                processed_intensities = smoothed_intensities

        if scale_to_peak:
            try:
                _, processed_intensities = normalize_to_peak(
                    processed_angles,
                    processed_intensities,
                    intensity_window,
                    intensity_reference,
                    align_to_reference=False,
                    scale_intensity=True,
                )
            except ValueError as exc:
                print(f"Skipping intensity normalization for {file_path.name}: {exc}")

        patterns.append((file_path, processed_angles, processed_intensities))

    if not patterns:
        raise RuntimeError(
            f"No usable XRD patterns found in {directory} for window {window}."
        )

    return patterns


def smooth_series(values: list[float], window_size: int = 11) -> list[float]:
    """Return a softened moving average of `values` with reflective padding."""
    length = len(values)
    if window_size <= 1 or window_size > length:
        return values[:]

    half_window = window_size // 2
    padded = [values[0]] * half_window + values + [values[-1]] * half_window

    window_sum = sum(padded[:window_size])
    smoothed: list[float] = [window_sum / window_size]

    for index in range(1, length):
        window_sum += padded[index + window_size - 1] - padded[index - 1]
        smoothed.append(window_sum / window_size)

    return smoothed


def _estimate_angle_error(angles: Sequence[float], index: int) -> float:
    """Approximate angle uncertainty from neighboring sampling distance."""
    if not angles:
        return math.nan

    candidates: list[float] = []
    if index > 0:
        left_step = angles[index] - angles[index - 1]
        if left_step > 0:
            candidates.append(left_step)
    if index + 1 < len(angles):
        right_step = angles[index + 1] - angles[index]
        if right_step > 0:
            candidates.append(right_step)

    if not candidates:
        return 0.0

    return min(candidates) / 2.0


def _estimate_intensity_error(
    intensities: Sequence[float],
    smoothed: Sequence[float],
    index: int,
    window: int = 6,
) -> float:
    """Use local residuals between raw and smoothed traces as an error proxy."""
    if not intensities or not smoothed:
        return math.nan

    half_window = max(window // 2, 1)
    start = max(index - half_window, 0)
    stop = min(index + half_window + 1, len(intensities))
    if stop <= start:
        return 0.0

    residuals = [
        intensities[i] - smoothed[i]
        for i in range(start, stop)
        if i < len(smoothed)
    ]
    if not residuals:
        return 0.0

    mean = sum(residuals) / len(residuals)
    variance = sum((value - mean) ** 2 for value in residuals) / len(residuals)
    return math.sqrt(max(variance, 0.0))


def _interval_error(
    angles: Sequence[float], left_index: int, right_index: int
) -> float:
    """Return half the spacing for the points used in interpolation."""
    if (
        not angles
        or left_index < 0
        or right_index >= len(angles)
        or right_index <= left_index
    ):
        return math.nan

    spacing = angles[right_index] - angles[left_index]
    if spacing <= 0:
        return 0.0
    return spacing / 2.0


def locate_calc_peaks(
    angles: list[float],
    intensities: list[float],
    min_height: float = 0.08,
    min_distance: float = 0.05,
    relative_intensity: float = 0.08,
    detection_profile: list[float] | None = None,
) -> list[PeakMeasurement]:
    """Identify prominent peaks and estimate their FWHM with uncertainties."""
    if len(angles) < 3 or len(angles) != len(intensities):
        return []

    profile = detection_profile if detection_profile is not None else intensities
    if len(profile) != len(intensities):
        raise ValueError("Detection profile length must match intensities")

    smoothed = smooth_series(profile)

    candidate_indices: list[int] = []
    for idx in range(1, len(intensities) - 1):
        current = smoothed[idx]
        if current <= smoothed[idx - 1] or current <= smoothed[idx + 1]:
            continue
        if current < min_height:
            continue
        candidate_indices.append(idx)

    candidate_indices.sort(key=lambda index: smoothed[index], reverse=True)

    selected: list[int] = []
    for idx in candidate_indices:
        angle = angles[idx]
        if any(abs(angle - angles[other]) < min_distance for other in selected):
            continue
        selected.append(idx)

    selected.sort()
    results: list[PeakMeasurement] = []

    for idx in selected:
        peak_intensity = intensities[idx]
        if peak_intensity < relative_intensity:
            continue
        half_height = smoothed[idx] * 0.5

        left_idx = idx
        while left_idx > 0 and smoothed[left_idx] > half_height:
            left_idx -= 1

        if left_idx == 0 and smoothed[left_idx] > half_height:
            left_half = None
        elif smoothed[left_idx] == half_height:
            left_half = angles[left_idx]
        else:
            x1 = angles[left_idx]
            y1 = smoothed[left_idx]
            x2 = angles[left_idx + 1]
            y2 = smoothed[left_idx + 1]
            if y2 == y1:
                left_half = x1
            else:
                left_half = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)

        right_idx = idx
        last_index = len(intensities) - 1
        while right_idx < last_index and smoothed[right_idx] > half_height:
            right_idx += 1

        if right_idx == last_index and smoothed[right_idx] > half_height:
            right_half = None
        elif smoothed[right_idx] == half_height:
            right_half = angles[right_idx]
        else:
            x1 = angles[right_idx - 1]
            y1 = smoothed[right_idx - 1]
            x2 = angles[right_idx]
            y2 = smoothed[right_idx]
            if y2 == y1:
                right_half = x2
            else:
                right_half = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)

        angle_error = _estimate_angle_error(angles, idx)
        intensity_error = _estimate_intensity_error(intensities, smoothed, idx)

        if left_half is None or right_half is None:
            fwhm = math.nan
            fwhm_error = math.nan
        else:
            fwhm = right_half - left_half
            if smoothed[left_idx] == half_height:
                left_error = _estimate_angle_error(angles, left_idx)
            else:
                left_error = _interval_error(
                    angles, left_idx, min(left_idx + 1, len(angles) - 1)
                )

            if smoothed[right_idx] == half_height:
                right_error = _estimate_angle_error(angles, right_idx)
            else:
                right_error = _interval_error(
                    angles, max(right_idx - 1, 0), right_idx
                )

            if math.isnan(left_error) or math.isnan(right_error):
                fwhm_error = math.nan
            else:
                fwhm_error = math.sqrt(left_error ** 2 + right_error ** 2)

        results.append(
            PeakMeasurement(
                angle=angles[idx],
                intensity=peak_intensity,
                fwhm=fwhm,
                angle_error=angle_error,
                intensity_error=intensity_error,
                fwhm_error=fwhm_error,
            )
        )

    return results


def baseline_correct(
    angles: list[float], intensities: list[float], window_size: int = 51
) -> list[float]:
    """Estimate and subtract a smooth baseline from the intensity profile."""
    length = len(intensities)
    if length == 0:
        return []
    if length < 3:
        return intensities[:]

    if window_size > length:
        window_size = length if length % 2 == 1 else max(length - 1, 1)

    if window_size % 2 == 0:
        window_size += 1

    if window_size <= 1:
        return intensities[:]

    smoothed_baseline = smooth_series(intensities, window_size)

    corrected: list[float] = []
    for original, baseline in zip(intensities, smoothed_baseline):
        corrected.append(max(original - baseline, 0.0))

    return corrected


def calculate_stack_step(
    patterns: Sequence[tuple[Path, list[float], list[float]]]
) -> float:
    """Return a consistent vertical offset for stacked traces."""
    if not patterns:
        raise ValueError("No patterns available to determine stack spacing")

    raw_max = max(max(intensities) for _, _, intensities in patterns)
    raw_min = min(min(intensities) for _, _, intensities in patterns)
    value_span = max(raw_max - raw_min, raw_max, 1.0)
    return value_span * STACK_SPACING_FACTOR


def plot_xrd_patterns(
    patterns: Iterable[tuple[Path, list[float], list[float]]],
    output_path: Path | None = None,
    xlim: tuple[float, float] | None = None, #xlim: tuple[float, float] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    show: bool = True,
    peaks_map: dict[Path, list[PeakMeasurement]] | None = None,
    highlight_file: Path | None = None,
    figure_size: tuple[float, float] = (7, 4),
    stack: bool = True,
    stack_step: float | None = None,
    legend_location: str | tuple[str, tuple[float, float]] | None = "upper right",
) -> None:
    patterns_list = list(patterns)
    if not patterns_list:
        raise RuntimeError("No plottable XRD patterns provided")

    order_lookup = {
        name: index
        for index, name in enumerate(reversed(STACK_ORDER_TOP_TO_BOTTOM))
    }
    patterns_list.sort(
        key=lambda item: (
            order_lookup.get(item[0].stem, len(order_lookup)),
            item[0].stem,
        )
    )

    plt.figure(figsize=figure_size)

    ax = plt.gca()
    line_colors: dict[Path, Any] = {}

    raw_max = max(max(intensities) for _, _, intensities in patterns_list)
    raw_min = min(min(intensities) for _, _, intensities in patterns_list)
    value_span = max(raw_max - raw_min, raw_max, 1.0)

    offset_step = 0.0
    if stack:
        default_step = value_span * STACK_SPACING_FACTOR
        offset_step = stack_step if stack_step is not None else default_step

    global_max = -float("inf")
    global_min = float("inf")

    for index, (file_path, angles, intensities) in enumerate(patterns_list):
        vertical_offset = offset_step * index
        shifted_intensities = (
            [value + vertical_offset for value in intensities] if stack else intensities
        )

        palette_color = GROUP_PALETTE.get(file_path.stem)
        line = ax.plot(
            angles,
            shifted_intensities,
            label=file_path.stem,
            color=palette_color if palette_color is not None else None,
        )[0]
        color = line.get_color()
        line_colors[file_path] = color

        global_max = max(global_max, max(shifted_intensities))
        global_min = min(global_min, min(shifted_intensities))

        if peaks_map is not None:
            peaks = peaks_map.get(file_path)
            if peaks:
                for peak in peaks:
                    peak_angle = peak.angle
                    peak_intensity = peak.intensity
                    if stack:
                        marker_bottom = max(vertical_offset - 0.05 * value_span, -0.4)
                        marker_top = peak_intensity + vertical_offset
                    else:
                        marker_bottom = -0.4
                        marker_top = -0.1
                    ax.vlines(
                        peak_angle,
                        ymin=marker_bottom,
                        ymax=marker_top,
                        colors=color,
                        linewidth=0.5,
                    )

    bottom_margin = value_span * 0.1
    top_margin = value_span * 0.05
    computed_bottom = min(global_min - bottom_margin, -0.4)
    computed_top = global_max + top_margin

    ylabel = "Intensity (a.u.)" if stack else "Intensity (a.u.)"
    plt.ylabel(ylabel)
    plt.xlabel(r"2$\theta$ (deg)")
    ax.set_yticks([])
    #plt.title("MA Spincoating Parameter XRD Patterns")
    if legend_location is not None:
        legend_kwargs = {
            "fontsize": "x-small",
            "frameon": True,
        }
        if isinstance(legend_location, tuple):
            loc, anchor = legend_location
            legend_kwargs["loc"] = loc
            legend_kwargs["bbox_to_anchor"] = anchor
            legend_kwargs["borderaxespad"] = 0.0
        else:
            legend_kwargs["loc"] = legend_location
        ax.legend(**legend_kwargs)
    plt.tight_layout()

    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        bottom, top = ylim
        if bottom is None:
            bottom = computed_bottom
        if top is None:
            top = computed_top
        ax.set_ylim(bottom=bottom, top=top)
    else:
        if stack:
            legend_padding = max(offset_step * 0.6, value_span * 0.2)
        else:
            legend_padding = value_span * 0.4
        ax.set_ylim(bottom=computed_bottom, top=computed_top + legend_padding)

    if (
        highlight_file is not None
        and peaks_map is not None
        and highlight_file in peaks_map
        and highlight_file in line_colors
    ):
        highlight_color = line_colors[highlight_file]
        for peak in peaks_map[highlight_file]:
            peak_angle = peak.angle
            ax.axvline(
                peak_angle,
                color=highlight_color,
                linestyle="--",
                linewidth=0.8,
            )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def plot_located_peaks(
    patterns: Iterable[tuple[Path, list[float], list[float]]],
    reference_peaks: Sequence[PeakMeasurement],
    reference_file: Path | None,
    output_dir: Path,
    min_span: float = 1.0,
    span_scale: float = 3.0,
    peaks_map: dict[Path, list[PeakMeasurement]] | None = None,
) -> None:
    patterns_list = list(patterns)
    if not patterns_list or not reference_peaks:
        return

    order_lookup = {
        name: index
        for index, name in enumerate(reversed(STACK_ORDER_TOP_TO_BOTTOM))
    }
    patterns_list.sort(
        key=lambda item: (
            order_lookup.get(item[0].stem, len(order_lookup)),
            item[0].stem,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_max = max(max(intensities) for _, _, intensities in patterns_list)
    raw_min = min(min(intensities) for _, _, intensities in patterns_list)
    value_span = max(raw_max - raw_min, raw_max, 1.0)
    stack_step = value_span * STACK_SPACING_FACTOR
    total_stack_height = stack_step * max(len(patterns_list) - 1, 0)

    for index, peak in enumerate(reference_peaks, start=1):
        center = peak.angle
        peak_height = peak.intensity
        fwhm = peak.fwhm
        if math.isnan(fwhm) or fwhm <= 0:
            span = min_span
        else:
            span = max(fwhm, min_span)

        half_window = (span * span_scale) / 2.0
        x_min = center - half_window
        x_max = center + half_window

        local_max = 0.0
        for _, angles, intensities in patterns_list:
            for angle, intensity in zip(angles, intensities):
                if x_min <= angle <= x_max and intensity > local_max:
                    local_max = intensity

        if local_max <= 0:
            local_max = 1.0

        reference_height = peak_height if peak_height > 0 else 1.0
        y_max = max(local_max, reference_height) * 1.1
        output_path = output_dir / f"peak_{index:02d}_{center:.2f}deg.png"
        top_limit = y_max + total_stack_height + value_span * 0.05

        plot_xrd_patterns(
            patterns_list,
            output_path=output_path,
            xlim=(x_min, x_max),#xlim=(x_min, x_max),
            ylim=(-0.4, top_limit),
            show=False,
            peaks_map=peaks_map,
            highlight_file=reference_file,
            figure_size=(5, 4),
            stack=True,
            stack_step=stack_step,
            legend_location=None,
        )


def plot_zoom_windows(
    patterns: Sequence[tuple[Path, list[float], list[float]]],
    zoom_windows: Iterable[tuple[float, float]],
    output_dir: Path,
    *,
    peaks_map: dict[Path, list[PeakMeasurement]] | None = None,
    stack_step: float | None = None,
    figure_size: tuple[float, float] = ZOOM_FIGURE_SIZE,
) -> None:
    """Create stacked zoom plots for the provided angle windows."""
    sanitized: list[tuple[float, float]] = []
    for window in zoom_windows:
        if len(window) != 2:
            continue
        left, right = window
        if left is None or right is None:
            continue
        if left >= right:
            continue
        sanitized.append((left, right))

    if not sanitized:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (left, right) in enumerate(sanitized, start=1):
        file_name = f"zoom_{index:02d}_{left:.2f}-{right:.2f}.png"
        output_path = output_dir / file_name
        plot_xrd_patterns(
            patterns,
            output_path=output_path,
            xlim=(left, right),
            show=False,
            peaks_map=peaks_map,
            figure_size=figure_size,
            stack=True,
            stack_step=stack_step,
            legend_location=None,
        )


def main() -> None:
    data_dir = DEFAULT_DATA_DIR
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1]).expanduser()

    normalized_patterns = prepare_normalized_patterns(
        data_dir,
        window=ITO_CALIBRATION_WINDOW,
        align_to_reference=CALIBRATE_WITH_ITO,
        scale_to_peak=NORMALIZE_TO_PEAK,
        reference_angle=ITO_REFERENCE_ANGLE,
        smoothing_window=SMOOTHING_WINDOW,
        intensity_window=FAPBI3_PEAK_WINDOW,
        intensity_reference=FAPBI3_REFERENCE_ANGLE,
    )
    stack_step = calculate_stack_step(normalized_patterns)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"result+{timestamp}"
    result_dir = data_dir / base_name

    if result_dir.exists():
        suffix = 1
        while True:
            candidate = data_dir / f"{base_name}_{suffix:02d}"
            if not candidate.exists():
                result_dir = candidate
                break
            suffix += 1

    result_dir.mkdir(parents=True, exist_ok=False)
    output_path = result_dir / "result.png"

    log_rows: list[tuple[str, float, float, float, float, float, float]] = []
    reference_peaks: list[PeakMeasurement] | None = None
    reference_file: Path | None = None
    peaks_map: dict[Path, list[PeakMeasurement]] = {}

    for file_path, angles, intensities in normalized_patterns:
        baseline_removed = baseline_correct(angles, intensities)
        peaks = locate_calc_peaks(
            angles,
            intensities,
            detection_profile=baseline_removed,
        )

        if (reference_peaks is None or not reference_peaks) and peaks:
            reference_peaks = peaks
            reference_file = file_path

        peaks_map[file_path] = peaks

        for peak in peaks:
            log_rows.append(
                (
                    file_path.name,
                    peak.angle,
                    peak.intensity,
                    peak.fwhm,
                    peak.angle_error,
                    peak.intensity_error,
                    peak.fwhm_error,
                )
            )

    log_path = result_dir / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file",
                "peak_angle_deg",
                "peak_angle_error_deg",
                "intensity",
                "intensity_error",
                "fwhm_deg",
                "fwhm_error_deg",
            ]
        )

        def format_float(value: float) -> str:
            return "" if math.isnan(value) else f"{value:.6f}"

        for (
            file_name,
            angle,
            intensity,
            fwhm,
            angle_error,
            intensity_error,
            fwhm_error,
        ) in log_rows:
            writer.writerow(
                [
                    file_name,
                    format_float(angle),
                    format_float(angle_error),
                    format_float(intensity),
                    format_float(intensity_error),
                    format_float(fwhm),
                    format_float(fwhm_error),
                ]
            )

    plot_xrd_patterns(
        normalized_patterns,
        output_path=output_path,
        xlim=(5, 30),
        peaks_map=peaks_map,
        ylim=(-0.4, 7),
        stack_step=stack_step,
        legend_location=None,
    )

    if ZOOM_WINDOWS:
        zoom_dir = result_dir / ZOOM_OUTPUT_DIRNAME
        plot_zoom_windows(
            normalized_patterns,
            ZOOM_WINDOWS,
            zoom_dir,
            peaks_map=peaks_map,
            stack_step=stack_step,
        )

    if reference_peaks:
        peaks_dir = result_dir / "Peaks"
        plot_located_peaks(
            normalized_patterns,
            reference_peaks,
            reference_file,
            peaks_dir,
            #peaks_map=peaks_map,
        )


if __name__ == "__main__":
    main()
