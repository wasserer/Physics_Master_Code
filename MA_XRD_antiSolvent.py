from __future__ import annotations

import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Tuple

import matplotlib.pyplot as plt

DEFAULT_DATA_DIR = Path(
    #'/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/MA_Data/Spincoating_Parameter_New'
    #'/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/MA_Data/Compare_CsI'
    '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/MA_Data/Final_Compare'
)

GROUP_PALETTE: dict[str, str] = {
    'R3-20s-with-Cs': '#EB1615',
    'R3-25s-with-Cs': '#EBBD15',
    'R5-25s-with-Cs': '#35B85B',
    'R5-30s-with-Cs': '#3C4CFF',
    'R3-25s-without-Cs': '#B835EB', 
}

NORMALIZE_PATTERNS: bool = True


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
) -> Tuple[list[float], list[float]]:
    """Scale intensities so the maximum within `window` becomes 1 and align the peak."""
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

    delta = reference_angle - angles[peak_index]
    shifted_angles = [angle + delta for angle in angles]
    normalized_intensities = [value / peak_intensity for value in intensities]

    return shifted_angles, normalized_intensities


def iter_xrd_files(directory: Path) -> Iterable[Path]:
    """Yield XRD text files within the provided directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Data directory missing: {directory}")

    for path in sorted(directory.glob("*.txt")):
        yield path


def prepare_normalized_patterns(
    directory: Path,
    window: Tuple[float, float] = (30.0, 31.0),
    normalize: bool = True,
) -> list[tuple[Path, list[float], list[float]]]:
    """Load XRD files within `directory` and optionally normalize them."""
    patterns: list[tuple[Path, list[float], list[float]]] = []

    for file_path in iter_xrd_files(directory):
        try:
            angles, intensities = load_xrd_pattern(file_path)
        except ValueError as exc:
            print(f"Skipping {file_path.name}: {exc}")
            continue

        smoothed_intensities = smooth_series(intensities, window_size=5)

        if normalize:
            try:
                processed_angles, processed_intensities = normalize_to_peak(
                    angles,
                    smoothed_intensities,
                    window,
                )
            except ValueError as exc:
                print(f"Skipping {file_path.name}: {exc}")
                continue
        else:
            processed_angles = angles
            processed_intensities = smoothed_intensities

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


def locate_calc_peaks(
    angles: list[float],
    intensities: list[float],
    min_height: float = 0.2,
    min_distance: float = 0.05,
    relative_intensity: float = 0.2,
    detection_profile: list[float] | None = None,
) -> list[tuple[float, float, float]]:
    """Identify prominent peaks and estimate their FWHM."""
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
    results: list[tuple[float, float, float]] = []

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

        if left_half is None or right_half is None:
            fwhm = math.nan
        else:
            fwhm = right_half - left_half

        results.append((angles[idx], peak_intensity, fwhm))

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


def plot_xrd_patterns(
    patterns: Iterable[tuple[Path, list[float], list[float]]],
    output_path: Path | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    show: bool = True,
    peaks_map: dict[Path, list[tuple[float, float, float]]] | None = None,
    highlight_file: Path | None = None,
    figure_size: tuple[float, float] = (7, 4),
) -> None:
    plt.figure(figsize=figure_size)

    plotted_any = False

    ax = plt.gca()
    line_colors: dict[Path, Any] = {}

    for file_path, angles, intensities in patterns:
        palette_color = GROUP_PALETTE.get(file_path.stem)
        line = ax.plot(
            angles,
            intensities,
            label=file_path.stem,
            color=palette_color if palette_color is not None else None,
        )[0]
        color = line.get_color()
        line_colors[file_path] = color

        if peaks_map is not None:
            peaks = peaks_map.get(file_path)
            if peaks:
                for peak_angle, *_ in peaks:
                    ax.vlines(
                        peak_angle,
                        ymin=-0.4,
                        ymax=-0.1,
                        colors=color,
                        linewidth=0.5,
                    )
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("No plottable XRD patterns provided")
    plt.ylim(bottom=-0.4)
    plt.xlabel(r"2$\theta$ (deg)")
    plt.ylabel("Intensity (a.u.)")
    #plt.title("MA Spincoating Parameter XRD Patterns")
    plt.legend()
    plt.tight_layout()

    if xlim is not None:
        plt.xlim(*xlim)

    if ylim is not None:
        bottom, top = ylim
        plt.ylim(bottom=bottom, top=top)
        current_bottom, current_top = ax.get_ylim()
    else:
        current_bottom, current_top = ax.get_ylim()

    if current_bottom > -0.4:
        ax.set_ylim(bottom=-0.4, top=current_top)
        current_bottom, current_top = ax.get_ylim()

    if (
        highlight_file is not None
        and peaks_map is not None
        and highlight_file in peaks_map
        and highlight_file in line_colors
    ):
        highlight_color = line_colors[highlight_file]
        for peak_angle, *_ in peaks_map[highlight_file]:
            ax.axvline(peak_angle, color=highlight_color, linestyle="--", linewidth=0.8)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


def plot_located_peaks(
    patterns: Iterable[tuple[Path, list[float], list[float]]],
    reference_peaks: list[tuple[float, float, float]],
    reference_file: Path | None,
    output_dir: Path,
    min_span: float = 1.0,
    span_scale: float = 3.0,
    peaks_map: dict[Path, list[tuple[float, float, float]]] | None = None,
) -> None:
    patterns_list = list(patterns)
    if not patterns_list or not reference_peaks:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (center, peak_height, fwhm) in enumerate(reference_peaks, start=1):
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

        plot_xrd_patterns(
            patterns_list,
            output_path=output_path,
            xlim=(x_min, x_max),
            ylim=(-0.4, y_max),
            show=False,
            peaks_map=peaks_map,
            highlight_file=reference_file,
            figure_size=(5, 5),
        )


def main() -> None:
    data_dir = DEFAULT_DATA_DIR
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1]).expanduser()

    normalized_patterns = prepare_normalized_patterns(
        data_dir,
        normalize=NORMALIZE_PATTERNS,
    )

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

    log_rows: list[tuple[str, float, float, float]] = []
    reference_peaks: list[tuple[float, float, float]] | None = None
    reference_file: Path | None = None
    peaks_map: dict[Path, list[tuple[float, float, float]]] = {}

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

        for angle, intensity, fwhm in peaks:
            log_rows.append((file_path.name, angle, intensity, fwhm))

    log_path = result_dir / "log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "peak_angle_deg", "intensity", "fwhm_deg"])
        for file_name, angle, intensity, fwhm in log_rows:
            writer.writerow(
                [
                    file_name,
                    f"{angle:.6f}",
                    f"{intensity:.6f}",
                    "" if math.isnan(fwhm) else f"{fwhm:.6f}",
                ]
            )

    plot_xrd_patterns(
        normalized_patterns,
        output_path=output_path,
        peaks_map=peaks_map,
        ylim=(-0.4, None),
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
