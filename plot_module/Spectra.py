import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from datetime import datetime
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from scipy.signal import savgol_filter
import re

#This is actually a UV spectroscopy, which was used in the thermal cycling setup!
class Spectroscopy:
    """
    This is a class for the spectroscopy data, which is used in the thermal cycling setup.
    It can import the data from the folder, and do some basic analysis on the data.
    The data is stored in the following format:
    - wavelengths: list of wavelengths
    - values: list of values
    - timestamps: list of timestamps
    - temperatures: list of temperatures
    - labels: list of labels (dark or light)"""
    def __init__(self, folderPath = None):
        self.folderPath = folderPath
        self.wavelengths = []
        self.values = []
        self.timestamps = []
        self.temperatures = []
        self.labels = []
        self.dat_files = None
        self.PeakPosition = []
        self.PeakHeight = []
        self.PeakFWHM = []
        self.wavelengths_interp = []
        self.values_interp = []
        self.values_calibrated = []
        self.transmissions = []
        self.absorbances = []
        self.valueTaucs = [] #Y axis of tauc plot
        self.bandGap = []
        self.tauc_slope = []
        self.tauc_slope_b = []
        self.tauc_fit_r2 = []
        self.tauc_fit_residual = []
        self.tauc_window_used = []
        self.tauc_fit_lines = []
        self.cycleNum = []
        self.cycleNum_avg = []
        self.fileNameRaw = []
        self.values_interpAvr = []
        self.values_interpErr = []
        self.fileNameAveraged = []
        self.StatusLabel = []
        self.tauc_last_mode = None
        self.file_group_map = {}
        self.group_members = []
        self.file_metadata = {}
        self.tauc_results = {}
        self.urbach_results = {}
        self.urbach_x = None
        self.urbach_y = []
        self.urbach_slope = []
        self.urbach_slope_b = []
        self.urbach_fit_r2 = []
        self.urbach_fit_residual = []
        self.urbach_window_used = []
        self.urbach_energy = []
        self.energyDifference = []
        #Import Data:

    def importDark(self, calculate = False, darkFilePath = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/02072025/R5_1_Neustart_2025-07-02_18-58-14/Spectra/Dark/20250702_185956_R5_1_Neustart_Spectrum_dark_Dark.dat'):
        '''This function imports the dark measurement data from the file, and subtracts it from the light measurement data.
        The dark measurement data is stored in the following format:
        - wavelengths: list of wavelengths
        - values: list of values
        The dark measurement data is used to calibrate the light measurement data.'''
        data = np.loadtxt(darkFilePath, skiprows=21)
        self.darkWavelength = data[:, 1]
        self.darkValue = data[:, 2]
        if calculate is True:
            for i in range (0, len(self.values)):
                self.values[i] = self.values[i] - self.darkValue

    def importData(self, newMode = True):
        '''Import the data from the folder, and do some basic analysis on the data.
        The data is stored in the following format:
        - wavelengths: list of wavelengths
        - values: list of values
        - timestamps: list of timestamps
        - temperatures: list of temperatures
        - labels: list of labels (dark or light)'''
        self.dat_files = [f for f in os.listdir(self.folderPath) if f.endswith('.dat')]
        for fileName in self.dat_files:
            print("Debug:",fileName)
            filePath = os.path.join(self.folderPath, fileName)
            data = np.loadtxt(filePath, skiprows=21)
            wavelength = data[:, 1].tolist()
            value = data[:, 2].tolist()
            #Label: dark or light
            fnameLower = fileName.lower()
            if "dark" in fnameLower:
                label = "Dark"
            elif "light" in fnameLower:
                label = "Light"
            else:
                label = "Unknown"
            #Import timestamp:
            with open(filePath, 'r') as f:
                lines = f.readlines()
                first_line = lines[0].strip()
                temp_line = lines[6].strip() #Old log(before August): lines[5]
                timestampStr = first_line.split(":", 1)[1].strip()
                timestamp = datetime.strptime(timestampStr, "%Y-%m-%d %H:%M:%S")
                #Import Cycle Number(New):
                cycleLine = lines[3].strip()
                match = re.search(r'\d+', cycleLine)
                if match:
                    cycleNum = int(match.group())
                    self.cycleNum.append(cycleNum)
                else:
                    print("No integer found")
                    self.cycleNum.append(np.nan)
            
            #Import temperature:
                temp_str = temp_line.split(":")[1].strip().split()[0]
                temperature = float(temp_str)
            #Log the peak position, height and the FWHM:(editing)
                for line in lines:
                    if "# INFO : no cal params" in line:
                        print(f"Bad data detected: '# INFO : no cal params' found at {fileName}, {timestamp}.")
                        self.StatusLabel.append(False)
                        continue  # Exit the function immediately
                # Convert to NumPy array for slicing
                self.StatusLabel.append(True)
                wavelength = np.array(wavelength)
                value = np.array(value)

                # Find peak
                max_index = np.argmax(value)
                peak_position = wavelength[max_index]
                peak_height = value[max_index]
                half_max = peak_height / 2

                # Find left half max index (closest from the left)
                left_indices = np.where(value[:max_index] < half_max)[0]
                if left_indices.size > 0:
                    left_index = left_indices[-1]
                    # Linear interpolation
                    x1, x2 = wavelength[left_index], wavelength[left_index + 1]
                    y1, y2 = value[left_index], value[left_index + 1]
                    left_half = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                else:
                    left_half = wavelength[0]

                # Find right half max index (closest from the right)
                right_indices = np.where(value[max_index:] < half_max)[0]
                if right_indices.size > 0:
                    right_index = right_indices[0] + max_index
                    # Linear interpolation
                    x1, x2 = wavelength[right_index - 1], wavelength[right_index]
                    y1, y2 = value[right_index - 1], value[right_index]
                    right_half = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
                else:
                    right_half = wavelength[-1]

                fwhm = right_half - left_half

            #Load the data in the module:
            if np.sum(wavelength) == 0:
                pass
            else:
                self.wavelengths.append(wavelength)
                self.values.append(value)
                self.temperatures.append(temperature)
                self.labels.append(label)
                self.timestamps.append(timestamp)
                self.PeakHeight.append(peak_height)
                self.PeakPosition.append(peak_position)
                self.PeakFWHM.append(fwhm)
                self.fileNameRaw.append(fileName)

        #Sort the data by the timestamp:
        combined = list(
            zip(
                self.timestamps,
                self.wavelengths,
                self.values,
                self.temperatures,
                self.labels,
                self.PeakPosition,
                self.PeakHeight,
                self.PeakFWHM,
                self.fileNameRaw,
                self.cycleNum,
            )
        )
        sorted_combined = sorted(combined, key=lambda x: x[0])
        (
            self.timestamps,
            self.wavelengths,
            self.values,
            self.temperatures,
            self.labels,
            self.PeakPosition,
            self.PeakHeight,
            self.PeakFWHM,
            self.fileNameRaw,
            self.cycleNum,
        ) = map(list, zip(*sorted_combined))
        start_time = self.timestamps[0]
        self.timestampAbsS = [(t - start_time).total_seconds() for t in self.timestamps]
        self.timestampAbsM = [s / 60 for s in self.timestampAbsS]
        self.timestampAbsH = [s / 3600 for s in self.timestampAbsS]
        self.file_metadata = {}
        for fname, timestamp, temperature, cycle in zip(
            self.fileNameRaw, self.timestamps, self.temperatures, self.cycleNum
        ):
            self.file_metadata[fname] = {
                "timestamp": timestamp,
                "temperature": temperature,
                "cycle": cycle,
            }
        #Find out which cycle it was on:
        '''
        for i in range(len(self.timestampAbsS)):
            self.cycleNum.append((self.timestampAbsS[i]//5520+1))
        '''

    def importLight(self, calculate = False, lightFilePath = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/LightValue/Rudong_Spectra_Light_500.dat'):
        data = np.loadtxt(lightFilePath, skiprows=21)
        self.lightWavelength = data[:, 1]
        self.lightValue = data[:, 2]
        if calculate is True:
            self.lightValue = self.lightValue - self.darkValue
            pass #Need to be done!!!
    
    def interpolate_masked(self, step=0.1, wl_min=334, wl_max=883):
        """Interpolate all spectra onto a common grid with bounded resolution.

        A coarse step keeps the memory footprint manageable given the large
        number of spectra in a run. The default 0.1 nm step yields ~5k points
        per spectrum instead of several hundred thousand.
        """
        if not self.wavelengths:
            return

        # Build a shared wavelength grid covering the requested range.
        grid = np.arange(wl_min, wl_max + step / 2, step)

        # Clear any previous interpolation results if Pipeline is re-used.
        self.wavelengths_interp = []
        self.values_interp = []

        for wavelength, value in zip(self.wavelengths, self.values):
            # Ensure ascending order for np.interp
            if wavelength[0] > wavelength[-1]:
                wavelength = wavelength[::-1]
                value = value[::-1]
            interp_values = np.interp(grid, wavelength, value)
            self.wavelengths_interp.append(grid)
            self.values_interp.append(interp_values)

        # Interpolate dark and light references onto the same grid.
        dark_wl, dark_val = self.darkWavelength, self.darkValue
        if dark_wl[0] > dark_wl[-1]:
            dark_wl = dark_wl[::-1]
            dark_val = dark_val[::-1]
        self.darkWavelength_interp = grid
        self.darkValue_interp = np.interp(grid, dark_wl, dark_val)

        light_wl, light_val = self.lightWavelength, self.lightValue
        if light_wl[0] > light_wl[-1]:
            light_wl = light_wl[::-1]
            light_val = light_val[::-1]
        self.lightWavelength_interp = grid
        self.lightValue_interp = np.interp(grid, light_wl, light_val)

    def average_value(self):
        """
        Average the spectra data for each 'cxxx' group from -1, -2, and -3 measurements.
        Store averaged values and error (standard deviation of the mean), and record new filenames.
        Also, compute and store averaged absolute timestamps in seconds as self.timestampAbsSN.
        """
        self.file_group_map = {}
        self.group_members = []
        group_dict = defaultdict(list)
        self.cycleNum_avg = []
        #Get wavelength, value and the name(After interpolate)
        for wl, val, fname in zip(self.wavelengths_interp, self.values_interp, self.fileNameRaw):
            # Extract group key from filename: e.g., "c000" from "..._c000-1.dat"
            try:
                group_key = fname.split("_")[-1].split("-")[0]
                group_dict[group_key].append((wl, val, fname))
                self.file_group_map[fname] = group_key
            except IndexError:
                print(f"Skipping file due to name parsing error: {fname}")
                continue

        for group_key, spectra_list in group_dict.items():
            if len(spectra_list) < 1:
                print(f"Skipping {group_key} due to no data.")
                continue

            wavelengths = spectra_list[0][0]
            all_values = np.array([val for wl, val, fname in spectra_list])

            # Sanity check: all wavelength arrays must be identical
            if not all(np.array_equal(wl, wavelengths) for wl, val, fname in spectra_list):
                print(f"Wavelength mismatch in group {group_key}, skipping.")
                continue

            avg_values = np.mean(all_values, axis=0)
            AVR = avg_values.tolist()

            if len(all_values) > 1:
                std_dev = np.std(all_values, axis=0, ddof=1)
                err = std_dev / np.sqrt(len(all_values))
            else:
                # Avoid numpy warnings when only a single trace is available for a group
                std_dev = np.zeros_like(avg_values)
                err = std_dev

            ERR = err.tolist()
            self.values_interpAvr.append(AVR)
            self.values_interpErr.append(ERR)
            avg_fname = f"average_{group_key}.dat"
            self.fileNameAveraged.append(avg_fname)
            self.file_group_map[avg_fname] = group_key
            group_filenames = [fname for _, _, fname in spectra_list]
            self.group_members.append({
                "group_key": group_key,
                "filenames": group_filenames,
            })

            # Averaged timestamp for this group: take the average of the corresponding timestamps
            # Find the indices in self.fileNameRaw corresponding to this group
            indices = []
            for _, _, fname in spectra_list:
                try:
                    idx = self.fileNameRaw.index(fname)
                    indices.append(idx)
                except ValueError:
                    continue

            if indices:
                timestamps_for_group = [self.timestamps[i] for i in indices]
                # Compute average datetime (as float seconds since epoch, then convert back)
                avg_timestamp_float = np.mean([dt.timestamp() for dt in timestamps_for_group])
                avg_timestamp_dt = datetime.fromtimestamp(avg_timestamp_float)
                if not hasattr(self, 'timestamps'):
                    self.timestamps = []
                self.timestamps.append(avg_timestamp_dt)
                # Now, append to self.timestampAbsSN the absolute timestamp in seconds relative to the first timestamp in self.timestamps
                if not hasattr(self, 'timestampAbsSN'):
                    self.timestampAbsSN = []
                start_time = self.timestamps[0]
                self.timestampAbsSN.append((avg_timestamp_dt - start_time).total_seconds())
                # Inserted: Store times in minutes and hours
                if not hasattr(self, 'timestampAbsMN'):
                    self.timestampAbsMN = []
                if not hasattr(self, 'timestampAbsHN'):
                    self.timestampAbsHN = []
                self.timestampAbsMN.append(self.timestampAbsSN[-1] / 60)
                self.timestampAbsHN.append(self.timestampAbsSN[-1] / 3600)

                cycles_for_group = []
                for idx in indices:
                    if idx < len(self.cycleNum):
                        cycles_for_group.append(self.cycleNum[idx])
                cycle_value = np.nan
                if cycles_for_group:
                    cycles_for_group = np.asarray(cycles_for_group, dtype=float)
                    valid_cycles = cycles_for_group[~np.isnan(cycles_for_group)]
                    if valid_cycles.size:
                        counts = defaultdict(int)
                        for val in valid_cycles:
                            counts[int(val)] += 1
                        max_count = max(counts.values())
                        candidates = [cycle for cycle, cnt in counts.items() if cnt == max_count]
                        cycle_value = int(min(candidates))

                self.cycleNum_avg.append(cycle_value)
            else:
                self.values_interpAvr.pop()
                self.values_interpErr.pop()
                self.fileNameAveraged.pop()

    def taucCalc(
        self,
        mean=True,
        *,
        auto_window=True,
        search_range=(1.50, 1.65),      # Search for the maximum slope within this range
        slope_half_width=0.01,          # +/- half-width used when computing the local slope
        fit_half_width=0.02,            # +/- half-width defining the final fitting window
        min_window_points=6,            # Minimum number of points allowed in the rolling fit
        epsilon=1e-10,                  # Prevent log10(0) issues
        tauc_power=2.0,                 # Tauc exponent; 2.0 corresponds to a direct band gap
        smooth_window=0,                # Savitzky-Golay window (set to 0 or None to disable)
        smooth_polyorder=2,             # Polynomial order for smoothing
    ):
        """
        Perform a Tauc calculation with automatic fitting-window selection.
        Steps:
        1) Within ``search_range`` perform a local linear fit with +/-``slope_half_width`` around every energy point to obtain a slope spectrum;
        2) Identify the energy ``E*`` with the maximum slope;
        3) Fit a straight line on ``[E* - fit_half_width, E* + fit_half_width]`` and use its x-intercept as the band gap.
        """
        self.tauc_last_mode = "mean" if mean else "raw"
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import pandas as pd
        import os
        from datetime import datetime

        def _prepare_energy_axis():
            # Keep using self.wavelengths_interp[1] to stay consistent with the original code
            eV = 1240 / np.array(self.wavelengths_interp[1], dtype=float)
            # Sort by ascending energy to make the calculation robust
            sort_idx = np.argsort(eV)
            return eV[sort_idx], sort_idx

        def _calc_tauc(trans, eV):
            absorb = -np.log10(np.maximum(trans, epsilon))
            return (absorb * eV) ** tauc_power

        def _apply_smoothing(array):
            """Apply Savitzky-Golay smoothing if the window is valid."""
            if not smooth_window or smooth_window < 3:
                return array
            data = np.asarray(array, dtype=float)
            n_points = data.size
            if n_points < 3:
                return data
            window = max(3, int(smooth_window))
            if window % 2 == 0:
                window += 1
            if window > n_points:
                window = n_points if n_points % 2 == 1 else n_points - 1
            if window <= 2:
                return data
            polyorder = min(int(smooth_polyorder), window - 2)
            if polyorder < 1:
                polyorder = 1
            if window <= polyorder:
                return data
            return savgol_filter(data, window_length=window, polyorder=polyorder)

        def _align_sequence(seq, target_len, fill_value=np.nan):
            """Return a list with a fixed length by trimming or padding."""
            if seq is None:
                return [fill_value] * target_len
            seq_list = list(seq)
            current_len = len(seq_list)
            if current_len >= target_len:
                return seq_list[:target_len]
            return seq_list + [fill_value] * (target_len - current_len)

        def _rolling_slope(x, y, half_width, min_pts):
            """
            For each ``x[i]`` perform a linear regression on ``[x[i]-half_width, x[i]+half_width]``.
            Return the slope for every center point; mark entries with too few points as NaN.
            """
            n = x.size
            slopes = np.full(n, np.nan, dtype=float)
            model = LinearRegression()
            # Use a two-pointer scheme to improve efficiency
            L = 0
            R = 0
            for i in range(n):
                xi = x[i]
                # Expand the window to the left boundary
                while L < n and x[L] < xi - half_width:
                    L += 1
                # Expand the window to the right boundary
                while R < n and x[R] <= xi + half_width:
                    R += 1
                if R - L >= min_pts:
                    Xw = x[L:R].reshape(-1, 1)
                    Yw = y[L:R].reshape(-1, 1)
                    # Filter out NaN values
                    mask = ~(np.isnan(Xw) | np.isnan(Yw)).ravel()
                    if np.count_nonzero(mask) >= min_pts:
                        model.fit(Xw[mask], Yw[mask])
                        slopes[i] = float(model.coef_.ravel()[0])
            return slopes

        def _run_tauc_for(series_values, series_labels, mode_name):
            """Return Tauc results for the provided series without mutating shared state."""
            result = {
                "mode": mode_name,
                "sources": [],
                "valueTaucs": [],
                "bandGap": [],
                "tauc_slope": [],
                "tauc_slope_b": [],
                "tauc_fit_r2": [],
                "tauc_fit_residual": [],
                "tauc_window_used": [],
                "tauc_fit_lines": [],
                "tauc_qc_pass": [],
                "tauc_qc_reason": [],
                "tauc_attempt": [],
                "debug_auto_window": [],
            }

            if not series_values:
                return result

            dark_sorted = np.array(self.darkValue_interp, dtype=float)[sort_idx]
            light_sorted = np.array(self.lightValue_interp, dtype=float)[sort_idx]
            denom = np.maximum(light_sorted - dark_sorted, epsilon)

            for i, vals in enumerate(series_values):
                name = (
                    series_labels[i]
                    if series_labels is not None and i < len(series_labels)
                    else f"{mode_name}_{i}"
                )

                result["sources"].append(name)

                vals_sorted = np.array(vals, dtype=float)[sort_idx]
                valueCalibrated = vals_sorted - dark_sorted
                Trans = valueCalibrated / denom
                tauc = _calc_tauc(Trans, eV_raw)
                tauc = _apply_smoothing(tauc)
                result["valueTaucs"].append(tauc)

                # ---------- Automatic window selection ----------
                if auto_window:
                    search_mask = (eV_raw >= min(search_range)) & (eV_raw <= max(search_range))
                    if np.count_nonzero(search_mask) < min_window_points:
                        fit_lo, fit_hi = search_range
                    else:
                        slopes = _rolling_slope(
                            eV_raw[search_mask], tauc[search_mask], slope_half_width, min_window_points
                        )
                        if np.all(np.isnan(slopes)):
                            fit_lo, fit_hi = search_range
                        else:
                            k = int(np.nanargmax(slopes))
                            E_star = eV_raw[search_mask][k]
                            fit_lo = E_star - fit_half_width
                            fit_hi = E_star + fit_half_width

                    fit_lo = max(fit_lo, float(np.nanmin(eV_raw)))
                    fit_hi = min(fit_hi, float(np.nanmax(eV_raw)))
                else:
                    fit_lo, fit_hi = (1.50, 1.60) if mode_name == "mean" else (1.55, 1.60)

                # ---------- Linear fit and calculate the band gap ----------
                fit_mask = (eV_raw >= fit_lo) & (eV_raw <= fit_hi)
                X_sel = eV_raw[fit_mask].reshape(-1, 1)
                Y_sel = tauc[fit_mask].reshape(-1, 1)

                ok = ~(np.isnan(X_sel) | np.isnan(Y_sel)).ravel()
                if np.count_nonzero(ok) >= 2:
                    model = LinearRegression()
                    model.fit(X_sel[ok], Y_sel[ok])
                    slope = float(model.coef_.ravel()[0])
                    b = float(model.intercept_.ravel()[0])
                    x_intercept = -b / slope if not np.isclose(slope, 0.0) else np.nan
                    y_pred = (slope * X_sel + b).ravel()
                    ss_res = float(np.sum((Y_sel.ravel()[ok] - y_pred[ok]) ** 2))
                    ss_tot = float(np.sum((Y_sel.ravel()[ok] - np.mean(Y_sel.ravel()[ok])) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    rmse = float(np.sqrt(ss_res / max(np.count_nonzero(ok), 1)))
                else:
                    slope = np.nan
                    b = np.nan
                    x_intercept = np.nan
                    y_pred = np.full_like(Y_sel.ravel(), np.nan)
                    r2 = np.nan
                    rmse = np.nan

                result["bandGap"].append(x_intercept)
                result["tauc_slope"].append(slope)
                result["tauc_slope_b"].append(b)
                result["tauc_fit_r2"].append(r2)
                result["tauc_fit_residual"].append(rmse)
                window_tuple = (
                    float(np.nanmin(X_sel)) if X_sel.size else float("nan"),
                    float(np.nanmax(X_sel)) if X_sel.size else float("nan"),
                )
                result["tauc_window_used"].append(window_tuple)
                is_ok = np.isfinite(x_intercept)
                result["tauc_qc_pass"].append(is_ok)
                result["tauc_qc_reason"].append("ok" if is_ok else "too_few_points")
                result["tauc_attempt"].append("primary")

                debug_entry = {
                    "source": name,
                    "x_start": float(window_tuple[0]),
                    "x_end": float(window_tuple[1]),
                    "y_start": float(y_pred[0]) if y_pred.size else float("nan"),
                    "y_end": float(y_pred[-1]) if y_pred.size else float("nan"),
                    "r2": float(r2) if np.isfinite(r2) else float("nan"),
                    "rmse": float(rmse) if np.isfinite(rmse) else float("nan"),
                    "Eg": float(x_intercept) if np.isfinite(x_intercept) else float("nan"),
                    "fit_window": window_tuple,
                }
                result["tauc_fit_lines"].append(debug_entry)
                result["debug_auto_window"].append(
                    {
                        "name": name,
                        "search_range": tuple(search_range),
                        "slope_half_width": slope_half_width,
                        "fit_half_width": fit_half_width,
                        "fit_window": (fit_lo, fit_hi),
                        "Eg": x_intercept,
                        "slope": slope,
                        "intercept": b,
                    }
                )

            return result

        eV_raw, sort_idx = _prepare_energy_axis()
        self.eV = eV_raw

        dataset_configs = []
        if self.values_interp:
            dataset_configs.append(("raw", self.values_interp, getattr(self, "fileNameRaw", [])))
        if mean and getattr(self, "values_interpAvr", None):
            dataset_configs.append(("mean", self.values_interpAvr, getattr(self, "fileNameAveraged", [])))

        if not dataset_configs:
            raise RuntimeError("No spectra available for Tauc calculation.")

        self.tauc_results = {}
        for mode_name, data_values, label_list in dataset_configs:
            self.tauc_results[mode_name] = _run_tauc_for(data_values, label_list, mode_name)

        active_mode = self.tauc_last_mode if self.tauc_last_mode in self.tauc_results else dataset_configs[0][0]
        active = self.tauc_results.get(active_mode, {})

        self.valueTaucs = list(active.get("valueTaucs", []))
        self.bandGap = list(active.get("bandGap", []))
        self.tauc_slope = list(active.get("tauc_slope", []))
        self.tauc_slope_b = list(active.get("tauc_slope_b", []))
        self.tauc_fit_r2 = list(active.get("tauc_fit_r2", []))
        self.tauc_fit_residual = list(active.get("tauc_fit_residual", []))
        self.tauc_window_used = list(active.get("tauc_window_used", []))
        self.tauc_fit_lines = list(active.get("tauc_fit_lines", []))
        self.tauc_qc_pass = list(active.get("tauc_qc_pass", []))
        self.tauc_qc_reason = list(active.get("tauc_qc_reason", []))
        self.tauc_attempt = list(active.get("tauc_attempt", []))
        self.debug_auto_window = list(active.get("debug_auto_window", []))

        # --------- Save Result ---------
        results_folder = os.path.join(self.folderPath, "Results")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        timestamp_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(results_folder, f"band_gap_results_{timestamp_now}.csv")

        entry_count = len(self.bandGap)
        cycle_number_avg = _align_sequence(getattr(self, "cycleNum_avg", []), entry_count)
        temperature_log = _align_sequence(getattr(self, "temperatures", []), entry_count)

        if mean:
            taucData = {
                "timestamp(in Hour):": getattr(self, "timestampAbsHN", []),
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b,
                "tauc_r2": self.tauc_fit_r2,
                "tauc_rmse": self.tauc_fit_residual,
                "window_start": [w[0] for w in self.tauc_window_used],
                "window_end": [w[1] for w in self.tauc_window_used],
                "cycle_number_avg": cycle_number_avg,
                "temperature": temperature_log,
            }
        else:
            taucData = {
                "timestamp": getattr(self, "timestamps", []),
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b,
                "tauc_r2": self.tauc_fit_r2,
                "tauc_rmse": self.tauc_fit_residual,
                "window_start": [w[0] for w in self.tauc_window_used],
                "window_end": [w[1] for w in self.tauc_window_used],
                "cycle_number_avg": cycle_number_avg,
                "temperature": temperature_log,
            }

        df = pd.DataFrame(taucData)
        df.to_csv(results_file, index=False)

        raw_result = self.tauc_results.get("raw")
        if raw_result and raw_result.get("bandGap"):
            self._log_band_gap_groups(raw_result, results_folder, timestamp_now)

    def _log_band_gap_groups(self, raw_result, results_folder, timestamp_suffix):
        """Persist per-measurement band gaps and grouped uncertainties."""
        sources = raw_result.get("sources", [])
        band_gaps = raw_result.get("bandGap", [])
        slopes = raw_result.get("tauc_slope", [])
        intercepts = raw_result.get("tauc_slope_b", [])
        r2_values = raw_result.get("tauc_fit_r2", [])
        rmses = raw_result.get("tauc_fit_residual", [])
        windows = raw_result.get("tauc_window_used", [])

        rows = []
        for source, eg, slope, intercept, r2, rmse, window in zip(
            sources, band_gaps, slopes, intercepts, r2_values, rmses, windows
        ):
            meta = self.file_metadata.get(source, {}) if hasattr(self, "file_metadata") else {}
            timestamp = meta.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            group_key = self.file_group_map.get(source, "") if hasattr(self, "file_group_map") else ""
            window_start = window[0] if isinstance(window, (tuple, list)) and window else np.nan
            window_end = window[1] if isinstance(window, (tuple, list)) and len(window) > 1 else np.nan
            rows.append(
                {
                    "group_key": group_key,
                    "source": source,
                    "timestamp": timestamp,
                    "temperature": meta.get("temperature"),
                    "cycle": meta.get("cycle"),
                    "band_gap": eg,
                    "slope": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "rmse": rmse,
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )

        if rows:
            raw_path = os.path.join(results_folder, f"band_gap_raw_measurements_{timestamp_suffix}.csv")
            pd.DataFrame(rows).to_csv(raw_path, index=False)

        grouped_values = defaultdict(list)
        grouped_sources = defaultdict(list)
        for row in rows:
            key = row["group_key"]
            val = row["band_gap"]
            if key and val is not None and not np.isnan(val):
                grouped_values[key].append(float(val))
                grouped_sources[key].append(row["source"])

        stats_rows = []
        for key, values in grouped_values.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            mean_val = float(np.mean(arr))
            if arr.size > 1:
                std_val = float(np.std(arr, ddof=1))
            else:
                std_val = 0.0
            sem_val = float(std_val / np.sqrt(arr.size)) if arr.size > 0 else float("nan")
            stats_rows.append(
                {
                    "group_key": key,
                    "n_measurements": int(arr.size),
                    "band_gap_mean": mean_val,
                    "band_gap_std": std_val,
                    "band_gap_sem": sem_val,
                    "sources": ";".join(grouped_sources[key]),
                }
            )

        if stats_rows:
            stats_path = os.path.join(results_folder, f"band_gap_group_stats_{timestamp_suffix}.csv")
            pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    def _log_urbach_groups(self, raw_result, results_folder, timestamp_suffix):
        """Persist raw Urbach measurements and grouped uncertainty."""
        sources = raw_result.get("sources", [])
        energies = raw_result.get("urbach_energy", [])
        slopes = raw_result.get("urbach_slope", [])
        intercepts = raw_result.get("urbach_slope_b", [])
        r2_values = raw_result.get("urbach_fit_r2", [])
        rmses = raw_result.get("urbach_fit_residual", [])
        windows = raw_result.get("urbach_window_used", [])

        rows = []
        for source, energy, slope, intercept, r2, rmse, window in zip(
            sources, energies, slopes, intercepts, r2_values, rmses, windows
        ):
            meta = self.file_metadata.get(source, {}) if hasattr(self, "file_metadata") else {}
            timestamp = meta.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            group_key = self.file_group_map.get(source, "") if hasattr(self, "file_group_map") else ""
            window_start = window[0] if isinstance(window, (tuple, list)) and window else np.nan
            window_end = window[1] if isinstance(window, (tuple, list)) and len(window) > 1 else np.nan
            rows.append(
                {
                    "group_key": group_key,
                    "source": source,
                    "timestamp": timestamp,
                    "temperature": meta.get("temperature"),
                    "cycle": meta.get("cycle"),
                    "urbach_energy": energy,
                    "slope": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "rmse": rmse,
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )

        if rows:
            raw_path = os.path.join(results_folder, f"urbach_raw_measurements_{timestamp_suffix}.csv")
            pd.DataFrame(rows).to_csv(raw_path, index=False)

        grouped_values = defaultdict(list)
        grouped_sources = defaultdict(list)
        for row in rows:
            key = row["group_key"]
            val = row["urbach_energy"]
            if key and val is not None and not np.isnan(val):
                grouped_values[key].append(float(val))
                grouped_sources[key].append(row["source"])

        stats_rows = []
        for key, values in grouped_values.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                continue
            mean_val = float(np.mean(arr))
            if arr.size > 1:
                std_val = float(np.std(arr, ddof=1))
            else:
                std_val = 0.0
            sem_val = float(std_val / np.sqrt(arr.size)) if arr.size > 0 else float("nan")
            stats_rows.append(
                {
                    "group_key": key,
                    "n_measurements": int(arr.size),
                    "urbach_mean": mean_val,
                    "urbach_std": std_val,
                    "urbach_sem": sem_val,
                    "sources": ";".join(grouped_sources[key]),
                }
            )

        if stats_rows:
            stats_path = os.path.join(results_folder, f"urbach_group_stats_{timestamp_suffix}.csv")
            pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    def calc_urbach(
        self,
        *,
        energy_range=(1.5, 1.6), 
        slope_half_width=0.025,
        fit_half_width=0.02,
        min_window_points=5,
        epsilon=1e-12,
        mode=None,
    ):
        """Calculate Urbach energies for the requested Tauc mode."""
        self.urbach_x = np.asarray(getattr(self, "eV", []), dtype=float)
        if self.urbach_x.size == 0:
            raise RuntimeError("Energy axis unavailable; run taucCalc before calc_urbach.")

        if not getattr(self, "tauc_results", None):
            raise RuntimeError("No Tauc results cached; run taucCalc first.")

        mode = mode or self.tauc_last_mode or "raw"
        if mode not in self.tauc_results:
            raise RuntimeError(f"No Tauc spectra found for mode '{mode}'.")

        spectra = self.tauc_results[mode].get("valueTaucs", [])
        sources = list(self.tauc_results[mode].get("sources", []))
        if not spectra:
            return

        def _rolling_slope(x, y, half_width, min_pts):
            slopes = np.full(x.size, np.nan, dtype=float)
            model = LinearRegression()
            left = 0
            right = 0
            for idx, xi in enumerate(x):
                while left < x.size and x[left] < xi - half_width:
                    left += 1
                while right < x.size and x[right] <= xi + half_width:
                    right += 1
                if right - left >= min_pts:
                    Xw = x[left:right].reshape(-1, 1)
                    Yw = y[left:right].reshape(-1, 1)
                    mask = ~(np.isnan(Xw) | np.isnan(Yw)).ravel()
                    if np.count_nonzero(mask) >= min_pts:
                        model.fit(Xw[mask], Yw[mask])
                        slopes[idx] = float(model.coef_.ravel()[0])
            return slopes

        result = {
            "mode": mode,
            "sources": sources,
            "urbach_y": [],
            "urbach_slope": [],
            "urbach_slope_b": [],
            "urbach_fit_r2": [],
            "urbach_fit_residual": [],
            "urbach_window_used": [],
            "urbach_energy": [],
        }

        eV = self.urbach_x

        for idx, tauc_values in enumerate(spectra):
            spectrum = np.asarray(tauc_values, dtype=float)
            ln_vals = np.log(np.clip(spectrum, epsilon, None))
            result["urbach_y"].append(ln_vals)

            search_mask = (eV >= energy_range[0]) & (eV <= energy_range[1])
            if np.count_nonzero(search_mask) < min_window_points:
                fit_lo, fit_hi = energy_range
            else:
                slopes = _rolling_slope(eV[search_mask], ln_vals[search_mask], slope_half_width, min_window_points)
                if np.all(np.isnan(slopes)):
                    fit_lo, fit_hi = energy_range
                else:
                    max_idx = int(np.nanargmax(slopes))
                    center = eV[search_mask][max_idx]
                    fit_lo = center - fit_half_width
                    fit_hi = center + fit_half_width

            fit_lo = max(fit_lo, float(np.nanmin(eV)))
            fit_hi = min(fit_hi, float(np.nanmax(eV)))

            fit_mask = (eV >= fit_lo) & (eV <= fit_hi)
            X_sel = eV[fit_mask].reshape(-1, 1)
            Y_sel = ln_vals[fit_mask].reshape(-1, 1)

            ok = ~(np.isnan(X_sel) | np.isnan(Y_sel)).ravel()
            if np.count_nonzero(ok) >= 2:
                model = LinearRegression()
                model.fit(X_sel[ok], Y_sel[ok])
                slope = float(model.coef_.ravel()[0])
                intercept = float(model.intercept_.ravel()[0])
                y_pred = model.predict(X_sel[ok]).ravel()
                residuals = Y_sel[ok].ravel() - y_pred
                ss_res = float(np.sum(residuals ** 2))
                ss_tot = float(np.sum((Y_sel[ok].ravel() - np.mean(Y_sel[ok])) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                rmse = float(np.sqrt(ss_res / max(np.count_nonzero(ok), 1)))
                intercept_energy = np.nan if np.isclose(slope, 0.0) else 1 / slope
            else:
                slope = np.nan
                intercept = np.nan
                r2 = np.nan
                rmse = np.nan
                intercept_energy = np.nan

            result["urbach_slope"].append(slope)
            result["urbach_slope_b"].append(intercept)
            result["urbach_fit_r2"].append(r2)
            result["urbach_fit_residual"].append(rmse)
            result["urbach_window_used"].append((fit_lo, fit_hi))
            result["urbach_energy"].append(intercept_energy)

        self.urbach_results[mode] = result
        if mode == self.tauc_last_mode:
            self.urbach_y = list(result["urbach_y"])
            self.urbach_slope = list(result["urbach_slope"])
            self.urbach_slope_b = list(result["urbach_slope_b"])
            self.urbach_fit_r2 = list(result["urbach_fit_r2"])
            self.urbach_fit_residual = list(result["urbach_fit_residual"])
            self.urbach_window_used = list(result["urbach_window_used"])
            self.urbach_energy = list(result["urbach_energy"])

        entry_count = len(result["urbach_energy"])
        if entry_count == 0:
            return

        def _align(seq, target_len, fill_value):
            items = list(seq) if seq is not None else []
            if len(items) >= target_len:
                return items[:target_len]
            return items + [fill_value] * (target_len - len(items))

        if mode == "mean":
            time_key = "timestamp(in Hour):"
            time_axis = _align(getattr(self, "timestampAbsHN", []), entry_count, np.nan)
            cycles = _align(getattr(self, "cycleNum_avg", []), entry_count, np.nan)
            source_names = _align(getattr(self, "fileNameAveraged", []), entry_count, "")
        else:
            time_key = "timestamp"
            time_axis = _align(getattr(self, "timestamps", []), entry_count, pd.NaT)
            cycles = _align(getattr(self, "cycleNum", []), entry_count, np.nan)
            source_names = _align(getattr(self, "fileNameRaw", []), entry_count, "")

        window_start = [w[0] if isinstance(w, (tuple, list)) and len(w) > 0 else np.nan for w in result["urbach_window_used"]]
        window_end = [w[1] if isinstance(w, (tuple, list)) and len(w) > 1 else np.nan for w in result["urbach_window_used"]]

        results_folder = os.path.join(self.folderPath, "Results")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        timestamp_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(results_folder, f"urbach_results_{mode}_{timestamp_now}.csv")

        df_payload = {
            "source": source_names,
            time_key: time_axis,
            "cycle": cycles,
            "urbach_energy": result["urbach_energy"],
            "urbach_slope": result["urbach_slope"],
            "urbach_intercept": result["urbach_slope_b"],
            "urbach_r2": result["urbach_fit_r2"],
            "urbach_rmse": result["urbach_fit_residual"],
            "window_start": window_start,
            "window_end": window_end,
        }

        pd.DataFrame(df_payload).to_csv(results_file, index=False)

        if mode == "raw":
            self._log_urbach_groups(result, results_folder, timestamp_now)

    def calc_urbach_energy(self):
        if not self.bandGap or not self.urbach_energy:
            self.energyDifference = []
            return

        band_gap_arr = np.asarray(self.bandGap, dtype=float)
        urbach_arr = np.asarray(self.urbach_energy, dtype=float)

        length = min(band_gap_arr.size, urbach_arr.size)
        if length == 0:
            self.energyDifference = []
            return

        diff = np.abs(band_gap_arr[:length] - urbach_arr[:length])
        self.energyDifference = diff.tolist()

    def Pipeline(
        self,
        darkFolder=None,
        lightFilePath=None,
        newMode=True,
        mean=True,
        calculate_dark=False,
        calculate_light=False,
        **tauc_kwargs,
    ):
        '''Run the full spectroscopy pipeline.

        Parameters
        - darkFolder: path to dark measurement file (back-compat name).
        - lightFilePath: path to light reference file (optional).
        - newMode: pass-through for importData(newMode).
        - mean: compute Tauc on averaged spectra if True; per-scan if False.
        - calculate_dark: subtract dark immediately on importDark if True.
        - calculate_light: subtract dark immediately on importLight if True.
        - Additional keyword arguments are forwarded to taucCalc.
        '''
        # Import dark reference (respect provided path and calc flag)
        if darkFolder is not None:
            self.importDark(calculate=calculate_dark, darkFilePath=darkFolder)
        else:
            self.importDark(calculate=calculate_dark)

        # Import measurement data
        self.importData(newMode=newMode)

        # Import light reference (respect provided path and calc flag)
        if lightFilePath is not None:
            self.importLight(calculate=calculate_light, lightFilePath=lightFilePath)
        else:
            self.importLight(calculate=calculate_light)

        # Process and analyze
        self.interpolate_masked()
        self.average_value()
        self.taucCalc(mean=mean, **tauc_kwargs)
        self.calc_urbach(mode="raw")
        if self.tauc_last_mode != "raw":
            self.calc_urbach(mode=self.tauc_last_mode)
        self.calc_urbach_energy()
