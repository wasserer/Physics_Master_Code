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
        #Import Data:

    def importDark(self, calculate = False, darkFilePath = ""):  # Enter your file path/folder path in this place
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
        #Find out which cycle it was on:
        '''
        for i in range(len(self.timestampAbsS)):
            self.cycleNum.append((self.timestampAbsS[i]//5520+1))
        '''

    def importLight(self, calculate = False, lightFilePath = ""):  # Enter your file path/folder path in this place
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
        group_dict = defaultdict(list)
        self.cycleNum_avg = []
        #Get wavelength, value and the name(After interpolate)
        for wl, val, fname in zip(self.wavelengths_interp, self.values_interp, self.fileNameRaw):
            # Extract group key from filename: e.g., "c000" from "..._c000-1.dat"
            try:
                group_key = fname.split("_")[-1].split("-")[0]
                group_dict[group_key].append((wl, val, fname))
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
            self.fileNameAveraged.append(f"average_{group_key}.dat")

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

    def taucCalc(self, mean = False): #Don't use this yet
        '''This is the Tauc calculation, which is used to calculate the band gap of the material.
        The Tauc plot is a plot of (Absorbance * photon energy)^0.5 vs photon energy.
        The band gap can be found by fitting a line to the Tauc plot and finding the x-intercept.
        The formula for the Tauc plot is:
        Tauc = (Absorbance * photon energy)^2
        Where photon energy = 1240 / wavelength (in nm)'''
        lightCalibrated = self.lightValue_interp - self.darkValue_interp
        epsilon = 1e-10
        #X-Axis for tauc plot
        self.eV = 1240 / np.array(self.wavelengths_interp[1])
        if mean == False:
            for i in range(0, len(self.timestampAbsS)):
                print("DEBUG:", self.fileNameRaw[i])
                #Calculate the Absorbance:
                print("The time stamp is:", self.timestampAbsS[i])
                valueCalibrated = np.array(self.values_interp[i]) - np.array(self.darkValue_interp)
                Trans = valueCalibrated / lightCalibrated
                #print("Trans:", Trans)
                Absorb = - np.log10(np.maximum(Trans, epsilon))
                #print("Abs:", Absorb)
                #Calculate the Tauc and Band Gap: photonEnergy = 4.135667696e-15 * 2.99792e8 / wavelength
                tauc = (Absorb * self.eV)**2 #0.5
                self.valueTaucs.append(tauc)
                #print("Tauc Plot:", tauc)
                #Do the band-gap fit:
                mask = (self.eV >= 1.55) & (self.eV <= 1.6) #Select the range for band gap fit
                x_selected = self.eV[mask]
                y_selected = tauc[mask]
                #if y_selected contains nan, return 0.
                y_selected[np.isnan(y_selected)] = 0
                #print("x and y selected", x_selected, y_selected)
                #Fit the data:
                model = LinearRegression()
                print("Label: ", self.labels[i], "x", x_selected, "y", y_selected)
                model.fit(x_selected.reshape(-1, 1), y_selected.reshape(-1, 1))
                b = model.intercept_[0] #This is the intercept with the y-axis!
                slope = model.coef_[0] #This is the x-axis!
                #print("b:", b, "slope:", slope, self.labels[i]) #
                intercept = model.intercept_
                x_intercept = - intercept / slope #Intercept with x axis
                #Store the data:
                self.bandGap.append(x_intercept.tolist()[0])
                self.tauc_slope.append(slope)
                self.tauc_slope_b.append(b)
        else:
            for i in range(0, len(self.values_interpAvr)):
                print("The file name is:", self.fileNameAveraged[i])
                print("DEBUG:", self.fileNameRaw[i])
                valueCalibrated = np.array(self.values_interpAvr[i]) - np.array(self.darkValue_interp)
                Trans = valueCalibrated / lightCalibrated
                #print("Trans:", Trans)
                Absorb = - np.log10(np.maximum(Trans, epsilon))
                #print("Abs:", Absorb)
                #Calculate the Tauc and Band Gap: photonEnergy = 4.135667696e-15 * 2.99792e8 / wavelength
                tauc = (Absorb * self.eV)**2
                self.valueTaucs.append(tauc)
                #print("Tauc Plot:", tauc)
                #Do the band-gap fit:
                mask = (self.eV >= 1.5) & (self.eV <= 1.6) #Select the range for band gap fit
                x_selected = self.eV[mask]
                y_selected = tauc[mask]
                #if y_selected contains nan, return 0.
                y_selected[np.isnan(y_selected)] = 0
                #print("x and y selected", x_selected, y_selected)
                #Fit the data:
                model = LinearRegression()
                #print("Label: ", self.labels[i], "x", x_selected, "y", y_selected)
                model.fit(x_selected.reshape(-1, 1), y_selected.reshape(-1, 1))
                b = model.intercept_[0] #This is the intercept with the y-axis!
                slope = model.coef_[0] #This is the x-axis!
                #print("b:", b, "slope:", slope, self.labels[i]) #
                intercept = model.intercept_
                x_intercept = - intercept / slope #Intercept with x axis
                #Store the data:
                self.bandGap.append(x_intercept.tolist()[0])
                self.tauc_slope.append(slope)
                self.tauc_slope_b.append(b)
        # Log the band gap and tauc slope in a .csv file, and save the data in another folder called "Results", if there is no such folder, create it.
        results_folder = os.path.join(self.folderPath, "Results")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        timestamp_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(results_folder, f"band_gap_results_{timestamp_now}.csv")
        # Remove incorrect early DataFrame creation
        # Compose the correct data dictionary for the DataFrame
        if mean == True:
            taucData = {
                "timestamp(in Hour):": self.timestampAbsHN,
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b
            }
        else:
            taucData = {
                "timestamp": self.timestamps,
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b
            }
        df = pd.DataFrame(taucData)
        df.to_csv(results_file, index=False)

    def taucCalc_v1_1(
        self,
        mean=True,
        *,
        auto_window=True,
        search_range=(1.50, 1.65),      # 在此范围内寻找最大斜率点
        slope_half_width=0.01,          # 计算局部斜率的 ± 半窗宽
        fit_half_width=0.02,            # 最终拟合窗口的 ± 半窗宽
        min_window_points=6,            # 滚动拟合的最少点数
        epsilon=1e-10,                  # 防止 log10(0)
        tauc_power=2.0                  # Tauc 幂次，直接带隙=2；若需平均可改
    ):
        """
        自动选择拟合窗口的 Tauc 计算。
        步骤：
        1) 在 search_range 内，对每个能量点做 ±slope_half_width 的局部线性拟合，得到斜率谱；
        2) 取最大斜率的能量 E*；
        3) 用 [E*-fit_half_width, E*+fit_half_width] 做最终直线拟合，求 x 截距作为带隙。
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import pandas as pd
        import os
        from datetime import datetime

        def _prepare_energy_axis():
            # 你原来用 self.wavelengths_interp[1]，这里保持一致
            eV = 1240 / np.array(self.wavelengths_interp[1], dtype=float)
            # 为了稳健计算，统一按能量升序排序
            sort_idx = np.argsort(eV)
            return eV[sort_idx], sort_idx

        def _calc_tauc(trans, eV):
            absorb = -np.log10(np.maximum(trans, epsilon))
            return (absorb * eV) ** tauc_power

        def _rolling_slope(x, y, half_width, min_pts):
            """
            对每个 x[i]，取 [x[i]-half_width, x[i]+half_width] 的窗口做线性回归，返回 slope[i]。
            若点数不足则置为 NaN。
            """
            n = x.size
            slopes = np.full(n, np.nan, dtype=float)
            model = LinearRegression()
            # 使用双指针提升效率
            L = 0
            R = 0
            for i in range(n):
                xi = x[i]
                # 扩展到左边界
                while L < n and x[L] < xi - half_width:
                    L += 1
                # 扩展到右边界
                while R < n and x[R] <= xi + half_width:
                    R += 1
                if R - L >= min_pts:
                    Xw = x[L:R].reshape(-1, 1)
                    Yw = y[L:R].reshape(-1, 1)
                    # 过滤 NaN
                    mask = ~(np.isnan(Xw) | np.isnan(Yw)).ravel()
                    if np.count_nonzero(mask) >= min_pts:
                        model.fit(Xw[mask], Yw[mask])
                        slopes[i] = float(model.coef_.ravel()[0])
            return slopes

        # 公共预处理
        lightCalibrated = np.array(self.lightValue_interp, dtype=float) - np.array(self.darkValue_interp, dtype=float)
        eV_raw, sort_idx = _prepare_energy_axis()

        # 按 mean 模式选择数据序列
        series_values = self.values_interpAvr if mean else self.values_interp
        series_labels = getattr(self, "fileNameAveraged", None) if mean else getattr(self, "labels", None)

        # 清空累积容器（防重复调用叠加）
        self.eV = eV_raw  # 保存能量轴供后续绘图
        self.valueTaucs = []
        self.bandGap = []
        self.tauc_slope = []
        self.tauc_slope_b = []
        self.tauc_fit_r2 = []
        self.tauc_fit_residual = []
        self.tauc_window_used = []
        self.tauc_fit_lines = []
        self.tauc_qc_pass = []
        self.tauc_qc_reason = []
        self.tauc_attempt = []
        self.debug_auto_window = []  # 保存每次的自动窗口与斜率信息以便追踪

        for i, vals in enumerate(series_values):
            # Debug 名称
            name = series_labels[i] if series_labels is not None and i < len(series_labels) else f"series_{i}"

            # 同步排序（能量升序）
            vals_sorted = np.array(vals, dtype=float)[sort_idx]
            dark_sorted = np.array(self.darkValue_interp, dtype=float)[sort_idx]
            light_sorted = np.array(self.lightValue_interp, dtype=float)[sort_idx]
            eV = eV_raw.copy()

            # 计算透过率与 Tauc
            valueCalibrated = vals_sorted - dark_sorted
            Trans = valueCalibrated / np.maximum(light_sorted - dark_sorted, epsilon)
            tauc = _calc_tauc(Trans, eV)
            self.valueTaucs.append(tauc)

            # ---------- 自动窗口选择 ----------
            if auto_window:
                # 在 search_range 中计算滚动斜率
                search_mask = (eV >= min(search_range)) & (eV <= max(search_range))
                if np.count_nonzero(search_mask) < min_window_points:
                    # 回退：如果点太少，直接用 search_range 拟合
                    fit_lo, fit_hi = search_range
                else:
                    slopes = _rolling_slope(eV[search_mask], tauc[search_mask], slope_half_width, min_window_points)
                    # 选出最大斜率位置（忽略 NaN）
                    if np.all(np.isnan(slopes)):
                        fit_lo, fit_hi = search_range
                    else:
                        k = np.nanargmax(slopes)
                        E_star = eV[search_mask][k]
                        fit_lo = E_star - fit_half_width
                        fit_hi = E_star + fit_half_width

                # 限制到数据边界
                fit_lo = max(fit_lo, float(np.nanmin(eV)))
                fit_hi = min(fit_hi, float(np.nanmax(eV)))
            else:
                # 固定窗口（保持你原来的做法）
                fit_lo, fit_hi = (1.55, 1.60) if not mean else (1.50, 1.60)

            # ---------- 线性拟合并求带隙 ----------
            fit_mask = (eV >= fit_lo) & (eV <= fit_hi)
            X_sel = eV[fit_mask].reshape(-1, 1)
            Y_sel = tauc[fit_mask].reshape(-1, 1)

            # NaN 处理
            ok = ~(np.isnan(X_sel) | np.isnan(Y_sel)).ravel()
            if np.count_nonzero(ok) >= 2:
                model = LinearRegression()
                model.fit(X_sel[ok], Y_sel[ok])
                slope = float(model.coef_.ravel()[0])
                b = float(model.intercept_.ravel()[0])
                # 防止除零
                if np.isclose(slope, 0.0):
                    x_intercept = np.nan
                else:
                    x_intercept = -b / slope
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

            # 存储
            self.bandGap.append(x_intercept)
            self.tauc_slope.append(slope)
            self.tauc_slope_b.append(b)
            self.tauc_fit_r2.append(r2)
            self.tauc_fit_residual.append(rmse)
            self.tauc_window_used.append((float(np.nanmin(X_sel)), float(np.nanmax(X_sel))))
            is_ok = np.isfinite(x_intercept)
            self.tauc_qc_pass.append(is_ok)
            self.tauc_qc_reason.append("ok" if is_ok else "too_few_points")
            self.tauc_attempt.append("primary")

            debug_entry = {
                "source": name,
                "x_start": float(np.nanmin(X_sel)) if X_sel.size else float("nan"),
                "x_end": float(np.nanmax(X_sel)) if X_sel.size else float("nan"),
                "y_start": float(y_pred[0]) if y_pred.size else float("nan"),
                "y_end": float(y_pred[-1]) if y_pred.size else float("nan"),
                "r2": float(r2) if np.isfinite(r2) else float("nan"),
                "rmse": float(rmse) if np.isfinite(rmse) else float("nan"),
                "Eg": float(x_intercept) if np.isfinite(x_intercept) else float("nan"),
                "fit_window": (float(np.nanmin(X_sel)), float(np.nanmax(X_sel))) if X_sel.size else (float("nan"), float("nan")),
            }
            self.tauc_fit_lines.append(debug_entry)

            # 保存调试信息
            self.debug_auto_window.append({
                "name": name,
                "search_range": tuple(search_range),
                "slope_half_width": slope_half_width,
                "fit_half_width": fit_half_width,
                "fit_window": (fit_lo, fit_hi),
                "Eg": x_intercept,
                "slope": slope,
                "intercept": b,
            })

        # --------- 结果落盘 ---------
        results_folder = os.path.join(self.folderPath, "Results")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        timestamp_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(results_folder, f"band_gap_results_{timestamp_now}.csv")

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
            }

        df = pd.DataFrame(taucData)
        df.to_csv(results_file, index=False)

    def calcTauc_v2(
        self,
        mean=True,
        search_lower=1.5,
        search_upper=1.65,
        window_width=0.05,
        step_energy=0.1,
        min_points=10,
        smoothing_window=0,
        smoothing_polyorder=2,
        min_r2=0.95,
        fallback_window=(1.55, 1.6),
        tauc_power=2.0,
        tauc_power_mean=2.0,
        store_debug=True,
        enable_qc=True,
        qc_energy_window=(-0.12, +0.1),
        qc_min_r2=None,
        qc_min_span=0.02,
        qc_max_extrapolation=0.18,
        qc_rmse_max=None,
        retry_on_fail=True,
        retry_step_energy=None,
        retry_window_width=None,
        retry_min_points=None,
        retry_min_span=None,
        retry_min_r2=None,
        retry_search_lower=None,
        retry_search_upper=None,
        retry_smoothing_window=None,
        use_robust_fit=False,
        robust_on_retry=True,
        **_,
    ):
        """Sliding-window Tauc fit: pick the steepest 0.05 eV segment between 1.5–1.65 eV."""

        import os
        from datetime import datetime
        import numpy as np
        import pandas as pd
        from scipy.signal import savgol_filter
        from sklearn.linear_model import LinearRegression, TheilSenRegressor

        if not self.wavelengths_interp:
            raise RuntimeError("No interpolated spectra available; run interpolate_masked first.")

        light_calibrated = np.array(self.lightValue_interp) - np.array(self.darkValue_interp)
        epsilon = 1e-10
        if fallback_window is None:
            fallback_window = (search_lower, search_upper)

        reference_wavelength = np.array(self.wavelengths_interp[0])
        if np.any(reference_wavelength == 0):
            raise ValueError("Wavelength array contains zero values; cannot compute photon energy.")
        self.eV = 1240 / reference_wavelength

        self.valueTaucs = []
        self.bandGap = []
        self.tauc_slope = []
        self.tauc_slope_b = []
        self.tauc_fit_r2 = []
        self.tauc_fit_residual = []
        self.tauc_window_used = []
        self.tauc_fit_lines = []
        self.tauc_qc_pass = []
        self.tauc_qc_reason = []
        self.tauc_attempt = []

        def _iter_spectra():
            if mean:
                for idx, values in enumerate(self.values_interpAvr):
                    name = self.fileNameAveraged[idx] if idx < len(self.fileNameAveraged) else f"avg_{idx}"
                    yield np.array(values), name
            else:
                for idx, values in enumerate(self.values_interp):
                    name = self.fileNameRaw[idx] if idx < len(self.fileNameRaw) else f"raw_{idx}"
                    yield np.array(values), name

        def _smooth(data, win_override=None):
            win = win_override if win_override is not None else smoothing_window
            if not win or win < 3:
                return data
            win = min(int(win), data.size)
            if win % 2 == 0:
                win -= 1
            if win < 3:
                return data
            poly = min(int(smoothing_polyorder), win - 1)
            if poly < 1:
                poly = 1
            try:
                return savgol_filter(data, win, poly, mode="interp")
            except ValueError:
                return data

        def _fit_line(x, y, robust=False):
            if robust:
                model = TheilSenRegressor(random_state=0)
                model.fit(x.reshape(-1, 1), y)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
            else:
                model = LinearRegression()
                model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
                slope = float(model.coef_[0][0])
                intercept = float(model.intercept_[0])
            y_pred = slope * x + intercept
            resid = y - y_pred
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = float(np.sqrt(ss_res / max(len(x), 1)))
            Eg = -intercept / slope if slope != 0 else np.nan
            return slope, intercept, r2, rmse, Eg, y_pred

        def _qc_check(Eg, x_sel, r2, rmse, energy_guess):
            if not enable_qc:
                return True, "ok"
            if not np.isfinite(Eg):
                return False, "Eg_nan"
            lo, hi = qc_energy_window
            if not (energy_guess + lo <= Eg <= energy_guess + hi):
                return False, "Eg_out_of_window"
            span = float(x_sel.max() - x_sel.min()) if x_sel.size else 0.0
            if span < qc_min_span:
                return False, "span_too_small"
            r2_min = qc_min_r2 if qc_min_r2 is not None else min_r2
            if r2_min is not None and (not np.isfinite(r2) or r2 < r2_min):
                return False, "r2_below_qc"
            if qc_rmse_max is not None and (not np.isfinite(rmse) or rmse > qc_rmse_max):
                return False, "rmse_too_high"
            leftmost = float(np.min(x_sel)) if x_sel.size else np.nan
            if np.isfinite(leftmost) and (leftmost - Eg) > qc_max_extrapolation:
                return False, "over_extrapolated"
            return True, "ok"

        target_power = tauc_power_mean if mean else tauc_power

        def _select_window(x_axis, y_axis, y_smoothed, step_e, win_width, min_pts, min_span_req, low, high):
            half = win_width / 2.0
            low_expanded = low - half
            high_expanded = high + half

            band_mask = (x_axis >= low_expanded) & (x_axis <= high_expanded)
            band_idx = np.where(band_mask)[0]
            if band_idx.size < max(min_pts, 2):
                return None, {"reason": "no_candidates"}

            x_band = x_axis[band_idx]
            y_band = y_smoothed[band_idx]

            step_e = float(step_e)
            if step_e <= 0:
                step_e = (high - low) / max(band_idx.size, 1)

            centers = np.arange(low, high + step_e * 0.5, step_e)
            if centers.size == 0:
                centers = np.array([(low + high) / 2])

            starts = np.searchsorted(x_band, centers - half, side="left")
            ends = np.searchsorted(x_band, centers + half, side="right")

            n = ends - starts
            valid = n >= max(min_pts, 2)
            if not np.any(valid):
                return None, {"reason": "no_valid_windows"}

            # prefix sums for fast statistics on smoothed data
            px = np.concatenate(([0.0], np.cumsum(x_band)))
            py = np.concatenate(([0.0], np.cumsum(y_band)))
            pxy = np.concatenate(([0.0], np.cumsum(x_band * y_band)))
            px2 = np.concatenate(([0.0], np.cumsum(x_band ** 2)))
            py2 = np.concatenate(([0.0], np.cumsum(y_band ** 2)))

            sum_x = px[ends] - px[starts]
            sum_y = py[ends] - py[starts]
            sum_xy = pxy[ends] - pxy[starts]
            sum_x2 = px2[ends] - px2[starts]
            sum_y2 = py2[ends] - py2[starts]

            n_float = n.astype(float)
            denom = n_float * sum_x2 - sum_x ** 2
            valid &= denom > 0

            slope = np.full_like(sum_x, np.nan, dtype=float)
            slope[valid] = (n_float[valid] * sum_xy[valid] - sum_x[valid] * sum_y[valid]) / denom[valid]

            intercept = np.full_like(sum_x, np.nan, dtype=float)
            intercept[valid] = (sum_y[valid] - slope[valid] * sum_x[valid]) / n_float[valid]

            span = np.full_like(sum_x, np.nan, dtype=float)
            valid_spans = valid & (starts < ends)
            last_idx = np.clip(ends - 1, 0, x_band.size - 1)
            span[valid_spans] = x_band[last_idx[valid_spans]] - x_band[starts[valid_spans]]
            valid &= np.nan_to_num(span >= min_span_req, nan=False)

            slope[~valid] = np.nan

            if not np.any(np.isfinite(slope)):
                return None, {"reason": "no_positive_slope"}

            # prefer positive slopes only
            slope[slope <= 0] = np.nan
            if not np.any(np.isfinite(slope)):
                return None, {"reason": "no_positive_slope"}

            # choose maximum slope
            best_idx = int(np.nanargmax(slope))
            start_idx = starts[best_idx]
            end_idx = ends[best_idx]
            selected_idx = band_idx[start_idx:end_idx]
            if selected_idx.size < max(min_pts, 2):
                return None, {"reason": "no_points_post_selection"}

            mask = np.zeros_like(x_axis, dtype=bool)
            mask[selected_idx] = True

            # compute candidate r2 on smoothed data for diagnostics
            num = (n_float * sum_xy - sum_x * sum_y) ** 2
            denom_y = n_float * sum_y2 - sum_y ** 2
            denom_r = denom * denom_y
            r2 = np.full_like(slope, np.nan)
            valid_r = denom_r > 0
            r2[valid_r] = num[valid_r] / denom_r[valid_r]

            meta = {
                "start": float(x_band[start_idx]),
                "end": float(x_band[end_idx - 1]),
                "span": float(span[best_idx]) if np.isfinite(span[best_idx]) else float(x_band[end_idx - 1] - x_band[start_idx]),
                "slope_candidate": float(slope[best_idx]),
                "r2_candidate": float(r2[best_idx]) if np.isfinite(r2[best_idx]) else float("nan"),
                "center": float(centers[best_idx]),
            }
            return mask, meta

        for spectrum_values, source_name in _iter_spectra():
            value_calibrated = np.array(spectrum_values) - np.array(self.darkValue_interp)
            with np.errstate(divide="ignore", invalid="ignore"):
                transmission = value_calibrated / light_calibrated
            absorbance = -np.log10(np.maximum(transmission, epsilon))
            tauc = np.power(np.maximum(absorbance * self.eV, 0), target_power)
            self.valueTaucs.append(tauc)

            def _run_pass(is_retry=False):
                cfg_step_e = retry_step_energy if (is_retry and retry_step_energy is not None) else step_energy
                cfg_width = retry_window_width if (is_retry and retry_window_width is not None) else window_width
                cfg_min_pts = retry_min_points if (is_retry and retry_min_points is not None) else min_points
                cfg_min_span_local = retry_min_span if (is_retry and retry_min_span is not None) else qc_min_span
                cfg_min_r2_local = retry_min_r2 if (is_retry and retry_min_r2 is not None) else min_r2
                cfg_low = retry_search_lower if (is_retry and retry_search_lower is not None) else search_lower
                cfg_high = retry_search_upper if (is_retry and retry_search_upper is not None) else search_upper
                cfg_smoothing = retry_smoothing_window if (is_retry and retry_smoothing_window is not None) else smoothing_window

                smoothed = _smooth(tauc, win_override=cfg_smoothing)
                mask, meta = _select_window(
                    self.eV,
                    tauc,
                    smoothed,
                    step_e=cfg_step_e,
                    win_width=float(cfg_width),
                    min_pts=max(int(cfg_min_pts), 2),
                    min_span_req=float(cfg_min_span_local) if cfg_min_span_local is not None else 0.0,
                    low=float(cfg_low),
                    high=float(cfg_high),
                )

                fallback_used = False
                if mask is None:
                    fallback_used = True
                    meta = meta or {}
                    meta.setdefault("reason", "fallback")
                    mask = (self.eV >= fallback_window[0]) & (self.eV <= fallback_window[1])

                x_selected = self.eV[mask]
                y_selected = tauc[mask]
                finite = np.isfinite(x_selected) & np.isfinite(y_selected)
                x_selected = x_selected[finite]
                y_selected = y_selected[finite]
                if x_selected.size < max(int(cfg_min_pts), 2):
                    meta = meta or {}
                    meta["fallback_used"] = fallback_used
                    return dict(ok=False, reason="too_few_points", meta=meta)

                robust_now = (use_robust_fit or (robust_on_retry and is_retry))
                slope, intercept, r2, rmse, Eg, y_pred = _fit_line(x_selected, y_selected, robust=robust_now)

                meta = meta or {}
                span_current = float(x_selected[-1] - x_selected[0])
                meta.setdefault("start", float(x_selected[0]))
                meta.setdefault("end", float(x_selected[-1]))
                meta.setdefault("span", span_current)
                meta.setdefault("fallback_used", fallback_used)
                meta["candidate_slope"] = float(slope)
                meta["candidate_r2"] = float(r2)

                # QC uses cfg_min_r2_local & cfg_min_span_local for tightened threshold when retrying
                ok, reason = _qc_check(Eg, x_selected, r2, rmse, (search_lower + search_upper) / 2)
                min_span_required = cfg_min_span_local if cfg_min_span_local is not None else qc_min_span
                if span_current < min_span_required:
                    ok = False
                    reason = "span_too_small"
                if cfg_min_r2_local is not None and r2 < cfg_min_r2_local:
                    ok = False
                    reason = "r2_below_cfg"

                return dict(
                    ok=ok,
                    reason=reason,
                    Eg=Eg,
                    slope=slope,
                    intercept=intercept,
                    r2=r2,
                    rmse=rmse,
                    xsel=x_selected,
                    ysel=y_selected,
                    ypred=y_pred,
                    meta=meta,
                    robust=robust_now,
                )

            res = _run_pass(is_retry=False)
            attempt_label = "primary"
            if (not res.get("ok", False)) and retry_on_fail:
                res_retry = _run_pass(is_retry=True)
                if res_retry.get("ok", False):
                    res = res_retry
                    attempt_label = "retry"

            if not res.get("ok", False):
                self.bandGap.append(np.nan)
                self.tauc_slope.append(np.nan)
                self.tauc_slope_b.append(np.nan)
                self.tauc_fit_r2.append(np.nan)
                self.tauc_fit_residual.append(np.nan)
                self.tauc_window_used.append(tuple(fallback_window))
                self.tauc_qc_pass.append(False)
                self.tauc_qc_reason.append(res.get("reason", "fail"))
                self.tauc_attempt.append(attempt_label)
                if store_debug:
                    self.tauc_fit_lines.append({
                        "source": source_name,
                        "x_start": float(fallback_window[0]),
                        "x_end": float(fallback_window[1]),
                        "qc_reason": res.get("reason", "fail"),
                        "attempt": attempt_label,
                        **{k: v for k, v in res.get("meta", {}).items() if isinstance(v, (int, float))},
                    })
                continue

            self.bandGap.append(float(res["Eg"]))
            self.tauc_slope.append(float(res["slope"]))
            self.tauc_slope_b.append(float(res["intercept"]))
            self.tauc_fit_r2.append(float(res["r2"]))
            self.tauc_fit_residual.append(float(res["rmse"]))
            self.tauc_window_used.append((float(res["xsel"][0]), float(res["xsel"][-1])))
            self.tauc_qc_pass.append(True)
            self.tauc_qc_reason.append("ok")
            self.tauc_attempt.append(attempt_label)

            if store_debug:
                r2_val = float(res["r2"])
                r_value = float(np.sqrt(max(r2_val, 0.0)))
                self.tauc_fit_lines.append({
                    "source": source_name,
                    "x_start": float(res["xsel"][0]),
                    "x_end": float(res["xsel"][-1]),
                    "y_start": float(res["ypred"][0]),
                    "y_end": float(res["ypred"][-1]),
                    "r": r_value,
                    "r2": r2_val,
                    "rmse": float(res["rmse"]),
                    "Eg": float(res["Eg"]),
                    "attempt": attempt_label,
                    **{k: v for k, v in res["meta"].items() if isinstance(v, (int, float))},
                })

        results_folder = os.path.join(self.folderPath, "Results")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        timestamp_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(results_folder, f"band_gap_results_{timestamp_now}.csv")

        if mean:
            tauc_data = {
                "timestamp(in Hour):": getattr(self, "timestampAbsHN", []),
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b,
                "tauc_r2": self.tauc_fit_r2,
                "tauc_rmse": self.tauc_fit_residual,
                "qc_pass": self.tauc_qc_pass,
                "qc_reason": self.tauc_qc_reason,
                "attempt": self.tauc_attempt,
            }
        else:
            tauc_data = {
                "timestamp": self.timestamps,
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b,
                "tauc_r2": self.tauc_fit_r2,
                "tauc_rmse": self.tauc_fit_residual,
                "qc_pass": self.tauc_qc_pass,
                "qc_reason": self.tauc_qc_reason,
                "attempt": self.tauc_attempt,
            }

        df = pd.DataFrame(tauc_data)
        df.to_csv(results_file, index=False)

    def calcTauc_v2_1(
        self,
        mean=True,
        energy_guess=1.55,
        search_half_width=0.2,
        min_window_points=10,
        min_window_span=0.03,
        smoothing_window=0,
        smoothing_polyorder=2,
        search_lower=1.5,
        search_upper=1.65,
        slope_percentile=70.0,
        curvature_percentile=40.0,
        linear_trim_fraction=0.2,
        min_r2=0.95,
        fallback_window=(1.55, 1.6),
        tauc_power=2.0,
        tauc_power_mean=2.0,
        store_debug=True,

        # ---- NEW: QC and robustness controls ----
        enable_qc=True,
        qc_energy_window=(-0.12, +0.1),      # Eg must lie in [guess+lo, guess+hi]
        qc_min_r2=None,                      # default: use min_r2
        qc_min_span=0.02,                    # selected window must span at least this (eV)
        qc_max_extrapolation=0.18,           # Eg should not be too far left of window
        qc_rmse_max=None,                    # optional absolute RMSE cap
        qc_curvature_cv_max=0.4,             # curvature gate (coefficient of variation of local slope)

        # retry with stricter settings if QC fails (ONE retry)
        retry_on_fail=True,
        retry_search_half_width=0.15,
        retry_min_window_points=None,
        retry_min_window_span=0.025,
        retry_smoothing_window=0,
        retry_search_lower=None,
        retry_search_upper=None,
        retry_slope_percentile=60.0,
        retry_curvature_percentile=55.0,
        retry_linear_trim_fraction=0.15,

        # fitter options
        use_robust_fit=False,                # switch to Theil–Sen (robust to outliers)
        robust_on_retry=True,                # automatically use robust at retry even if use_robust_fit=False
        **_,
    ):
        """
        Improved Tauc analysis with adaptive window selection + automatic QC/Retry.

        - Stronger window selection: curvature gate, right-edge cap, short-window preference.
        - QC after fit; if failed, re-run ONCE with stricter settings (then NaN).
        """

        # --- imports (kept local to avoid external dependencies leaking) ---
        import os
        from datetime import datetime
        import numpy as np
        import pandas as pd
        from scipy.signal import savgol_filter
        from sklearn.linear_model import LinearRegression, TheilSenRegressor

        if not self.wavelengths_interp:
            raise RuntimeError("No interpolated spectra available; run interpolate_masked first.")

        light_calibrated = np.array(self.lightValue_interp) - np.array(self.darkValue_interp)
        epsilon = 1e-10
        if fallback_window is None:
            fallback_window = (energy_guess - 0.05, energy_guess + 0.05)

        # Photon energy axis in eV (shared for every spectrum)
        reference_wavelength = np.array(self.wavelengths_interp[0])
        if np.any(reference_wavelength == 0):
            raise ValueError("Wavelength array contains zero values; cannot compute photon energy.")
        self.eV = 1240 / reference_wavelength

        # Reset containers
        self.valueTaucs = []
        self.bandGap = []
        self.tauc_slope = []
        self.tauc_slope_b = []
        self.tauc_fit_r2 = []
        self.tauc_fit_residual = []
        self.tauc_window_used = []
        self.tauc_fit_lines = []
        # NEW: QC bookkeeping
        self.tauc_qc_pass = []
        self.tauc_qc_reason = []
        self.tauc_attempt = []   # "primary" or "retry"

        # ---- helpers ----
        def _iter_spectra():
            if mean:
                for idx, values in enumerate(self.values_interpAvr):
                    name = self.fileNameAveraged[idx] if idx < len(self.fileNameAveraged) else f"avg_{idx}"
                    yield np.array(values), name
            else:
                for idx, values in enumerate(self.values_interp):
                    name = self.fileNameRaw[idx] if idx < len(self.fileNameRaw) else f"raw_{idx}"
                    yield np.array(values), name

        def _find_linear_segment(
            x_axis,
            y_axis,
            *,
            center,
            half_width,
            search_lower,
            search_upper,
            min_points,
            min_span,
            slope_percentile,
            curvature_percentile,
            trim_fraction,
            smoothing_window_cfg,
            smoothing_polyorder_cfg,
        ):
            """Locate a low-curvature, high-slope segment around the band edge."""

            low = center - half_width
            high = center + half_width
            if search_lower is not None:
                low = max(low, float(search_lower))
            if search_upper is not None:
                high = min(high, float(search_upper))
            if high <= low:
                return None, {"reason": "search_window_empty"}

            search_mask = (x_axis >= low) & (x_axis <= high)
            candidate_idx = np.where(search_mask)[0]
            if candidate_idx.size < max(min_points, 3):
                return None, {"reason": "insufficient_points"}

            subset_x = x_axis[candidate_idx]
            subset_y = y_axis[candidate_idx]

            finite_mask = np.isfinite(subset_x) & np.isfinite(subset_y)
            if np.count_nonzero(finite_mask) < max(min_points, 3):
                return None, {"reason": "nonfinite_values"}

            subset_x = subset_x[finite_mask]
            subset_y = subset_y[finite_mask]
            candidate_idx = candidate_idx[finite_mask]

            if subset_x.size < max(min_points, 3):
                return None, {"reason": "subset_too_small"}

            # Optional smoothing to stabilise derivatives
            smoothed = subset_y.astype(float)
            if smoothing_window_cfg and smoothed.size >= 3:
                win = min(int(smoothing_window_cfg), smoothed.size)
                if win % 2 == 0:
                    win -= 1
                if win >= 3:
                    poly = min(int(smoothing_polyorder_cfg), win - 1)
                    try:
                        smoothed = savgol_filter(smoothed, win, poly, mode="interp")
                    except ValueError:
                        pass

            if smoothed.size < max(min_points, 3):
                return None, {"reason": "smoothed_too_small"}

            # First and second derivatives
            dy = np.gradient(smoothed, subset_x)
            positive_slope = dy[dy > 0]
            if positive_slope.size == 0:
                return None, {"reason": "no_positive_slope"}

            slope_thresh = np.percentile(positive_slope, np.clip(slope_percentile, 0, 100))
            slope_thresh = max(slope_thresh, 0.0)

            curvature = np.gradient(dy, subset_x)
            abs_curv = np.abs(curvature)
            curvature_thresh = np.percentile(abs_curv, np.clip(curvature_percentile, 0, 100))
            if not np.isfinite(curvature_thresh) or curvature_thresh <= 0:
                curvature_thresh = np.max(abs_curv)

            candidate = (dy >= slope_thresh) & (abs_curv <= curvature_thresh)
            candidate_idx_local = np.where(candidate)[0]
            if candidate_idx_local.size < min_points:
                return None, {"reason": "no_segment_after_thresholds", "slope_thresh": float(slope_thresh), "curv_thresh": float(curvature_thresh)}

            # Split into contiguous segments
            splits = np.where(np.diff(candidate_idx_local) > 1)[0] + 1
            segments = np.split(candidate_idx_local, splits)

            best = None
            best_score = None
            for seg in segments:
                if seg.size < min_points:
                    continue
                span = float(subset_x[seg[-1]] - subset_x[seg[0]])
                if span < min_span:
                    continue
                mean_slope = float(np.mean(dy[seg]))
                score = mean_slope * span
                if best is None or score > best_score:
                    best = seg
                    best_score = score

            if best is None:
                return None, {"reason": "no_segment_after_scoring"}

            trim_fraction = float(np.clip(trim_fraction, 0.0, 0.45))
            trim_count = int(np.floor(best.size * trim_fraction))
            if trim_count * 2 >= best.size:
                trim_count = max(0, best.size // 4)

            trimmed = best[trim_count: best.size - trim_count if trim_count else best.size]
            if trimmed.size < min_points:
                trimmed = best
                if trimmed.size < min_points:
                    return None, {"reason": "segment_too_short"}

            selected_indices = candidate_idx[trimmed]
            mask = np.zeros_like(x_axis, dtype=bool)
            mask[selected_indices] = True

            meta = {
                "start": float(subset_x[trimmed[0]]),
                "end": float(subset_x[trimmed[-1]]),
                "span": float(subset_x[trimmed[-1]] - subset_x[trimmed[0]]),
                "mean_slope": float(np.mean(dy[trimmed])),
                "slope_threshold": float(slope_thresh),
                "curvature_threshold": float(curvature_thresh),
                "trim_count": int(trim_count),
                "search_low": float(low),
                "search_high": float(high),
            }
            return mask, meta

        def _fit_line(x, y, robust=False):
            if robust:
                model = TheilSenRegressor(random_state=0)
                model.fit(x.reshape(-1,1), y)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
            else:
                model = LinearRegression()
                model.fit(x.reshape(-1,1), y.reshape(-1,1))
                slope = float(model.coef_[0][0])
                intercept = float(model.intercept_[0])
            y_pred = slope * x + intercept
            resid = y - y_pred
            ss_res = float(np.sum(resid**2))
            ss_tot = float(np.sum((y - np.mean(y))**2))
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
            rmse = float(np.sqrt(ss_res/max(len(x),1)))
            x0 = -intercept / slope if slope != 0 else np.nan
            return slope, intercept, r2, rmse, x0, y_pred

        def _qc_check(Eg, x_sel, r2, rmse, energy_guess_local):
            if not enable_qc:
                return True, "ok"
            if not np.isfinite(Eg):
                return False, "Eg_nan"
            # energy window check
            lo, hi = qc_energy_window
            if not (energy_guess_local + lo <= Eg <= energy_guess_local + hi):
                return False, "Eg_out_of_window"
            # span
            if (x_sel.max() - x_sel.min()) < qc_min_span:
                return False, "span_too_small"
            # r2
            if qc_min_r2 is not None and (not np.isfinite(r2) or r2 < qc_min_r2):
                return False, "r2_below_qc"
            # rmse
            if (qc_rmse_max is not None) and (not np.isfinite(rmse) or rmse > qc_rmse_max):
                return False, "rmse_too_high"
            # extrapolation distance
            if np.isfinite(Eg):
                leftmost = float(np.min(x_sel))
                if (leftmost - Eg) > qc_max_extrapolation:
                    return False, "over_extrapolated"
            return True, "ok"

        # ---- main loop ----
        target_power = tauc_power_mean if mean else tauc_power

        for spectrum_values, source_name in _iter_spectra():
            value_calibrated = np.array(spectrum_values) - np.array(self.darkValue_interp)
            with np.errstate(divide="ignore", invalid="ignore"):
                transmission = value_calibrated / light_calibrated
            absorbance = -np.log10(np.maximum(transmission, epsilon))
            tauc = np.power(np.maximum(absorbance * self.eV, 0), target_power)
            self.valueTaucs.append(tauc)

            def _run_pass(is_retry=False):
                cfg_half_width = retry_search_half_width if is_retry else search_half_width
                cfg_min_points = (
                    retry_min_window_points if (is_retry and retry_min_window_points is not None)
                    else min_window_points
                )
                cfg_min_span = retry_min_window_span if is_retry else min_window_span
                cfg_smoothing = retry_smoothing_window if is_retry else smoothing_window
                cfg_slope_pct = retry_slope_percentile if is_retry else slope_percentile
                cfg_curv_pct = retry_curvature_percentile if is_retry else curvature_percentile
                cfg_trim = retry_linear_trim_fraction if is_retry else linear_trim_fraction
                cfg_search_low = (
                    retry_search_lower if (is_retry and retry_search_lower is not None)
                    else search_lower
                )
                cfg_search_high = (
                    retry_search_upper if (is_retry and retry_search_upper is not None)
                    else search_upper
                )

                mask, meta = _find_linear_segment(
                    self.eV,
                    tauc,
                    center=energy_guess,
                    half_width=cfg_half_width,
                    search_lower=cfg_search_low,
                    search_upper=cfg_search_high,
                    min_points=max(int(cfg_min_points), 2),
                    min_span=float(cfg_min_span),
                    slope_percentile=float(cfg_slope_pct),
                    curvature_percentile=float(cfg_curv_pct),
                    trim_fraction=float(cfg_trim),
                    smoothing_window_cfg=cfg_smoothing,
                    smoothing_polyorder_cfg=smoothing_polyorder,
                )

                fallback_used = False
                if mask is None:
                    fallback_used = True
                    meta = meta or {}
                    meta.setdefault("reason", "no_segment")
                    mask = (self.eV >= fallback_window[0]) & (self.eV <= fallback_window[1])

                x_selected = self.eV[mask]
                y_selected = tauc[mask]
                finite_selected = np.isfinite(x_selected) & np.isfinite(y_selected)
                x_selected = x_selected[finite_selected]
                y_selected = y_selected[finite_selected]
                if x_selected.size < 2 or np.allclose(y_selected, y_selected[0]):
                    meta = meta or {}
                    meta["fallback_used"] = fallback_used
                    return dict(ok=False, reason="too_few_points", meta=meta)

                meta = meta or {}
                meta.setdefault("start", float(x_selected[0]))
                meta.setdefault("end", float(x_selected[-1]))
                meta.setdefault("span", float(x_selected[-1] - x_selected[0]))
                meta["fallback_used"] = fallback_used

                robust_now = (use_robust_fit or (robust_on_retry and is_retry))
                slope, intercept, r2, rmse, Eg, y_pred = _fit_line(x_selected, y_selected, robust=robust_now)

                ok, reason = _qc_check(Eg, x_selected, r2, rmse, energy_guess)
                meta["r2_window"] = float(r2) if np.isfinite(r2) else float("nan")
                meta["rmse"] = float(rmse)
                return dict(
                    ok=ok,
                    reason=reason,
                    Eg=Eg,
                    slope=slope,
                    intercept=intercept,
                    r2=r2,
                    rmse=rmse,
                    xsel=x_selected,
                    ysel=y_selected,
                    ypred=y_pred,
                    meta=meta,
                    robust=robust_now,
                )

            # primary pass
            res = _run_pass(is_retry=False)
            attempt_label = "primary"

            # retry pass if needed
            if (not res.get("ok", False)) and retry_on_fail:
                res_retry = _run_pass(is_retry=True)
                if res_retry.get("ok", False):
                    res = res_retry
                    attempt_label = "retry"

            # finalize record
            if not res.get("ok", False):
                # give up, record NaN but keep debug
                self.bandGap.append(np.nan)
                self.tauc_slope.append(np.nan)
                self.tauc_slope_b.append(np.nan)
                self.tauc_fit_r2.append(np.nan)
                self.tauc_fit_residual.append(np.nan)
                self.tauc_window_used.append(tuple(fallback_window))
                self.tauc_qc_pass.append(False)
                self.tauc_qc_reason.append(res.get("reason", "fail"))
                self.tauc_attempt.append(attempt_label)
                if store_debug:
                    self.tauc_fit_lines.append({
                        "source": source_name,
                        "x_start": float(fallback_window[0]),
                        "x_end": float(fallback_window[1]),
                        "y_start": float("nan"),
                        "y_end": float("nan"),
                        "qc_reason": res.get("reason", "fail"),
                        "attempt": attempt_label,
                        **{k:v for k,v in res.get("meta", {}).items() if isinstance(v,(int,float))}
                    })
                continue

            # success
            self.bandGap.append(float(res["Eg"]))
            self.tauc_slope.append(float(res["slope"]))
            self.tauc_slope_b.append(float(res["intercept"]))
            self.tauc_fit_r2.append(float(res["r2"]))
            self.tauc_fit_residual.append(float(res["rmse"]))
            self.tauc_window_used.append((float(res["xsel"][0]), float(res["xsel"][-1])))
            self.tauc_qc_pass.append(True)
            self.tauc_qc_reason.append("ok")
            self.tauc_attempt.append(attempt_label)

            if store_debug:
                r2_val = float(res["r2"])
                r_value = float(np.sqrt(max(r2_val, 0.0)))
                self.tauc_fit_lines.append({
                    "source": source_name,
                    "x_start": float(res["xsel"][0]),
                    "x_end": float(res["xsel"][-1]),
                    "y_start": float(res["ypred"][0]),
                    "y_end": float(res["ypred"][-1]),
                    "r": r_value,
                    "r2": r2_val,
                    "r2_exact": r2_val,
                    "rmse": float(res["rmse"]),
                    "Eg": float(res["Eg"]),
                    "attempt": attempt_label,
                    **{k:v for k,v in res["meta"].items() if isinstance(v,(int,float))}
                })

        # ---- persist results ----
        results_folder = os.path.join(self.folderPath, "Results")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        timestamp_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(results_folder, f"band_gap_results_{timestamp_now}.csv")

        if mean:
            tauc_data = {
                "timestamp(in Hour):": getattr(self, "timestampAbsHN", []),
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b,
                "tauc_r2": self.tauc_fit_r2,
                "tauc_rmse": self.tauc_fit_residual,
                "qc_pass": self.tauc_qc_pass,
                "qc_reason": self.tauc_qc_reason,
                "attempt": self.tauc_attempt,
            }
        else:
            tauc_data = {
                "timestamp": self.timestamps,
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b,
                "tauc_r2": self.tauc_fit_r2,
                "tauc_rmse": self.tauc_fit_residual,
                "qc_pass": self.tauc_qc_pass,
                "qc_reason": self.tauc_qc_reason,
                "attempt": self.tauc_attempt,
            }

        df = pd.DataFrame(tauc_data)
        df.to_csv(results_file, index=False)

    def calcTauc_v3(
        self,
        mean=True,
        logistic_range=(1.45, 1.65),
        fit_half_window=0.02,
        min_points=6,
        smoothing_window=0,
        smoothing_polyorder=2,
        tauc_power=2.0,
        tauc_power_mean=2.0,
        fallback_window=(1.55, 1.6),
        store_debug=True,
        **_,
    ):
        """Hybrid logistics + local linear fitting for band-gap extraction."""

        import os
        from datetime import datetime
        import numpy as np
        import pandas as pd
        from scipy.optimize import curve_fit
        from scipy.signal import savgol_filter
        from sklearn.linear_model import LinearRegression

        if not self.wavelengths_interp:
            raise RuntimeError("No interpolated spectra available; run interpolate_masked first.")

        low, high = sorted(logistic_range)
        half_window = float(fit_half_window)
        if half_window <= 0:
            raise ValueError("fit_half_window must be positive")

        lightCalibrated = np.array(self.lightValue_interp) - np.array(self.darkValue_interp)
        epsilon = 1e-10
        if fallback_window is None:
            fallback_window = (low, high)

        reference_wavelength = np.array(self.wavelengths_interp[0])
        if np.any(reference_wavelength == 0):
            raise ValueError("Wavelength array contains zero values; cannot compute photon energy.")
        self.eV = 1240 / reference_wavelength

        self.valueTaucs = []
        self.bandGap = []
        self.tauc_slope = []
        self.tauc_slope_b = []
        self.tauc_fit_r2 = []
        self.tauc_fit_residual = []
        self.tauc_window_used = []
        self.tauc_fit_lines = []
        self.tauc_qc_pass = []
        self.tauc_qc_reason = []
        self.tauc_attempt = []

        def _iter_spectra():
            if mean:
                for idx, values in enumerate(self.values_interpAvr):
                    name = self.fileNameAveraged[idx] if idx < len(self.fileNameAveraged) else f"avg_{idx}"
                    yield np.array(values), name
            else:
                for idx, values in enumerate(self.values_interp):
                    name = self.fileNameRaw[idx] if idx < len(self.fileNameRaw) else f"raw_{idx}"
                    yield np.array(values), name

        def _smooth(data):
            if not smoothing_window or smoothing_window < 3:
                return data
            win = min(int(smoothing_window), data.size)
            if win % 2 == 0:
                win -= 1
            if win < 3:
                return data
            poly = min(int(smoothing_polyorder), win - 1)
            if poly < 1:
                poly = 1
            try:
                return savgol_filter(data, win, poly, mode="interp")
            except ValueError:
                return data

        def logistic_plus_linear(x, b, c):
            return 1.0 / (1.0 + np.exp(-(x - c))) + b * x

        def logistic_plus_linear_slope(x, b, c):
            exp_term = np.exp(-(x - c))
            logistic_der = exp_term / (1.0 + exp_term) ** 2
            return logistic_der + b

        target_power = tauc_power_mean if mean else tauc_power
        center_base = 0.5 * (low + high)

        for spectrum_values, source_name in _iter_spectra():
            value_calibrated = np.array(spectrum_values, dtype=float) - np.array(self.darkValue_interp, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                transmission = value_calibrated / lightCalibrated
            absorbance = -np.log10(np.maximum(transmission, epsilon))
            tauc = np.power(np.maximum(absorbance * self.eV, 0), target_power)
            self.valueTaucs.append(tauc)

            mask_range = (self.eV >= low) & (self.eV <= high)
            x_range = self.eV[mask_range]
            y_range = tauc[mask_range]

            if x_range.size < max(min_points, 3):
                fallback_used = True
                meta = {"reason": "insufficient_points"}
                x_fit = self.eV[(self.eV >= fallback_window[0]) & (self.eV <= fallback_window[1])]
                y_fit = tauc[(self.eV >= fallback_window[0]) & (self.eV <= fallback_window[1])]
            else:
                y_smooth = _smooth(y_range)
                x_scaled = x_range - center_base
                p0_b = (y_smooth[-1] - y_smooth[0]) / max(x_range[-1] - x_range[0], 1e-6)
                p0_c = 0.0
                try:
                    params, _ = curve_fit(
                        logistic_plus_linear,
                        x_scaled,
                        y_smooth,
                        p0=[p0_b, p0_c],
                        maxfev=10000,
                    )
                    b_opt, c_opt = params
                    slopes = logistic_plus_linear_slope(x_scaled, b_opt, c_opt)
                    if np.all(np.isnan(slopes)):
                        raise RuntimeError("logistic slopes NaN")
                    idx_peak = int(np.nanargmax(slopes))
                    x_peak = float(x_range[idx_peak])
                    fallback_used = False
                    meta = {
                        "b": float(b_opt),
                        "c": float(c_opt),
                        "peak_slope": float(slopes[idx_peak]) if np.isfinite(slopes[idx_peak]) else float("nan"),
                        "peak_energy": x_peak,
                    }
                    fit_low = x_peak - half_window
                    fit_high = x_peak + half_window
                except Exception as exc:
                    fallback_used = True
                    meta = {"reason": f"logistic_fit_fail: {exc}"}
                    fit_low, fit_high = fallback_window
                x_fit = self.eV[(self.eV >= fit_low) & (self.eV <= fit_high)]
                y_fit = tauc[(self.eV >= fit_low) & (self.eV <= fit_high)]

            finite = np.isfinite(x_fit) & np.isfinite(y_fit)
            x_fit = x_fit[finite]
            y_fit = y_fit[finite]

            if x_fit.size < max(min_points, 2):
                mask_fb = (self.eV >= fallback_window[0]) & (self.eV <= fallback_window[1])
                x_fit = self.eV[mask_fb]
                y_fit = tauc[mask_fb]
                finite = np.isfinite(x_fit) & np.isfinite(y_fit)
                x_fit = x_fit[finite]
                y_fit = y_fit[finite]
                fallback_used = True
                meta.setdefault("reason", "fallback_window")

            if x_fit.size < max(min_points, 2):
                self.bandGap.append(np.nan)
                self.tauc_slope.append(np.nan)
                self.tauc_slope_b.append(np.nan)
                self.tauc_fit_r2.append(np.nan)
                self.tauc_fit_residual.append(np.nan)
                self.tauc_window_used.append(tuple(fallback_window))
                self.tauc_qc_pass.append(False)
                self.tauc_qc_reason.append("too_few_points")
                self.tauc_attempt.append("primary")
                if store_debug:
                    self.tauc_fit_lines.append({
                        "source": source_name,
                        "x_start": float(fallback_window[0]),
                        "x_end": float(fallback_window[1]),
                        "qc_reason": "too_few_points",
                        "fallback_used": True,
                        **{k: float(v) for k, v in meta.items() if isinstance(v, (int, float))},
                    })
                continue

            model = LinearRegression()
            X = x_fit.reshape(-1, 1)
            Y = y_fit.reshape(-1, 1)
            model.fit(X, Y)
            slope = float(model.coef_[0][0])
            intercept = float(model.intercept_[0])
            if np.isclose(slope, 0.0):
                Eg = np.nan
            else:
                Eg = -intercept / slope
            y_pred = (slope * X + intercept).ravel()
            res = y_fit - y_pred
            ss_res = float(np.sum(res ** 2))
            ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = float(np.sqrt(ss_res / max(len(y_fit), 1)))

            ok = np.isfinite(Eg)
            self.bandGap.append(float(Eg) if ok else np.nan)
            self.tauc_slope.append(slope)
            self.tauc_slope_b.append(intercept)
            self.tauc_fit_r2.append(r2)
            self.tauc_fit_residual.append(rmse)
            self.tauc_window_used.append((float(x_fit[0]), float(x_fit[-1])))
            self.tauc_qc_pass.append(ok)
            self.tauc_qc_reason.append("ok" if ok else "Eg_nan")
            self.tauc_attempt.append("primary")

            if store_debug:
                self.tauc_fit_lines.append({
                    "source": source_name,
                    "x_start": float(x_fit[0]),
                    "x_end": float(x_fit[-1]),
                    "y_start": float(y_pred[0]),
                    "y_end": float(y_pred[-1]),
                    "r": float(np.sqrt(max(r2, 0.0))) if np.isfinite(r2) else float("nan"),
                    "r2": float(r2) if np.isfinite(r2) else float("nan"),
                    "rmse": float(rmse),
                    "Eg": float(Eg) if ok else float("nan"),
                    "fallback_used": fallback_used,
                    **{k: float(v) for k, v in meta.items() if isinstance(v, (int, float))},
                })

        results_folder = os.path.join(self.folderPath, "Results")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        timestamp_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(results_folder, f"band_gap_results_{timestamp_now}.csv")

        if mean:
            tauc_data = {
                "timestamp(in Hour):": getattr(self, "timestampAbsHN", []),
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b,
                "tauc_r2": self.tauc_fit_r2,
                "tauc_rmse": self.tauc_fit_residual,
                "qc_pass": self.tauc_qc_pass,
                "qc_reason": self.tauc_qc_reason,
                "attempt": self.tauc_attempt,
            }
        else:
            tauc_data = {
                "timestamp": self.timestamps,
                "band_gap": self.bandGap,
                "tauc_slope": self.tauc_slope,
                "tauc_slope_b": self.tauc_slope_b,
                "tauc_r2": self.tauc_fit_r2,
                "tauc_rmse": self.tauc_fit_residual,
                "qc_pass": self.tauc_qc_pass,
                "qc_reason": self.tauc_qc_reason,
                "attempt": self.tauc_attempt,
            }

        df = pd.DataFrame(tauc_data)
        df.to_csv(results_file, index=False)

    def Pipeline(
        self,
        darkFolder=None,
        lightFilePath=None,
        newMode=True,
        mean=True,
        calculate_dark=False,
        calculate_light=False,
        tauc_method="v1_1", #v1_1, v1
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
        if tauc_method == "v2":
            self.calcTauc_v2(mean=mean, **tauc_kwargs)
        elif tauc_method == "v2_1":
            self.calcTauc_v2_1(mean=mean, **tauc_kwargs)
        elif tauc_method == "v1_1":
            self.taucCalc_v1_1(mean=mean, **tauc_kwargs)
        elif tauc_method == "v3":
            self.calcTauc_v3(mean=mean, **tauc_kwargs)
        elif tauc_method == "v1":
            self.taucCalc(mean=mean)
        else:
            raise ValueError("tauc_method must be 'v1', 'v1_1', 'v2', 'v2_1', or 'v3'.")
