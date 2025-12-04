import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
"""
Th
"""
class UV_VIS_Analyzer:
    def __init__(self, folderPath, filePath = None, GermanMode = True, tauc = True):
        self.folderPath = folderPath
        self.dark = None
        self.light = None
        self.data = None
        self.waveLength = None
        self.absorption = None
        self.tauc = None
        self.eVolt = None
        self.tauc_x = None
        self.tauc_slope_data = None
        self.tauc_intercept = None
        self.label = []
        self.y_intercept = None
        self.slope = None
        self.x_at_max_slope = []
        self.bandGap = []
        self._german_mode = GermanMode
        self._use_tauc = tauc

        csv_files = []
        for root, _, files in os.walk(self.folderPath):
            for file in files:
                if file.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
        csv_files.sort()

        for file_path in csv_files:
            wave_length, absorption = self._load_numeric_spectrum(file_path)
            if wave_length is None or absorption is None:
                continue

            relative_name = os.path.relpath(file_path, self.folderPath)
            key = os.path.splitext(relative_name)[0]
            base_lower = os.path.basename(key).lower()
            label = key.replace(os.sep, " / ")

            if base_lower.startswith("dark"):
                self.dark = absorption
                continue
            if base_lower.startswith("light"):
                self.light = absorption
                continue

            if self.waveLength is None:
                self.waveLength = wave_length
                self.absorption = absorption.reshape(1, -1)
                self.label.append(label)
                if self._use_tauc:
                    self.eVolt = 1240 / self.waveLength
                    self.tauc = (self.absorption * self.eVolt.reshape(1, -1)) ** 2
                continue

            if wave_length.shape != self.waveLength.shape or not np.allclose(
                wave_length, self.waveLength
            ):
                print(
                    f"Skipping '{fileName}' because its wavelength grid does not match the first spectrum."
                )
                continue

            self.label.append(label)
            self.absorption = np.vstack((self.absorption, absorption.reshape(1, -1)))
            if self._use_tauc:
                tauc_values = (absorption.reshape(1, -1) * self.eVolt.reshape(1, -1)) ** 2
                self.tauc = np.vstack((self.tauc, tauc_values))

        if self._use_tauc and self.tauc is not None:
            self.tauc_x = self.eVolt
            self.tauc_slope_data = None
            self.tauc_intercept = None

    def _load_numeric_spectrum(self, filePath):
        primary = {
            "decimal": "," if self._german_mode else ".",
            "sep": ";" if self._german_mode else ",",
        }

        fallbacks = [
            {"decimal": ",", "sep": ";"},
            {"decimal": ".", "sep": ","},
            {"decimal": ",", "sep": ","},
            {"decimal": ".", "sep": ";"},
        ]

        parse_options = [primary] + [opt for opt in fallbacks if opt != primary]

        for option in parse_options:
            try:
                df = pd.read_csv(
                    filePath,
                    decimal=option["decimal"],
                    sep=option["sep"],
                    skipinitialspace=True,
                )
            except Exception as exc:
                last_error = exc
                continue

            if df.empty or df.shape[1] < 2:
                continue

            x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            y = pd.to_numeric(df.iloc[:, 1], errors="coerce")

            valid_mask = ~(x.isna() | y.isna())
            if not valid_mask.any():
                continue

            wave_length = x[valid_mask].to_numpy(dtype=float)
            absorption = y[valid_mask].to_numpy(dtype=float)

            if wave_length.size == 0 or absorption.size == 0:
                continue

            return wave_length, absorption

        if 'last_error' in locals():
            print(
                f"Skipping '{os.path.basename(filePath)}' due to read error after trying multiple formats: {last_error}"
            )
        else:
            print(
                f"Skipping '{os.path.basename(filePath)}' because no numeric wavelength/absorption columns were detected."
            )
        return None, None

    def UV_multiPlot(self, saveName = "Result_1909.png", figColor = None):
        savePath = os.path.join(self.folderPath, saveName)
        plt.figure(figsize = (9, 7), dpi = 300)
        if self.absorption is None or len(self.label) == 0:
            raise ValueError("No spectra loaded. Make sure the folder contains valid CSV spectra files.")

        if figColor is None:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        else:
            color_cycle = list(figColor)

        if not color_cycle:
            color_cycle = [None] * len(self.label)

        for i in range(len(self.label)):
            color = color_cycle[i % len(color_cycle)]
            plt.plot(self.waveLength, self.absorption[i], color=color, label=self.label[i])
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Reflection [%]") # or absorption etc.
        plt.legend()
        plt.grid()
        plt.savefig(savePath)

    def compute_tauc(self, energy_window=(1.55, 1.65), exponent=2.0):
        if self.absorption is None or len(self.label) == 0:
            raise ValueError("No spectra loaded. Make sure the folder contains valid CSV spectra files.")

        epsilon = 1e-10
        self.eVolt = 1240 / self.waveLength
        light_calibrated = None
        if self.light is not None and self.dark is not None:
            light_calibrated = np.asarray(self.light, dtype=float) - np.asarray(self.dark, dtype=float)

        tauc_rows = []
        slopes = []
        intercepts = []
        band_gaps = []
        x_midpoints = []
        fit_rows = []

        window_mask = (self.eVolt >= energy_window[0]) & (self.eVolt <= energy_window[1])

        slope_window_half_width = 0.02

        for idx, label in enumerate(self.label):
            spectrum = np.asarray(self.absorption[idx], dtype=float)

            if light_calibrated is not None and spectrum.shape == light_calibrated.shape:
                value_calibrated = spectrum - np.asarray(self.dark, dtype=float)
                safe_light = np.where(np.abs(light_calibrated) < epsilon, np.sign(light_calibrated) * epsilon, light_calibrated)
                transmission = value_calibrated / safe_light
                transmission = np.clip(transmission, epsilon, None)
                absorbance = -np.log10(transmission)
            else:
                absorbance = np.clip(spectrum, 0, None)

            tauc = np.power(np.clip(absorbance * self.eVolt, epsilon, None), exponent)
            tauc_rows.append(tauc)

            x_selected = self.eVolt[window_mask]
            y_selected = tauc[window_mask]

            if x_selected.size >= 2 and not np.allclose(y_selected, y_selected[0]):
                best_slope = None
                best_intercept = None
                best_midpoint = np.nan

                for x_center in x_selected:
                    local_mask = np.abs(x_selected - x_center) <= slope_window_half_width
                    if np.count_nonzero(local_mask) < 2:
                        continue

                    x_local = x_selected[local_mask]
                    y_local = y_selected[local_mask]
                    if np.allclose(y_local, y_local[0]):
                        continue

                    model = LinearRegression()
                    model.fit(x_local.reshape(-1, 1), y_local)
                    local_slope = float(model.coef_[0])
                    local_intercept = float(model.intercept_)

                    if best_slope is None or local_slope > best_slope:
                        best_slope = local_slope
                        best_intercept = local_intercept
                        best_midpoint = float(np.mean(x_local))

                if best_slope is not None:
                    slope = best_slope
                    intercept = best_intercept
                    band_gap = -intercept / slope if slope != 0 else np.nan
                    fit_line = slope * self.eVolt + intercept
                    x_midpoint = best_midpoint
                else:
                    slope = np.nan
                    intercept = np.nan
                    band_gap = np.nan
                    fit_line = np.full_like(self.eVolt, np.nan)
                    x_midpoint = np.nan
            else:
                slope = np.nan
                intercept = np.nan
                band_gap = np.nan
                fit_line = np.full_like(self.eVolt, np.nan)
                x_midpoint = np.nan

            slopes.append(slope)
            intercepts.append(intercept)
            band_gaps.append(band_gap)
            fit_rows.append(fit_line)
            x_midpoints.append(x_midpoint if x_selected.size else np.nan)

        self.tauc = np.vstack(tauc_rows)
        self.tauc_slope_data = np.vstack(fit_rows)
        self.slope = slopes
        self.y_intercept = intercepts
        self.bandGap = band_gaps
        self.x_at_max_slope = x_midpoints
        self.tauc_x = self.eVolt
        self.tauc_intercept = intercepts
        self.tauc_energy_window = energy_window
        self.tauc_exponent = exponent

    def tau_Plot(self, saveName = "tau_result1.png", figColor = None, fit = None, energy_window=(1.55, 1.6), exponent=2.0):
        savePath = os.path.join(self.folderPath, saveName)
        plt.figure(figsize = (7, 5), dpi = 300)
        self.compute_tauc(energy_window=energy_window, exponent=exponent)

        if figColor is None:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        else:
            color_cycle = list(figColor)
        if not color_cycle:
            color_cycle = [None] * len(self.label)

        for i in range(len(self.label)):
            color = color_cycle[i % len(color_cycle)]
            plt.plot(self.eVolt, self.tauc[i], color = color, label = self.label[i])
            if fit:
                fit_line = self.tauc_slope_data[i]
                if not np.all(np.isnan(fit_line)):
                    plt.plot(self.eVolt, fit_line, linestyle="--", color = color, label = f"Fit: Eg = {self.bandGap[i]:.3f} eV")
        plt.xlim(1.4, 1.7)
        plt.ylim(0, 10)
        #plt.yscale("log")
        plt.xlabel("Photon energy [eV]")
        plt.ylabel(f"(Absorbance·E)$^{2}$")
        plt.legend()
        #plt.grid()
        plt.savefig(savePath)

    def logData(self, saveName="fit_results.csv"):
        savePath = os.path.join(self.folderPath, saveName)
        
        # 构建字典数据
        result_dict = {
            "Label": self.label,
            "Slope": self.slope,
            "Y_Intercept": self.y_intercept,
            "X_at_max_slope": self.x_at_max_slope,
            "X_intercept (Eg)": self.bandGap
        }

        # 转换为 DataFrame
        df_log = pd.DataFrame(result_dict)

        # 保存为 CSV
        df_log.to_csv(savePath, index=False)
        print(f"Fit data saved to: {savePath}")

if __name__ == "__main__":
    figcolor = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
    df = UV_VIS_Analyzer(folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/UV-VIS/After_TC_1909/Data', GermanMode=False)
    #print("Light",df.light, type(df.light))
    #print("Dark:", df.dark)
    #print(df.absorption.shape)
    #print(df.label)
    df.UV_multiPlot(figColor=figcolor)
    #f.fit()
    df.tau_Plot(figColor=figcolor, fit=True)
    #df.logData()
    
