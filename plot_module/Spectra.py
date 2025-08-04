import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from datetime import datetime
from sklearn.linear_model import LinearRegression
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
        self.cycleNum = []
        self.fileNameRaw = []
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

    def importData(self):
        '''Import the data from the folder, and do some basic analysis on the data.
        The data is stored in the following format:
        - wavelengths: list of wavelengths
        - values: list of values
        - timestamps: list of timestamps
        - temperatures: list of temperatures
        - labels: list of labels (dark or light)'''
        self.dat_files = [f for f in os.listdir(self.folderPath) if f.endswith('.dat')]
        for fileName in self.dat_files:
            #print("Debug:",fileName)
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
                temp_line = lines[5].strip()
                timestampStr = first_line.split(":", 1)[1].strip()
                timestamp = datetime.strptime(timestampStr, "%Y-%m-%d %H:%M:%S")
            #Import temperature:
                temp_str = temp_line.split(":")[1].strip().split()[0]
                temperature = float(temp_str)
            #Log the peak position, height and the FWHM:(editing)
                for line in lines:
                    if "# INFO : no cal params" in line:
                        print("Bad data detected: '# INFO : no cal params' found.")
                        continue  # Exit the function immediately
                # Convert to NumPy array for slicing
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
        combined = list(zip(self.timestamps, self.wavelengths, self.values, self.temperatures, self.labels, self.PeakPosition, self.PeakHeight, self.PeakFWHM, self.fileNameRaw))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        self.timestamps, self.wavelengths, self.values, self.temperatures, self.labels, self.PeakPosition, self.PeakHeight, self.PeakFWHM, self.fileNameRaw= zip(*sorted_combined)
        start_time = self.timestamps[0]
        self.timestampAbsS = [(t - start_time).total_seconds() for t in self.timestamps]
        self.timestampAbsM = [s / 60 for s in self.timestampAbsS]
        self.timestampAbsH = [s / 3600 for s in self.timestampAbsS]
        #Find out which cycle it was on:
        for i in range(len(self.timestampAbsS)):
            self.cycleNum.append((self.timestampAbsS[i]//5520+1))

    def importLight(self, calculate = False, lightFilePath = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/Folder_Samuel_Felix_10072025/Felix/Ivonne/2025-06-26_I26H_spectra/light_100percent/Lichtmessung2.dat'):
        data = np.loadtxt(lightFilePath, skiprows=21)
        self.lightWavelength = data[:, 1]
        self.lightValue = data[:, 2]
        if calculate is True:
            self.lightValue = self.lightValue - self.darkValue
            pass #Need to be done!!!
    
    def interpolate_masked(self): #Don't use this yet
        #Interpolate Data:
        for i in range (0, len(self.timestampAbsS)):
            wavelength_interp = np.arange(self.wavelengths[i].min(), self.wavelengths[i].max(), 0.001)

            value_interp = np.interp(wavelength_interp, self.wavelengths[i], self.values[i])
            #Mask the value:
            mask = (wavelength_interp >= 334) & (wavelength_interp <= 883) #313 to 883
            wavelength_interp_mask = wavelength_interp[mask]
            value_interp_mask = value_interp[mask]
            #Store the value:
            self.wavelengths_interp.append(wavelength_interp_mask)
            self.values_interp.append(value_interp_mask)

        #Find the average data from the 3 raw data, use the data name as the identifier, and generate a new identifier without the ending "-1, -2 or -3"
            #Add an array for errors:

        #Dark measurement and light measurement:
        darkWavelength_interp = np.arange(self.darkWavelength.min(), self.darkWavelength.max(), 0.001)
        darkValue_interp = np.interp(darkWavelength_interp, self.darkWavelength, self.darkValue)
        
        dMask = (darkWavelength_interp >= 334) & (darkWavelength_interp <= 883) #313 to 883
        self.darkWavelength_interp = darkWavelength_interp[dMask]
        self.darkValue_interp = darkValue_interp[dMask]

        lightWavelength_interp = np.arange(self.lightWavelength.min(), self.lightWavelength.max(), 0.001)
        lightValue_interp = np.interp(lightWavelength_interp, self.lightWavelength, self.lightValue)

        lMask = (lightWavelength_interp >= 334) & (lightWavelength_interp <= 883) #313 to 883
        self.lightWavelength_interp = lightWavelength_interp[lMask]
        self.lightValue_interp = lightValue_interp[lMask]

    def average_value(self): #Do this tonight!!!
        pass
    def taucCalc_old(self): #Don't use this yet
        '''This is the Tauc calculation, which is used to calculate the band gap of the material.
        The Tauc plot is a plot of (Absorbance * photon energy)^0.5 vs photon energy.
        The band gap can be found by fitting a line to the Tauc plot and finding the x-intercept.
        The formula for the Tauc plot is:
        Tauc = (Absorbance * photon energy)^0.5
        Where photon energy = 1240 / wavelength (in nm)'''
        lightCalibrated = self.lightValue_interp - self.darkValue_interp
        epsilon = 1e-10
        #X-Axis for tauc plot
        self.eV = 1240 / np.array(self.wavelengths_interp[1])
        for i in range(0, len(self.timestampAbsS)):
            #Calculate the Absorbance:
            #print("The time stamp is:", self.timestampAbsS[i])
            valueCalibrated = np.array(self.values_interp[i]) - np.array(self.darkValue_interp)
            Trans = valueCalibrated / lightCalibrated
            #print("Trans:", Trans)
            Absorb = - np.log10(np.maximum(Trans, epsilon))
            #print("Abs:", Absorb)
            #Calculate the Tauc and Band Gap: photonEnergy = 4.135667696e-15 * 2.99792e8 / wavelength
            tauc = (Absorb * self.eV)**0.5
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
        data = {
            "timestamp": self.timestamps,
            "band_gap": self.bandGap,
            "tauc_slope": self.tauc_slope,
            "tauc_slope_b": self.tauc_slope_b
        }
        df = pd.DataFrame(data)
        df.to_csv(results_file, index=False)

    def taucCalc(self): #Don't use this yet
        '''This is the Tauc calculation, which is used to calculate the band gap of the material.
        The Tauc plot is a plot of (Absorbance * photon energy)^0.5 vs photon energy.
        The band gap can be found by fitting a line to the Tauc plot and finding the x-intercept.
        The formula for the Tauc plot is:
        Tauc = (Absorbance * photon energy)^0.5
        Where photon energy = 1240 / wavelength (in nm)'''
        lightCalibrated = self.lightValue_interp - self.darkValue_interp
        epsilon = 1e-10
        #X-Axis for tauc plot
        self.eV = 1240 / np.array(self.wavelengths_interp[1])
        for i in range(0, len(self.timestampAbsS)):
            #Calculate the Absorbance:
            #print("The time stamp is:", self.timestampAbsS[i])
            valueCalibrated = np.array(self.values_interp[i]) - np.array(self.darkValue_interp)
            Trans = valueCalibrated / lightCalibrated
            #print("Trans:", Trans)
            Absorb = - np.log10(np.maximum(Trans, epsilon))
            #print("Abs:", Absorb)
            #Calculate the Tauc and Band Gap: photonEnergy = 4.135667696e-15 * 2.99792e8 / wavelength
            tauc = (Absorb * self.eV)**0.5
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
        data = {
            "timestamp": self.timestamps,
            "band_gap": self.bandGap,
            "tauc_slope": self.tauc_slope,
            "tauc_slope_b": self.tauc_slope_b
        }
        df = pd.DataFrame(data)
        df.to_csv(results_file, index=False)

    def Pipeline(self):
        '''This is the pipeline for the spectroscopy data, which is used in the thermal cycling setup.
        It will import the data, do the calibration, and calculate the Tauc plot.   '''
        self.importDark(self)
        self.importData(self)
        self.importLight(self)
        self.interpolate_masked(self)
        self.taucCalc(self)