#Simple import and plot PL etc.
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit

class PL_Analyzer:
    def __init__(self, folderPath = None):
        pass
        self.folderPath = folderPath
        self.wavelengths = []
        self.values = []
        self.gaussian_Wavelength = []
        self.gaussianFit = []
        self.gaussianFit_Parameter = []
        self.gaussianFit_Error = []

    def gaussian(self, x, A, mu, sigma, offset):
        return offset + A * np.exp(-(x - mu)**2 / (2 * sigma**2))


    def importFolder(self, log = False):
        self.dat_files = [f for f in os.listdir(self.folderPath) if f.endswith('.dat')]
        for fileName in self.dat_files:
            filePath = os.path.join(self.folderPath, fileName)
            data = np.loadtxt(filePath, skiprows=13).T
            wavelength = data[1].tolist()
            value = data[2].tolist()
            self.wavelengths.append(wavelength)
            self.values.append(value)
            #FindPeak:
            wavelength = np.array(wavelength)
            value = np.array(value)
            mask = (wavelength >=720 ) & (wavelength <= 870)
            wavelength_mask = wavelength[mask]
            value_mask = value[mask]
            #Do gaussian fit:
            offset0 = np.min(value_mask)
            A0 = np.max(value_mask) - offset0
            mu0 = wavelength_mask[np.argmax(value_mask)]
            sigma0 = 10
            popt, pcov = curve_fit(self.gaussian, wavelength_mask, value_mask, p0 = [A0, mu0, sigma0, offset0])
            A_fit, mu_fit, sigma_fit, offset_fit = popt
            self.gaussianFit_Parameter.append(popt)
            self.gaussianFit_Error.append(pcov)
            self.gaussian_Wavelength.append(wavelength_mask)
            self.gaussianFit.append(self.gaussian(wavelength_mask, A_fit, mu_fit, sigma_fit, offset_fit))
            



if __name__ == "__main__":
    folderPath = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/21072025/PL'
    analyzer = PL_Analyzer(folderPath=folderPath)
    analyzer.importFolder()


    plt.figure(figsize = (5, 3))
    plt.plot(analyzer.wavelengths[0], analyzer.values[0])
    plt.plot(analyzer.gaussian_Wavelength[0], analyzer.gaussianFit[0])
    plt.show()
    #plt.savefig("Result_PL.png")
