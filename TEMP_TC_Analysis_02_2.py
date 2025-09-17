#Import relevant libs:
import numpy as np
from plot_module.tc_analyzer import ThermalCycling
from plot_module.solsim_analyzer import solarSimulator
from plot_module.Spectra import Spectroscopy
import matplotlib.pyplot as plt

Spectra_FolderPath = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TEMP_TC_1709/Spectra'
Spectra = Spectroscopy(folderPath=Spectra_FolderPath)

#Spectra.importDark(darkFilePath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TEMP_TC16092025/Spectra/Dark/20250915_155012_Thermal_Cycling_Spectrum_dark_Dark.dat')
#spectra.importDark(darkFilePath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/06082025/Thermal_Cycling_2025-08-05_14-59-47/Spectra/Dark/20250805_150250_Thermal_Cycling_Spectrum_dark_Dark.dat')
Spectra.Pipeline(darkFolder='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/TEMP_TC_1709/Spectra/Dark/20250915_155012_Thermal_Cycling_Spectrum_dark_Dark.dat')

#import matplotlib.pyplot as plt
plt.figure()
plt.scatter(Spectra.timestampAbsHN, Spectra.bandGap)
#plt.ylim(1.45,)
plt.xlim(0, 18)
plt.show()