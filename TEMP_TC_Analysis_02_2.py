#Import relevant libs:
import numpy as np
from plot_module.tc_analyzer import ThermalCycling
from plot_module.solsim_analyzer import solarSimulator
from plot_module.Spectra import Spectroscopy
import matplotlib.pyplot as plt

Spectra_FolderPath = ""  # Enter your file path/folder path in this place
Spectra = Spectroscopy(folderPath=Spectra_FolderPath)

#Spectra.importDark(darkFilePath="")  # Enter your file path/folder path in this place
#spectra.importDark(darkFilePath="")  # Enter your file path/folder path in this place
Spectra.Pipeline(darkFolder="")  # Enter your file path/folder path in this place

#import matplotlib.pyplot as plt
plt.figure()
plt.scatter(Spectra.timestampAbsHN, Spectra.bandGap)
#plt.ylim(1.45,)
plt.xlim(0, 18)
plt.show()