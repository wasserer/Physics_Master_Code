from  plot_module.xrd_analyzer import XRDAnalyzer
import numpy as np
from plot_module.colors import color

file = ""  # Enter your file path/folder path in this place

analyzer = XRDAnalyzer()
analyzer.import_xrd_data(pathName=file)
analyzer.find_peaks_and_FWHM(log=True)
#print()
figcolor = color.matlab()
print(analyzer.peakAngles)
print(analyzer.FWHM)
print(analyzer.grainSizes)
#print(analyzer.angles)
#print(analyzer.intensities)
analyzer.plotXRD(save_path="PeakTest.png", findPeaks=True, graphColor=figcolor)