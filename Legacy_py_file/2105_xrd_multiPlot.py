from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color

figColor = color.matlab(multiData=True)
masterPath = ""  # Enter your file path/folder path in this place
master_curve = XRDAnalyzer()
master_curve.import_xrd_data(pathName=masterPath)
master_curve.find_peaks_and_FWHM()
folder = ""  # Enter your file path/folder path in this place
xrd_data = XRDAnalyzer()
xrd_data.import_xrd_folder(folderPath=folder)
xrd_data.peakAngles = master_curve.peakAngles
xrd_data.multiXRD(fileName="result_multiXRD_withAngle.png", graphColor=figColor)
