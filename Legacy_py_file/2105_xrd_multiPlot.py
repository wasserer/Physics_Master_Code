from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color

figColor = color.matlab(multiData=True)
masterPath = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/data/25sIPA.txt'
master_curve = XRDAnalyzer()
master_curve.import_xrd_data(pathName=masterPath)
master_curve.find_peaks_and_FWHM()
folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/data'
xrd_data = XRDAnalyzer()
xrd_data.import_xrd_folder(folderPath=folder)
xrd_data.peakAngles = master_curve.peakAngles
xrd_data.multiXRD(fileName="result_multiXRD_withAngle.png", graphColor=figColor)
