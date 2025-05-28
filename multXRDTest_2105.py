from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
analyzer = XRDAnalyzer()
figColor = color.matlab(multiData=True)
analyzer.import_xrd_folder(folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/data')
analyzer.normalize()
analyzer2 = XRDAnalyzer()
analyzer2.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/data/25sIPA.txt')
analyzer2.find_peaks_and_FWHM()
analyzer.peakAngles = analyzer2.peakAngles
#print(type(analyzer.xrd_data), analyzer.xrd_data.shape)
print("peak Array shape", analyzer.peakAngles.shape)
print("Peak array list",analyzer.peakAngles)
analyzer.zoomInMultiPlot(graphColor=figColor)
#analyzer.multiXRD(graphColor=figColor, savePath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/data/MultiPlot_Auto.png')