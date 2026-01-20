from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
analyzer = XRDAnalyzer()
#figColor = color.matlab(multiData=True)
#analyzer.import_xrd_folder(folderPath="")  # Enter your file path/folder path in this place
#analyzer.normalize()
analyzer2 = XRDAnalyzer()
analyzer2.import_xrd_data(pathName="")  # Enter your file path/folder path in this place
analyzer2.find_peaks_and_FWHM(log=True)
#analyzer.peakAngles = analyzer2.peakAngles
#print(type(analyzer.xrd_data), analyzer.xrd_data.shape)
#print("peak Array shape", analyzer.peakAngles.shape)
#print("Peak array list",analyzer.peakAngles)
#analyzer.multiXRD(graphColor=figColor)
#analyzer.zoomInMultiPlot(graphColor=figColor)
#analyzer.multiXRD(graphColor=figColor, savePath="")  # Enter your file path/folder path in this place