from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color

figColor = color.matlab(multiData=True)

#fileNew =""  # Enter your file path/folder path in this place
folder = ""  # Enter your file path/folder path in this place
#fileOld =""  # Enter your file path/folder path in this place
#folder = ""  # Enter your file path/folder path in this place
analyzer_new = XRDAnalyzer()
peaks = XRDAnalyzer()
peaks.import_xrd_data(pathName="")  # Enter your file path/folder path in this place
peaks.find_peaks_and_FWHM(log = True)
peaks.plotXRD(saveFolderPath="")  # Enter your file path/folder path in this place
#analyzer_new.peakAngles = peaks.peakAngles
#analyzer_new.import_xrd_data(pathName=fileNew)
#analyzer_new.import_xrd_folder("")  # Enter your file path/folder path in this place
#analyzer.baselineCorrection()
#analyzer.import_xrd_folder(folderPath=folder)
#analyzer_new.find_peaks_and_FWHM(log=True)
#analyzer_new.multiXRD(graphColor=figColor, findPeaks=True)
#analyzer.plotXRD(graphColor=figColor, saveFolderPath="")  # Enter your file path/folder path in this place
#analyzer.multiXRD(graphColor = figColor)
#analyzer_new.plotXRD(graphColor=figColor, saveFolderPath="")  # Enter your file path/folder path in this place