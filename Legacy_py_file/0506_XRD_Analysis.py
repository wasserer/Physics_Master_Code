from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color

figColor = color.matlab(multiData=True)

#fileNew ='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/CB.txt'
folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/Results'
#fileOld ='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/Compare_Wth_Old/25sIPA_old.txt'
#folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/Compare_Wth_Old'
analyzer_new = XRDAnalyzer()
peaks = XRDAnalyzer()
peaks.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/26062025/N.txt')
peaks.find_peaks_and_FWHM(log = True)
peaks.plotXRD(saveFolderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/26062025/result.png')
#analyzer_new.peakAngles = peaks.peakAngles
#analyzer_new.import_xrd_data(pathName=fileNew)
#analyzer_new.import_xrd_folder('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/Results')
#analyzer.baselineCorrection()
#analyzer.import_xrd_folder(folderPath=folder)
#analyzer_new.find_peaks_and_FWHM(log=True)
#analyzer_new.multiXRD(graphColor=figColor, findPeaks=True)
#analyzer.plotXRD(graphColor=figColor, saveFolderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/result.png')
#analyzer.multiXRD(graphColor = figColor)
#analyzer_new.plotXRD(graphColor=figColor, saveFolderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/resultCB.png')