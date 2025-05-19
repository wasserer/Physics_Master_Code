from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
analyzer = XRDAnalyzer()
figColor = color.matlab(multiData=True)
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD05152025/CB/CB.txt', labelName="CB")
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD05152025/IPA/IPA.txt', labelName="IPA")
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/28022025/2StepPerovskite.txt', labelName="2 Step CB")
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/28022025/1StepPerovskite.txt', labelName="Onestep CB old")
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/14042025/Result.txt', labelName="Onestep 14.4")
print(analyzer.xrd_data, analyzer.xrd_data.shape)
print(figColor)
analyzer.multiXRD(graphColor=figColor)