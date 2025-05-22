from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
analyzer = XRDAnalyzer()
figColor = color.matlab(multiData=True)
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/20sIPA.txt', labelName="20s_IPA")
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/25sCB.txt', labelName="25s_CB")
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/25sIPA.txt', labelName="25sIPA")
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/R5.txt', labelName="RPM5000_IPA")
analyzer.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD05152025/IPA/IPA.txt', labelName="IPA_Old")
analyzer.import_vesta_data(vestaPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/Material Project/FAPbI3.xy')
print(analyzer.xrd_data, analyzer.xrd_data.shape)
print(figColor)
analyzer.multiXRD(graphColor=figColor, savePath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/21052025/MultiPlot1.png')