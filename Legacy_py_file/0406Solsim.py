#from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
from plot_module.solsim_analyzer import solarSimulator

figColor = color.matlab(multiData=True)
print(0.6914389233954451*(-0.00322581))
#file = ""  # Enter your file path/folder path in this place
#debugger = solarSimulator(filePath="")  # Enter your file path/folder path in this place
#debugger.loadFileData()
#print(debugger.PCE)
folder = ""  # Enter your file path/folder path in this place
'''
#PTAA25 = solarSimulator(folderPath="")  # Enter your file path/folder path in this place
#PTAA50 = solarSimulator(folderPath="")  # Enter your file path/folder path in this place
#PTAA25.loadFolderData()
#PTAA25.logData()
#PTAA50.loadFolderData()
#PTAA50.logData()
#PTAA25.boxPlot(data1=PTAA25.PCE, data2=PTAA50.PCE)
#print(PTAA50.PCE)
analyzer = solarSimulator(folderPath=folder)
analyzer.loadFolderData()
analyzer.logData()
analyzer.IVMultiPlot(colorMode=figColor)
'''
#Plot Boxplot:
analyzer = solarSimulator(folderPath=folder)
analyzer.loadFolderData()
analyzer.histoPlot(color="#0367A6")