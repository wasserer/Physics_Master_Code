#from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
from plot_module.solsim_analyzer import solarSimulator

#debugger = solarSimulator(filePath="")  # Enter your file path/folder path in this place
#debugger.loadFileData()
#print(debugger.PCE)
folder = ""  # Enter your file path/folder path in this place
M25 = solarSimulator(folderPath="")  # Enter your file path/folder path in this place
M30 = solarSimulator(folderPath="")  # Enter your file path/folder path in this place
M25.loadFolderData()
M25.logData()
M30.loadFolderData()
M30.logData()
M25.boxPlot(data1=M25.PCE, data2=M30.PCE)
#print(PTAA50.PCE)