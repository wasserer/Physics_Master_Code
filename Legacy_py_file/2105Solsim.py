#from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
from plot_module.solsim_analyzer import solarSimulator

#figColor = color.matlab(multiData=True)
print(0.6914389233954451*(-0.00322581))
#file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/PIN_06_03_2025_PTAATest/25/Ruodong_cell1_25_px3_Light_forward_0.dat'
#debugger = solarSimulator(filePath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/PIN_06_03_2025_PTAATest/25/Ruodong_cell1_25_px3_Light_forward_0.dat')
#debugger.loadFileData()
#print(debugger.PCE)
folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/BestCells'
PTAA25 = solarSimulator(folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/PIN_06_03_2025_PTAATest/25')
PTAA50 = solarSimulator(folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/PIN_06_03_2025_PTAATest/50')
PTAA25.loadFolderData()
#PTAA25.logData()
PTAA50.loadFolderData()
#PTAA50.logData()
PTAA25.boxPlot(data1=PTAA25.PCE, data2=PTAA50.PCE)
#print(PTAA50.PCE)