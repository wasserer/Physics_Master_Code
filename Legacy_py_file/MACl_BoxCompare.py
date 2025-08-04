#from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
from plot_module.solsim_analyzer import solarSimulator

#debugger = solarSimulator(filePath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/PIN_06_03_2025_PTAATest/25/Ruodong_cell1_25_px3_Light_forward_0.dat')
#debugger.loadFileData()
#print(debugger.PCE)
folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/BestCells'
M25 = solarSimulator(folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/35_20_MACl/MACl_3')
M30 = solarSimulator(folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/35_20_MACl/MACl30')
M25.loadFolderData()
M25.logData()
M30.loadFolderData()
M30.logData()
M25.boxPlot(data1=M25.PCE, data2=M30.PCE)
#print(PTAA50.PCE)