from plot_module.solsim_analyzer import solarSimulator
from plot_module.colors import color

figColor=color.matlab(multiData=True)

PTAA = solarSimulator(folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/PIN_06_03_2025_PTAATest/Best_IV_Comparison')
PTAA.loadFolderData()
PTAA.IVMultiPlot(colorMode=figColor, saveName="result.png")