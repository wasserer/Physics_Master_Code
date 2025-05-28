#from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
from plot_module.solsim_analyzer import solarSimulator

figColor = color.red(multiData=True)

folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/BestCells'
analyzer = solarSimulator(folderPath=folder)
analyzer.loadFolderData()
analyzer.IVMultiPlot(saveName="IV_BestCells.png", colorMode = figColor)
#analyzer.logData()
#analyzer.histoPlot(saveName="BestCellsIV_21052025.png")