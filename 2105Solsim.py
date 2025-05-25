#from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.colors import color
from plot_module.solsim_analyzer import solarSimulator

figColor = color.matlab()

folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest'
analyzer = solarSimulator(folderPath=folder)
analyzer.loadFolderData()
analyzer.histoPlot(saveName="PCE_Histo.png", color = figColor)