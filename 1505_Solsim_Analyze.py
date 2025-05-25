#from plot_module.colors import color
from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.solsim_analyzer import solarSimulator

folderCB = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/15_05_2025_1Step_CB_IPA/CB'
folderIPA = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/15_05_2025_1Step_CB_IPA/IPA'

CBsolsim = solarSimulator(folderPath=folderCB)
CBsolsim.loadFolderData()
CBsolsim.logData()
CBsolsim.histoPlot(data=CBsolsim.PCE, color="orange")
IPAsolsim = solarSimulator(folderPath=folderIPA)
IPAsolsim.loadFolderData()
IPAsolsim.logData()
IPAsolsim.histoPlot(data=IPAsolsim.PCE, color = "blue")



