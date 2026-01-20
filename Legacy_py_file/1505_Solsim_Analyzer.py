#from plot_module.colors import color
from plot_module.xrd_analyzer import XRDAnalyzer
from plot_module.solsim_analyzer import solarSimulator
import matplotlib.pyplot as plt

folderCB = ""  # Enter your file path/folder path in this place
folderIPA = ""  # Enter your file path/folder path in this place

CBsolsim = solarSimulator(folderPath=folderCB)
CBsolsim.loadFolderData()
CBsolsim.logData()
#CBsolsim.histoPlot(data=CBsolsim.PCE, color="orange")
IPAsolsim = solarSimulator(folderPath=folderIPA)
IPAsolsim.loadFolderData()
IPAsolsim.logData()
#IPAsolsim.histoPlot(data=IPAsolsim.PCE, color = "blue")
CB =CBsolsim.PCE
IPA = IPAsolsim.PCE
plt.figure(figsize=(9, 6), dpi = 300)
plt.boxplot(CB)
plt.savefig("Box.png")



