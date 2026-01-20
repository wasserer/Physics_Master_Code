from plot_module.solsim_analyzer import solarSimulator
from plot_module.colors import color

figColor=color.matlab(multiData=True)

PTAA = solarSimulator(folderPath="")  # Enter your file path/folder path in this place
PTAA.loadFolderData()
PTAA.IVMultiPlot(colorMode=figColor, saveName="result.png")