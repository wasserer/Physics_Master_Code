from plot_module.UVVIS_analyzer import UV_VIS_Analyzer
from plot_module.colors import color

folder = ""  # Enter your file path/folder path in this place
colors = color.matlab(multiData=True)
df = UV_VIS_Analyzer(GermanMode=False, folderPath=folder)
df.UV_multiPlot(figColor=colors)