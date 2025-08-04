from plot_module.UVVIS_analyzer import UV_VIS_Analyzer
from plot_module.colors import color

folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/FoPra_WS2425/FoPra_50/data/UV_VIS'
colors = color.matlab(multiData=True)
df = UV_VIS_Analyzer(GermanMode=False, folderPath=folder)
df.UV_multiPlot(figColor=colors)