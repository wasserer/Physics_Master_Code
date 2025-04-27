# Author: Ruodong Yang
# Add module in the plot_module folder
# Add functions to call different modules with the functions under it

from .XRD_Plot import plotXRDData
from .IV_Plot import IV_Plot
#from .IV_Sweep_Post_Analysis import plot_iv_sweep
#from .analyse_result_sum import plot_summary

def import_and_plot(fileLocation, mode):
    if mode == 'xrd':
        return plotXRDData(fileName = fileLocation)
    elif mode == 'iv':
        return IV_Plot(filePath = fileLocation)
    #elif mode == 'iv_sweep':
    #    return plot_iv_sweep(fileLocation)
    #elif mode == 'summary':
    #    return plot_summary(fileLocation)
    #else:
    #    raise ValueError(f"Unknown mode: {mode}. Choose from 'xrd', 'iv', 'iv_sweep', 'summary'.")
    