# Physics_Master_Code
## Introduction:
This is a code collection for my master thesis. The use of it is to replace programs like Origin or Matlab, to treat and evaluate the data from Solar Simulator, XRD etc.
## Installation:
Just download it and run it. The file location needs to be manually adjusted.
## Functioning Modules:
All the modules are written in classes, some of them works together. Some examples can be found in the jupyter notebooks.
#### Solar simulator from glovebox and the IV-Curve(Thermal Cycling Setup) analyzer: 
Use **solsim_analyzer.py**
- from plot_module.solsim_analyzer import solarSimulator, the difference of each function: The ones for Thermal Cycling has a "cycle" after it.
#### UV-VIS Spectrometer in the Polymer Lab:
Use **UVVIS_analyzer.py**
- from plot_module.UVVIS_analyzer import UV_VIS_Analyzer
#### Spectra from the Thermal Cycling setup:
Use **Spectra.py**
- from plot_module.Spectra import Spectroscopy
