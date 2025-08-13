from plot_module.Spectra import Spectroscopy
from plot_module.solsim_analyzer import solarSimulator
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import re

file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/06082025_Temp/Test1_8_Orbit_2025-07-28_18-26-05/px2/20250728_182632_Test1_8_Orbit_px2_Light_forward_c000.dat'

with open(file, 'r') as data:
    lines = data.readlines()

# Get the 4th line
line = lines[3]

# Extract the first integer from the line
import re
match = re.search(r'\d+', line)
if match:
    cycle_number = int(match.group())
    print(cycle_number)
else:
    print("No integer found.")