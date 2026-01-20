from plot_module.Spectra import Spectroscopy
from plot_module.solsim_analyzer import solarSimulator
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import re

file = ""  # Enter your file path/folder path in this place

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