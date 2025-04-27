#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 18:02:01 2025

@author: ruodongyang

This is a piece of script trying to replot the I-V curve based on the Solsim raw data
"""

#Here are the parameter needed to adjust the settings:

import numpy as np
import pandas as pd    
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
# Import the path of the .dat file
#filePath = r'/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/10_02_2025/Old_10022025Cell3/Old_10022025'
#cellType = 'pin' #pin or nip

def importData(filePath):
    files = pd.read_csv(filePath, skiprows=range(0, 21), header=None, names=['voltage', 'current', 'power', 'time'], sep='\s+')
    data = np.asarray(files)
    return data

def SortData(filePath, cellType = 'pin'):
    data = importData(filePath)
    data = data.T
    voltage = data[0]
    current = data[1]
    power = data[2]
    time = data[3]
    current = -current
    return voltage, current, power, time

def IV_Plot(filePath, saveName = "IV-Plot2.png"): # Subject to change
    #Import Data
    #files = pd.read_csv(filePath, skiprows=range(0, 21), header=None, names=['voltage', 'current', 'power', 'time'], sep='\s+')
    #data = np.asarray(files)
    #SortData
    #data = data.T
    #voltage = data[0]
    #current = data[1]
    #power = data[2]
    #time = data[3]
    voltage, current, power, time = np.loadtxt(
            filePath,
            dtype=float,      # Expect float data
            #delimiter='\s',  # Use one or more whitespace as delimiter
            skiprows=21,      # Skip the first 21 rows
            usecols=(0, 1, 2, 3), # Select the 4 columns by index (0-based)
            unpack=True       # Assign columns to separate variables
        )

    #current = -current

    
    
    plt.figure(figsize=(7, 5), dpi = 300)
    plt.plot(voltage, current*1000)
    plt.ylabel("Current [mA]")
    plt.xlabel("Voltage [V]")
    plt.grid()
    plt.savefig(saveName)
    plt.close()
    
def MultiPlot(filePath):
    dat_files = [f for f in os.listdir(filePath) if f.endswith('.dat')]
    plt.figure()

    for file_name in dat_files:
        full_path = os.path.join(filePath, file_name)
        voltage, current, power, time = SortData(full_path)
        label = os.path.splitext(file_name)[0]
        plt.plot(voltage, current * 1000, label=label)

    plt.ylabel("Current [mA]")
    plt.xlabel("Voltage [V]")
    plt.legend()
    plt.title("IV-Curve of the Samples")
    plt.grid()
    plt.tight_layout()
    #plt.savefig(os.path.join(filePath, "IV_combined_plot.png"))
    plt.show()
        
        
def GetVOC(filePath):
    voltage, current, power, time = SortData(filePath)
    V = np.array(voltage)
    I = np.array(current)
# Sort data by voltage (just in case it's not sorted)
    sorted_indices = np.argsort(V)
    V = V[sorted_indices]
    I = I[sorted_indices]
# Interpolators
    current_vs_voltage = interp1d(V, I, kind='linear', fill_value="extrapolate")
    voltage_vs_current = interp1d(I, V, kind='linear', fill_value="extrapolate")
# Short-circuit current (I at V=0)
    Isc = current_vs_voltage(0)
# Open-circuit voltage (V at I=0)
    Voc = voltage_vs_current(0)
    return Isc, Voc
if __name__ == "__main__":
    IV_Plot(filePath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/10_04_2025_1Step_100C/Cell_PTAA150_px2_Light_forward_0.dat')
        