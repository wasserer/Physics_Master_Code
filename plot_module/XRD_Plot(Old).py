#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:36:19 2025

@author: ruodongyang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
#Change the parameters accordingly
#pathName = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/28022025/2StepPerovskite.txt'
#peaksPresets = ()
#pathNameVesta = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/Material Project/FAPbI3.xy'

def importData(pathName):
    with open(pathName, "r") as file:
        lines = file.readlines()
 
    data_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("[Data]"):
            data_start_index = i + 1  # The actual data starts after this line
            break
    
    data_lines = lines[data_start_index:]
    data = np.genfromtxt(data_lines, delimiter=",", dtype=float)
    
    angles = data[:, 0]
    intensities = data[:, 1]
    return angles, intensities

def importVesta(pathNameVesta):
    data = np.loadtxt(pathNameVesta)
    data = data.T
    return data

def findPeaks(data):
    angles = data[0]
    intensities = data[1]
    peak_indices = signal.find_peaks(intensities, height = 25, distance = 50)
    return peak_indices
'''
def peakTreatment(data):
    peaks = findPeaks(data)
    x = np.zeros(len(peaks))
    for i in len(peaks[0]):
        x[i] = peaks[i]
'''

def getPeakAngle(data):
    angle = data[0]
    peaks= findPeaks(data)[0]
    output = angle[peaks]
    print('The 2 theta positions are:', output)
    return output
'''
def getPeakIntensity(data):
    intensity = data[1]
    peaks, _ = findPeaks(data)
    output = intensity[peaks]
    return intensity    
'''    
def printPeaks(data):
    #Import the data:
    x = findPeaks(data)
    peakPositions, peakLabels = x
    for i in range (len(peakPositions)):
        print('Peak Position:', peakPositions[i])
            
def plotXRDData(fileName):
    #Import the data
    with open(fileName, "r") as file:
        lines = file.readlines()
 
    data_start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("[Data]"):
            data_start_index = i + 1  # The actual data starts after this line
            break
    
    data_lines = lines[data_start_index:]
    data = np.genfromtxt(data_lines, delimiter=",", dtype=float)
    
    angle = data[:, 0]
    intensity = data[:, 1]

    #FindPeaks
    peak_indices = signal.find_peaks(intensity, height = 25, distance = 50)
    peaks, _ = peak_indices
    #Plot the figure
    plt.figure(figsize=(7, 5), dpi = 300)
    plt.plot(angle, intensity, linestyle='-', markersize=2, label="XRD Intensity")
    y_min, _ = plt.ylim()
    plt.xlabel("Angle (2θ)")
    plt.ylabel("Intensity (counts)")
    plt.title("XRD Measurement: Angle vs Intensity")
    plt.legend()
    plt.savefig('result_Test.png')
    #plt.show()
    plt.close()
#    plt.grid(True)

def plotDataOld(data):
    angle = data[0]
    intensity = data[1]
    peaks, _ = findPeaks(data)
    plt.figure(figsize=(7, 5), dpi = 300)
    plt.plot(angle, intensity, linestyle='-', markersize=2, label="XRD Intensity")
    y_min, _ = plt.ylim()
#    plt.ylim(10, plt.y_max()*10)
#    plt.yscale('log')
#    plt.xlim(5, 40)
#    plt.ylim(10,1000000)
#    plt.plot(angle[peaks], intensity[peaks], 'x')
#    plt.vlines(angle[peaks], intensity[peaks]*1.1, 40000, color = 'black') #This is to draw the line
#    plt.axvline(intensity[peaks])
    plt.xlabel("Angle (2θ)")
    plt.ylabel("Intensity (counts)")
    plt.title("XRD Measurement: Angle vs Intensity")
    plt.legend()
    plt.savefig('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/14042025/result.png')
    #plt.show()
    plt.close()
#    plt.grid(True)

def PlotCompareData(data, dataVesta):
    angle = data[0]
    intensity = data[1]
    print("Here!",  data, intensity.min(), "End here!")
    intensityMax = np.nanmax(intensity)
    intensity = intensity/intensityMax
    #print("type:", type(intensity))
    #print(intensity)
    angleVesta = dataVesta[0]
    intensityVesta = dataVesta[1]
    print(type(intensityVesta))
    intensityVesta = intensityVesta/ max(intensityVesta)
    peaks, _ = findPeaks(data)
    plt.figure(figsize=(10, 5), dpi = 300)
    plt.plot(angle, intensity, linestyle='-', markersize=2, label="XRD Intensity", color = "b", alpha = 0.7)
    plt.plot(angleVesta, intensityVesta, linestyle='-', markersize=2, label="Simulated", color = "r", alpha = 0.7)
    y_min, _ = plt.ylim()
#    plt.ylim(10, plt.y_max()*10)
#    plt.yscale('log')
#    plt.xlim(5, 40)
#    plt.ylim(10,1000000)
#    plt.plot(angle[peaks], intensity[peaks], 'x')
    plt.vlines(angle[peaks], 0, intensity[peaks]*0.9, color = 'black') #This is to draw the line
#    plt.axvline(intensity[peaks])
    plt.xlabel("Angle (2θ)")
    plt.ylabel("Intensity (Noramalized)")
    plt.title("XRD Measurement: Angle vs Intensity")
    plt.legend()
    plt.savefig('/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/14042025/result_2step.png')
    #plt.show()
    plt.close()
#    plt.grid(True)

if __name__ == '__main__':
    data = importData(pathName = pathName)
    dataVesta = importVesta(pathNameVesta = pathNameVesta)
    PlotCompareData(data = data, dataVesta=dataVesta)
#    plotData(data = data)
#    plotDataLog(data = data)
#    printPeaks(data)
    x = getPeakAngle(data)
    #print(x)
    #print(dataVesta)