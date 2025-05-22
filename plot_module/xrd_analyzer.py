#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:36:19 2025
@author: ruodongyang
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from pybaselines import Baseline, utils
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.ext.matproj import MPRester
import re
from io import StringIO
#from colors import *
#import pandas as pd
#import importFile.importXRD as file

class XRDAnalyzer:
    def __init__(self, vesta_path=None, importer = None):
        #self.xrd_path = xrd_path
        self.vesta_path = vesta_path
        self.xrd_data = None
        self.vesta_data = None
        self.importer = importer
        self.peaks = None
        self.simuPeaks = None
        self.peakAngles = None
        self.angles = None
        self.intensities = None
        self.MP_API = "3ZAPXUD7Z91YReSGsjl1snF1d8uNkLDc"
        self.label = []

    def import_xrd_data(self, labelName ="NewData", pathName = ""):
        with open(pathName, "r") as file:
            lines = file.readlines()

        data_start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("[Data]"):
                data_start_index = i + 1 #Old version: data_start_index = i + 1
                break
        data_lines = lines[data_start_index:]
        data = np.genfromtxt(data_lines, delimiter=",", dtype=float, skip_header=1)
        data[:, 1] = data [:, 1] / np.max(data[:, 1])
        data = data[:, :2]
        data = data.T
        if self.xrd_data is None:
            self.xrd_data = data
            self.angles = data[0].reshape(1, -1)
            self.intensities = data[1].reshape(1, -1)
        else:
            self.xrd_data = np.vstack((self.xrd_data, data))
            self.angles = np.vstack((self.angles, data[0]))
            self.intensities = np.vstack((self.intensities, data[1]))

        self.label.append(labelName)
        
    def import_vesta_data(self, vestaPath = ""):
        data = np.loadtxt(vestaPath).T
        self.vesta_data = (data[0], data[1])
        return self.vesta_data
    #Newly added function:
    def find_peaks_and_fwhm(self, height=20, prominence=20, distance=5, output_file="peaks_and_fwhm.txt"):
        """
        Identify peaks in the XRD data and calculate their FWHM.
        Save results to a .txt file and store them in self.peaks_with_fwhm.

        Parameters:
        - height: minimum peak height
        - prominence: minimum prominence of the peaks
        - distance: minimum distance between peaks
        - output_file: file name to save peak positions and FWHMs
        """
        if self.xrd_data is None:
            raise ValueError("XRD data not loaded. Use import_xrd_data() first.")

        x, y = self.xrd_data
        peaks, properties = signal.find_peaks(y, height=height, prominence=prominence, distance=distance)

        # Calculate FWHM using scipy.signal.peak_widths
        widths_result = signal.peak_widths(y, peaks, rel_height=0.5)
        fwhms = widths_result[0] * (x[1] - x[0])  # convert from samples to x-units

        # Save data
        results = np.column_stack((x[peaks], fwhms))
        np.savetxt(output_file, results, header="Peak_Position  FWHM", fmt="%.4f")

        # Save to class attribute
        self.peaks_with_fwhm = results

        return results

    def baselineCorrection(self): 
        #baselineFilter = Baseline(x_data = self.xrd_data[0])
        #newY, para1 = (baselineFilter.mor(self.xrd_data[1], half_window=30))
        #newY = np.abs(self.xrd_data[1]-newY)
        #self.xrd_data = (self.xrd_data[0], newY)
        x = self.xrd_data[0]
        y = self.xrd_data[1]

        # 找到既不是NaN又是有限数值的位置
        valid_mask = np.isfinite(x) & np.isfinite(y)

        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        # 下面正常进行
        baselineFilter = Baseline(x_data=x_clean)
        newY, para1 = baselineFilter.mor(y_clean, half_window=30)
        newY = np.abs(y_clean - newY)
        self.xrd_data = (x_clean, newY)

    def MPAnalyzer(self):
        pass


    def find_peaks(self, height=10, distance=5):
        if self.xrd_data is None:
            raise ValueError("XRD data not loaded.")
        angle, intensity = self.xrd_data
        peaks, properties = signal.find_peaks(intensity, height=height, distance=distance, prominence = 20)
        return peaks, properties

    def getPeakAngles(self):
        peaks, _ = self.find_peaks()
        angles = self.xrd_data[0][peaks]
        print("The 2θ peak positions are:", angles)
        return angles

    def getFWHM(self):
        pass

    def plotXRD(self, save_path="result_Test.png", graphColor = None, labelName="Label", save = True):
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(self.angles[0], self.intensities[0], label=labelName, color = graphColor)
        plt.xlabel("Angle (2θ)")
        plt.ylabel("Intensity (counts)(or normalized)")
        plt.title("XRD Measurement")
        plt.legend()
        if save == True:
            plt.savefig(save_path)
            plt.close()

    def plotVesta(self, save_path="result_comparison.png"):
        angle, intensity = self.xrd_data
        angle_vesta, intensity_vesta = self.vesta_data
        #Normalizing the data
        intensity = intensity / np.nanmax(intensity)
        intensity_vesta = intensity_vesta / np.max(intensity_vesta)

        peaks = self.getPeakAngles()
        graphColor = color.red()
        colorVesta = color.blue()
        plt.figure(figsize=(10, 5), dpi=300)
        plt.plot(angle, intensity, label="XRD Intensity", alpha=0.9, color = graphColor)
        plt.plot(angle_vesta, intensity_vesta, label="Simulated", alpha=0.9, color = colorVesta)
        for i in peaks:
            plt.vlines(i, 0, intensity[self.peaks]*0.9, color="black")

        plt.xlabel("Angle (2θ)")
        plt.ylabel("Intensity (Normalized)")
        plt.title("XRD vs Simulated Data")
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def multiXRD(self, savePath = "multiResult.png", graphColor = None, direction = True):
        num = self.angles.shape[0]
        plt.figure(figsize=(10,10), dpi = 300)
        for i in range (0, num):
            angle = self.angles[i]
            intensity = self.intensities[i] + 1.1 * i
            plt.plot(angle, intensity, label = self.label[i], color = graphColor[i])
        #Add sth to plot the simulated vesta peaks
        
        plt.xlabel("Angle (2θ)")
        plt.ylabel("Intensity (normalized)")
        plt.title("XRD Measurement")
        plt.legend()
        plt.grid()
        plt.savefig(savePath)
        plt.close()
        
    def savePeaks(self, file_path = 'Peaks.txt'):
        angles = self.getPeakAngles()
        np.savetxt(fname = file_path, X=angles, delimiter=",")
        self.peakAngles = angles

if __name__ == "__main__":
    from colors import *
    xrd_path = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/14042025/Result.txt'
    vesta_path = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/Material Project/FAPbI3.xy'
    analyzer = XRDAnalyzer()
    analyzer.import_xrd_data(pathName=xrd_path)
    x = analyzer.angles.shape
    figColor = color.matlab
    print(x[0])
    print(analyzer.xrd_data, type(analyzer.xrd_data), analyzer.xrd_data.shape)
    print(analyzer.angles, type(analyzer.angles), analyzer.angles.shape)
    print(analyzer.intensities, type(analyzer.intensities), analyzer.intensities.shape)
    analyzer.import_vesta_data(vestaPath=vesta_path)
    #analyzer.getPeakAngles()
    #analyzer.baselineCorrection()
    #analyzer.savePeaks()
    labelName = ["IPA"]
    analyzer.plotXRD(save_path = "testNor.png", graphColor= "red", labelName="IPA")
    #analyzer.plotVesta("xrd_comparison.png")
    #x, y = analyzer.find_peaks()
    #print(x, y)
    #print(analyzer.xrd_data[0])
    #print(analyzer.xrd_data[1])
    #analyzer.savePeaks()