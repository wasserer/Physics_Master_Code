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
from colors import *
#import pandas as pd
#import importFile.importXRD as file

class XRDAnalyzer:
    def __init__(self, xrd_path=None, vesta_path=None, importer = None):
        self.xrd_path = xrd_path
        self.vesta_path = vesta_path
        self.xrd_data = None
        self.vesta_data = None
        self.importer = importer
        self.peaks = None
        self.simuPeaks = None
        self.peakAngles = None
    def import_xrd_data(self):
        with open(self.xrd_path, "r") as file:
            lines = file.readlines()

        data_start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("[Data]"):
                data_start_index = i + 1 #Old version: data_start_index = i + 1
                break
        data_lines = lines[data_start_index:]
        data = np.genfromtxt(data_lines, delimiter=",", dtype=float, skip_header=1)
        self.xrd_data = (data[:, 0], data[:, 1])
        
    def import_vesta_data(self):
        data = np.loadtxt(self.vesta_path).T
        self.vesta_data = (data[0], data[1])
        return self.vesta_data
    






    def find_peaks_and_fwhm(self, height=None, prominence=None, distance=5, output_file="peaks_and_fwhm.txt"):
        """
        Identify peaks in the XRD data, calculate their FWHM and heights,
        and save results to a .txt file. Automatically adjusts threshold if not given.

        Parameters:
        - height: minimum peak height (auto-calculated if None)
        - prominence: minimum peak prominence (auto if None)
        - distance: minimum distance between peaks
        - output_file: output file name
        """
        if self.xrd_data is None:
            raise ValueError("XRD data not loaded. Use import_xrd_data() first.")

        x, y = self.xrd_data

        # Auto height: 10% of max intensity
        if height is None:
            height = 0.1 * np.max(y)

        # First find all peaks to analyze prominence
        if prominence is None:
            all_peaks, all_props = signal.find_peaks(y, height=height, distance=distance, prominence=0.01)
            prominences = all_props["prominences"]
            if len(prominences) > 0:
                prominence = np.percentile(prominences, 85)  # keep only top 15% most prominent
            else:
                prominence = 0.01

        # Now find peaks with final thresholds
        peaks, properties = signal.find_peaks(y, height=height, prominence=prominence, distance=distance)

        # Calculate FWHM
        widths_result = signal.peak_widths(y, peaks, rel_height=0.5)
        fwhms = widths_result[0] * (x[1] - x[0])

        peak_positions = x[peaks]
        peak_heights = properties["peak_heights"]

        results = np.column_stack((peak_positions, fwhms, peak_heights))
        np.savetxt(output_file, results, header="Peak_Position  FWHM  Peak_Height", fmt="%.4f")

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

    def Normalize(self):
        ##data == xrd or vesta
        pass

    def getFWHM(self):
        pass

    def plotXRD(self, save_path="result_Test.png", graphColor = color.matlab()):
        angle, intensity = self.xrd_data
        peaks = self.find_peaks()
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(angle, intensity, label="XRD Intensity", color = graphColor)
        plt.xlabel("Angle (2θ)")
        plt.ylabel("Intensity (counts)")
        plt.title("XRD Measurement")
        plt.legend()
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

        plt.xlabel("Angle (2θ)")
        plt.ylabel("Intensity (Normalized)")
        plt.title("XRD vs Simulated Data")
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def savePeaks(self, file_path = 'Peaks.txt'):
        angles = self.getPeakAngles()
        np.savetxt(fname = file_path, X=angles, delimiter=",")
        self.peakAngles = angles

    def storePlots(self, extraData):
        #Store extra data in the xrdData
        dataExtra = self.loadVestaData(extraData)

        
# Example usage:
if __name__ == "__main__":
    xrd_path = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/14042025/Result.txt'
    vesta_path = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/Material Project/FAPbI3.xy'
    analyzer = XRDAnalyzer(xrd_path, vesta_path)
    analyzer.import_xrd_data()
    analyzer.import_vesta_data()
    analyzer.getPeakAngles()
    analyzer.baselineCorrection()
    analyzer.savePeaks()
    analyzer.find_peaks_and_fwhm()
    analyzer.plotXRD(save_path = "xrd_result_bscTest8.png")
    analyzer.plotVesta("xrd_comparison.png")
    x, y = analyzer.find_peaks()
    print(x, y)
    print(analyzer.xrd_data[0])
    print(analyzer.xrd_data[1])
    #analyzer.savePeaks()