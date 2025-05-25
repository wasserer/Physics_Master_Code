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
import os
#from colors import *
import pandas as pd
#import importFile.importXRD as file

class XRDAnalyzer:
    def __init__(self, vesta_path=None, importer = None):
        self.xrd_path = None
        self.xrd_folderPath = None
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
        self.multiData = False
        self.grainSizes = None

    def import_xrd_data(self, labelName ="NewData", pathName = ""):
        self.xrd_path = pathName
        self.xrd_folderPath = os.path.dirname(self.xrd_path)
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
            self.multiData = True

        self.label.append(labelName)
        
    def import_vesta_data(self, vestaPath = ""):
        data = np.loadtxt(vestaPath).T
        self.vesta_data = (data[0], data[1])
        return self.vesta_data

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


    def find_peaks_and_FWHM(self, height=0.01, distance=1, log = False): #Peaks angle not always constant, use one of the time
        if self.xrd_data is None:
            raise ValueError("XRD data not loaded.")
        if self.multiData == False:
            #Find Peaks
            intensity = np.array(self.intensities).reshape(-1)
            angle_array = np.array(self.angles).reshape(-1)
            peaksIdx, properties = signal.find_peaks(intensity, height=height, distance=distance, prominence = 0.01)
            angleTemp = self.angles.reshape(-1)
            self.peakAngles = angleTemp[peaksIdx]
            self.peakHeights = properties["peak_heights"]
            self.peakProperties = properties
            #Find FWHM:
            peakWidth = signal.peak_widths(intensity, peaksIdx, rel_height=0.5)
            widths = peakWidth[0]                    # 宽度（以 index 计）
            left_ips = peakWidth[2]                  # 左边界（float index）
            right_ips = peakWidth[3]
            left_angles = np.interp(left_ips, np.arange(len(angle_array)), angle_array)
            right_angles = np.interp(right_ips, np.arange(len(angle_array)), angle_array)
            self.FWHM = right_angles - left_angles
            #Calculate the grain size:
            fwhm_deg = np.array(self.FWHM)
            two_theta = np.array(self.peakAngles)
            theta_rad = np.deg2rad(two_theta / 2)
            beta_rad = np.deg2rad(fwhm_deg)
            grain_sizes_A = (0.9 * 1.5406) / (beta_rad * np.cos(theta_rad))  # Unit: Amstrong, Scherrer equation
            self.grainSizes = grain_sizes_A *0.1  # Unit: nm
            if log == True:
                df = pd.DataFrame({
                    "2θ (deg)": self.peakAngles,
                    "Relative Intensity (%)": self.peakHeights,
                    "FWHM (deg)": self.FWHM,
                    "Grain Size (nm)": self.grainSizes
                })
                fileName = "Peaks_and_FWHM.csv"
                saveName = os.path.join(self.xrd_folderPath, fileName)
                df.to_csv(saveName, index=False)
                
                

    def plotXRD(self, save_path="result_Test.png", graphColor = None, labelName="Label", save = True, findPeaks = False):
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(self.angles[0], self.intensities[0], label=labelName, color = graphColor)
        if findPeaks == True:
            for i in self.peakAngles:
                plt.vlines(i, 0, -0.01, color = "black")
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