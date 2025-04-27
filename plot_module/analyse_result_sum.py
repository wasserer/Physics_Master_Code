#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 10:24:42 2024

@author: ruodongyang
"""

#This is a test file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
#import seaborn as sns

filePath = r'/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/10_04_2025_1Step_100C/result_sum.csv'
savePath = r'/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/Summery_03042025'
saveName = "PCE of 10mg"
title = '10.04.2025, 5mg PTAA, one step Nr.2'
fullPath = os.path.join(savePath, saveName)


df = pd.read_csv(filePath)
def ExtractPCE(filePath):
    x = df.iloc[:, 0]
    y = df.iloc[:, 6]
    #y = df.iloc[:, 3]
    length = len(y)
    return x, y
    
def ExtractFF(filePath):
    x = df.iloc[:, 0]
    y = df.iloc[:, 5]
    #y = df.iloc[:, 3]
    length = len(y)
    return x, y
#ave = []
#for i in range(0, len(y)):
#    if y[i] > 2:
#        ave.append(y[i])

#ave = sum(ave)/len(ave)


#ax, fig = plt.subplots(figsize = (9, 7),dpi = 300)

#for i in range (1, length):
#    if y[i] > 1:
#        plt.scatter(x[i], y[i], color = "green")
        
#plt.xlabel("number")
#plt.ylabel("efficiency")

def HistoPlot(filePath, Type='PCE'):
    fsize = 24
    ax, fig = plt.subplots(figsize = (9, 7),dpi = 300)
    match Type:
        case "PCE":
            x, y = ExtractPCE(filePath)
            plt.hist(y, bins=100, edgecolor='black')
            plt.xlim(0.05, 10)
            #plt.ylim(0, 20)
            plt.xticks(fontsize=0.8 * fsize)
            plt.yticks(fontsize=0.8 * fsize)
            plt.xlabel('PCE[%]', fontsize = fsize)
            plt.ylabel('Number of pins', fontsize=fsize)
            plt.title(title, fontsize=15)
            plt.savefig(fullPath)
        case "FF":
            plt.hist(df.iloc[:, 5], bins=30, edgecolor='black')
            #plt.xlim(0, 100)
            #plt.ylim(0, 20)
            plt.xticks(fontsize=0.8 * fsize)
            plt.yticks(fontsize=0.8 * fsize)
            plt.xlabel('FF[%]', fontsize = fsize)
            plt.ylabel('Number of pins', fontsize=fsize)
            plt.title(title, fontsize=15)
            plt.savefig(fullPath)
        case "Voc":
            plt.hist(df.iloc[:, 2], bins=20, edgecolor='black')
            plt.xlim(0.05, )
            #plt.ylim(0, 20)
            plt.xticks(fontsize=0.8 * fsize)
            plt.yticks(fontsize=0.8 * fsize)
            plt.xlabel('Voc[V]', fontsize = fsize)
            plt.ylabel('Number of pins', fontsize=fsize)
            plt.title(title, fontsize=15)
            plt.savefig(fullPath)
        case "Isc":
            plt.hist(df.iloc[:, 1], bins=20, edgecolor='black')
            plt.xlim(0.05, )
            #plt.ylim(0, 20)
            plt.xticks(fontsize=0.8 * fsize)
            plt.yticks(fontsize=0.8 * fsize)
            plt.xlabel('Isc[mA]', fontsize = fsize)
            plt.ylabel('Number of pins', fontsize=fsize)
            plt.title(title, fontsize=15)
            plt.savefig(fullPath)

if __name__ == "__main__":
    HistoPlot(filePath, Type="PCE")