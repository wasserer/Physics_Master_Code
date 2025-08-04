import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
"""
Th
"""
class UV_VIS_Analyzer:
    def __init__(self, folderPath, filePath = None, GermanMode = True, tauc = True):
        self.folderPath = folderPath
        self.dark = None
        self.light = None
        self.data = None
        self.waveLength = None
        self.absorption = None
        self.tauc = None
        self.eVolt = None
        self.tauc_x = None
        self.tauc_slope_data = None
        self.tauc_intercept = None
        self.label = []
        self.y_intercept = None
        self.slope = None
        self.x_at_max_slope = []
        for fileName in os.listdir(self.folderPath):
            if fileName.endswith(".csv"):
                filePath = os.path.join(self.folderPath, fileName)
                if GermanMode == True:
                    df = pd.read_csv(filePath, decimal=",", sep=";")
                else:
                    df = pd.read_csv(filePath, decimal=".", sep=",")
                data = df.values.T
                key = fileName.replace(".csv", "")
                match key:
                    case k if k.startswith("Dark"):
                        self.dark = data[1]
                    case k if k.startswith("Light"):
                        self.light = data[1]
                    case _:
                        self.label.append(key)
                        if self.waveLength is None:
                            self.waveLength = data[0]
                            self.absorption = data[1].reshape(1, -1)
                            #print(self.waveLength)
                            if tauc == True:                                
                                self.eVolt = (1240 / self.waveLength) # Conversion x axis
                                self.tauc = (self.absorption * self.eVolt.reshape(1, -1))**0.5# 
                            else:
                                pass
                        else:
                            #print(key)
                            self.absorption = np.vstack((self.absorption, data[1]))
                            if tauc == True:
                                taucT = (data[1].reshape(1, -1)*self.eVolt.reshape(1, -1))**0.5 # Could use: \alpha = 2.303 * A / d , d = 600 nm for PIN-Perovskite
                                self.tauc = np.vstack((self.tauc, taucT))#
                            else:
                                pass

    def UV_multiPlot(self, saveName = "Result.png", figColor = None):
        savePath = os.path.join(self.folderPath, saveName)
        plt.figure(figsize = (9, 7), dpi = 300)
        for i in range(len(self.label)):
            plt.plot(self.waveLength, self.absorption[i], color = figColor[i], label = self.label[i])
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Reflection [%]") # or absorption etc.
        plt.legend()
        plt.grid()
        plt.savefig(savePath)

    def fit(self, start=1.4, end=1.8): #Rewrite!!!! Make it auto detectable!!!!
        for i in range(0, len(self.label)):
            #Import Data:
            y = self.tauc[i] #The y value of the tauc plot
            x = self.eVolt  #X value of the tauc plot(Convert to electron volt)
            #Classic Method:
        #New Method:
            searchRange = (x>=start)&(x<end)
            x_preSelected = x[searchRange].flatten()
            y_preSelected = y[searchRange].flatten()
            #print("x_pre",x_preSelected)
            dy = np.diff(y_preSelected)
            dx = np.diff(x_preSelected)
            slope = dy / dx  # 相当于 dy/dx
            #print("Slope:",slope)
            max_index = np.argmax(slope)
            #print("type max", type(max_index))
            x_at_max_slope = (x_preSelected[max_index] + x_preSelected[max_index + 1]) / 2
            #self.x_at_max_slope.append(x_at_max_slope)
            # Step 2: 找最大斜率的位置（注意 slope 长度是 n-1）
            max_index = np.argmax(x)
            #print("max:",x_at_max_slope)
            realMask = (x>=(x_at_max_slope-0.02))&(x<=(x_at_max_slope+0.02))
            x_selected = x[realMask].reshape(1, -1).T
            y_selected = y[realMask].reshape(1, -1).T
            #print("x_selected", x_selected)
            model = LinearRegression()
            print(self.label[i], "x", x_selected, "y", y_selected)
            model.fit(x_selected, y_selected)
            b = model.intercept_[0]
            slope = model.coef_[0]
            #print("b:", b, "slope:", slope, self.label[i]) #
            intercept = model.intercept_
            x_intercept = - intercept/ slope #Intercept with x axis

            
            #print(x_intercept)
            #Store Data:
            if self.tauc_x is None:
                self.tauc_x = x_selected
                tauc_slope_data = self.tauc_x*slope + b
                self.tauc_slope_data = tauc_slope_data.reshape(1, -1)
                self.slope = [slope[0]]
                #print("Slope first:", self.slope, type(self.slope))#Debug
                self.y_intercept = [b]
            else:
                tauc_slope_data = self.tauc_x*slope + b
                self.tauc_slope_data = np.vstack((self.tauc_slope_data, tauc_slope_data.reshape(1, -1)))
                #print((slope[0], type(slope[0])), self.label[i], "Rest before line")
                self.slope.append(slope[0])
                self.y_intercept.append(b)
                #self.tauc_intercept = self.tauc_intercept.append(intercept)
                
        
    def tau_Plot(self, saveName = "tau_result1.png", figColor = None, fit = None):
        savePath = os.path.join(self.folderPath, saveName)
        plt.figure(figsize = (9, 7), dpi = 300)
        for i in range(len(self.label)):
            plt.plot(self.eVolt, self.tauc[i], color = figColor[i], label = self.label[i])
            if fit is True:
                print("Fitting right now!")
                plt.plot(self.eVolt, (self.eVolt*self.slope[i] + self.y_intercept[i]), color = figColor[i]) #Change the slope as a equation based on self.tauc_x, and make it - - - like
        plt.xlim(1, 2.0)
        plt.ylim(-0.25, 2)
        plt.xlabel("Wavelength")
        plt.ylabel("Intensity")
        plt.legend()
        plt.grid()
        plt.savefig(savePath)

    def logData(self, saveName="fit_results.csv"):
        savePath = os.path.join(self.folderPath, saveName)
        
        # 构建字典数据
        result_dict = {
            "Label": self.label,
            "Slope": self.slope,
            "Y_Intercept": self.y_intercept,
            "X_at_max_slope": self.x_at_max_slope,
            "X_intercept (Eg)": [-b / a for a, b in zip(self.slope, self.y_intercept)]
        }

        # 转换为 DataFrame
        df_log = pd.DataFrame(result_dict)

        # 保存为 CSV
        df_log.to_csv(savePath, index=False)
        print(f"Fit data saved to: {savePath}")

if __name__ == "__main__":
    figcolor = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
    df = UV_VIS_Analyzer(folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/UV-VIS/0506/Compare_Cs', GermanMode=True)
    #print("Light",df.light, type(df.light))
    #print("Dark:", df.dark)
    #print(df.absorption.shape)
    #print(df.label)
    df.UV_multiPlot(figColor=figcolor)
    df.fit()
    df.tau_Plot(figColor=figcolor, fit=True)
    df.logData()
    