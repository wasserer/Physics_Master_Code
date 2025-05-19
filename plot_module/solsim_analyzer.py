#Module to 
import numpy as np
import matplotlib.pyplot as plt
import os
from colors import color

class solarSimulator:
    def __init__(self, filePath=None, folderPath = None, CSV = None):
        self.filePath = filePath
        self.folderPath = folderPath
        self.CSV = CSV
        self.data = None
        self.dataT = None
        self.dat_files = None
        self.counter = 0

    def loadFileData(self): #Automatically detect the file type!
        files = pd.read_csv(self.filePath, skiprows=range(0, 21), header=None, names=['voltage', 'current', 'power', 'time'], sep='\s+')
        data = np.asarray(files)
        data = data.T

    def loadFolderData(self):
        self.dat_files = [f for f in os.listdir(self.folderPath) if f.endswith('.dat')]

    def IVCurve(self, saveName = "saveFig.png", colorMode = color.matlab()):
        voltage = self.data[0]
        current = -self.data[1]
        power = self.data[2]
        time = self.data[3]
        colorUsed = colorMode
        plt.figure(figsize=(7, 5), dpi = 300)
        plt.plot(voltage, current*1000, color = colorUsed)
        plt.ylabel("Current [mA]")
        plt.xlabel("Voltage [V]")
        plt.grid()
        plt.savefig(saveName)
        plt.close()

    def multiIVCurve(self, colorMode = color.matlab(multiData=True)):
        plt.figure(figsize=(9, 6), dpi = 300)
        for file_name in self.dat_files:
            full_path = os.path.join(self.folderPath, file_name)
            #Load the single data
            voltage = self.dataT[0]
            current = -self.dataT[1]
            power = self.dataT[2]
            time = self.dataT[3]
            label = os.path.splitext(file_name)[0]
            plt.plot(voltage, current * 1000, color = colorMode[self.counter], label=label)
            self.counter += 1
        self.counter = 0
        plt.ylabel("Current [mA]")
        plt.xlabel("Voltage [V]")
        plt.legend()
        plt.title("IV-Curve of the Samples")
        plt.grid()
        plt.tight_layout()
        #plt.savefig(os.path.join(filePath, "IV_combined_plot.png"))
        plt.show()
        plt.close()

    def HistoPlot(self):
        pass

    def boxPlot(self):
        pass
