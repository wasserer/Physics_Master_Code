#Module to 
import numpy as np
import matplotlib.pyplot as plt

class solarSimulator:
    def __init__(self, filePath=None, folderPath = None, CSV = None):
        self.filePath = filePath
        self.folderPath = folderPath
        self.CSV = CSV
        self.data = None

    def loadFileData(self): #Automatically detect the file type!
        files = pd.read_csv(self.filePath, skiprows=range(0, 21), header=None, names=['voltage', 'current', 'power', 'time'], sep='\s+')
        data = np.asarray(files)
        data = data.T

    def IVCurve(self, saveName = "saveFig.png"):
        voltage = self.data[0]
        current = -self.data[1]
        power = self.data[2]
        time = self.data[3]
        plt.figure(figsize=(7, 5), dpi = 300)
        plt.plot(voltage, current*1000)
        plt.ylabel("Current [mA]")
        plt.xlabel("Voltage [V]")
        plt.grid()
        plt.savefig(saveName)
        plt.close()

    def multiIVCurve(self):
        pass

    def HistoPlot(self):
        pass

    def boxPlot(self):
        pass
