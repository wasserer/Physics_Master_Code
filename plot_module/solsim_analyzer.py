#Module to 
import numpy as np
import matplotlib.pyplot as plt
import os
from colors import color
import pandas as pd

class solarSimulator:
    def __init__(self, filePath=None, folderPath = None, CSV = None):
        self.filePath = filePath
        self.folderPath = folderPath
        self.CSV = CSV
        self.data = None
        self.currents = None
        self.CDC = 1000/0.0966
        self.voltages = None
        self.powers = None
        self.times = None
        #self.counter = 0
        self.dat_files = None
        self.labels = []
        #Important Points in solSim:
        self.Voc = None
        self.Isc = None
        self.I_MPP = None
        self.V_MPP = None
        self.FF = None
        self.PCE = None
        self.intensity = 100*36.5/47.98 #Unit: mW/cm^2

    def calcIsc(self, voltage, current):
        idx = np.argmin(np.abs(voltage))
        Isc = current[idx]
        return Isc
    
    def calcVoc(self, voltage, current):
        idx = np.argmin(np.abs(current))
        Voc = voltage[idx]
        return Voc

    def calcMPP(self, current, voltage):
        Isc = self.calcIsc(voltage = voltage, current = current)
        Voc = self.calcVoc(voltage=voltage, current=current)
        mask = (current >=0 ) & (current <= Isc)
        V = voltage[mask]
        I = current[mask]
        P = V * I
        idx = np.argmax(np.abs(P))
        I_MPP = I[idx] #Unit: Current density mA/cm^2
        V_MPP = V[idx] #Unit: V
        P_MPP = P[idx] #mW/cm^2
        FF = np.abs((I_MPP*V_MPP)/(Isc*Voc))
        PCE = np.abs(P_MPP/self.intensity)*100
        return I_MPP, V_MPP, FF, PCE

    def loadFileData(self): #Automatically detect the file type!
        files = pd.read_csv(self.filePath, skiprows=range(0, 21), header=None, names=['voltage', 'current', 'power', 'time'], sep='\s+')
        self.data = np.asarray(files)
        self.data = self.data.T
        self.voltages = self.data[0]
        self.currents = self.data[1]* self.CDC
        self.powers = self.data[2]
        self.times = self.data[3]
        self.Isc = self.calcIsc(voltage = self.voltages, current = self.currents)
        self.Voc = self.calcVoc(voltage=self.voltages, current = self.currents)
        self.I_MPP, self.V_MPP, self.FF, self.PCE = self.calcMPP(voltage=self.voltages, current = self.currents)
        self.labels = self.filePath

    def loadFolderData(self):
        #if self.folderPath is None:
        #    self.loadFileData()
        self.dat_files = [f for f in os.listdir(self.folderPath) if f.endswith('.dat')]
        self.Isc, self.Voc, self.I_MPP, self.V_MPP, self.FF, self.PCE = [[] for _ in range(6)]
        for name in self.dat_files:
            file = os.path.join(self.folderPath, name)
            labelName = name
            files = pd.read_csv(file, skiprows=range(0, 21), header=None, names=['voltage', 'current', 'power', 'time'], sep='\s+')
            if self.currents is None:
                self.data = np.asarray(files)
                self.data = self.data.T
                self.voltages = self.data[0].reshape(1, -1)
                self.currents = self.data[1].reshape(1, -1) * self.CDC # Convert the current to current density
                self.powers = self.data[2].reshape(1, -1)
                self.times = self.data[3].reshape(1, -1)
                self.labels.append (labelName)
                #Calculations:                
                Isc = self.calcIsc(voltage = self.data[0], current=self.data[1])
                Voc = self.calcVoc(voltage = self.data[0], current=self.data[1])                   
                I_MPP, V_MPP, FF, PCE = self.calcMPP(voltage = self.data[0], current=self.data[1]*self.CDC)
            else:
                data = np.asarray(files)
                data = data.T
                
                self.data = np.vstack((self.data, data))
                self.voltages = np.vstack((self.voltages, data[0]))
                self.currents = np.vstack((self.currents, data[1]*self.CDC))
                self.powers = np.vstack((self.powers, data[2]))
                self.times = np.vstack((self.times, data[3]))
                self.labels.append(labelName)
                #Calculation for cell data:
                Isc = self.calcIsc(voltage = data[0], current=data[1]*self.CDC)
                Voc = self.calcVoc(voltage = data[0], current=data[1]*self.CDC)
                I_MPP, V_MPP, FF, PCE = self.calcMPP(voltage = data[0], current=data[1]*self.CDC)
            self.Isc.append(Isc)
            self.Voc.append(Voc)
            self.I_MPP.append(I_MPP)
            self.V_MPP.append(V_MPP)
            self.FF.append(FF)
            self.PCE.append(PCE)

    def logData(self):
        fileName = "result_log.csv"
        dir = self.folderPath
        target = os.path.join(dir, fileName)
        data = {
            'Cell Label':  self.labels,
            'Isc [mA/cm2]':    self.Isc,
            'Voc [V]':    self.Voc,
            'I_MPP':  self.I_MPP,
            'V_MPP':  self.V_MPP,
            'FF':     self.FF,
            'PCE':    self.PCE
        }
        df = pd.DataFrame(data)
        df.to_csv(target, index=False)

    def IVCurve(self, saveName = "saveFig.png", colorMode = "red"):
        voltage = self.data[0]
        current = self.data[1]
        power = self.data[2]
        time = self.data[3]
        colorUsed = colorMode
        plt.figure(figsize=(7, 5), dpi = 300)
        plt.plot(voltage, current*1000, color = colorMode)
        plt.ylabel("Current [mA]")
        plt.xlabel("Voltage [V]")
        plt.grid()
        plt.savefig(saveName)
        plt.close()

    def IVMultiPlot(self, colorMode = None, saveName = "multiIV.png"):
        plt.figure(figsize=(9, 6), dpi = 300)
        for i in range(0, self.voltages.shape[0]):
            voltage = self.voltages[i]
            #print("The type of it", type(voltage)) 
            current = self.currents[i]
            #current = current.tolist()
            #voltage = voltage.tolist()
            lab = self.labels[i]
            plt.plot(voltage, current, label=lab) #mA? 
        #self.counter = 0
        plt.ylabel("Current Density [mA/cm2]")#Too big!
        plt.xlabel("Voltage [V]")
        #plt.ylim(-10, 20)
        #plt.xlim(-0.121, -0.8)
        plt.legend()
        plt.title("IV-Curve of the Samples")
        plt.grid()
        plt.tight_layout()
        plt.savefig(saveName)
        #plt.show()
        plt.close()

    def HistoPlot(self):
        pass

    def boxPlot(self):
        pass

if __name__ == "__main__":
    from colors import color
    figColor = color.matlab()
    file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/IPARPM5_1_px5_Light_forward_0.dat'
    folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/'
    analyzer = solarSimulator(filePath = None, folderPath=folder)
    #analyzer.loadFileData()
    analyzer.loadFolderData()
    #print(analyzer.current.shape)
    #print(analyzer.voltages)
    #print("Cell Label:", analyzer.labels)
    print(analyzer.voltages.shape)
    print(type(analyzer.voltages))
    print("Isc:", analyzer.Isc, "Voc:", analyzer.Voc, "VMPP:", analyzer.V_MPP, "IMPP:",analyzer.I_MPP, "FF:", analyzer.FF, "PCE:", analyzer.PCE)
    analyzer.logData()
    analyzer.IVMultiPlot(saveName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/BestCells/IV_Multi3.png')
    #analyzer.IVCurve(saveName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/BestCells/IV_Single.png')