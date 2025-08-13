#Module to 
import numpy as np
import matplotlib.pyplot as plt
import os
#from colors import color
import pandas as pd
import seaborn as sns
import re
from datetime import datetime

class solarSimulator:
    ''''
    This is the class for the solar simulator data analysis.
    It will load the data from the file or folder, and calculate the important parameters.
    The data is in the form of a .dat file, which contains the voltage, current, power, and time.
    The data is in the form of a 2D array, where the first column is the voltage, the second column is the current, the third column is the power, and the fourth column is the time.
    The data is in the form of a .dat file, which contains the voltage, current, power, and time.
    The data is in the form of a 2D array, where the first column is the voltage, the second column is the current, the third column is the power, and the fourth column is the time.
    The class will also calculate the important parameters such as Isc, Voc, I_MPP, V_MPP, FF, and PCE.
    The class will also plot the IV curve and the multi IV curve.'''
    def __init__(self, filePath=None, folderPath = None, CSV = None):
        self.filePath = filePath
        self.folderPath = folderPath
        #self.saveLocation = os.path.join(folderPath, "Result")
        self.CSV = CSV
        #The raw data from the measurement
        self.data = None
        self.currents = None
        self.CDC = 1000/0.079 #Calculation where? 0.14, length: 0.69, old: 0.145
        self.voltages = None
        self.powers = None
        self.times = None
        #self.counter = 0
        self.dat_files = None
        self.labels = []
        #Important Results in solSim:
        self.Voc = None
        self.Isc = None
        self.I_MPP = None
        self.V_MPP = None
        self.FF = None
        self.PCE = None
        self.intensity = 100*37.0/47.98 #Unit: mW/cm^2
        self.cycleNum = []

    def calcIsc(self, voltage, current):
        try:
            idx = np.argmin(np.abs(voltage))
            Isc = current[idx]
            return Isc
        except Exception as e:
            print(f"Error:", e)
            return 0
    
    def calcVoc(self, voltage, current):
        try:
            idx = np.argmin(np.abs(current))
            Voc = voltage[idx]
            return Voc
        except Exception as e:
            print(f"Error:", e)
            return 0

    def calcMPP(self, current, voltage):
        try:
            Isc = self.calcIsc(voltage = voltage, current = current)
            Voc = self.calcVoc(voltage=voltage, current=current)
            #Test
            #("Isc:",self.Isc)
            #print("Voc", self.Voc)
            mask = (current >=0 ) & (current <= Isc)
            V = voltage[mask]
            I = current[mask]
            P = V * I
            print(P)
            idx = np.argmax(np.abs(P))
            I_MPP = I[idx] #Unit: Current density mA/cm^2
            V_MPP = V[idx] #Unit: V
            P_MPP = P[idx] #mW/cm^2
            FF = np.abs((I_MPP*V_MPP)/(Isc*Voc))
            PCE = np.abs(P_MPP/self.intensity)*100
            return I_MPP, V_MPP, FF, PCE
        except Exception as e:
            print(f"Error:", e)
            return 0, 0, 0, 0

    def loadFileData(self):
        
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

    def loadFolderData_Cycling(self):
        '''This function is used to load the data from the folder, and calculate the important parameters.
        The data is in the form of a .dat file, which contains the voltage, current, power, and time.
        The data is in the form of a 2D array, where the first column is the voltage, the second column is the current, the third column is the power, and the fourth column is the time.
        The class will also calculate the important parameters such as Isc, Voc, I_MPP, V_MPP, FF, and PCE.
        The class will also plot the IV curve and the multi IV curve.'''
        cycleCounter = 1
        self.temperature = []
        self.timestamp = []
        self.dat_files = [f for f in os.listdir(self.folderPath) if f.endswith('.dat')]
        self.Isc, self.Voc, self.I_MPP, self.V_MPP, self.FF, self.PCE = [[] for _ in range(6)]
        for name in self.dat_files:
            file = os.path.join(self.folderPath, name)
            mainName, ext = name, ext = os.path.splitext(name) #Newly added
            labelName = name
            #Read the time from the file:
            with open(file, 'r') as data:
                firstLine = data.readline().strip()
            #print("First line:", firstLine)
            dateTimeStr = firstLine.split(":", 1)[1].strip()
            # Extract datetime using known format (modify format if needed)
            timestamp = datetime.strptime(dateTimeStr, "%Y-%m-%d %H:%M:%S")
            #print("Timestamp:", timestamp)
            self.timestamp.append(timestamp)
      
            #Read the temperature from the file:                    
            with open(file, 'r') as data:
                lines = data.readlines()
            if len(lines) >= 6:
                line6 = lines[5]  # 0-based index
                print("Line 6:", line6.strip())
                # Extract the first floating-point number in the line
                match = re.search(r"[-+]?\d*\.\d+|\d+", line6)
                if match:
                    temperature = float(match.group())
                    self.temperature.append(temperature)
                else:
                    print("No number found in line 6.")
            else:
                print("File has fewer than 6 lines.")
            '''
            cycleLine = lines[3]
            match = re.search(r'\d+', cycleLine)
            if match:
                cycle_number = int(match.group())
                print(cycle_number)
            else:
                print("No integer found.")
            self.cycleNum.append(cycle_number)
            '''
            files = pd.read_csv(file, skiprows=range(0, 20), header=None, names=['voltage', 'current', 'power', 'time'], sep=';')
            if self.currents is None:
                self.data = np.asarray(files)
                self.data = self.data.T
                self.voltages = self.data[0].reshape(1, -1)
                self.currents = self.data[1].reshape(1, -1) * self.CDC # Convert the current to current density
                self.powers = self.data[2].reshape(1, -1)
                self.times = self.data[3].reshape(1, -1)
                self.labels.append(labelName)
                #Calculations:                
                Isc = self.calcIsc(voltage = self.data[0], current=self.data[1])
                Voc = self.calcVoc(voltage = self.data[0], current=self.data[1])                   
                I_MPP, V_MPP, FF, PCE = self.calcMPP(voltage = self.data[0], current=self.data[1]*self.CDC)
            else:
                data = np.asarray(files)
                data = data.T
                print(name)###
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

        #Sort Data
        combined = list(zip(self.timestamp, self.Isc, self.voltages, self.currents, self.data, self.Voc, self.PCE, self.I_MPP, self.V_MPP, self.FF, self.labels, self.temperature))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        self.timestamp, self.Isc, self.voltages, self.currents, self.data, self.Voc, self.PCE, self.I_MPP, self.V_MPP, self.FF, self.labels, self.temperature = zip(*sorted_combined)
        #Normalizing all the self.PCE values to its highest value, because the solar simulator is not calibrated.
        maxPCE = max(self.PCE)
        self.PCE = [pce / maxPCE * 100 for pce in self.PCE]
        start_time = self.timestamp[0]
        self.timestampAbsS = [(t - start_time).total_seconds() for t in self.timestamp]
        self.timestampAbsM = [s / 60 for s in self.timestampAbsS]
        self.timestampAbsH = [s / 3600 for s in self.timestampAbsS]
        #Find out which cycle it was on: THIS IS THE OLD METHOD, THE NEW WAY IS JUST TO LOOK UP IN THE FILE
        for i in range(len(self.timestampAbsS)):
            self.cycleNum.append((self.timestampAbsS[i]//5520+1))

        #Now save the data in a .csv file called f"result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv", and save the data in a folder called "Result", if the folder "result" does not exist, create it.
        result_folder = os.path.join(self.folderPath, "Result")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        result_file = os.path.join(result_folder, f"result_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        data = {
            'Timestamp[s]': self.timestampAbsS,
            'Labels': self.labels,
            'Temperature [C]': self.temperature,
            'Cycle Number': self.cycleNum,
            'Isc[mA/cm2]': self.Isc,
            'Voc [V]': self.Voc,
            'I_MPP [mA/cm2]': self.I_MPP,
            'V_MPP [V]': self.V_MPP,
            'FF': self.FF,
            'PCE [%]': self.PCE,
        }
        df = pd.DataFrame(data)
        df.to_csv(result_file, index=False)

    def loadFolderData(self):
        '''This is a function to do classical solar simulator data analysis, which is performed in the Solar simulator outside the glove box. The data is in the form of a .dat file, which contains the voltage, current, power, and time.
        The data is in the form of a 2D array, where the first column is the voltage, the second column is the current, the third column is the power, and the fourth column is the time.
        The class will also calculate the important parameters such as Isc, Voc, I_MPP, V_MPP, FF, and PCE.'''
        self.dat_files = [f for f in os.listdir(self.folderPath) if f.endswith('.dat')]
        self.Isc, self.Voc, self.I_MPP, self.V_MPP, self.FF, self.PCE = [[] for _ in range(6)]
        for name in self.dat_files:
            file = os.path.join(self.folderPath, name)
            mainName, ext = name, ext = os.path.splitext(name) #Newly added
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
                print(name)###
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
        '''This function is used to plot the IV curve of the solar cell, and save the figure in the folder.
        The data is in the form of a .dat file, which contains the voltage, current, power, and time.
        The data is in the form of a 2D array, where the first column is the voltage, the second column is the current, the third column is the power, and the fourth'''
        saveLocation = os.path.join(self.filePath, saveName)
        voltage = self.data[0]
        current = self.data[1]
        power = self.data[2]
        time = self.data[3]
        colorUsed = colorMode
        plt.figure(figsize=(7, 5), dpi = 300)
        plt.plot(voltage, current*1000, color = colorMode)
        plt.ylabel("Current [mA]")
        plt.ylim(0, )
        plt.xlabel("Voltage [V]")
        plt.grid()
        plt.savefig(saveLocation)
        plt.close()

    def IVMultiPlot(self, colorMode = None, saveName = "multiIV.png", labelMode = True):
        '''This function is used to plot the IV curve of the solar cell, and save the figure in the folder.
        The data is in the form of a .dat file, which contains the voltage, current, power, and time.
        The data is in the form of a 2D array, where the first column is the voltage, the second column is the current, the third column is the power, and the fourth'''
        saveLocation = os.path.join(self.folderPath, saveName)
        plt.figure(figsize=(9, 6), dpi = 300)
        for i in range(0, self.voltages.shape[0]):
            voltage = self.voltages[i]
            #print("The type of it", type(voltage)) 
            current = self.currents[i]
            #current = current.tolist()
            #voltage = voltage.tolist()
            lab = self.labels[i]
            if colorMode == None:
                plt.plot(voltage, current, label=lab) #mA?
            else:
                figColor = colorMode[i] 
                plt.plot(voltage, current, label=lab, color = figColor)
        #self.counter = 0
        plt.ylabel("Current Density [mA/cm2]")#Too big!
        plt.xlabel("Voltage [V]")
        plt.ylim(0, 20)
        #plt.ylim(-10, 20)
        plt.xlim(-1,)
        if labelMode is True:
            plt.legend()
        plt.title("IV-Curve of the Samples")
        plt.grid()
        plt.tight_layout()
        plt.savefig(saveLocation)
        #plt.show()
        plt.close()

    def histoPlot(self, saveName = "histoPlot.png", type = "PCE[%]", color = "blue"): #Not finished
        target = os.path.join(self.folderPath, saveName)
        plt.figure(figsize=(9, 6), dpi=300)
        plt.hist(self.PCE, bins=20, edgecolor='black', color = color)
        plt.ylabel("Number")
        plt.xlabel(type)
        plt.savefig(target)
        #plt.show()


    def boxPlot(self, data1=None, data2=None, names=["Low concentration of MACl", "High concentration of MACl"]):
        batch_1 = data1
        batch_2 = data2
        # Create DataFrame with dynamic batch names
        df = pd.DataFrame({
            'PCE': np.concatenate([batch_1, batch_2]),
            'Batch': [names[0]] * len(batch_1) + [names[1]] * len(batch_2)
        })

        # Save location
        savePlace = os.path.join(self.folderPath, "result.png")

        # Plot style
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6), dpi=300)

        # Boxplot and scatter overlay
        sns.boxplot(x='Batch', y='PCE', data=df, palette=["#0072BD", "#FF0A2D"])
        sns.stripplot(x='Batch', y='PCE', data=df, color='black', size=4, jitter=True, alpha=0.8)

        # Labels and title
        #plt.title("PCE Distribution by Batch", fontsize=14)
        plt.ylabel("PCE (%)", fontsize=12)
        plt.xlabel("Batch of cells")
        plt.title("PCE Distribution by Batch", fontsize=14)
        plt.tight_layout()
        plt.savefig(savePlace)
        #plt.show()

if __name__ == "__main__":
    from colors import color
    figColor = color.matlab()
    file = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/04_06_2025_Cs_Test/BestCells/test/25S_5_px3_Light_forward_0.dat'
    folder = '/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/BestCells'
    analyzer = solarSimulator(filePath = None, folderPath=folder)
    analyzer.loadFileData()
    #analyzer.loadFolderData()
    #print(analyzer.current.shape)
    #print(analyzer.voltages)
    #print("Cell Label:", analyzer.labels)
    analyzer.IVCurve(saveName="")
    print(analyzer.voltages.shape)
    print(type(analyzer.voltages))
    print("Isc:", analyzer.Isc, "Voc:", analyzer.Voc, "VMPP:", analyzer.V_MPP, "IMPP:",analyzer.I_MPP, "FF:", analyzer.FF, "PCE:", analyzer.PCE)
    analyzer.logData()
    #analyzer.IVMultiPlot(saveName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/BestCells/IV_Multi3.png')
    #analyzer.histoPlot()
    #analyzer.IVCurve(saveName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/SolSim/21_05_2025_IPATest/BestCells/IV_Single.png')