#The module to import different types of data, and treat them into a standard form afterwards
#Author: Ruodong Yang
import numpy as np

class importFile:
    def __init__(self, fileName, parameter=None):
        self.fileName = fileName
        self.parameter = parameter

    def importXRD(self):
#        if self.fileName is None:
#            raise ValueError("No XRD file is given!")
        with open(self.fileName, "r") as file:
            lines = file.readlines()
        data_start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("[Data]"):
                data_start_index = i + 1
                break
        data_lines = lines[data_start_index:]
        data = np.genfromtxt(data_lines, delimiter=",", dtype=float)
        xrd_data = (data[:, 0], data[:, 1])
        return xrd_data

    def importVesta(self):
        if self.fileName is None:
            raise ValueError("No Vesta file is given!")
        data = np.loadtxt(self.fileName).T
        vesta_data = (data[0], data[1])
        return vesta_data
