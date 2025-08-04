import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import re
import shutil

class ThermalCycling:
    def __init__(self, folderPath):
        self.folderPath = folderPath

    def sortData(self):
        identifiers = [f"px{i}" for i in range(1, 7)]
        #identifiersSpectra = "Spectrum"
        for filename in os.listdir(self.folderPath):
            file_path = os.path.join(self.folderPath, filename)
            if not os.path.isfile(file_path):
                continue
            # Process only .dat files
            if filename.endswith('.dat'):
                matched = False
                for identifier in identifiers:
                    if identifier in filename:
                        # Create folder if it doesn't exist
                        target_folder = os.path.join(self.folderPath, identifier)
                        os.makedirs(target_folder, exist_ok=True)

                        # Move file
                        shutil.move(file_path, os.path.join(target_folder, filename))
                        matched = True
                        break
                #for id in identifiersSpectra
                # If no identifier is found, leave the file in place
                if not matched:
                    target_folder = os.path.join(self.folderPath, "Spectra")
                    os.makedirs(target_folder, exist_ok=True)
                    shutil.move(file_path, os.path.join(target_folder, filename))
                    continue

class PreTreat_Spectra:
    def __init__(self, folder):
        self.folder = folder
        pattern = re.compile(r"^(?P<base>.+)-(?P<trial>[123])\.dat$")

        for fname in os.listdir(self.folder):
            match = pattern.match(fname)
            if not match:
                continue
            base = match
        

if __name__ == "__main__":
    analyzer = ThermalCycling(folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/ThermalCycling/02072025/R5_1_Neustart_2025-07-02_18-58-14')
    analyzer.sortData()