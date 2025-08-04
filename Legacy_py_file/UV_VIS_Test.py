import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folderPath='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/UV-VIS/0206/CSV/20IPA.csv'

df = pd.read_csv(folderPath, decimal=",", sep=";")
print(df.head())
data = df.values.T
print(data.shape)
abs = data[1].reshape(1, -1)
output = np.vstack((abs, data[1]))

print(output.shape)
#Try to calculate the tau plot
hv =1240 / data[0]
print("hv", hv.shape)
hv2 = hv.reshape(1, -1)
temp2 = (abs*hv2)**0.5
print("Temp2", temp2)

