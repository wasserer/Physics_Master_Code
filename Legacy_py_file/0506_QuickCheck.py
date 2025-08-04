from plot_module.xrd_analyzer import XRDAnalyzer
import matplotlib.pyplot as plt


X20s = XRDAnalyzer()
X20s.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/Results/20S_IPA.txt')
X25s = XRDAnalyzer()
X25s.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/Results/25S_IPA.txt')
XCBs = XRDAnalyzer()
XCBs.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/Results/CB.txt')
XR5s = XRDAnalyzer()
XR5s.import_xrd_data(pathName='/Users/ruodongyang/Documents/Resilio_Sync/TUM Master Physik/Pervoskite Space(Master)/Data/XRD/XRD_05062025/Results/R5.txt')
X20s.intensities = X20s.intensities / 0.027347053848322526
X25s.intensities = X25s.intensities / 0.19322344322344323# 0.19322344322344323
XCBs.intensities = XCBs.intensities / 0.7468643101482326
XR5s.intensities = XR5s.intensities / 0.1913335367714373

plt.figure(figsize=(4, 3))
#plt.plot(10, 20)
plt.plot(X20s.angles, X20s.intensities, color = "r")
plt.plot(X25s.angles, X25s.intensities)
plt.plot(XCBs.angles, XCBs.intensities)
plt.plot(XR5s.angles, XR5s.intensities)
plt.xlabel("Angle")
plt.ylabel("Intensity")
plt.grid()
plt.show()