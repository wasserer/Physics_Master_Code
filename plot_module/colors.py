#Author: Ruodong Yang
#A tiny database for colors
import numpy as np
import matplotlib.pyplot as plt

class color:
    def __init__(self, multiData):
        self.multiData = multiData

    def cold(multiData = False):
        if multiData == True:
            return ("#13DDF0", "#1397F0", "#638CF0", "#1352F0", "#1B13F0", "#6313F0")
        else:
            return "#1352F0"
        
    def warm(multiData = False):
        if multiData == True:
            return ["#FF0A2D", "#E82009", "#FE5217", "#E86109", "#FF8F0A"]
        else:
            return "#FE5217"
        
    
    def red(multiData = False):
        if multiData == True:
            return ("#F25C69", "#F23D3D", "#BF0F0F", "#8C0808", "#590202")
        else:
            return "#BF0F0F"
        
    def yellow(multiData = False):
        if multiData == True:
            return ["#F2DA5E", "#F2BC1B", "#8C6E14","#F2A516", "#594114"]
        else:
            return "#F2BC1B"
        
    def matlab(multiData = False):
        if multiData == True:
            return ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
        
if __name__ == "__main__":
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = x * 2.5
    y4 = x * 2
    y5 = x * 1.1
    y = [y1, y2, y3, y4, y5]
    labels = ['y1', 'y2', 'y3', 'y4', 'y5']
    plt.figure(figsize=(9, 6))
    color1 = color.matlab(multiData = True)
    for i in range(len(y)):
        plt.plot(x, y[i], color = color1[i], label = labels[i])
    plt.legend()
    plt.show()