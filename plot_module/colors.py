#Author: Ruodong Yang
#A tiny database for colors
import numpy as np
import matplotlib.pyplot as plt

class color:
    def __init__(self, multiData):
        self.multiData = multiData

    def cold(multiData = False):
        if multiData == True:
            return ["#13DDF0", "#1397F0", "#638CF0", "#1352F0", "#1B13F0", "#6313F0"]
        else:
            return "#1352F0"
        
    def warm(multiData = False):
        if multiData == True:
            return ["#FF0A2D", "#E82009", "#FE5217", "#E86109", "#FF8F0A"]
        else:
            return "#FE5217"
        
    
    def red(multiData = False):
        if multiData == True:
            return ["#F25C69", "#F23D3D", "#BF0F0F", "#8C0808", "#590202"]
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
        else:
            return "#0072BD"
        
    def blue(multiData = False):
        if multiData == True:
            return ["#034C8C", "#0367A6", "#2B96D9", "#4AA2D9", "#4AB0D9", "#2685BF", "#3D9DD9"]
        else:
            return "#2B96D9"

    def green(multiData = False):
        if multiData == True:
            return ["#002619", "#034525", "#147534", "#36BA45", "#7EE66C"]
        else:
            return "#034525"

    def rainbow(multiData = True):
        return ["#CC6666", "#CC9966",  "#CCCC66", "#66CC66", "#6699CC", "#6666CC", "#9966CC"]

#Example:

if __name__ == "__main__":
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = x * 2.5
    y4 = x * 2
    y5 = x * 1.1
    y6 = np.sin(x) + 1
    y7 = np.cos(x) + 1
    y = [y1, y2, y3, y4, y5, y6, y7]
    labels = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7']
    plt.figure(figsize=(9, 6))
    color1 = color.rainbow(multiData = True)
    for i in range(len(y)):
        plt.plot(x, y[i], color = color1[i], label = labels[i])
    plt.legend()
    plt.show()