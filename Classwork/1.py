import matplotlib.pyplot as plt
# import tkinter
import numpy as np

with open("norm.txt", "r") as file:
    data = file.read().replace(",", ".").split()  
    data = list(map(float, data))  

plt.hist(data, bins=7)

plt.show()