import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

with open("norm.txt", "r") as file:
    data = file.read().replace(",", ".").split()  
    data = list(map(float, data))  

unique, counts = np.unique(data, return_counts=True)
relative_freq = counts / len(data)

print("Варіаційний ряд:")
print("x_i:", unique)
print("n_i:", counts)
print("p_i:", relative_freq)

def plot_histogram(data, bins=None):
    if bins is None:
        bins = int(np.sqrt(len(data))) 
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel("Значення")
    plt.ylabel("Частота")
    plt.title(f"Гістограма (bins={bins})")
    plt.show()

plot_histogram(data)

manual_bins = int(input("Введіть кількість класів для гістограми: "))
plot_histogram(data, bins=manual_bins)
