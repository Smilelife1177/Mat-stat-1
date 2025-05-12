import matplotlib.pyplot as plt
# import tkinter
import numpy as np

with open("norm.txt", "r") as file:
    data = file.read().replace(",", ".").split()  
    data = list(map(float, data))  

plt.hist(data, bins=500, color='skyblue', edgecolor='black')

# Compute variation series (sorted unique values and frequencies)
unique_values, counts = np.unique(data, return_counts=True)
variation_series = list(zip(unique_values, counts))

print("Variation Series (Value, Frequency):")
for value, freq in variation_series:
    print(f"{value}: {freq}")

plt.show()
