import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

with open("norm.txt", "r") as file:
    data = file.read().replace(",", ".").split()  
    data = np.array(list(map(float, data)))  

unique, counts = np.unique(data, return_counts=True)
relative_freq = counts / len(data)

mean_value = np.mean(data)  # Середнє арифметичне
std_dev = np.std(data, ddof=1)  # середньоквадратичне відхилення
skewness = np.mean(((data - mean_value) / std_dev) ** 3)  # Коефіцієнт асиметрії
kurtosis = np.mean(((data - mean_value) / std_dev) ** 4) - 3  # Ексцес 
counterkurtosis = -kurtosis  # Контрексцес 
pearson_variation = std_dev / mean_value  # Варіація Пірсона


stats_df = pd.DataFrame({
    "Характеристика": [
        "Середнє арифметичне",
        "Середньоквадратичне відхилення",
        "Коефіцієнт асиметрії",
        "Ексцес",
        "Контрексцес",
        "Варіація Пірсона"
    ],
    "Значення": [
        mean_value,
        std_dev,
        skewness,
        kurtosis,
        counterkurtosis,
        pearson_variation
    ]
})

print(stats_df)

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
