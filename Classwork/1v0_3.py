import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import t

with open("norm.txt", "r") as file:
    data = file.read().replace(",", ".").split()  
    data = np.array(list(map(float, data)))  

# Перевірка на наявність пропущених або некоректних значень
data = data[~np.isnan(data)]  # Видаляємо NaN значення

# Формування варіаційного ряду
unique, counts = np.unique(data, return_counts=True)
relative_freq = counts / len(data)

# Розрахунок незсунених характеристик
mean_value = np.mean(data)  # Середнє арифметичне
std_dev = np.std(data, ddof=1)  # середньоквадратичне відхилення
skewness = np.mean(((data - mean_value) / std_dev) ** 3)  # Коефіцієнт асиметрії
kurtosis = np.mean(((data - mean_value) / std_dev) ** 4) - 3  # Ексцес 
counterkurtosis = -kurtosis  # Контрексцес 
pearson_variation = std_dev / mean_value  # Варіація Пірсона

# Створення DataFrame для виведення характеристик
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

# Виведення результатів з заданою точністю
precision = int(input("Введіть точність (кількість знаків після коми): "))
stats_df["Значення"] = stats_df["Значення"].apply(lambda x: round(x, precision))
print(stats_df)

# Функція для побудови гістограми
def plot_histogram(data, bins=None, method='auto'):
    if bins is None:
        if method == 'auto':
            bins = int(np.sqrt(len(data)))  # Автоматичний вибір
        elif method == 'sturges':
            bins = int(np.ceil(np.log2(len(data))) + 1)  # Правило Стреджеса
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel("Значення")
    plt.ylabel("Частота")
    plt.title(f"Гістограма (bins={bins})")
    plt.show()

# Виклик функції з автоматичним вибором кількості класів
plot_histogram(data, method='auto')

# Виклик функції з ручним вибором кількості класів
manual_bins = int(input("Введіть кількість класів для гістограми: "))
plot_histogram(data, bins=manual_bins)

# Функція для розрахунку довірчих інтервалів
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    t_value = t.ppf((1 + confidence) / 2, df=n-1)
    margin_of_error = t_value * std_err
    return mean - margin_of_error, mean + margin_of_error

# Візуалізація довірчих інтервалів
def plot_confidence_intervals(data, confidence=0.95):
    ci_mean = confidence_interval(data, confidence)
    plt.errorbar(0, np.mean(data), yerr=(ci_mean[1] - ci_mean[0])/2, fmt='o', label='Середнє арифметичне')
    plt.legend()
    plt.title("Довірчий інтервал для середнього арифметичного")
    plt.show()

plot_confidence_intervals(data)