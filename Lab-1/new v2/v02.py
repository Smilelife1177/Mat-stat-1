import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return
    
    with open(file_path, "r") as file:
        data = file.readlines()
    
    # Заміна ком на крапки та перетворення в числа
    data = [float(line.replace(",", ".")) for line in data]
    
    update_histogram(data)
    update_table(data)

def update_histogram(data):
    ax.clear()
    ax.hist(data, bins='auto', color='blue', alpha=0.7)
    ax.set_title("Гістограма")
    canvas.draw()

def update_table(data):
    # Розрахунок характеристик
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    skewness = skew(data)
    excess_kurtosis = kurtosis(data)
    counter_kurtosis = excess_kurtosis + 3
    variation_pearson = std_dev / mean if mean != 0 else 0
    mad = np.median(np.abs(data - np.median(data)))
    med = np.median(data)
    
    # Видаляємо старі значення з таблиці
    for i in table.get_children():
        table.delete(i)
    
    # Додаємо нові значення
    values = [
        ("Середнє арифметичне", mean),
        ("Середньоквадратичне відхилення", std_dev),
        ("Коефіцієнти асиметрії", skewness),
        ("Ексцес", excess_kurtosis),
        ("Контрексцес", counter_kurtosis),
        ("Варіація Пірсона", variation_pearson),
        ("MAD", mad),
        ("MED", med)
    ]
    
    for name, value in values:
        table.insert("", "end", values=(name, f"{value:.5f}"))

# Створення головного вікна
root = tk.Tk()
root.title("Статистичний аналіз")

# Створення полотна для графіка
fig, ax = plt.subplots(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=0, rowspan=4)

# Кнопка для завантаження файлу
btn_load = tk.Button(root, text="Завантажити файл", command=load_file)
btn_load.grid(row=0, column=1, padx=10, pady=5)

# Таблиця для відображення статистичних характеристик
table = ttk.Treeview(root, columns=("Характеристика", "Значення"), show="headings")
table.heading("Характеристика", text="Характеристика")
table.heading("Значення", text="Значення")
table.grid(row=1, column=1, padx=10, pady=5)

root.mainloop()
