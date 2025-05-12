import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.readlines()
    
    # Замінюємо коми на крапки та конвертуємо в числа
    clean_data = [float(line.replace(",", ".").strip()) for line in data if line.strip()]
    
    # Оновлюємо гістограму
    update_histogram(clean_data)

def update_histogram(data):
    ax.clear()
    ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
    ax.set_title("Гістограма даних")
    ax.set_xlabel("Значення")
    ax.set_ylabel("Частота")
    canvas.draw()

# Головне вікно
root = tk.Tk()
root.title("Статистичний аналіз")
root.geometry("600x500")

# Кнопка завантаження
btn_load = tk.Button(root, text="Завантажити файл", command=load_file)
btn_load.pack()

# Поле для графіка
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

root.mainloop()