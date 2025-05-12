import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# 
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not file_path:
        return
    
    with open(file_path, "r") as file:
        data = file.read().replace(",", ".").split()
        try:
            data = list(map(float, data))
        except ValueError:
            status_label.config(text="Помилка: файл містить некоректні дані")
            return
    
    update_histogram(data)

def update_histogram(data):
    ax.clear()
    bins = int(bin_entry.get()) if bin_entry.get().isdigit() else 20
    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Значення")
    ax.set_ylabel("Частота")
    ax.set_title("Гістограма")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    canvas.draw()
    
    num_classes = bins
    data_range = max(data) - min(data)
    step = data_range / num_classes if num_classes else 0
    status_label.config(text=f"Класів: {num_classes}, Крок: {step:.2f}, Розмах: {data_range:.2f}, Кількість: {len(data)}")

root = tk.Tk()
root.title("Побудова гістограми")
root.geometry("500x500")

title_label = tk.Label(root, text="Оберіть файл для побудови гістограми")
title_label.pack(pady=10)

open_button = tk.Button(root, text="Відкрити файл", command=open_file)
open_button.pack()

bin_label = tk.Label(root, text="Кількість класів:")
bin_label.pack()

bin_entry = tk.Entry(root)
bin_entry.pack()
bin_entry.insert(0, "20")

fig, ax = plt.subplots(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

status_label = tk.Label(root, text="")
status_label.pack(pady=10)

root.mainloop()
