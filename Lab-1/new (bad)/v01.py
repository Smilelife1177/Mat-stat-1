import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

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
    
    plot_histogram(data)

def plot_histogram(data):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Значення")
    plt.ylabel("Частота")
    plt.title("Гістограма")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

root = tk.Tk()
root.title("Побудова гістограми")
root.geometry("300x150")

title_label = tk.Label(root, text="Оберіть файл для побудови гістограми")
title_label.pack(pady=10)

open_button = tk.Button(root, text="Відкрити файл", command=open_file)
open_button.pack()

status_label = tk.Label(root, text="")
status_label.pack(pady=10)

root.mainloop()
