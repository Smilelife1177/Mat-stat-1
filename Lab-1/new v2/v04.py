import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import skew, kurtosis, variation, median_abs_deviation

def load_data():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    with open(file_path, 'r') as file:
        data = file.read().replace(',', '.')
    global values
    values = np.array(list(map(float, data.split())))
    update_histogram()
    update_statistics()
    update_characteristics()

def update_histogram():
    global values, bin_count_var
    bin_count = bin_count_var.get()
    if bin_count == 0:
        bin_count = int(np.sqrt(len(values)))
    hist_ax.clear()
    hist_ax.hist(values, bins=bin_count, color='blue', alpha=0.7, edgecolor='black')
    hist_ax.set_title('Гістограма')
    hist_canvas.draw()
    
    range_val = np.ptp(values)
    bin_width = range_val / bin_count
    info_text.set(f'Кількість класів: {bin_count}\nКрок розбиття: {bin_width:.3f}\nРозмах: {range_val:.3f}\nКількість даних: {len(values)}')

def update_statistics():
    global values
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)
    skewness = skew(values)
    excess_kurtosis = kurtosis(values, fisher=True)
    counter_kurtosis = 1 / (excess_kurtosis + 3) if excess_kurtosis + 3 != 0 else 0
    pearson_var = std_dev / mean if mean != 0 else 0
    nonparam_var = variation(values)
    mad_val = median_abs_deviation(values)
    med_val = np.median(values)
    
    stat_text.set(f"Середнє: {mean:.3f}\nСередньокв. відхилення: {std_dev:.3f}\nАсиметрія: {skewness:.3f}\n"
                 f"Ексцес: {excess_kurtosis:.3f}\nКонтрексцес: {counter_kurtosis:.3f}\nВаріація Пірсона: {pearson_var:.3f}\n"
                 f"Непарам. коеф. вар.: {nonparam_var:.3f}\nMAD: {mad_val:.3f}\nMED: {med_val:.3f}")

def update_characteristics():
    global values
    n = len(values)
    
    # Розрахунок характеристик
    characteristics = {
        "Середнє (зсунене)": np.mean(values),
        "Дисперсія (зсунена)": np.var(values, ddof=0),
        "Середньокв. відхилення (зсунене)": np.std(values, ddof=0),
        "Середнє (незсунене)": np.mean(values),
        "Дисперсія (незсунена)": np.var(values, ddof=1),
        "Середньокв. відхилення (незсунене)": np.std(values, ddof=1),
    }
    
    # Очистити таблицю
    for item in char_table.get_children():
        char_table.delete(item)
    
    # Заповнити таблицю
    for name, value in characteristics.items():
        char_table.insert("", "end", values=(name, f"{value:.4f}"))

root = tk.Tk()
root.title("Статистичний аналіз")

# Створюємо notebook для вкладок
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Створюємо три вкладки
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
tab3 = ttk.Frame(notebook)

# Додаємо вкладки до notebook
notebook.add(tab1, text='Основний аналіз')
notebook.add(tab2, text='Точкові характеристики')
notebook.add(tab3, text='Графіки')

# Вкладка 1: Основний аналіз
frame = tk.Frame(tab1)
frame.pack(side=tk.LEFT, padx=10, pady=10)

load_button = tk.Button(frame, text="Завантажити дані", command=load_data)
load_button.pack()

bin_label = tk.Label(frame, text="Введіть кількість класів для гістограми:")
bin_label.pack()

bin_count_var = tk.IntVar(value=0)
bin_entry = tk.Entry(frame, textvariable=bin_count_var)
bin_entry.pack()

update_button = tk.Button(frame, text="Оновити гістограму", command=update_histogram)
update_button.pack()

info_text = tk.StringVar()
info_label = tk.Label(frame, textvariable=info_text, justify=tk.LEFT)
info_label.pack()

stat_text = tk.StringVar()
stat_label = tk.Label(frame, textvariable=stat_text, justify=tk.LEFT)
stat_label.pack()

fig, hist_ax = plt.subplots()
hist_canvas = FigureCanvasTkAgg(fig, master=tab1)
hist_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Вкладка 2: Точкові характеристики
frame2 = tk.Frame(tab2)
frame2.pack(fill='both', expand=True, padx=10, pady=10)

# Створення таблиці для характеристик
char_table = ttk.Treeview(frame2, columns=("characteristic", "value"), show="headings")
char_table.heading("characteristic", text="Характеристика")
char_table.heading("value", text="Значення")
char_table.column("characteristic", width=300)
char_table.column("value", width=150)
char_table.pack(fill='both', expand=True)

root.mainloop()