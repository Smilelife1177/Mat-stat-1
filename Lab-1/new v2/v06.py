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
    global values, bin_count_var, hist_ax
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

def plot_empirical_distribution():
    global values, hist_ax
    
    # Створюємо новий графік
    plt.figure(figsize=(10, 6))  # Збільшуємо розмір графіка
    emp_ax = plt.gca()
    
    # Використовуємо меншу кількість бінів (наприклад, корінь з кількості значень)
    n_bins = int(np.sqrt(len(values)))
    
    # Розрахунок емпіричної функції розподілу
    bin_dt, bin_gr = np.histogram(values, bins=n_bins)
    Y = np.cumsum(bin_dt) / len(values)  # Нормалізуємо значення
    
    # Додаємо початкову точку
    emp_ax.plot([bin_gr[0], bin_gr[0]], [0, Y[0]], color='green', linewidth=2)
    
    # Побудова ступінчастого графіка
    for i in range(len(Y)):
        # Горизонтальна лінія
        emp_ax.plot([bin_gr[i], bin_gr[i+1]], [Y[i], Y[i]], color='green', linewidth=2)
        # Вертикальна лінія до наступного рівня (якщо це не остання точка)
        # if i < len(Y) - 1:
        #     emp_ax.plot([bin_gr[i+1], bin_gr[i+1]], [Y[i], Y[i+1]], color='green', linewidth=2)
    
    # Додаємо останню горизонтальну лінію
    emp_ax.plot([bin_gr[-1], bin_gr[-1]], [Y[-1], 1], color='green', linewidth=2)
    
    emp_ax.set_title('Емпірична функція розподілу')
    emp_ax.set_xlabel('Значення')
    emp_ax.set_ylabel('Ймовірність')
    emp_ax.grid(True, linestyle='--', alpha=0.7)
    emp_ax.set_ylim(-0.05, 1.05)  # Додаємо невеликий відступ по осі Y
    
    # Відображення графіка
    # Очищаємо попередній графік на вкладці, якщо він є
    for widget in tab3.winfo_children():
        widget.destroy()
    
    emp_canvas = FigureCanvasTkAgg(plt.gcf(), master=tab3)
    emp_canvas.draw()
    emp_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
    
    # Очистити таблицю
    for item in char_table.get_children():
        char_table.delete(item)
    

    # stat_text.set(f"Середнє: {mean:.3f}\nСередньокв. відхилення: {std_dev:.3f}\nАсиметрія: {skewness:.3f}\n"
    #              f"Ексцес: {excess_kurtosis:.3f}\nКонтрексцес: {counter_kurtosis:.3f}\nВаріація Пірсона: {pearson_var:.3f}\n"
    #              f"Непарам. коеф. вар.: {nonparam_var:.3f}\nMAD: {mad_val:.3f}\nMED: {med_val:.3f}")
    # # Додати дані до таблиці

    characteristics = [
        ("Середнє", np.mean(values), np.mean(values)),
        ("Дисперсія", np.var(values, ddof=0), np.var(values, ddof=1)),
        ("Середньокв. відхилення", np.std(values, ddof=0), np.std(values, ddof=1)),
        ("Асиметрія", skew(values), skew(values)),
        ("Ексцес", kurtosis(values, fisher=True), kurtosis(values, fisher=True)),
        ("Контрексцес", (1 / (kurtosis(values, fisher=True) + 3) if kurtosis(values, fisher=True) + 3 != 0 else 0), (1 / (kurtosis(values, fisher=True) + 3) if kurtosis(values, fisher=True) + 3 != 0 else 0)),
        ("Варіація Пірсона", np.std(values, ddof=1) / np.mean(values) if np.mean(values) != 0 else 0), kurtosis(values, fisher=True),
    ]
    
    for char, biased, unbiased in characteristics:
        char_table.insert("", "end", values=(char, f"{biased:.4f}", f"{unbiased:.4f}"))

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
notebook.add(tab2, text='Додатковий аналіз')
notebook.add(tab3, text='Емпірична функція розподілу')

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

# Додаємо кнопку для побудови емпіричної функції розподілу
emp_button = tk.Button(frame, text="Побудувати емпіричну функцію розподілу", command=plot_empirical_distribution)
emp_button.pack(pady=5)

info_text = tk.StringVar()
info_label = tk.Label(frame, textvariable=info_text, justify=tk.LEFT)
info_label.pack()

# Створення та налаштування таблиці характеристик
char_frame = ttk.LabelFrame(frame, text="Точкові характеристики", padding=(5, 5))
char_frame.pack(fill='x', pady=10)

char_table = ttk.Treeview(char_frame, columns=("characteristic", "biased", "unbiased"), show="headings", height=5)
char_table.heading("characteristic", text="Характеристика")
char_table.heading("biased", text="Зсунена")
char_table.heading("unbiased", text="Незсунена")
char_table.column("characteristic", width=150)
char_table.column("biased", width=100)
char_table.column("unbiased", width=100)
char_table.pack(fill='x')

stat_text = tk.StringVar()
stat_label = tk.Label(frame, textvariable=stat_text, justify=tk.LEFT)
stat_label.pack()

# Графік
fig, hist_ax = plt.subplots()
hist_canvas = FigureCanvasTkAgg(fig, master=tab1)
hist_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

root.mainloop()