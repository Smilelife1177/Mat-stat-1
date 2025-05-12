import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import scipy.stats as stats
from scipy.stats import norm

from data_manager import values

def update_histogram(hist_ax, hist_canvas, bin_count_var, lower_bound_var, upper_bound_var, info_text):
    global values
    bin_count = bin_count_var.get()
    if bin_count == 0:
        bin_count = int(np.sqrt(len(values)))
    hist_ax.clear()
    hist_ax.hist(values, bins=bin_count, color='blue', alpha=0.7, edgecolor='black')
    hist_ax.set_title('Гістограма')
    
    # Додаємо границі на графік, якщо вони встановлені
    try:
        lower = float(lower_bound_var.get())
        upper = float(upper_bound_var.get())
        if lower < upper:
            hist_ax.axvline(x=lower, color='r', linestyle='--', label='Нижня границя')
            hist_ax.axvline(x=upper, color='g', linestyle='--', label='Верхня границя')
            hist_ax.legend()
    except (ValueError, TypeError):
        pass
    
    hist_canvas.draw()
    
    range_val = np.ptp(values)
    bin_width = range_val / bin_count
    info_text.set(f'Кількість класів: {bin_count}\nКрок розбиття: {bin_width:.3f}\nРозмах: {range_val:.3f}\nКількість даних: {len(values)}')

def plot_empirical_distribution(tab3, lower_bound_var, upper_bound_var):
    global values
    
    # Створюємо новий графік
    plt.figure(figsize=(10, 6))
    emp_ax = plt.gca()
    
    # Використовуємо меншу кількість бінів
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
    
    # Додаємо останню горизонтальну лінію
    emp_ax.plot([bin_gr[-1], bin_gr[-1]], [Y[-1], 1], color='green', linewidth=2)
    
    # Додаємо границі на графік, якщо вони встановлені
    try:
        lower = float(lower_bound_var.get())
        upper = float(upper_bound_var.get())
        if lower < upper:
            emp_ax.axvline(x=lower, color='r', linestyle='--', label='Нижня границя')
            emp_ax.axvline(x=upper, color='g', linestyle='--', label='Верхня границя')
            emp_ax.legend()
    except (ValueError, TypeError):
        pass
    
    emp_ax.set_title('Емпірична функція розподілу')
    emp_ax.set_xlabel('Значення')
    emp_ax.set_ylabel('Ймовірність')
    emp_ax.grid(True, linestyle='--', alpha=0.7)
    emp_ax.set_ylim(-0.05, 1.05)
    
    # Відображення графіка
    for widget in tab3.winfo_children():
        widget.destroy()
    
    emp_canvas = FigureCanvasTkAgg(plt.gcf(), master=tab3)
    emp_canvas.draw()
    emp_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def plot_distribution_functions(tab2, lower_bound_var, upper_bound_var):
    global values
    
    # Очистимо вміст вкладки 2
    for widget in tab2.winfo_children():
        widget.destroy()
    
    # Створюємо новий графік
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Розрахунок емпіричної функції розподілу
    x = np.sort(values)
    y = np.arange(1, len(values) + 1) / len(values)
    
    # Визначення параметрів для теоретичних розподілів
    mean, std = np.mean(values), np.std(values)
    
    # Створюємо масив x для побудови теоретичних розподілів
    x_theor = np.linspace(min(values), max(values), 1000)
    
    # Побудова графіка нормального розподілу
    ax.plot(x_theor, norm.cdf(x_theor, loc=mean, scale=std), label='Нормальний розподіл', color='red')
    
    # Додаємо границі на графік, якщо вони встановлені
    try:
        lower = float(lower_bound_var.get())
        upper = float(upper_bound_var.get())
        if lower < upper:
            ax.axvline(x=lower, color='r', linestyle='--', label='Нижня границя')
            ax.axvline(x=upper, color='g', linestyle='--', label='Верхня границя')
    except (ValueError, TypeError):
        pass
    
    # Налаштування графіка
    ax.set_title('Порівняння функцій розподілу')
    ax.set_xlabel('Значення')
    ax.set_ylabel('Ймовірність')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Відображення графіка
    canvas = FigureCanvasTkAgg(fig, master=tab2)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Додаємо тест Колмогорова-Смірнова для перевірки нормальності
    ks_statistic, ks_pvalue = stats.kstest(values, 'norm', args=(mean, std))
    ks_text = f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\np-значення: {ks_pvalue:.4f}"
    
    # Додаємо тест Шапіро-Вілка для перевірки нормальності
    if len(values) <= 5000:  # Тест Шапіро-Вілка працює тільки для вибірок розміром до 5000
        shapiro_statistic, shapiro_pvalue = stats.shapiro(values)
        ks_text += f"\n\nТест Шапіро-Вілка:\nСтатистика: {shapiro_statistic:.4f}\np-значення: {shapiro_pvalue:.4f}"
    
    # Додаємо тест Андерсона-Дарлінга
    ad_result = stats.anderson(values, 'norm')
    ks_text += f"\n\nТест Андерсона-Дарлінга:\nСтатистика: {ad_result.statistic:.4f}"
    
    # Відображення результатів тестів
    info_frame = tk.Frame(tab2)
    info_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=10, pady=10)
    
    info_label = tk.Label(info_frame, text=ks_text, justify=tk.LEFT)
    info_label.pack()