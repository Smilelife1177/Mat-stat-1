import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from scipy import stats
import math

class StatisticalAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Статистичний аналіз даних")
        self.root.geometry("1200x900")
        
        self.data = None
        self.precision = 4  # Default precision for calculations
        self.create_gui()
    
    def create_gui(self):
        # Створення фреймів
        self.left_frame = ttk.Frame(self.root, padding="5")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        
        self.right_frame = ttk.Frame(self.root, padding="5")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Налаштування для гістограми
        self.hist_frame = ttk.LabelFrame(self.left_frame, text="Налаштування гістограми", padding="5")
        self.hist_frame.grid(row=0, column=0, pady=5, sticky="ew")
        
        self.bins_var = tk.StringVar(value="auto")
        ttk.Radiobutton(self.hist_frame, text="Авто", variable=self.bins_var, value="auto").grid(row=0, column=0)
        ttk.Radiobutton(self.hist_frame, text="Вручну", variable=self.bins_var, value="manual").grid(row=0, column=1)
        self.bins_entry = ttk.Entry(self.hist_frame, width=10)
        self.bins_entry.grid(row=0, column=2)
        self.bins_entry.insert(0, "10")
        
        # Налаштування точності
        ttk.Label(self.left_frame, text="Точність:").grid(row=1, column=0, sticky="w")
        self.precision_entry = ttk.Entry(self.left_frame, width=5)
        self.precision_entry.grid(row=1, column=0, sticky="e")
        self.precision_entry.insert(0, str(self.precision))
        
        # Кнопки для аналізу
        ttk.Button(self.left_frame, text="Завантажити дані", command=self.load_data).grid(row=2, column=0, pady=5)
        ttk.Button(self.left_frame, text="Варіаційний ряд", command=self.create_variation_series).grid(row=3, column=0, pady=5)
        ttk.Button(self.left_frame, text="Побудувати гістограму", command=self.plot_histogram).grid(row=4, column=0, pady=5)
        ttk.Button(self.left_frame, text="Функція розподілу", command=self.plot_distribution).grid(row=5, column=0, pady=5)
        ttk.Button(self.left_frame, text="Розрахувати характеристики", command=self.calculate_statistics).grid(row=6, column=0, pady=5)
        ttk.Button(self.left_frame, text="Довірчі інтервали", command=self.calculate_confidence_intervals).grid(row=7, column=0, pady=5)
        
        # Кнопки для редагування даних
        edit_frame = ttk.LabelFrame(self.left_frame, text="Редагування даних", padding="5")
        edit_frame.grid(row=8, column=0, pady=5, sticky="ew")
        
        ttk.Button(edit_frame, text="Стандартизація", command=self.standardize_data).grid(row=0, column=0, pady=2)
        ttk.Button(edit_frame, text="Логарифмування", command=self.log_transform).grid(row=1, column=0, pady=2)
        ttk.Button(edit_frame, text="Видалити викиди", command=self.remove_outliers).grid(row=2, column=0, pady=2)
        
        # Область результатів
        self.result_text = tk.Text(self.right_frame, width=60, height=30)
        self.result_text.grid(row=0, column=0, padx=5, pady=5)
        
        # Область для графіків
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)
    
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])
        if file_path:
            try:
                if file_path.endswith('.txt'):
                    with open(file_path, 'r') as file:
                        content = file.read().replace(",", ".")
                        self.data = np.array([float(x) for x in content.split()])
                else:
                    df = pd.read_csv(file_path)
                    self.data = np.array([float(str(x).replace(",", ".")) for x in df.iloc[:, 0]])
                
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Завантажено {len(self.data)} значень\n")
                self.result_text.insert(tk.END, f"Перші 5 значень: {self.data[:5]}\n")
                self.plot_distribution()
            except Exception as e:
                messagebox.showerror("Помилка", f"Помилка завантаження даних: {str(e)}")

    def create_variation_series(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        sorted_data = np.sort(self.data)
        unique, counts = np.unique(sorted_data, return_counts=True)
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Варіаційний ряд:\n\n")
        
        prec = int(self.precision_entry.get())
        for value, count in zip(unique, counts):
            self.result_text.insert(tk.END, f"Значення: {value:.{prec}f}, Частота: {count}\n")

    def plot_histogram(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        self.ax.clear()
        
        # Визначення кількості класів
        if self.bins_var.get() == "auto":
            bins = 'auto'
        else:
            try:
                bins = int(self.bins_entry.get())
            except ValueError:
                messagebox.showerror("Помилка", "Некоректна кількість класів")
                return
        
        # Побудова гістограми та KDE
        hist = sns.histplot(data=self.data, bins=bins, kde=True, ax=self.ax)
        
        # Розрахунок та виведення параметрів гістограми
        if isinstance(bins, str):
            bins = len(hist.patches)
        
        data_range = np.ptp(self.data)
        bin_width = data_range / bins
        
        stats_text = f"Кількість класів: {bins}\n"
        stats_text += f"Крок розбиття: {bin_width:.4f}\n"
        stats_text += f"Розмах: {data_range:.4f}\n"
        stats_text += f"Кількість даних: {len(self.data)}"
        
        self.ax.text(0.95, 0.95, stats_text,
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.ax.set_title("Гістограма та KDE")
        self.canvas.draw()

    def calculate_statistics(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        prec = int(self.precision_entry.get())
        
        # Розрахунок базових статистик
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)  # незсунена оцінка
        std_biased = np.std(self.data, ddof=0)  # зсунена оцінка
        skew = stats.skew(self.data)
        kurt = stats.kurtosis(self.data)
        
        # Додаткові характеристики
        mad = np.mean(np.abs(self.data - mean))
        med = np.median(np.abs(self.data - np.median(self.data)))
        cv = (std / mean) * 100  # коефіцієнт варіації Пірсона
        
        # Контрексцес
        contrkurt = 1 / (kurt + 1)
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Статистичні характеристики:\n\n")
        self.result_text.insert(tk.END, f"Середнє арифметичне: {mean:.{prec}f}\n")
        self.result_text.insert(tk.END, f"Стандартне відхилення (незсунене): {std:.{prec}f}\n")
        self.result_text.insert(tk.END, f"Стандартне відхилення (зсунене): {std_biased:.{prec}f}\n")
        self.result_text.insert(tk.END, f"Асиметрія: {skew:.{prec}f}\n")
        self.result_text.insert(tk.END, f"Ексцес: {kurt:.{prec}f}\n")
        self.result_text.insert(tk.END, f"Контрексцес: {contrkurt:.{prec}f}\n")
        self.result_text.insert(tk.END, f"MAD: {mad:.{prec}f}\n")
        self.result_text.insert(tk.END, f"MED: {med:.{prec}f}\n")
        self.result_text.insert(tk.END, f"Коефіцієнт варіації Пірсона: {cv:.{prec}f}%\n")

    def calculate_confidence_intervals(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        n = len(self.data)
        confidence = 0.95
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        prec = int(self.precision_entry.get())
        
        # Довірчі інтервали для всіх характеристик
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)
        
        # Для середнього
        mean_margin = t_value * (std / np.sqrt(n))
        mean_ci = (mean - mean_margin, mean + mean_margin)
        
        # Для стандартного відхилення
        chi2_lower = stats.chi2.ppf((1-confidence)/2, n-1)
        chi2_upper = stats.chi2.ppf((1+confidence)/2, n-1)
        std_ci = (std * np.sqrt((n-1)/chi2_upper), std * np.sqrt((n-1)/chi2_lower))
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Довірчі інтервали (рівень довіри {confidence*100}%):\n\n")
        self.result_text.insert(tk.END, f"Середнє: [{mean_ci[0]:.{prec}f}, {mean_ci[1]:.{prec}f}]\n")
        self.result_text.insert(tk.END, f"Стандартне відхилення: [{std_ci[0]:.{prec}f}, {std_ci[1]:.{prec}f}]\n")

    def standardize_data(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        self.data = (self.data - np.mean(self.data)) / np.std(self.data)
        self.plot_histogram()
        messagebox.showinfo("Інформація", "Дані стандартизовано")

    def log_transform(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        if np.any(self.data <= 0):
            messagebox.showerror("Помилка", "Неможливо взяти логарифм від від'ємних чисел")
            return
        
        self.data = np.log(self.data)
        self.plot_histogram()
        messagebox.showinfo("Інформація", "Дані прологарифмовано")

    def remove_outliers(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        mask = (self.data >= lower_bound) & (self.data <= upper_bound)
        removed = len(self.data) - np.sum(mask)
        self.data = self.data[mask]
        
        self.plot_histogram()
        messagebox.showinfo("Інформація", f"Видалено {removed} викидів")

    def plot_distribution(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        self.ax.clear()
        
        # Побудова ем
        
        # Побудова емпіричної функції розподілу
        sorted_data = np.sort(self.data)
        n = len(sorted_data)
        cumprob = np.arange(1, n + 1) / n
        
        # Побудова графіка
        self.ax.step(sorted_data, cumprob, where='post', label='Емпірична функція розподілу')
        
        # Додавання теоретичної нормальної функції розподілу для порівняння
        mean = np.mean(sorted_data)
        std = np.std(sorted_data)
        x = np.linspace(min(sorted_data), max(sorted_data), 100)
        self.ax.plot(x, stats.norm.cdf(x, mean, std), 'r--', label='Нормальний розподіл')
        
        # Налаштування графіка
        self.ax.set_title('Функція розподілу')
        self.ax.set_xlabel('Значення')
        self.ax.set_ylabel('Ймовірність')
        self.ax.grid(True)
        self.ax.legend()
        
        # Оновлення полотна
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = StatisticalAnalysisApp(root)
    root.mainloop()