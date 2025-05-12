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
        ####
        # Дані
        self.data = None
        
        # Створення основного інтерфейсу
        self.create_gui()
    
    def create_gui(self):
        # Створення фреймів
        self.left_frame = ttk.Frame(self.root, padding="5")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        
        self.right_frame = ttk.Frame(self.root, padding="5")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Кнопки для завантаження та обробки даних
        ttk.Button(self.left_frame, text="Завантажити дані", command=self.load_data).grid(row=0, column=0, pady=5)
        ttk.Button(self.left_frame, text="Варіаційний ряд", command=self.create_variation_series).grid(row=1, column=0, pady=5)
        ttk.Button(self.left_frame, text="Побудувати гістограму", command=self.plot_histogram).grid(row=2, column=0, pady=5)
        ttk.Button(self.left_frame, text="Розрахувати характеристики", command=self.calculate_statistics).grid(row=3, column=0, pady=5)
        ttk.Button(self.left_frame, text="Довірчі інтервали", command=self.calculate_confidence_intervals).grid(row=4, column=0, pady=5)
        
        # Поле для виведення результатів
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
                        # Читаємо дані та замінюємо коми на крапки
                        content = file.read().replace(",", ".")
                        # Розділяємо по пробілах та конвертуємо в числа
                        self.data = np.array([float(x) for x in content.split()])
                else:  # CSV файл
                    df = pd.read_csv(file_path)
                    # Конвертуємо значення першого стовпця в строки, замінюємо коми на крапки
                    self.data = np.array([float(str(x).replace(",", ".")) for x in df.iloc[:, 0]])
                
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Завантажено {len(self.data)} значень\n")
                self.result_text.insert(tk.END, f"Перші 5 значень: {self.data[:5]}\n")
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
        for value, count in zip(unique, counts):
            self.result_text.insert(tk.END, f"Значення: {value:.4f}, Частота: {count}\n")

    def plot_histogram(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        self.ax.clear()
        
        # Побудова гістограми
        sns.histplot(data=self.data, kde=True, ax=self.ax)
        
        # Додавання KDE
        sns.kdeplot(data=self.data, ax=self.ax, color='red', linewidth=2)
        
        self.ax.set_title("Гістограма та KDE")
        self.canvas.draw()

    def calculate_statistics(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        # Розрахунок статистик
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)  # незсунена оцінка
        skew = stats.skew(self.data)
        kurt = stats.kurtosis(self.data)
        
        # MAD (Mean Absolute Deviation)
        mad = np.mean(np.abs(self.data - mean))
        
        # MED (Median Absolute Deviation)
        med = np.median(np.abs(self.data - np.median(self.data)))
        
        # Коефіцієнт варіації Пірсона
        cv = (std / mean) * 100
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Статистичні характеристики:\n\n")
        self.result_text.insert(tk.END, f"Середнє: {mean:.4f}\n")
        self.result_text.insert(tk.END, f"Стандартне відхилення: {std:.4f}\n")
        self.result_text.insert(tk.END, f"Асиметрія: {skew:.4f}\n")
        self.result_text.insert(tk.END, f"Ексцес: {kurt:.4f}\n")
        self.result_text.insert(tk.END, f"MAD: {mad:.4f}\n")
        self.result_text.insert(tk.END, f"MED: {med:.4f}\n")
        self.result_text.insert(tk.END, f"Коефіцієнт варіації: {cv:.2f}%\n")

    def calculate_confidence_intervals(self):
        if self.data is None:
            messagebox.showwarning("Попередження", "Спочатку завантажте дані")
            return
        
        n = len(self.data)
        confidence = 0.95
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        
        # Довірчий інтервал для середнього
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)
        margin = t_value * (std / np.sqrt(n))
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Довірчі інтервали (рівень довіри {confidence*100}%):\n\n")
        self.result_text.insert(tk.END, f"Середнє: [{mean-margin:.4f}, {mean+margin:.4f}]\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = StatisticalAnalysisApp(root)
    root.mainloop()