import numpy as np
from tkinter import messagebox
import matplotlib.pyplot as plt

def generate_and_build_series(mean, std, results_text, ax, canvas):
    """
    Генерує 100 нормально розподілених чисел, будує інтервальний статистичний ряд (10 інтервалів),
    гістограму, полігон частот, огіву та емпіричну функцію розподілу (ECDF).
    Виводить результати в текстовий блок і відображає графіки.
    """
    try:
        # Перевірка параметрів
        if std <= 0:
            raise ValueError("Стандартне відхилення повинно бути додатним.")
        
        # Генерація 100 нормально розподілених чисел
        data = np.random.normal(mean, std, 100)
        
        # Побудова інтервального статистичного ряду (10 інтервалів)
        hist, bin_edges = np.histogram(data, bins=10)
        
        # Очищення текстового блоку
        results_text.delete(1.0, 'end')
        
        # Вивід згенерованих даних
        results_text.insert('end', "Згенеровані дані (100 чисел):\n")
        results_text.insert('end', ", ".join(f"{x:.4f}" for x in data) + "\n\n")
        
        # Вивід інтервального ряду
        results_text.insert('end', "Інтервальний статистичний ряд (10 інтервалів):\n")
        results_text.insert('end', f"{'Інтервал':<20} {'Частота':<10}\n")
        results_text.insert('end', "-" * 30 + "\n")
        for i in range(len(hist)):
            interval = f"[{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f})"
            freq = hist[i]
            results_text.insert('end', f"{interval:<20} {freq:<10}\n")
        results_text.insert('end', "\nСумарна частота: " + str(sum(hist)) + "\n")
        
        # Очищення попереднього графіку
        ax.clear()
        
        # Побудова гістограми
        ax.hist(data, bins=10, density=False, alpha=0.7, color='blue', edgecolor='black', label='Гістограма')
        
        # Побудова полігону частот
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, hist, 'r-', marker='o', label='Полігон частот')
        
        # Побудова огіви (кумулятивної гістограми)
        cumulative_hist = np.cumsum(hist)
        ax.step(bin_edges[1:], cumulative_hist, 'g-', where='post', label='Огіва (кумулятивна гістограма)')
        
        # Побудова емпіричної функції розподілу (ECDF)
        sorted_data = np.sort(data)
        ecdf_y = np.arange(1, len(data) + 1) / len(data)
        ax.step(sorted_data, ecdf_y, 'm-', where='post', label='Емпірична функція розподілу')
        
        # Налаштування графіку
        ax.set_title('Гістограма, Полігон частот, Огіва та ECDF')
        ax.set_xlabel('Значення')
        ax.set_ylabel('Частота / Кумулятивна частота / Ймовірність')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Оновлення canvas
        canvas.draw()
    
    except ValueError as ve:
        messagebox.showerror("Помилка", str(ve))
    except Exception as e:
        messagebox.showerror("Помилка", f"Виникла помилка: {str(e)}")