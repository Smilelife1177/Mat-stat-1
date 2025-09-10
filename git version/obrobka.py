import numpy as np
from tkinter import messagebox

def generate_and_build_series(mean, std, results_text):
    """
    Генерує 100 нормально розподілених чисел і будує інтервальний статистичний ряд (10 інтервалів).
    Виводить результати в текстовий блок.
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
    
    except ValueError as ve:
        messagebox.showerror("Помилка", str(ve))
    except Exception as e:
        messagebox.showerror("Помилка", f"Виникла помилка: {str(e)}")