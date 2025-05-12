import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ui_elements import create_ui_elements
from data_manager import values, original_values, load_data
from visualizations import update_histogram, plot_distribution_functions, plot_empirical_distribution

def main():
    # Створення основного вікна
    root = tk.Tk()
    root.title("Статистичний аналіз")
    root.state('zoomed')  # Максимізуємо вікно

    # Створюємо notebook для вкладок
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # Створюємо три вкладки
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)

    # Додаємо вкладки до notebook
    notebook.add(tab1, text='Основний аналіз')
    notebook.add(tab2, text='Функції розподілу')
    notebook.add(tab3, text='Емпірична функція розподілу')

    # Створення фігури для гістограми
    fig, hist_ax = plt.subplots(figsize=(8, 6))
    hist_canvas = FigureCanvasTkAgg(fig, master=tab1)
    hist_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Створюємо елементи інтерфейсу
    ui_elements = create_ui_elements(tab1, tab2, tab3, hist_ax, hist_canvas)
    
    # Запуск основного циклу
    root.mainloop()

if __name__ == "__main__":
    main()