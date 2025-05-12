import tkinter as tk
from tkinter import ttk

from data_manager import load_data, apply_bounds, update_data_box
from statistics import update_statistics, update_characteristics
from visualizations import update_histogram, plot_distribution_functions, plot_empirical_distribution
from data_transformations import standardize_data, log_transform, shift_data, remove_outliers, reset_data

def create_ui_elements(tab1, tab2, tab3, hist_ax, hist_canvas):
    # Вкладка 1: Основний аналіз
    frame = tk.Frame(tab1)
    frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
    
    # Створення текстових змінних
    info_text = tk.StringVar()
    bin_count_var = tk.IntVar(value=10)  # Changed default from 0 to 10
    lower_bound_var = tk.StringVar()
    upper_bound_var = tk.StringVar()
    
    # Створення елементів інтерфейсу
    # Завантаження даних
    load_button = tk.Button(frame, text="Завантажити дані", 
                          command=lambda: load_data(
                              lambda: update_histogram(hist_ax, hist_canvas, bin_count_var, lower_bound_var, upper_bound_var, info_text),
                              update_statistics,
                              lambda: update_characteristics(char_table),
                              lambda: update_data_box(data_box),
                              editing_buttons, 
                              plot_btn, 
                              cdf_btn,
                              lower_bound_var, 
                              upper_bound_var))
    load_button.pack(fill=tk.X, pady=5)
    
    # Гістограма
    bin_label = tk.Label(frame, text="Введіть кількість класів для гістограми:")
    bin_label.pack()
    
    bin_entry = tk.Entry(frame, textvariable=bin_count_var)
    bin_entry.pack()
    
    update_button = tk.Button(frame, text="Оновити гістограму", 
                            command=lambda: update_histogram(hist_ax, hist_canvas, bin_count_var, lower_bound_var, upper_bound_var, info_text))
    update_button.pack(fill=tk.X, pady=5)
    
    # Інформація про дані
    info_label = tk.Label(frame, textvariable=info_text, justify=tk.LEFT)
    info_label.pack()
    
    # Секція встановлення границь
    bounds_frame = ttk.LabelFrame(frame, text="Встановлення границь", padding=(5, 5))
    bounds_frame.pack(fill='x', pady=10)
    
    # Нижня границя
    lower_frame = tk.Frame(bounds_frame)
    lower_frame.pack(fill='x', pady=2)
    lower_label = tk.Label(lower_frame, text="Нижня границя:", width=15, anchor='w')
    lower_label.pack(side=tk.LEFT)
    lower_entry = tk.Entry(lower_frame, textvariable=lower_bound_var)
    lower_entry.pack(side=tk.LEFT, fill='x', expand=True)
    
    # Верхня границя
    upper_frame = tk.Frame(bounds_frame)
    upper_frame.pack(fill='x', pady=2)
    upper_label = tk.Label(upper_frame, text="Верхня границя:", width=15, anchor='w')
    upper_label.pack(side=tk.LEFT)
    upper_entry = tk.Entry(upper_frame, textvariable=upper_bound_var)
    upper_entry.pack(side=tk.LEFT, fill='x', expand=True)
    
    # Кнопка застосування границь
    apply_bounds_btn = tk.Button(bounds_frame, text="Застосувати границі", 
                               command=lambda: apply_bounds(
                                   lower_bound_var, 
                                   upper_bound_var, 
                                   lambda: update_histogram(hist_ax, hist_canvas, bin_count_var, lower_bound_var, upper_bound_var, info_text),
                                   update_statistics,
                                   lambda: update_characteristics(char_table),
                                   lambda: update_data_box(data_box)))
    apply_bounds_btn.pack(fill=tk.X, pady=5)
    apply_bounds_btn.config(state=tk.DISABLED)
    
    # Секція редагування даних
    edit_frame = ttk.LabelFrame(frame, text="Редагування даних", padding=(5, 5))
    edit_frame.pack(fill='x', pady=10)
    
    # Кнопки для редагування даних
    standardize_btn = tk.Button(edit_frame, text="Стандартизувати", 
                             command=lambda: standardize_data(
                                 lambda: update_histogram(hist_ax, hist_canvas, bin_count_var, lower_bound_var, upper_bound_var, info_text),
                                 update_statistics,
                                 lambda: update_characteristics(char_table),
                                 lambda: update_data_box(data_box)),
                             state=tk.DISABLED)
    standardize_btn.pack(fill=tk.X, pady=2)
    
    log_btn = tk.Button(edit_frame, text="Логарифмувати", 
                      command=lambda: log_transform(
                          lambda: update_histogram(hist_ax, hist_canvas, bin_count_var, lower_bound_var, upper_bound_var, info_text),
                          update_statistics,
                          lambda: update_characteristics(char_table),
                          lambda: update_data_box(data_box)),
                      state=tk.DISABLED)
    log_btn.pack(fill=tk.X, pady=2)
    
    shift_btn = tk.Button(edit_frame, text="Зсунути", 
                        command=lambda: shift_data(
                            lambda: update_histogram(hist_ax, hist_canvas, bin_count_var, lower_bound_var, upper_bound_var, info_text),
                            update_statistics,
                            lambda: update_characteristics(char_table),
                            lambda: update_data_box(data_box)),
                        state=tk.DISABLED)
    shift_btn.pack(fill=tk.X, pady=2)
    
    outliers_btn = tk.Button(edit_frame, text="Вилучити аномальні дані", 
                          command=lambda: remove_outliers(
                              lambda: update_histogram(hist_ax, hist_canvas, bin_count_var, lower_bound_var, upper_bound_var, info_text),
                              update_statistics,
                              lambda: update_characteristics(char_table),
                              lambda: update_data_box(data_box)),
                          state=tk.DISABLED)
    outliers_btn.pack(fill=tk.X, pady=2)
    
    reset_btn = tk.Button(edit_frame, text="Скинути до початкових", 
                        command=lambda: reset_data(
                            lambda: update_histogram(hist_ax, hist_canvas, bin_count_var, lower_bound_var, upper_bound_var, info_text),
                            update_statistics,
                            lambda: update_characteristics(char_table),
                            lambda: update_data_box(data_box),
                            lower_bound_var,
                            upper_bound_var),
                        state=tk.DISABLED)
    reset_btn.pack(fill=tk.X, pady=2)
    
    # Список кнопок для активації/деактивації
    editing_buttons = [standardize_btn, log_btn, shift_btn, outliers_btn, reset_btn, apply_bounds_btn]
    
    # Додаємо кнопки для побудови функцій розподілу
    plot_btn = tk.Button(frame, text="Побудувати функції розподілу", 
                       command=lambda: plot_distribution_functions(tab2, lower_bound_var, upper_bound_var), 
                       state=tk.DISABLED)
    plot_btn.pack(fill=tk.X, pady=5)
    
    cdf_btn = tk.Button(frame, text="Побудувати емпіричну функцію розподілу", 
                      command=lambda: plot_empirical_distribution(tab3, lower_bound_var, upper_bound_var), 
                      state=tk.DISABLED)
    cdf_btn.pack(fill=tk.X, pady=5)
    
    # Створення та налаштування таблиці характеристик
    char_frame = ttk.LabelFrame(frame, text="Точкові характеристики", padding=(5, 5))
    char_frame.pack(fill='x', pady=10)
    
    char_table = ttk.Treeview(char_frame, columns=("value"), show="headings", height=10)
######################################################################
    char_table.heading("value", text="Значення")
    char_table.column("value", width=100, anchor="center")
    char_table.pack(fill=tk.X, pady=5)
    
    # Права сторона: вивід даних та статистик
    right_frame = tk.Frame(tab1)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Рамка для виводу даних
    data_frame = ttk.LabelFrame(right_frame, text="Дані", padding=(5, 5))
    data_frame.pack(fill='both', expand=True, pady=10)
    
    # Текстове поле для виводу даних з прокруткою
    data_scroll = tk.Scrollbar(data_frame)
    data_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    data_box = tk.Text(data_frame, height=20, width=50, yscrollcommand=data_scroll.set)
    data_box.pack(fill=tk.BOTH, expand=True)
    data_scroll.config(command=data_box.yview)
    
    # Рамка для виводу статистик
    stats_frame = ttk.LabelFrame(right_frame, text="Статистичні показники", padding=(5, 5))
    stats_frame.pack(fill='both', expand=True, pady=10)
    
    # Текстове поле для виводу статистик з прокруткою
    stats_scroll = tk.Scrollbar(stats_frame)
    stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    stats_box = tk.Text(stats_frame, height=10, width=50, yscrollcommand=stats_scroll.set)
    stats_box.pack(fill=tk.BOTH, expand=True)
    stats_scroll.config(command=stats_box.yview)
    
    # Зберігаємо всі важливі елементи для використання в інших функціях
    return {
        "info_text": info_text,
        "bin_count_var": bin_count_var,
        "lower_bound_var": lower_bound_var,
        "upper_bound_var": upper_bound_var,
        "data_box": data_box,
        "stats_box": stats_box,
        "char_table": char_table,
        "editing_buttons": editing_buttons,
        "plot_btn": plot_btn,
        "cdf_btn": cdf_btn
    }

def main():
    # Створення головного вікна
    root = tk.Tk()
    root.title("Статистичний аналіз даних")
    root.geometry("1200x700")
    
    # Створення вкладок
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)
    
    notebook.add(tab1, text="Основний аналіз")
    notebook.add(tab2, text="Функції розподілу")
    notebook.add(tab3, text="Емпірична функція розподілу")
    
    # Створення полотна для гістограми
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
    # Створення елементів для гістограми
    fig, hist_ax = plt.subplots(figsize=(6, 4))
    hist_canvas = FigureCanvasTkAgg(fig, tab1)
    hist_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Ініціалізація UI елементів
    ui_elements = create_ui_elements(tab1, tab2, tab3, hist_ax, hist_canvas)
    
    # Збереження важливих елементів у глобальних змінних для використання в інших модулях
    global stats_box, info_text
    stats_box = ui_elements["stats_box"]
    info_text = ui_elements["info_text"]
    
    # Запуск головного циклу програми
    root.mainloop()

if __name__ == "__main__":
    main()