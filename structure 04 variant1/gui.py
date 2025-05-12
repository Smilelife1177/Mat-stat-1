import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def create_gui(root):
    # Створюємо notebook для вкладок
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # Створюємо дві вкладки
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)

    # Додаємо вкладки до notebook
    notebook.add(tab1, text='Основний аналіз')
    notebook.add(tab2, text='Функції розподілу')

    # Вкладка 1: Основний аналіз
    # Створюємо полотно з прокруткою
    canvas = tk.Canvas(tab1)
    scrollbar = tk.Scrollbar(tab1, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    # Налаштовуємо прокрутку
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    # Оновлюємо розміри прокрутки при зміні вмісту
    def configure_scroll(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scrollable_frame.bind("<Configure>", configure_scroll)

    # Завантаження даних
    load_button = tk.Button(scrollable_frame, text="Завантажити дані")
    load_button.pack(fill=tk.X, pady=5)

    # Гістограма
    bin_label = tk.Label(scrollable_frame, text="Введіть кількість класів для гістограми:")
    bin_label.pack()

    bin_count_var = tk.IntVar(value=0)
    bin_entry = tk.Entry(scrollable_frame, textvariable=bin_count_var)
    bin_entry.pack()

    update_button = tk.Button(scrollable_frame, text="Оновити гістограму")
    update_button.pack(fill=tk.X, pady=5)

    # Додаємо нову кнопку "Оновити графік"
    refresh_graph_button = tk.Button(scrollable_frame, text="Оновити графік")
    refresh_graph_button.pack(fill=tk.X, pady=5)

    # Інформація про дані
    info_text = tk.StringVar()
    info_label = tk.Label(scrollable_frame, textvariable=info_text, justify=tk.LEFT)
    info_label.pack()

    # Поля для рівня довіри та точності
    confidence_label = tk.Label(scrollable_frame, text="Рівень довіри (%):")
    confidence_label.pack()
    confidence_var = tk.DoubleVar(value=95.0)
    confidence_entry = tk.Entry(scrollable_frame, textvariable=confidence_var)
    confidence_entry.pack()

    precision_label = tk.Label(scrollable_frame, text="Точність (знаки після коми):")
    precision_label.pack()
    precision_var = tk.IntVar(value=4)
    precision_entry = tk.Entry(scrollable_frame, textvariable=precision_var)
    precision_entry.pack()

    # Секція встановлення границь
    bounds_frame = ttk.LabelFrame(scrollable_frame, text="Встановлення границь", padding=(5, 5))
    bounds_frame.pack(fill='x', pady=10)

    lower_frame = tk.Frame(bounds_frame)
    lower_frame.pack(fill='x', pady=2)
    lower_label = tk.Label(lower_frame, text="Нижня границя:", width=15, anchor='w')
    lower_label.pack(side=tk.LEFT)
    lower_bound_var = tk.StringVar()
    lower_entry = tk.Entry(lower_frame, textvariable=lower_bound_var)
    lower_entry.pack(side=tk.LEFT, fill='x', expand=True)

    upper_frame = tk.Frame(bounds_frame)
    upper_frame.pack(fill='x', pady=2)
    upper_label = tk.Label(upper_frame, text="Верхня границя:", width=15, anchor='w')
    upper_label.pack(side=tk.LEFT)
    upper_bound_var = tk.StringVar()
    upper_entry = tk.Entry(upper_frame, textvariable=upper_bound_var)
    upper_entry.pack(side=tk.LEFT, fill='x', expand=True)

    apply_bounds_btn = tk.Button(bounds_frame, text="Застосувати границі")
    apply_bounds_btn.pack(fill=tk.X, pady=5)

    # Секція редагування даних
    edit_frame = ttk.LabelFrame(scrollable_frame, text="Редагування даних", padding=(5, 5))
    edit_frame.pack(fill='x', pady=10)

    standardize_btn = tk.Button(edit_frame, text="Стандартизувати", state=tk.DISABLED)
    standardize_btn.pack(fill=tk.X, pady=2)

    log_btn = tk.Button(edit_frame, text="Логарифмувати", state=tk.DISABLED)
    log_btn.pack(fill=tk.X, pady=2)

    shift_btn = tk.Button(edit_frame, text="Зсунути", state=tk.DISABLED)
    shift_btn.pack(fill=tk.X, pady=2)

    outliers_btn = tk.Button(edit_frame, text="Вилучити аномальні дані", state=tk.DISABLED)
    outliers_btn.pack(fill=tk.X, pady=2)

    reset_btn = tk.Button(edit_frame, text="скидання кнопка НАЖМИНАМЕНЕ", state=tk.DISABLED)
    reset_btn.pack(fill=tk.X, pady=2)

    # Список кнопок для активації/деактивації
    editing_buttons = [standardize_btn, log_btn, shift_btn, outliers_btn, reset_btn, apply_bounds_btn]

    # Додаємо кнопки для побудови функцій розподілу
    plot_btn = tk.Button(scrollable_frame, text="Побудувати функції розподілу", state=tk.DISABLED)
    plot_btn.pack(fill=tk.X, pady=5)

    # cdf_btn більше не потрібна, але якщо хочете залишити місце, можете закоментувати
    cdf_btn = tk.Button(scrollable_frame, text=" ", state=tk.DISABLED)
    cdf_btn.pack(fill=tk.X, pady=5)

    # Створення та налаштування таблиці характеристик
    char_frame = ttk.LabelFrame(scrollable_frame, text="Точкові характеристики", padding=(5, 5))
    char_frame.pack(fill='x', pady=10)

    char_table = ttk.Treeview(char_frame, columns=("characteristic", "biased", "unbiased"), show="headings", height=10)
    char_table.heading("characteristic", text="Характеристика")
    char_table.heading("biased", text="Зсунена")
    char_table.heading("unbiased", text="Незсунена")
    char_table.column("characteristic", width=150)
    char_table.column("biased", width=100)
    char_table.column("unbiased", width=100)
    char_table.pack(fill='x')

    # Додаємо текстове поле для відображення і редагування даних
    data_frame = ttk.LabelFrame(scrollable_frame, text="Дані", padding=(5, 5))
    data_frame.pack(fill='both', expand=True, pady=10)

    data_box = tk.Text(data_frame, height=10, width=30, wrap=tk.WORD)
    data_box.pack(fill='both', expand=True)
    data_scroll = tk.Scrollbar(data_box, command=data_box.yview)
    data_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    data_box.config(yscrollcommand=data_scroll.set)

    save_btn = tk.Button(data_frame, text="Зберегти дані", state=tk.DISABLED)
    save_btn.pack(fill=tk.X, pady=2)
    editing_buttons.append(save_btn)

    # Графік на вкладці 1
    fig, hist_ax = plt.subplots(figsize=(8, 6))
    hist_canvas = FigureCanvasTkAgg(fig, master=tab1)
    hist_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Повертаємо словник із необхідними об’єктами
    return {
        'bin_count_var': bin_count_var,
        'info_text': info_text,
        'lower_bound_var': lower_bound_var,
        'upper_bound_var': upper_bound_var,
        'editing_buttons': editing_buttons,
        'plot_btn': plot_btn,
        'cdf_btn': cdf_btn,  # Закоментовано, якщо кнопка більше не потрібна
        'char_table': char_table,
        'data_box': data_box,
        'fig': fig,
        'hist_ax': hist_ax,
        'hist_canvas': hist_canvas,
        'tab2': tab2,
        'tab3': tab2,  # Залишаємо tab3 як tab2, якщо третя вкладка не потрібна
        'load_button': load_button,
        'update_button': update_button,
        'apply_bounds_btn': apply_bounds_btn,
        'standardize_btn': standardize_btn,
        'log_btn': log_btn,
        'shift_btn': shift_btn,
        'outliers_btn': outliers_btn,
        'reset_btn': reset_btn,
        'save_btn': save_btn,
        'confidence_var': confidence_var,
        'precision_var': precision_var,
        'refresh_graph_button': refresh_graph_button  # Додаємо нову кнопку
    }