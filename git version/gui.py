import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def create_gui(root):
    # Створюємо notebook для вкладок
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # Створюємо п'ять вкладок
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)
    tab5 = ttk.Frame(notebook)

    # Додаємо вкладки до notebook
    notebook.add(tab1, text='Основний аналіз')
    notebook.add(tab2, text='Функції розподілу')
    notebook.add(tab3, text='Експоненціальний розподіл')
    notebook.add(tab5, text='Гістограма та розподіли')

    # Вкладка 1: Основний аналіз
    canvas = tk.Canvas(tab1)
    scrollbar = tk.Scrollbar(tab1, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def configure_scroll(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    scrollable_frame.bind("<Configure>", configure_scroll)

    load_button = tk.Button(scrollable_frame, text="Завантажити дані")
    load_button.pack(fill=tk.X, pady=5)

    bin_label = tk.Label(scrollable_frame, text="Введіть кількість класів для гістограми:")
    bin_label.pack()

    bin_count_var = tk.IntVar(value=0)
    bin_entry = tk.Entry(scrollable_frame, textvariable=bin_count_var)
    bin_entry.pack()

    # Новий чекбокс для функції щільності
    density_var = tk.BooleanVar(value=False)
    density_check = tk.Checkbutton(scrollable_frame, text="Показати функцію щільності", variable=density_var)
    density_check.pack(pady=5)

    update_button = tk.Button(scrollable_frame, text="Оновити гістограму")
    update_button.pack(fill=tk.X, pady=5)

    refresh_graph_button = tk.Button(scrollable_frame, text="Оновити графік")
    refresh_graph_button.pack(fill=tk.X, pady=5)

    info_text = tk.StringVar()
    info_label = tk.Label(scrollable_frame, textvariable=info_text, justify=tk.LEFT)
    info_label.pack()

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

    outliers_skew_btn = tk.Button(edit_frame, text="Вилучити аномальні дані за асиметрією", state=tk.DISABLED)
    outliers_skew_btn.pack(fill=tk.X, pady=2)

    reset_btn = tk.Button(edit_frame, text="скидання кнопка", state=tk.DISABLED)
    reset_btn.pack(fill=tk.X, pady=2)

    editing_buttons = [standardize_btn, log_btn, shift_btn, outliers_btn, outliers_skew_btn, reset_btn]

    plot_btn = tk.Button(scrollable_frame, text="Побудувати функції розподілу", state=tk.DISABLED)
    plot_btn.pack(fill=tk.X, pady=5)

    cdf_btn = tk.Button(scrollable_frame, text="Побудувати експоненціальний розподіл", state=tk.DISABLED)
    cdf_btn.pack(fill=tk.X, pady=5)

    # call_type_btn = tk.Button(scrollable_frame, text="Аналіз за типами дзвінків", state=tk.DISABLED)
    # call_type_btn.pack(fill=tk.X, pady=5)

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

    fig, hist_ax = plt.subplots(figsize=(8, 6))
    hist_canvas = FigureCanvasTkAgg(fig, master=tab1)
    hist_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Вкладка 5: Гістограма та розподіли
    control_frame = ttk.Frame(tab5)
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    normal_var = tk.BooleanVar(value=False)
    normal_check = tk.Checkbutton(control_frame, text="Нормальний розподіл", variable=normal_var)
    normal_check.pack(side=tk.LEFT, padx=5)

    exponential_var = tk.BooleanVar(value=False)
    exponential_check = tk.Checkbutton(control_frame, text="Експоненціальний розподіл", variable=exponential_var)
    exponential_check.pack(side=tk.LEFT, padx=5)

    weibull_var = tk.BooleanVar(value=False)
    weibull_check = tk.Checkbutton(control_frame, text="Розподіл Вейбулла", variable=weibull_var)
    weibull_check.pack(side=tk.LEFT, padx=5)

    uniform_var = tk.BooleanVar(value=False)
    uniform_check = tk.Checkbutton(control_frame, text="Рівномірний розподіл", variable=uniform_var)
    uniform_check.pack(side=tk.LEFT, padx=5)

    rayleigh_var = tk.BooleanVar(value=False)
    rayleigh_check = tk.Checkbutton(control_frame, text="Розподіл Релея", variable=rayleigh_var)
    rayleigh_check.pack(side=tk.LEFT, padx=5)

    update_graph_btn = tk.Button(control_frame, text="Оновити графік")
    update_graph_btn.pack(side=tk.LEFT, padx=5)

    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    dist_canvas = FigureCanvasTkAgg(fig_dist, master=tab5)
    dist_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    info_frame_tab5 = ttk.Frame(tab5)
    info_frame_tab5.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
    dist_info_text = tk.Text(info_frame_tab5, height=15, width=60, wrap=tk.WORD)
    dist_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    dist_info_scroll = tk.Scrollbar(info_frame_tab5, command=dist_info_text.yview)
    dist_info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    dist_info_text.config(yscrollcommand=dist_info_scroll.set)

    return {
        'dist_info_text': dist_info_text,
        'bin_count_var': bin_count_var,
        'info_text': info_text,
        'editing_buttons': editing_buttons,
        'plot_btn': plot_btn,
        'cdf_btn': cdf_btn,
        # 'call_type_btn': call_type_btn,
        'char_table': char_table,
        'data_box': data_box,
        'fig': fig,
        'hist_ax': hist_ax,
        'hist_canvas': hist_canvas,
        'tab2': tab2,
        'tab3': tab3,
        'tab5': tab5,
        'load_button': load_button,
        'update_button': update_button,
        'standardize_btn': standardize_btn,
        'log_btn': log_btn,
        'shift_btn': shift_btn,
        'outliers_btn': outliers_btn,
        'outliers_skew_btn': outliers_skew_btn,
        'reset_btn': reset_btn,
        'save_btn': save_btn,
        'confidence_var': confidence_var,
        'precision_var': precision_var,
        'refresh_graph_button': refresh_graph_button,
        'normal_var': normal_var,
        'exponential_var': exponential_var,
        'weibull_var': weibull_var,
        'uniform_var': uniform_var,
        'rayleigh_var': rayleigh_var,
        'update_graph_btn': update_graph_btn,
        'fig_dist': fig_dist,
        'ax_dist': ax_dist,
        'dist_canvas': dist_canvas,
        'density_var': density_var
    }