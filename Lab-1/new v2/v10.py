import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import skew, kurtosis, variation, median_abs_deviation, norm
import scipy.stats as stats

def load_data():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    try:
        with open(file_path, 'r') as file:
            data = file.read().replace(',', '.')
        global values, original_values
        values = np.array(list(map(float, data.split())))
        original_values = values.copy()  # Зберігаємо оригінальні значення
        update_histogram()
        update_statistics()
        update_characteristics()
        update_data_box()
        # Активуємо кнопки редагування
        for btn in editing_buttons:
            btn.config(state=tk.NORMAL)
        plot_btn.config(state=tk.NORMAL)
        cdf_btn.config(state=tk.NORMAL)
        # Оновлюємо значення верхньої та нижньої границі
        min_val, max_val = np.min(values), np.max(values)
        lower_bound_var.set(min_val)
        upper_bound_var.set(max_val)
    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося завантажити дані: {str(e)}")

def update_histogram():
    global values, bin_count_var, hist_ax
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

def plot_empirical_distribution():
    global values, hist_ax
    
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
        # Вертикальна лінія до наступного рівня (якщо це не остання точка)
        # if i < len(Y) - 1:
        #     emp_ax.plot([bin_gr[i+1], bin_gr[i+1]], [Y[i], Y[i+1]], color='green', linewidth=2)
    
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

def plot_distribution_functions():
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
    
    # Побудова графіка емпіричної функції розподілу
    # ax.step(x, y, label='Емпірична функція розподілу', where='post', color='blue')
    
    # Побудова графіка нормального розподілу
    ax.plot(x_theor, norm.cdf(x_theor, loc=mean, scale=std), label='Нормальний розподіл', color='red')
    
    # Побудова інших теоретичних розподілів
    # Логнормальний
    # if min(values) > 0:  # Логнормальний розподіл тільки для додатних значень
    #     ax.plot(x_theor, stats.lognorm.cdf(x_theor, s=np.std(np.log(values)), scale=np.exp(np.mean(np.log(values)))), 
    #             label='Логнормальний розподіл', color='green')
    
    # Експоненційний
    # if min(values) >= 0:  # Експоненційний розподіл тільки для невід'ємних значень
    #     ax.plot(x_theor, stats.expon.cdf(x_theor, loc=0, scale=mean), 
    #             label='Експоненційний розподіл', color='purple')
    
    # # Рівномірний
    # ax.plot(x_theor, stats.uniform.cdf(x_theor, loc=min(values), scale=max(values) - min(values)), 
    #         label='Рівномірний розподіл', color='orange')
    
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

def update_characteristics():
    global values
    
    # Очистити таблицю
    for item in char_table.get_children():
        char_table.delete(item)
    
    # Розрахунок всіх характеристик
    mean = np.mean(values)
    variance_biased = np.var(values, ddof=0)
    variance_unbiased = np.var(values, ddof=1)
    std_dev_biased = np.std(values, ddof=0)
    std_dev_unbiased = np.std(values, ddof=1)
    skewness = skew(values)
    excess_kurtosis = kurtosis(values, fisher=True)
    counter_kurtosis = 1 / (excess_kurtosis + 3) if excess_kurtosis + 3 != 0 else 0
    pearson_var = std_dev_unbiased / mean if mean != 0 else 0
    nonparam_var = variation(values)
    mad_val = median_abs_deviation(values)
    med_val = np.median(values)
    
    # Додаємо всі характеристики до таблиці
    characteristics = [
        ("Середнє", mean, mean),
        ("Дисперсія", variance_biased, variance_unbiased),
        ("Середньокв. відхилення", std_dev_biased, std_dev_unbiased),
        ("Асиметрія", skewness, skewness),
        ("Ексцес", excess_kurtosis, excess_kurtosis),
        ("Контрексцес", counter_kurtosis, counter_kurtosis),
        ("Варіація Пірсона", pearson_var, pearson_var),
        ("Непарам. коеф. вар.", nonparam_var, nonparam_var),
        ("MAD", mad_val, mad_val),
        ("Медіана", med_val, med_val)
    ]
    
    for char, biased, unbiased in characteristics:
        char_table.insert("", "end", values=(char, f"{biased:.4f}", f"{unbiased:.4f}"))

def update_data_box():
    global values
    # Очищаємо текстове поле
    data_box.delete(1.0, tk.END)
    # Додаємо дані
    formatted_values = ', '.join([f"{val:.4f}" for val in values])
    data_box.insert(tk.END, formatted_values)

def standardize_data():
    global values
    if len(values) == 0:
        return
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    values = (values - mean) / std
    
    update_histogram()
    update_statistics()
    update_characteristics()
    update_data_box()
    messagebox.showinfo("Стандартизація", "Дані успішно стандартизовано")

def log_transform():
    global values
    if len(values) == 0:
        return
    
    if np.min(values) <= 0:
        messagebox.showerror("Помилка", "Логарифмування можливе тільки для додатних значень")
        return
    
    values = np.log(values)
    
    update_histogram()
    update_statistics()
    update_characteristics()
    update_data_box()
    messagebox.showinfo("Логарифмування", "Дані успішно логарифмовано")

def shift_data():
    global values
    if len(values) == 0:
        return
    
    shift_value = simpledialog.askfloat("Зсув даних", "Введіть значення зсуву:")
    if shift_value is not None:
        values = values + shift_value
        
        update_histogram()
        update_statistics()
        update_characteristics()
        update_data_box()
        messagebox.showinfo("Зсув", f"Дані успішно зсунуто на {shift_value}")

def remove_outliers():
    global values
    if len(values) == 0:
        return
    
    # Метод ІQR (міжквартильний розмах)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    original_count = len(values)
    values = values[(values >= lower_bound) & (values <= upper_bound)]
    removed_count = original_count - len(values)
    
    update_histogram()
    update_statistics()
    update_characteristics()
    update_data_box()
    messagebox.showinfo("Видалення викидів", f"Видалено {removed_count} викидів")

def reset_data():
    global values, original_values
    if hasattr(globals(), 'original_values'):
        values = original_values.copy()
        
        # Оновлюємо значення верхньої та нижньої границі до початкових
        min_val, max_val = np.min(values), np.max(values)
        lower_bound_var.set(min_val)
        upper_bound_var.set(max_val)
        
        update_histogram()
        update_statistics()
        update_characteristics()
        update_data_box()
        messagebox.showinfo("Скидання", "Дані повернуто до початкового стану")

def apply_bounds():
    global values, original_values
    if len(values) == 0:
        return
    
    try:
        lower = float(lower_bound_var.get())
        upper = float(upper_bound_var.get())
        
        if lower >= upper:
            messagebox.showerror("Помилка", "Нижня границя має бути менше верхньої")
            return
        
        # Застосовуємо границі до оригінальних даних
        filtered_values = original_values[(original_values >= lower) & (original_values <= upper)]
        
        if len(filtered_values) == 0:
            messagebox.showerror("Помилка", "За вказаними границями немає даних")
            return
        
        values = filtered_values
        
        update_histogram()
        update_statistics()
        update_characteristics()
        update_data_box()
        messagebox.showinfo("Границі", f"Застосовано границі: [{lower:.4f}, {upper:.4f}]")
    except ValueError:
        messagebox.showerror("Помилка", "Введіть числові значення для границь")

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
notebook.add(tab3, text='ЕМпірична функція розподілу')

# Вкладка 1: Основний аналіз
frame = tk.Frame(tab1)
frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

# Завантаження даних
load_button = tk.Button(frame, text="Завантажити дані", command=load_data)
load_button.pack(fill=tk.X, pady=5)

# Гістограма
bin_label = tk.Label(frame, text="Введіть кількість класів для гістограми:")
bin_label.pack()

bin_count_var = tk.IntVar(value=0)
bin_entry = tk.Entry(frame, textvariable=bin_count_var)
bin_entry.pack()

update_button = tk.Button(frame, text="Оновити гістограму", command=update_histogram)
update_button.pack(fill=tk.X, pady=5)

# Інформація про дані
info_text = tk.StringVar()
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
lower_bound_var = tk.StringVar()
lower_entry = tk.Entry(lower_frame, textvariable=lower_bound_var)
lower_entry.pack(side=tk.LEFT, fill='x', expand=True)

# Верхня границя
upper_frame = tk.Frame(bounds_frame)
upper_frame.pack(fill='x', pady=2)
upper_label = tk.Label(upper_frame, text="Верхня границя:", width=15, anchor='w')
upper_label.pack(side=tk.LEFT)
upper_bound_var = tk.StringVar()
upper_entry = tk.Entry(upper_frame, textvariable=upper_bound_var)
upper_entry.pack(side=tk.LEFT, fill='x', expand=True)

# Кнопка застосування границь
apply_bounds_btn = tk.Button(bounds_frame, text="Застосувати границі", command=apply_bounds)
apply_bounds_btn.pack(fill=tk.X, pady=5)

# Секція редагування даних
edit_frame = ttk.LabelFrame(frame, text="Редагування даних", padding=(5, 5))
edit_frame.pack(fill='x', pady=10)

# Кнопки для редагування даних
standardize_btn = tk.Button(edit_frame, text="Стандартизувати", command=standardize_data, state=tk.DISABLED)
standardize_btn.pack(fill=tk.X, pady=2)

log_btn = tk.Button(edit_frame, text="Логарифмувати", command=log_transform, state=tk.DISABLED)
log_btn.pack(fill=tk.X, pady=2)

shift_btn = tk.Button(edit_frame, text="Зсунути", command=shift_data, state=tk.DISABLED)
shift_btn.pack(fill=tk.X, pady=2)

outliers_btn = tk.Button(edit_frame, text="Вилучити аномальні дані", command=remove_outliers, state=tk.DISABLED)
outliers_btn.pack(fill=tk.X, pady=2)

reset_btn = tk.Button(edit_frame, text="Скинути до початкових", command=reset_data, state=tk.DISABLED)
reset_btn.pack(fill=tk.X, pady=2)

# Список кнопок для активації/деактивації
editing_buttons = [standardize_btn, log_btn, shift_btn, outliers_btn, reset_btn, apply_bounds_btn]

# Додаємо кнопки для побудови функцій розподілу
plot_btn = tk.Button(frame, text="Побудувати функції розподілу", command=plot_distribution_functions, state=tk.DISABLED)
plot_btn.pack(fill=tk.X, pady=5)

cdf_btn = tk.Button(frame, text="Побудувати емпіричну функцію розподілу", command=plot_empirical_distribution, state=tk.DISABLED)
cdf_btn.pack(fill=tk.X, pady=5)

# Створення та налаштування таблиці характеристик
char_frame = ttk.LabelFrame(frame, text="Точкові характеристики", padding=(5, 5))
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
data_frame = ttk.LabelFrame(frame, text="Дані", padding=(5, 5))
data_frame.pack(fill='both', expand=True, pady=10)

data_box = tk.Text(data_frame, height=10, width=30, wrap=tk.WORD)
data_box.pack(fill='both', expand=True)
data_scroll = tk.Scrollbar(data_box, command=data_box.yview)
data_scroll.pack(side=tk.RIGHT, fill=tk.Y)
data_box.config(yscrollcommand=data_scroll.set)

# Графік на вкладці 1
fig, hist_ax = plt.subplots(figsize=(8, 6))
hist_canvas = FigureCanvasTkAgg(fig, master=tab1)
hist_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Запуск основного циклу
root.mainloop()