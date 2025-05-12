import numpy as np
import tkinter as tk  # Імпортуємо tkinter і створюємо псевдонім tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import skew, kurtosis, variation, median_abs_deviation, norm, stats

# Глобальні змінні для даних
values = np.array([])
original_values = np.array([])

# Змінні GUI, які будуть ініціалізовані пізніше
gui_objects = {}

def initialize_logic(objects):
    global gui_objects
    gui_objects = objects
    
    # Прив'язуємо команди до кнопок
    gui_objects['load_button'].config(command=load_data)
    gui_objects['update_button'].config(command=update_histogram)
    gui_objects['apply_bounds_btn'].config(command=apply_bounds)
    gui_objects['standardize_btn'].config(command=standardize_data)
    gui_objects['log_btn'].config(command=log_transform)
    gui_objects['shift_btn'].config(command=shift_data)
    gui_objects['outliers_btn'].config(command=remove_outliers)
    gui_objects['reset_btn'].config(command=reset_data)
    gui_objects['plot_btn'].config(command=plot_distribution_functions)
    gui_objects['cdf_btn'].config(command=plot_empirical_distribution)

def load_data():
    global values, original_values
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    try:
        with open(file_path, 'r') as file:
            data = file.read().replace(',', '.')
        values = np.array(list(map(float, data.split())))
        original_values = values.copy()  # Зберігаємо оригінальні значення
        update_histogram()
        update_statistics()
        update_characteristics()
        update_data_box()
        # Активуємо кнопки редагування
        for btn in gui_objects['editing_buttons']:
            btn.config(state=tk.NORMAL)  # Використовуємо tk.NORMAL
        gui_objects['plot_btn'].config(state=tk.NORMAL)
        gui_objects['cdf_btn'].config(state=tk.NORMAL)
        # Оновлюємо значення верхньої та нижньої границі
        min_val, max_val = np.min(values), np.max(values)
        gui_objects['lower_bound_var'].set(min_val)
        gui_objects['upper_bound_var'].set(max_val)
    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося завантажити дані: {str(e)}")

def update_histogram():
    global values
    bin_count = gui_objects['bin_count_var'].get()
    if bin_count == 0:
        bin_count = int(np.sqrt(len(values)))
    gui_objects['hist_ax'].clear()
    gui_objects['hist_ax'].hist(values, bins=bin_count, color='blue', alpha=0.7, edgecolor='black')
    gui_objects['hist_ax'].set_title('Гістограма')
    
    # Додаємо границі на графік, якщо вони встановлені
    try:
        lower = float(gui_objects['lower_bound_var'].get())
        upper = float(gui_objects['upper_bound_var'].get())
        if lower < upper:
            gui_objects['hist_ax'].axvline(x=lower, color='r', linestyle='--', label='Нижня границя')
            gui_objects['hist_ax'].axvline(x=upper, color='g', linestyle='--', label='Верхня границя')
            gui_objects['hist_ax'].legend()
    except (ValueError, TypeError):
        pass
    
    gui_objects['hist_canvas'].draw()
    
    range_val = np.ptp(values)
    bin_width = range_val / bin_count
    gui_objects['info_text'].set(f'Кількість класів: {bin_count}\nКрок розбиття: {bin_width:.3f}\nРозмах: {range_val:.3f}\nКількість даних: {len(values)}')

def plot_empirical_distribution():
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
        emp_ax.plot([bin_gr[i], bin_gr[i+1]], [Y[i], Y[i]], color='green', linewidth=2)
    
    # Додаємо останню горизонтальну лінію
    emp_ax.plot([bin_gr[-1], bin_gr[-1]], [Y[-1], 1], color='green', linewidth=2)
    
    # Додаємо границі на графік, якщо вони встановлені
    try:
        lower = float(gui_objects['lower_bound_var'].get())
        upper = float(gui_objects['upper_bound_var'].get())
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
    for widget in gui_objects['tab3'].winfo_children():
        widget.destroy()
    
    emp_canvas = FigureCanvasTkAgg(plt.gcf(), master=gui_objects['tab3'])
    emp_canvas.draw()
    emp_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def plot_distribution_functions():
    global values
    
    # Очистимо вміст вкладки 2
    for widget in gui_objects['tab2'].winfo_children():
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
        lower = float(gui_objects['lower_bound_var'].get())
        upper = float(gui_objects['upper_bound_var'].get())
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
    canvas = FigureCanvasTkAgg(fig, master=gui_objects['tab2'])
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Додаємо тест Колмогорова-Смірнова для перевірки нормальності
    ks_statistic, ks_pvalue = stats.kstest(values, 'norm', args=(mean, std))
    ks_text = f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\np-значення: {ks_pvalue:.4f}"
    
    # Додаємо тест Шапіро-Вілка для перевірки нормальності
    if len(values) <= 5000:
        shapiro_statistic, shapiro_pvalue = stats.shapiro(values)
        ks_text += f"\n\nТест Шапіро-Вілка:\nСтатистика: {shapiro_statistic:.4f}\np-значення: {shapiro_pvalue:.4f}"
    
    # Додаємо тест Андерсона-Дарлінга
    ad_result = stats.anderson(values, 'norm')
    ks_text += f"\n\nТест Андерсона-Дарлінга:\nСтатистика: {ad_result.statistic:.4f}"
    
    # Відображення результатів тестів
    info_frame = tk.Frame(gui_objects['tab2'])
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
    for item in gui_objects['char_table'].get_children():
        gui_objects['char_table'].delete(item)
    
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
        gui_objects['char_table'].insert("", "end", values=(char, f"{biased:.4f}", f"{unbiased:.4f}"))

def update_data_box():
    global values
    # Очищаємо текстове поле
    gui_objects['data_box'].delete(1.0, tk.END)
    # Додаємо дані
    formatted_values = ', '.join([f"{val:.4f}" for val in values])
    gui_objects['data_box'].insert(tk.END, formatted_values)

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
        gui_objects['lower_bound_var'].set(min_val)
        gui_objects['upper_bound_var'].set(max_val)
        
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
        lower = float(gui_objects['lower_bound_var'].get())
        upper = float(gui_objects['upper_bound_var'].get())
        
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



