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

def save_data():
    global values
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'w') as file:
            file.write('\n'.join(map(str, values)))
        messagebox.showinfo("Збереження", "Дані успішно збережено")

def update_from_data_box(event=None):
    global values
    try:
        text = gui_objects['data_box'].get(1.0, tk.END).strip()
        values = np.array([float(x) for x in text.split(',')])
        update_histogram()
        update_statistics()
        update_characteristics()
    except ValueError:
        messagebox.showerror("Помилка", "Некоректний формат даних")

def calculate_confidence_interval(data, statistic, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Стандартна похибка
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)  # Критичне значення t-розподілу
    if statistic == "mean":
        return mean - h, mean + h
    elif statistic == "variance":
        var = np.var(data, ddof=1)
        chi2_lower = stats.chi2.ppf((1 - confidence) / 2, n - 1)
        chi2_upper = stats.chi2.ppf((1 + confidence) / 2, n - 1)
        return (n - 1) * var / chi2_upper, (n - 1) * var / chi2_lower
    elif statistic == "std":
        var_lower, var_upper = calculate_confidence_interval(data, "variance", confidence)
        return np.sqrt(var_lower), np.sqrt(var_upper)
    return None, None  # Для інших характеристик поки не розраховуємо

def update_characteristics():
    global values
    confidence = gui_objects['confidence_var'].get() / 100
    precision = gui_objects['precision_var'].get()
    
    for item in gui_objects['char_table'].get_children():
        gui_objects['char_table'].delete(item)
    
    mean = np.mean(values)
    variance_unbiased = np.var(values, ddof=1)
    std_dev_unbiased = np.std(values, ddof=1)
    skewness = skew(values)
    excess_kurtosis = kurtosis(values, fisher=True)
    counter_kurtosis = 1 / (excess_kurtosis + 3) if excess_kurtosis + 3 != 0 else 0
    pearson_var = std_dev_unbiased / mean if mean != 0 else 0
    nonparam_var = variation(values)
    mad_val = median_abs_deviation(values)
    med_val = np.median(values)
    
    mean_ci_lower, mean_ci_upper = calculate_confidence_interval(values, "mean", confidence)
    var_ci_lower, var_ci_upper = calculate_confidence_interval(values, "variance", confidence)
    std_ci_lower, std_ci_upper = calculate_confidence_interval(values, "std", confidence)
    
    fmt = f".{precision}f"
    characteristics = [
        ("Середнє", mean, f"[{mean_ci_lower:{fmt}}, {mean_ci_upper:{fmt}}]"),
        ("Дисперсія", variance_unbiased, f"[{var_ci_lower:{fmt}}, {var_ci_upper:{fmt}}]"),
        ("Середньокв. відхилення", std_dev_unbiased, f"[{std_ci_lower:{fmt}}, {std_ci_upper:{fmt}}]"),
        ("Асиметрія", skewness, "-"),
        ("Ексцес", excess_kurtosis, "-"),
        ("Контрексцес", counter_kurtosis, "-"),
        ("Варіація Пірсона", pearson_var, "-"),
        ("Непарам. коеф. вар.", nonparam_var, "-"),
        ("MAD", mad_val, "-"),
        ("Медіана", med_val, "-")
    ]
    
    for char, value, ci in characteristics:
        gui_objects['char_table'].insert("", "end", values=(char, f"{value:{fmt}}", ci))

def initialize_logic(objects):
    global gui_objects
    gui_objects = objects
    
    # Прив'язуємо команди до кнопок
    gui_objects['save_btn'].config(command=save_data)
    gui_objects['data_box'].bind('<FocusOut>', update_from_data_box)
    gui_objects['load_button'].config(command=load_data)
    gui_objects['update_button'].config(command=update_histogram)
    gui_objects['apply_bounds_btn'].config(command=apply_bounds)
    gui_objects['standardize_btn'].config(command=standardize_data)
    gui_objects['log_btn'].config(command=log_transform)
    gui_objects['shift_btn'].config(command=shift_data)
    gui_objects['outliers_btn'].config(command=remove_outliers)
    gui_objects['reset_btn'].config(command=reset_data)
    gui_objects['plot_btn'].config(command=plot_distribution_functions)
    # Видаляємо прив'язку для cdf_btn, якщо вона більше не потрібна
    # gui_objects['cdf_btn'].config(command=plot_empirical_distribution)

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
    gui_objects['hist_ax'].hist(values, bins=bin_count, color='blue', alpha=0.7, edgecolor='black', density=True)
    gui_objects['hist_ax'].set_title('Гістограма та щільність')
    
    # Додаємо графік щільності
    mean, std = np.mean(values), np.std(values)
    x = np.linspace(min(values), max(values), 100)
    density = norm.pdf(x, mean, std)
    gui_objects['hist_ax'].plot(x, density, 'r-', label='Щільність')
    
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
    for widget in gui_objects['tab2'].winfo_children():
        widget.destroy()
    
    

    fig, ax = plt.subplots(figsize=(10, 6))
    n_bins = int(np.sqrt(len(values)))
    bin_dt, bin_gr = np.histogram(values, bins=n_bins)
    Y = np.cumsum(bin_dt) / len(values)
    ax.plot([bin_gr[0], bin_gr[0]], [0, Y[0]], color='green', linewidth=2, label='Емпіричний розподіл')
    for i in range(len(Y)):
        ax.plot([bin_gr[i], bin_gr[i+1]], [Y[i], Y[i]], color='green', linewidth=2)
    ax.plot([bin_gr[-1], bin_gr[-1]], [Y[-1], 1], color='green', linewidth=2)
    
    mean, std = np.mean(values), np.std(values)
    confidence = gui_objects['confidence_var'].get() / 100
    x_theor = np.linspace(min(values), max(values), 1000)
    cdf = norm.cdf(x_theor, mean, std)
    ax.plot(x_theor, cdf, label='Нормальний розподіл', color='red')
    
    # Довірчі інтервали для CDF
    n = len(values)
    epsilon = np.sqrt(1/(2*n) * np.log(2/(1-confidence)))
    ax.fill_between(x_theor, np.maximum(cdf - epsilon, 0), np.minimum(cdf + epsilon, 1), color='red', alpha=0.2, label='Довірчий інтервал')
    
    try:
        lower = float(gui_objects['lower_bound_var'].get())
        upper = float(gui_objects['upper_bound_var'].get())
        if lower < upper:
            ax.axvline(x=lower, color='r', linestyle='--', label='Нижня границя')
            ax.axvline(x=upper, color='g', linestyle='--', label='Верхня границя')
    except (ValueError, TypeError):
        pass
    
    ax.set_title('Порівняння емпіричного та нормального розподілів')
    ax.set_xlabel('Значення')
    ax.set_ylabel('Ймовірність')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    
    canvas = FigureCanvasTkAgg(fig, master=gui_objects['tab2'])
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    ks_statistic, ks_pvalue = stats.kstest(values, 'norm', args=(mean, std))
    ks_text = f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\np-значення: {ks_pvalue:.4f}"
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



