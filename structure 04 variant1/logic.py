import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import skew, kurtosis, variation, median_abs_deviation, norm, t, chi2, sem, kstest
import pandas as pd

# Глобальні змінні для даних
values = np.array([])
original_values = np.array([])

# Змінні GUI
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
        data_list = [x.strip() for x in text.split(',') if x.strip()]
        new_values = []
        for item in data_list:
            try:
                new_values.append(float(item))
            except ValueError:
                new_values.append(np.nan)
        new_values = np.array(new_values)
        
        nan_mask = pd.isna(new_values)
        if np.any(nan_mask):
            mean_value = np.nanmean(new_values)
            if np.isnan(mean_value):
                raise ValueError("Усі значення є пропущеними. Неможливо обчислити середнє.")
            new_values = np.where(nan_mask, mean_value, new_values)
            messagebox.showinfo("Обробка даних", f"Замінено {np.sum(nan_mask)} пропущених значень середнім: {mean_value:.4f}")
        
        if not np.array_equal(values, new_values):
            values = new_values
            print("Оновлені значення з текстового поля:", values)
            update_statistics()
            update_characteristics()
            # Видаляємо виклик update_histogram()
        else:
            print("Дані не змінилися, пропускаємо оновлення.")
    except ValueError as e:
        messagebox.showerror("Помилка", f"Некоректний формат даних: {str(e)}")

def calculate_confidence_interval(data, statistic, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)  # Використовуємо sem безпосередньо
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)  # Використовуємо t.ppf
    if statistic == "mean":
        return mean - h, mean + h
    elif statistic == "variance":
        var = np.var(data, ddof=1)
        chi2_lower = chi2.ppf((1 - confidence) / 2, n - 1)  # Використовуємо chi2.ppf
        chi2_upper = chi2.ppf((1 + confidence) / 2, n - 1)
        return (n - 1) * var / chi2_upper, (n - 1) * var / chi2_lower
    elif statistic == "std":
        var_lower, var_upper = calculate_confidence_interval(data, "variance", confidence)
        return np.sqrt(var_lower), np.sqrt(var_upper)
    return None, None

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
    gui_objects['refresh_graph_button'].config(command=update_histogram)  # Прив’язуємо команду до нової кнопки

def load_data():
    global values, original_values
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls"), ("Text files", "*.txt")])
    if not file_path:
        messagebox.showwarning("Попередження", "Файл не вибрано")
        return
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            wait_time_col = "Час очікування (хв)"
            if wait_time_col not in df.columns:
                possible_cols = [col for col in df.columns if "час" in col.lower() or "wait" in col.lower()]
                if not possible_cols:
                    raise ValueError("У файлі немає стовпця з часом очікування. Перевірте структуру даних.")
                wait_time_col = possible_cols[0]
            
            df[wait_time_col] = df[wait_time_col].replace(r'^\s*$', np.nan, regex=True)
            df[wait_time_col] = pd.to_numeric(df[wait_time_col], errors='coerce')
            values = df[wait_time_col].to_numpy()
            
            print("Завантажені значення (до заміни):", values)
            
            nan_mask = pd.isna(values)
            if np.any(nan_mask):
                mean_value = np.nanmean(values)
                if np.isnan(mean_value):
                    raise ValueError("Усі значення в стовпці є пропущеними. Неможливо обчислити середнє.")
                values = np.where(nan_mask, mean_value, values)
                nan_count = np.sum(nan_mask)
                messagebox.showinfo("Обробка даних", f"Замінено {nan_count} пропущених значень середнім: {mean_value:.4f}")
            
            print("Значення після заміни:", values)
            
        else:
            with open(file_path, 'r') as file:
                data = file.read().replace(',', '.').strip()
            data_list = [x for x in data.split() if x]
            values = []
            for item in data_list:
                try:
                    values.append(float(item))
                except ValueError:
                    continue
            values = np.array(values)
            if len(values) == 0:
                raise ValueError("Дані порожні або некоректні")
        
        if len(values) == 0:
            raise ValueError("Дані порожні або некоректні")
        
        original_values = values.copy()
        update_statistics()
        update_characteristics()
        update_data_box()
        # Видаляємо виклик update_histogram()
        for btn in gui_objects['editing_buttons']:
            btn.config(state=tk.NORMAL)
        gui_objects['plot_btn'].config(state=tk.NORMAL)
        min_val, max_val = np.min(values), np.max(values)
        gui_objects['lower_bound_var'].set(str(min_val))
        gui_objects['upper_bound_var'].set(str(max_val))
        
        mean_wait_time = np.mean(values)
        recommendation = ("Рекомендується збільшити кількість операторів у пікові години, "
                        "а також розглянути впровадження IVR для автоматичних відповідей.") if mean_wait_time > 5 else "Поточна кількість операторів достатня."
        messagebox.showinfo("Аналіз", f"Середній час очікування: {mean_wait_time:.2f} хв\nРекомендація: {recommendation}")
    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося завантажити дані: {str(e)}")
        values = np.array([])
        original_values = np.array([])

def update_histogram():
    global values
    if len(values) == 0:
        return
    print("Дані для гістограми:", values)  # Логування для перевірки
    bin_count = gui_objects['bin_count_var'].get()
    if bin_count == 0:
        bin_count = int(np.sqrt(len(values)))
    
    # Очищаємо графік
    gui_objects['hist_ax'].clear()
    
    # Будуємо гістограму
    hist, bins, _ = gui_objects['hist_ax'].hist(values, bins=bin_count, color='blue', alpha=0.7, edgecolor='black', density=True)
    gui_objects['hist_ax'].set_title('Гістограма часу очікування та щільність')
    
    # Додаємо криву щільності
    mean, std = np.mean(values), np.std(values)
    x = np.linspace(min(values), max(values), 100)
    density = norm.pdf(x, mean, std)
    gui_objects['hist_ax'].plot(x, density, 'r-', label='Щільність')
    
    # Видаляємо код для нижньої і верхньої границь
    # Більше не додаємо axvline і відповідні елементи легенди
    gui_objects['hist_ax'].legend()  # Залишаємо легенду лише для "Щільність"
    
    # Масштабуємо графік
    x_min, x_max = np.min(values), np.max(values)
    x_range = x_max - x_min
    gui_objects['hist_ax'].set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)  # Додаємо 10% запасу з обох боків
    
    y_max = max(np.max(hist), np.max(density))  # Максимальна висота гістограми або кривої щільності
    gui_objects['hist_ax'].set_ylim(0, y_max * 1.1)  # Додаємо 10% запасу зверху
    
    # Примусово оновлюємо полотно
    gui_objects['hist_canvas'].draw()
    gui_objects['hist_canvas'].get_tk_widget().update()
    gui_objects['fig'].canvas.flush_events()
    
    # Оновлюємо інформаційний текст
    range_val = np.ptp(values)
    bin_width = range_val / bin_count
    gui_objects['info_text'].set(f'Кількість класів: {bin_count}\nКрок розбиття: {bin_width:.3f}\nРозмах: {range_val:.3f}\nКількість даних: {len(values)}')

def plot_distribution_functions():
    global values
    if len(values) == 0:
        return
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
    ax.set_xlabel('Час очікування (хв)')
    ax.set_ylabel('Ймовірність')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    
    canvas = FigureCanvasTkAgg(fig, master=gui_objects['tab2'])
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    ks_statistic, ks_pvalue = kstest(values, 'norm', args=(mean, std))  # Використовуємо kstest
    critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
    conclusion = "Розподіл нормальний" if ks_statistic < critical_value else "Розподіл не нормальний"
    ks_text = (f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
               f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
               f"Висновок: {conclusion}")
    info_frame = tk.Frame(gui_objects['tab2'])
    info_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=10, pady=10)
    info_label = tk.Label(info_frame, text=ks_text, justify=tk.LEFT)
    info_label.pack()

def update_statistics():
    global values
    if len(values) == 0:
        return
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)
    skewness = skew(values)
    excess_kurtosis = kurtosis(values, fisher=True)
    counter_kurtosis = 1 / (excess_kurtosis + 3) if excess_kurtosis + 3 != 0 else 0
    pearson_var = std_dev / mean if mean != 0 else 0
    nonparam_var = variation(values)
    mad_val = median_abs_deviation(values)
    med_val = np.median(values)

def update_data_box():
    global values
    gui_objects['data_box'].delete(1.0, tk.END)
    formatted_values = ', '.join([f"{val:.4f}" for val in values])
    gui_objects['data_box'].insert(tk.END, formatted_values)
    update_histogram()  # Гарантуємо оновлення гістограми

def standardize_data():
    global values
    if len(values) == 0:
        return
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    values = (values - mean) / std
    update_statistics()
    update_characteristics()
    update_data_box()
    messagebox.showinfo("Стандартизація", "Дані успішно стандартизовано")
    # Видаляємо виклик update_histogram()

def log_transform():
    global values
    if len(values) == 0:
        return
    if np.min(values) <= 0:
        messagebox.showerror("Помилка", "Логарифмування можливе тільки для додатних значень")
        return
    values = np.log(values)
    update_statistics()
    update_characteristics()
    update_data_box()
    messagebox.showinfo("Логарифмування", "Дані успішно логарифмовано")
    # Видаляємо виклик update_histogram()

def shift_data():
    global values
    if len(values) == 0:
        return
    shift_value = simpledialog.askfloat("Зсув даних", "Введіть значення зсуву:")
    if shift_value is not None:
        values = values + shift_value
        update_statistics()
        update_characteristics()
        update_data_box()
        messagebox.showinfo("Зсув", f"Дані успішно зсунуто на {shift_value}")
    # Видаляємо виклик update_histogram()

def remove_outliers():
    global values
    if len(values) == 0:
        return
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    original_count = len(values)
    values = values[(values >= lower_bound) & (values <= upper_bound)]
    removed_count = original_count - len(values)
    update_statistics()
    update_characteristics()
    update_data_box()
    update_histogram()  # Залишаємо виклик update_histogram()
    messagebox.showinfo("Видалення викидів", f"Видалено {removed_count} викидів")

def reset_data():
    global values, original_values
    if len(original_values) == 0:
        messagebox.showwarning("Попередження", "Немає початкових даних для скидання. Завантажте дані спочатку.")
        return
    
    values = original_values.copy()
    min_val, max_val = np.min(values), np.max(values)
    gui_objects['lower_bound_var'].set(str(min_val))
    gui_objects['upper_bound_var'].set(str(max_val))
    update_statistics()
    update_characteristics()
    update_data_box()
    for widget in gui_objects['tab2'].winfo_children():
        widget.destroy()
    messagebox.showinfo("Скидання", "Дані повернуто до початкового стану")
    # Видаляємо виклик update_histogram()

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
        filtered_values = original_values[(original_values >= lower) & (original_values <= upper)]
        if len(filtered_values) == 0:
            messagebox.showerror("Помилка", "За вказаними границями немає даних")
            return
        values = filtered_values
        update_statistics()
        update_characteristics()
        update_data_box()
        messagebox.showinfo("Границі", f"Застосовано границі: [{lower:.4f}, {upper:.4f}]")
    except ValueError:
        messagebox.showerror("Помилка", "Введіть числові значення для границь")
    # Видаляємо виклик update_histogram()