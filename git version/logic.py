# logic.py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, variation, median_abs_deviation, norm, t, chi2, sem, kstest, expon, weibull_min, rayleigh, uniform, chi2_contingency, ttest_1samp
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Глобальні змінні для даних
values = np.array([])
original_values = np.array([])
call_types = np.array([])
original_call_types = np.array([])

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
    global values, call_types
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
            call_types = np.array(['Невідомо'] * len(values))
            print("Оновлені значення з текстового поля:", values)
            update_statistics()
            update_characteristics()
            update_data_box()
            update_distribution_plot()  # Оновлюємо графік розподілів
        else:
            print("Дані не змінилися, пропускаємо оновлення.")
    except ValueError as e:
        messagebox.showerror("Помилка", f"Некоректний формат даних: {str(e)}")

def calculate_confidence_interval(data, statistic, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    if statistic == "mean":
        return mean - h, mean + h
    elif statistic == "variance":
        var = np.var(data, ddof=1)
        chi2_lower = chi2.ppf((1 - confidence) / 2, n - 1)
        chi2_upper = chi2.ppf((1 + confidence) / 2, n - 1)
        return (n - 1) * var / chi2_upper, (n - 1) * var / chi2_lower
    elif statistic == "std":
        var_lower, var_upper = calculate_confidence_interval(data, "variance", confidence)
        return np.sqrt(var_lower), np.sqrt(var_upper)
    return None, None

def update_characteristics():
    global values
    if len(values) == 0:
        return
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
    update_histogram()
    update_distribution_plot()  # Оновлюємо графік розподілів

def load_data():
    global values, original_values, call_types, original_call_types
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls"), ("Text files", "*.txt")])
    if not file_path:
        messagebox.showwarning("Попередження", "Файл не вибрано")
        return
    try:
        if file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            wait_time_col = "Час очікування (хв)"
            call_type_col = "Тип дзвінка"
            
            if wait_time_col not in df.columns:
                possible_cols = [col for col in df.columns if "час" in col.lower() or "wait" in col.lower()]
                if not possible_cols:
                    raise ValueError("У файлі немає стовпця з часом очікування. Перевірте структуру даних.")
                wait_time_col = possible_cols[0]
            
            if call_type_col not in df.columns:
                possible_type_cols = [col for col in df.columns if "тип" in col.lower() or "type" in col.lower()]
                if not possible_type_cols:
                    messagebox.showwarning("Попередження", "Стовпець 'Тип дзвінка' не знайдено. Аналіз за типами недоступний.")
                    call_types = np.array(['Невідомо'] * len(df))
                else:
                    call_type_col = possible_type_cols[0]
            
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
            
            if call_type_col in df.columns:
                call_types = df[call_type_col].fillna('Невідомо').to_numpy()
            else:
                call_types = np.array(['Невідомо'] * len(values))
            
            print("Типи дзвінків:", call_types)
            
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
            call_types = np.array(['Невідомо'] * len(values))
            if len(values) == 0:
                raise ValueError("Дані порожні або некоректні")
        
        if len(values) == 0:
            raise ValueError("Дані порожні або некоректні")
        
        original_values = values.copy()
        original_call_types = call_types.copy()
        update_statistics()
        update_characteristics()
        update_data_box()
        for btn in gui_objects['editing_buttons']:
            btn.config(state=tk.NORMAL)
        gui_objects['plot_btn'].config(state=tk.NORMAL)
        gui_objects['cdf_btn'].config(state=tk.NORMAL)
        gui_objects['call_type_btn'].config(state=tk.NORMAL)
        
        mean_wait_time = np.mean(values)
        recommendation = ("Рекомендується збільшити кількість операторів у пікові години, "
                        "а також розглянути впровадження IVR для автоматичних відповідей.") if mean_wait_time > 5 else "Поточна кількість операторів достатня."
        messagebox.showinfo("Аналіз", f"Середній час очікування: {mean_wait_time:.2f} хв\nРекомендація: {recommendation}")
    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося завантажити дані: {str(e)}")
        values = np.array([])
        original_values = np.array([])
        call_types = np.array([])
        original_call_types = np.array([])

def update_histogram():
    global values 
    if len(values) == 0:
        return
    print("Дані для гістограми:", values)
    
    bin_count = gui_objects['bin_count_var'].get()
    if bin_count == 0:
        n = len(values)
        if n < 100:
            m = int(np.sqrt(n))
            bin_count = m if m % 2 != 0 else m - 1
        else:
            m = int(np.cbrt(n))
            bin_count = m if m % 2 != 0 else m - 1
    
    gui_objects['hist_ax'].clear()
    
    hist, bins, _ = gui_objects['hist_ax'].hist(values, bins=bin_count, color='blue', alpha=0.7, edgecolor='black', density=True)
    gui_objects['hist_ax'].set_title('Гістограма часу очікування та щільність')
    
    mean, std = np.mean(values), np.std(values)
    x = np.linspace(min(values), max(values), 100)
    density = norm.pdf(x, mean, std)
    gui_objects['hist_ax'].plot(x, density, 'r-', label='Щільність')
    
    gui_objects['hist_ax'].legend()
    
    x_min, x_max = np.min(values), np.max(values)
    x_range = x_max - x_min if x_max != x_min else 1
    gui_objects['hist_ax'].set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    
    y_max = max(np.max(hist), np.max(density))
    gui_objects['hist_ax'].set_ylim(0, y_max * 1.1)
    
    gui_objects['hist_canvas'].draw()
    
    range_val = np.ptp(values)
    bin_width = range_val / bin_count
    gui_objects['info_text'].set(f'Кількість класів: {bin_count}\nКрок розбиття: {bin_width:.3f}\nРозмах: {range_val:.3f}\nКількість даних: {len(values)}')

def pearson_chi2_test(data, dist_name, params, bins):
    """Виконує тест Пірсона для перевірки відповідності розподілу."""
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    expected = []
    n = len(data)
    for i in range(len(bin_edges) - 1):
        if dist_name == 'norm':
            p = norm.cdf(bin_edges[i + 1], *params) - norm.cdf(bin_edges[i], *params)
        elif dist_name == 'expon':
            p = expon.cdf(bin_edges[i + 1], *params) - expon.cdf(bin_edges[i], *params)
        elif dist_name == 'weibull':
            p = weibull_min.cdf(bin_edges[i + 1], *params) - weibull_min.cdf(bin_edges[i], *params)
        elif dist_name == 'uniform':
            p = uniform.cdf(bin_edges[i + 1], *params) - uniform.cdf(bin_edges[i], *params)
        elif dist_name == 'rayleigh':
            p = (1 - np.exp(-bin_edges[i + 1]**2 / (2 * params[0]**2))) - (1 - np.exp(-bin_edges[i]**2 / (2 * params[0]**2)))
        expected.append(p * n)
    expected = np.array(expected)
    observed = hist
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = len(hist) - 1 - len(params)
    # p_value = 1 - chi2.cdf(chi2_stat'>";
    # df)
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return chi2_stat, p_value, df

def estimate_weibull_params(data):
    """Оцінка параметрів розподілу Вейбулла методом максимальної правдоподібності."""
    def weibull_loglik(params):
        c, scale = params
        if c <= 0 or scale <= 0:
            return np.inf
        return -np.sum(weibull_min.logpdf(data, c, loc=0, scale=scale))
    
    initial_guess = [1.0, np.mean(data)]
    result = minimize(weibull_loglik, initial_guess, method='Nelder-Mead')
    if result.success:
        c, scale = result.x
        std_err_c = np.sqrt(result.hess_inv[0, 0]) if hasattr(result, 'hess_inv') else np.nan
        std_err_scale = np.sqrt(result.hess_inv[1, 1]) if hasattr(result, 'hess_inv') else np.nan
        return c, scale, std_err_c, std_err_scale
    return np.nan, np.nan, np.nan, np.nan

def plot_distribution_functions():
    global values
    if len(values) == 0:
        messagebox.showwarning("Попередження", "Немає даних для побудови функцій розподілу")
        return
    
    for widget in gui_objects['tab2'].winfo_children():
        widget.destroy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_bins = gui_objects['bin_count_var'].get()
    if n_bins == 0:
        n_bins = int(np.sqrt(len(values)))
    
    bin_dt, bin_gr = np.histogram(values, bins=n_bins)
    Y = np.cumsum(bin_dt) / len(values)
    # Початок з першого інтервалу, прибираємо вертикальний стрибок на початку
    for i in range(len(Y)):
        ax.plot([bin_gr[i], bin_gr[i+1]], [Y[i], Y[i]], color='green', linewidth=2, label='Емпіричний розподіл' if i == 0 else "")
    ax.plot([bin_gr[-1], bin_gr[-1]], [Y[-1], 1], color='green', linewidth=2)
    
    mean, std = np.mean(values), np.std(values)
    confidence = gui_objects['confidence_var'].get() / 100
    
    x_min, x_max = np.min(values), np.max(values)
    x_range = x_max - x_min if x_max != x_min else 1
    x_lower = x_min - 0.1 * x_range
    x_upper = x_max + 0.1 * x_range
    
    x_theor = np.linspace(x_lower, x_upper, 1000)
    cdf = norm.cdf(x_theor, mean, std)
    ax.plot(x_theor, cdf, label=f'Нормальний розподіл (μ={mean:.4f}, σ={std:.4f})', color='red')
    
    n = len(values)
    epsilon = np.sqrt(1/(2*n) * np.log(2/(1-confidence)))
    ax.fill_between(x_theor, np.maximum(cdf - epsilon, 0), np.minimum(cdf + epsilon, 1), 
                    color='red', alpha=0.2, label='Довірчий інтервал')
    
    ax.set_title('Порівняння емпіричного та нормального розподілів')
    ax.set_xlabel('Час очікування (хв)')
    ax.set_ylabel('Ймовірність')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(-0.05, 1.05)
    
    canvas = FigureCanvasTkAgg(fig, master=gui_objects['tab2'])
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    ks_statistic, ks_pvalue = kstest(values, 'norm', args=(mean, std))
    chi2_stat, chi2_p, df = pearson_chi2_test(values, 'norm', (mean, std), n_bins)
    critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(n)
    conclusion = "Розподіл нормальний" if ks_statistic < critical_value else "Розподіл не нормальний"
    info_text = (f"Нормальний розподіл:\nμ={mean:.4f} (±{sem(values):.4f})\nσ={std:.4f}\n"
                 f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
                 f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
                 f"Висновок: {conclusion}\n"
                 f"Тест Пірсона:\nχ²={chi2_stat:.4f}, p={chi2_p:.4f}, df={df}")
    info_frame = tk.Frame(gui_objects['tab2'])
    info_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=10, pady=10)
    info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
    info_label.pack()

def plot_exponential_distribution():
    global values
    if len(values) == 0:
        messagebox.showwarning("Попередження", "Немає даних для побудови експоненціального розподілу")
        return
    
    if np.any(values < 0):
        messagebox.showerror("Помилка", "Експоненціальний розподіл можливий лише для невід’ємних значень")
        return
    
    for widget in gui_objects['tab3'].winfo_children():
        widget.destroy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    empirical_probs = np.arange(1, n + 1) / (n + 1)
    y_values = -np.log(1 - empirical_probs)
    
    y_max = np.max(y_values)
    if y_max == 0:
        messagebox.showerror("Помилка", "Максимальне значення y дорівнює нулю. Неможливо нормалізувати.")
        return
    y_values_normalized = y_values / y_max
    
    mean = np.mean(values)
    if mean == 0:
        messagebox.showerror("Помилка", "Середнє значення дорівнює нулю. Неможливо оцінити параметр.")
        return
    lambda_param = 1 / mean
    
    ax.scatter(sorted_values, y_values_normalized, color='green', label='Дані', s=50)
    
    x_theor = np.linspace(0, np.max(sorted_values) * 1.2, 100)
    y_theor = lambda_param * x_theor
    y_theor_normalized = y_theor / y_max
    ax.plot(x_theor, y_theor_normalized, color='blue', linestyle='--', label=f'Експоненціальний розподіл (λ={lambda_param:.4f})')
    
    x_min, x_max = np.min(values), np.max(values)
    x_range = x_max - x_min if x_max != x_min else 1
    x_lower = max(0, x_min - 0.1 * x_range)
    x_upper = x_max + 0.1 * x_range
    
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(0, 1)  # Встановлюємо межі y-осі від 0 до 1
    
    ax.set_title('Імовірнісна сітка експоненціального розподілу')
    ax.set_xlabel('Значення (Час очікування, хв)')
    ax.set_ylabel('Нормалізоване -ln(1 - F(x))')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    canvas = FigureCanvasTkAgg(fig, master=gui_objects['tab3'])
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    confidence = gui_objects['confidence_var'].get() / 100
    n_bins = gui_objects['bin_count_var'].get() or int(np.sqrt(len(values)))
    ks_statistic, ks_pvalue = kstest(values, 'expon', args=(0, mean))
    chi2_stat, chi2_p, df = pearson_chi2_test(values, 'expon', (0, mean), n_bins)
    critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(len(values))
    conclusion = "Розподіл відповідає експоненціальному" if ks_statistic < critical_value else "Розподіл не відповідає експоненціальному"
    info_text = (f"Експоненціальний розподіл:\nλ={lambda_param:.4f} (±{1/(np.sqrt(n)*mean):.4f})\n"
                 f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
                 f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
                 f"Висновок: {conclusion}\n"
                 f"Тест Пірсона:\nχ²={chi2_stat:.4f}, p={chi2_p:.4f}, df={df}")
    info_frame = tk.Frame(gui_objects['tab3'])
    info_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=10, pady=10)
    info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
    info_label.pack()

def analyze_call_types():
    global values, call_types
    if len(values) == 0 or len(call_types) == 0:
        messagebox.showwarning("Попередження", "Немає даних для аналізу за типами дзвінків")
        return
    
    for widget in gui_objects['tab4'].Winfo_children():
        widget.destroy()
    
    unique_types = np.unique(call_types)
    
    info_frame = tk.Frame(gui_objects['tab4'])
    info_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    table_frame = ttk.LabelFrame(info_frame, text="Характеристики за типами дзвінків", padding=(5, 5))
    table_frame.pack(fill='x', pady=5)
    
    char_table = ttk.Treeview(table_frame, columns=("type", "count", "mean", "median", "std"), show="headings")
    char_table.heading("type", text="Тип дзвінка")
    char_table.heading("count", text="Кількість")
    char_table.heading("mean", text="Середнє (хв)")
    char_table.heading("median", text="Медіана (хв)")
    char_table.heading("std", text="Стд. відхилення")
    char_table.column("type", width=150)
    char_table.column("count", width=100)
    char_table.column("mean", width=100)
    char_table.column("median", width=100)
    char_table.column("std", width=100)
    char_table.pack(fill='x')
    
    recommendations = []
    precision = gui_objects['precision_var'].get()
    fmt = f".{precision}f"
    
    for call_type in unique_types:
        mask = call_types == call_type
        type_values = values[mask]
        if len(type_values) == 0:
            continue
        count = len(type_values)
        mean = np.mean(type_values)
        median = np.median(type_values)
        std = np.std(type_values, ddof=1)
        
        char_table.insert("", "end", values=(call_type, str(count), f"{mean:{fmt}}", f"{median:{fmt}}", f"{std:{fmt}}"))
        
        if mean > 5:
            recommendations.append(f"Для типу '{call_type}': середній час очікування ({mean:.2f} хв) перевищує 5 хвилин. "
                                 f"Рекомендується додати операторів або впровадити IVR.")
        else:
            recommendations.append(f"Для типу '{call_type}': середній час очікування ({mean:.2f} хв) у нормі. "
                                 f"Оптимізація не потрібна.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, call_type in enumerate(unique_types):
        mask = call_types == call_type
        type_values = values[mask]
        if len(type_values) == 0:
            continue
        bin_count = max(int(np.sqrt(len(type_values))), 1)
        ax.hist(type_values, bins=bin_count, alpha=0.5, label=call_type, color=colors[i % len(colors)], 
                density=True, edgecolor='black')
    
    x_min, x_max = np.min(values), np.max(values)
    x_range = x_max - x_min if x_max != x_min else 1
    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    
    ax.set_title('Гістограми часу очікування за типами дзвінків')
    ax.set_xlabel('Час очікування (хв)')
    ax.set_ylabel('Щільність')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    canvas = FigureCanvasTkAgg(fig, master=info_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    rec_frame = ttk.LabelFrame(info_frame, text="Рекомендації", padding=(5, 5))
    rec_frame.pack(fill='x', pady=5)
    rec_text = "\n".join(recommendations)
    rec_label = ttk.Label(rec_frame, text=rec_text, justify=tk.LEFT)
    rec_label.pack()

def standardize_data():
    global values, call_types
    if len(values) == 0:
        messagebox.showwarning("Попередження", "Немає даних для стандартизації")
        return
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if std == 0:
        messagebox.showerror("Помилка", "Стандартне відхилення дорівнює нулю. Неможливо стандартизувати.")
        return
    values = (values - mean) / std
    call_types = np.array(['Невідомо'] * len(values))
    update_statistics()
    update_characteristics()
    update_data_box()
    messagebox.showinfo("Стандартизація", "Дані успішно стандартизовано")

def log_transform():
    global values, call_types
    if len(values) == 0:
        messagebox.showwarning("Попередження", "Немає даних для логарифмування")
        return
    if np.min(values) <= 0:
        messagebox.showerror("Помилка", "Логарифмування можливе тільки для додатних значень")
        return
    values = np.log(values)
    call_types = np.array(['Невідомо'] * len(values))
    update_statistics()
    update_characteristics()
    update_data_box()
    messagebox.showinfo("Логарифмування", "Дані успішно логарифмовано")

def shift_data():
    global values, call_types
    if len(values) == 0:
        messagebox.showwarning("Попередження", "Немає даних для зсуву")
        return
    shift_value = simpledialog.askfloat("Зсув даних", "Введіть значення зсуву:")
    if shift_value is not None:
        values = values + shift_value
        call_types = np.array(['Невідомо'] * len(values))
        update_statistics()
        update_characteristics()
        update_data_box()
        messagebox.showinfo("Зсув", f"Дані успішно зсунуто на {shift_value}")

def remove_outliers():
    global values, call_types
    if len(values) == 0:
        messagebox.showwarning("Попередження", "Немає даних для обробки")
        return

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    if std == 0:
        messagebox.showerror("Помилка", "Стандартне відхилення дорівнює нулю. Неможливо виявити викиди.")
        return
    
    z_scores = np.abs((values - mean) / std)
    threshold = 3
    
    outlier_indices = np.where(z_scores > threshold)[0]
    
    if len(outlier_indices) == 0:
        messagebox.showinfo("Аномалії", "Аномальних значень не знайдено")
        return
    
    outlier_info = [f"Індекс: {i}, Значення: {values[i]:.4f}, Z-оцінка: {z_scores[i]:.4f}" 
                    for i in outlier_indices]
    
    dialog = tk.Toplevel()
    dialog.title("Підтвердження видалення аномалій")
    dialog.geometry("400x300")
    
    tk.Label(dialog, text="Виявлені аномалії. Оберіть, які видалити:").pack(pady=5)
    
    listbox = tk.Listbox(dialog, selectmode=tk.MULTIPLE, height=10)
    for info in outlier_info:
        listbox.insert(tk.END, info)
    listbox.pack(pady=5, fill=tk.BOTH, expand=True)
    
    selected_indices = []
    
    def confirm():
        selected_indices.extend([outlier_indices[int(i)] for i in listbox.curselection()])
        dialog.destroy()
    
    def cancel():
        dialog.destroy()
    
    tk.Button(dialog, text="Видалити обрані", command=confirm).pack(pady=5)
    tk.Button(dialog, text="Скасувати", command=cancel).pack(pady=5)
    
    dialog.grab_set()
    dialog.wait_window()
    
    if selected_indices:
        original_count = len(values)
        values = np.delete(values, selected_indices)
        call_types = np.delete(call_types, selected_indices)
        removed_count = original_count - len(values)
        
        update_statistics()
        update_characteristics()
        update_data_box()
        update_histogram()
        update_distribution_plot()
        messagebox.showinfo("Видалення викидів", f"Видалено {removed_count} викидів")
    else:
        messagebox.showinfo("Видалення викидів", "Жодних викидів не видалено")

def reset_data():
    global values, original_values, call_types, original_call_types
    if len(original_values) == 0:
        messagebox.showwarning("Попередження", "Немає початкових даних для скидання. Завантажте дані спочатку.")
        return
    
    values = original_values.copy()
    call_types = original_call_types.copy()
    update_statistics()
    update_characteristics()
    update_data_box()
    for widget in gui_objects['tab2'].winfo_children():
        widget.destroy()
    messagebox.showinfo("Скидання", "Дані повернуто до початкового стану")

def update_distribution_plot():
    global values
    if len(values) == 0:
        messagebox.showwarning("Попередження", "Немає даних для побудови розподілів")
        return

    gui_objects['ax_dist'].clear()
    results_text = []  # To store text for dist_info_text widget

    # Гістограма
    bin_count = gui_objects['bin_count_var'].get()
    if bin_count == 0:
        n = len(values)
        if n < 100:
            m = int(np.sqrt(n))
            bin_count = m if m % 2 != 0 else m - 1
        else:
            m = int(np.cbrt(n))
            bin_count = m if m % 2 != 0 else m - 1

    hist, bins, _ = gui_objects['ax_dist'].hist(values, bins=bin_count, color='blue', alpha=0.7, edgecolor='black', density=True, label='Гістограма')
    max_density = np.max(hist)

    confidence = gui_objects['confidence_var'].get() / 100
    n = len(values)
    x_min, x_max = np.min(values), np.max(values)
    x_range = x_max - x_min if x_max != x_min else 1
    x_lower = x_min - 0.1 * x_range
    x_upper = x_max + 0.1 * x_range
    x_theor = np.linspace(x_lower, x_upper, 1000)

    # Calculate critical value for KS test once
    critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(n)

    # Helper function for confidence bands
    def add_confidence_band(cdf, ax, x, label):
        epsilon = np.sqrt(1/(2*n) * np.log(2/(1-confidence)))
        ax.fill_between(x, np.maximum(cdf - epsilon, 0), np.minimum(cdf + epsilon, 1),
                        color='gray', alpha=0.2, label=f'Довірчий інтервал {label}')

    # Helper function for Pearson's Chi-Square test
    def pearson_chi2_test(data, dist, params, bins):
        hist, bin_edges = np.histogram(data, bins=bins, density=False)
        expected = []
        for i in range(len(bin_edges)-1):
            if dist == 'norm':
                p = norm.cdf(bin_edges[i+1], *params) - norm.cdf(bin_edges[i], *params)
            elif dist == 'expon':
                p = expon.cdf(bin_edges[i+1], *params) - expon.cdf(bin_edges[i], *params)
            elif dist == 'weibull_min':
                p = weibull_min.cdf(bin_edges[i+1], *params) - weibull_min.cdf(bin_edges[i], *params)
            elif dist == 'uniform':
                p = uniform.cdf(bin_edges[i+1], *params) - uniform.cdf(bin_edges[i], *params)
            elif dist == 'rayleigh':
                p = rayleigh.cdf(bin_edges[i+1], *params) - rayleigh.cdf(bin_edges[i], *params)
            expected.append(p * n)
        expected = np.array(expected)
        # Ensure no zero expected frequencies and at least 5 for validity
        expected = np.where(expected < 5, 5, expected)
        chi2_stat, p_value = chi2_contingency([hist, expected])[0:2]
        return chi2_stat, p_value

    # Helper function for T-test bootstrap
    def t_test_bootstrap(data, sample_sizes=[20, 50, 100, 400, 1000, 2000, 5000], bootstrap_samples=1000):
        results = {}
        pop_mean = np.mean(data)  # Hypothesized population mean
        for n in sample_sizes:
            t_stats = []
            for _ in range(bootstrap_samples):
                sample = np.random.choice(data, size=n, replace=True)
                t_stat, _ = ttest_1samp(sample, pop_mean)
                if not np.isnan(t_stat):
                    t_stats.append(t_stat)
            if t_stats:
                mean_t = np.mean(t_stats)
                std_t = np.std(t_stats, ddof=1)
                results[n] = (mean_t, std_t)
        return results

    # Flag to check if any distribution is plotted
    any_distribution_plotted = False

    # Normal Distribution
    if gui_objects['normal_var'].get():
        any_distribution_plotted = True
        mean, std = np.mean(values), np.std(values, ddof=1)
        density = norm.pdf(x_theor, mean, std)
        gui_objects['ax_dist'].plot(x_theor, density, 'r-', label='Нормальний розподіл')
        max_density = max(max_density, np.max(density))

        # Parameter estimation
        mean_ci = calculate_confidence_interval(values, "mean", confidence)
        std_ci = calculate_confidence_interval(values, "std", confidence)
        results_text.append(f"Нормальний розподіл:\n"
                           f"  Оцінка середнього: {mean:.4f} (ДІ: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}])\n"
                           f"  Оцінка стд. відхилення: {std:.4f} (ДІ: [{std_ci[0]:.4f}, {std_ci[1]:.4f}])\n")

        # CDF and confidence band
        cdf = norm.cdf(x_theor, mean, std)
        add_confidence_band(cdf, gui_objects['ax_dist'], x_theor, 'Нормальний')

        # Goodness-of-fit tests
        ks_stat, ks_pval = kstest(values, 'norm', args=(mean, std))
        chi2_stat, chi2_pval = pearson_chi2_test(values, 'norm', (mean, std), bins)
        results_text.append(f"  Тест Колмогорова-Смірнова: Статистика = {ks_stat:.4f}, p-значення = {ks_pval:.4f}, "
                           f"Критичне значення = {critical_value:.4f}, "
                           f"{'Нормальний' if ks_stat < critical_value else 'Не нормальний'}\n"
                           f"  Тест Пірсона: Статистика = {chi2_stat:.4f}, p-значення = {chi2_pval:.4f}, "
                           f"{'Нормальний' if chi2_pval > 0.05 else 'Не нормальний'}\n")

        # T-test bootstrap
        t_results = t_test_bootstrap(values)
        results_text.append("  T-тест (середнє T-статистики та стд. відхилення):\n")
        for n, (mean_t, std_t) in t_results.items():
            results_text.append(f"    Обсяг вибірки {n}: Середнє = {mean_t:.4f}, Стд. відхилення = {std_t:.4f}\n")

    # Exponential Distribution
    if gui_objects['exponential_var'].get():
        if np.any(values < 0):
            messagebox.showerror("Помилка", "Експоненціальний розподіл можливий лише для невід'ємних значень")
            gui_objects['exponential_var'].set(False)
        else:
            mean = np.mean(values)
            if mean == 0:
                messagebox.showerror("Помилка", "Середнє значення дорівнює нулю. Неможливо оцінити параметр.")
                gui_objects['exponential_var'].set(False)
            else:
                any_distribution_plotted = True
                lambda_param = 1 / mean
                density = expon.pdf(x_theor, scale=mean)
                gui_objects['ax_dist'].plot(x_theor, density, 'b-', label=f'Експоненціальний розподіл (λ={lambda_param:.4f})')
                max_density = max(max_density, np.max(density))

                # Parameter estimation
                lambda_se = lambda_param / np.sqrt(n)
                lambda_ci = (lambda_param - norm.ppf(1-(1-confidence)/2)*lambda_se,
                            lambda_param + norm.ppf(1-(1-confidence)/2)*lambda_se)
                results_text.append(f"Експоненціальний розподіл:\n"
                                   f"  Оцінка λ: {lambda_param:.4f} (ДІ: [{lambda_ci[0]:.4f}, {lambda_ci[1]:.4f}])\n")

                # CDF and confidence band
                cdf = expon.cdf(x_theor, scale=mean)
                add_confidence_band(cdf, gui_objects['ax_dist'], x_theor, 'Експоненціальний')

                # Goodness-of-fit tests
                ks_stat, ks_pval = kstest(values, 'expon', args=(0, mean))
                chi2_stat, chi2_pval = pearson_chi2_test(values, 'expon', (0, mean), bins)
                results_text.append(f"  Тест Колмогорова-Смірнова: Статистика = {ks_stat:.4f}, p-значення = {ks_pval:.4f}, "
                                   f"Критичне значення = {critical_value:.4f}, "
                                   f"{'Експоненціальний' if ks_stat < critical_value else 'Не експоненціальний'}\n"
                                   f"  Тест Пірсона: Статистика = {chi2_stat:.4f}, p-значення = {chi2_pval:.4f}, "
                                   f"{'Експоненціальний' if chi2_pval > 0.05 else 'Не експоненціальний'}\n")

                # T-test bootstrap
                t_results = t_test_bootstrap(values)
                results_text.append("  T-тест (середнє T-статистики та стд. відхилення):\n")
                for n, (mean_t, std_t) in t_results.items():
                    results_text.append(f"    Обсяг вибірки {n}: Середнє = {mean_t:.4f}, Стд. відхилення = {std_t:.4f}\n")

    # Weibull Distribution
    if gui_objects['weibull_var'].get():
        if np.any(values < 0):
            messagebox.showerror("Помилка", "Розподіл Вейбулла можливий лише для невід'ємних значень")
            gui_objects['weibull_var'].set(False)
        else:
            try:
                any_distribution_plotted = True
                shape, loc, scale = weibull_min.fit(values, floc=0)
                density = weibull_min.pdf(x_theor, shape, loc=loc, scale=scale)
                gui_objects['ax_dist'].plot(x_theor, density, 'm-', label=f'Розподіл Вейбулла (k={shape:.4f}, λ={scale:.4f})')
                max_density = max(max_density, np.max(density))

                # Parameter estimation (approximate CI using bootstrap)
                bootstrap_samples = 1000
                shape_samples = []
                scale_samples = []
                for _ in range(bootstrap_samples):
                    sample = np.random.choice(values, size=n, replace=True)
                    try:
                        s, _, sc = weibull_min.fit(sample, floc=0)
                        shape_samples.append(s)
                        scale_samples.append(sc)
                    except:
                        continue
                shape_ci = (np.percentile(shape_samples, 2.5), np.percentile(shape_samples, 97.5))
                scale_ci = (np.percentile(scale_samples, 2.5), np.percentile(scale_samples, 97.5))
                results_text.append(f"Розподіл Вейбулла:\n"
                                   f"  Оцінка форми (k): {shape:.4f} (ДІ: [{shape_ci[0]:.4f}, {shape_ci[1]:.4f}])\n"
                                   f"  Оцінка масштабу (λ): {scale:.4f} (ДІ: [{scale_ci[0]:.4f}, {scale_ci[1]:.4f}])\n")

                # CDF and confidence band
                cdf = weibull_min.cdf(x_theor, shape, loc=loc, scale=scale)
                add_confidence_band(cdf, gui_objects['ax_dist'], x_theor, 'Вейбулла')

                # Goodness-of-fit tests
                ks_stat, ks_pval = kstest(values, weibull_min.cdf, args=(shape, loc, scale))
                chi2_stat, chi2_pval = pearson_chi2_test(values, 'weibull_min', (shape, loc, scale), bins)
                results_text.append(f"  Тест Колмогорова-Смірнова: Статистика = {ks_stat:.4f}, p-значення = {ks_pval:.4f}, "
                                   f"Критичне значення = {critical_value:.4f}, "
                                   f"{'Вейбулла' if ks_stat < critical_value else 'Не Вейбулла'}\n"
                                   f"  Тест Пірсона: Статистика = {chi2_stat:.4f}, p-значення = {chi2_pval:.4f}, "
                                   f"{'Вейбулла' if chi2_pval > 0.05 else 'Не Вейбулла'}\n")

                # T-test bootstrap
                t_results = t_test_bootstrap(values)
                results_text.append("  T-тест (середнє T-статистики та стд. відхилення):\n")
                for n, (mean_t, std_t) in t_results.items():
                    results_text.append(f"    Обсяг вибірки {n}: Середнє = {mean_t:.4f}, Стд. відхилення = {std_t:.4f}\n")
            except Exception as e:
                messagebox.showerror("Помилка", f"Не вдалося підігнати розподіл Вейбулла: {str(e)}")
                gui_objects['weibull_var'].set(False)

    # Uniform Distribution
    if gui_objects['uniform_var'].get():
        any_distribution_plotted = True
        x_min, x_max = np.min(values), np.max(values)
        if x_max == x_min:
            x_max += 1
        range_width = x_max - x_min
        uniform_density = 1 / range_width if range_width > 0 else 0
        gui_objects['ax_dist'].plot(x_theor, [uniform_density] * len(x_theor), 'c-', label='Рівномірний розподіл')
        max_density = max(max_density, uniform_density)

        # Parameter estimation
        loc, scale = x_min, range_width
        loc_se = np.std(values, ddof=1) / np.sqrt(n)
        scale_se = np.std(values, ddof=1) / np.sqrt(n)
        loc_ci = (loc - norm.ppf(1-(1-confidence)/2)*loc_se, loc + norm.ppf(1-(1-confidence)/2)*loc_se)
        scale_ci = (scale - norm.ppf(1-(1-confidence)/2)*scale_se, scale + norm.ppf(1-(1-confidence)/2)*scale_se)
        results_text.append(f"Рівномірний розподіл:\n"
                           f"  Оцінка нижньої межі (a): {loc:.4f} (ДІ: [{loc_ci[0]:.4f}, {loc_ci[1]:.4f}])\n"
                           f"  Оцінка масштабу (b-a): {scale:.4f} (ДІ: [{scale_ci[0]:.4f}, {scale_ci[1]:.4f}])\n")

        # CDF and confidence band
        cdf = uniform.cdf(x_theor, loc=loc, scale=scale)
        add_confidence_band(cdf, gui_objects['ax_dist'], x_theor, 'Рівномірний')

        # Goodness-of-fit tests
        ks_stat, ks_pval = kstest(values, 'uniform', args=(loc, scale))
        chi2_stat, chi2_pval = pearson_chi2_test(values, 'uniform', (loc, scale), bins)
        results_text.append(f"  Тест Колмогорова-Смірнова: Статистика = {ks_stat:.4f}, p-значення = {ks_pval:.4f}, "
                           f"Критичне значення = {critical_value:.4f}, "
                           f"{'Рівномірний' if ks_stat < critical_value else 'Не рівномірний'}\n"
                           f"  Тест Пірсона: Статистика = {chi2_stat:.4f}, p-значення = {chi2_pval:.4f}, "
                           f"{'Рівномірний' if chi2_pval > 0.05 else 'Не рівномірний'}\n")

        # T-test bootstrap
        t_results = t_test_bootstrap(values)
        results_text.append("  T-тест (середнє T-статистики та стд. відхилення):\n")
        for n, (mean_t, std_t) in t_results.items():
            results_text.append(f"    Обсяг вибірки {n}: Середнє = {mean_t:.4f}, Стд. відхилення = {std_t:.4f}\n")

    # Rayleigh Distribution
    if gui_objects['rayleigh_var'].get():
        if np.any(values < 0):
            messagebox.showerror("Помилка", "Розподіл Релея можливий лише для невід'ємних значень")
            gui_objects['rayleigh_var'].set(False)
        else:
            any_distribution_plotted = True
            sigma = np.sqrt(np.mean(values**2) / 2)
            density = rayleigh.pdf(x_theor, scale=sigma)
            gui_objects['ax_dist'].plot(x_theor, density, 'y-', label=f'Розподіл Релея (σ={sigma:.4f})')
            max_density = max(max_density, np.max(density))

            # Parameter estimation
            sigma_se = sigma / np.sqrt(2 * n)
            sigma_ci = (sigma - norm.ppf(1-(1-confidence)/2)*sigma_se, sigma + norm.ppf(1-(1-confidence)/2)*sigma_se)
            results_text.append(f"Розподіл Релея:\n"
                               f"  Оцінка масштабу (σ): {sigma:.4f} (ДІ: [{sigma_ci[0]:.4f}, {sigma_ci[1]:.4f}])\n")

            # CDF and confidence band
            cdf = rayleigh.cdf(x_theor, scale=sigma)
            add_confidence_band(cdf, gui_objects['ax_dist'], x_theor, 'Релея')

            # Goodness-of-fit tests
            ks_stat, ks_pval = kstest(values, 'rayleigh', args=(0, sigma))
            chi2_stat, chi2_pval = pearson_chi2_test(values, 'rayleigh', (0, sigma), bins)
            results_text.append(f"  Тест Колмогорова-Смірнова: Статистика = {ks_stat:.4f}, p-значення = {ks_pval:.4f}, "
                               f"Критичне значення = {critical_value:.4f}, "
                               f"{'Релея' if ks_stat < critical_value else 'Не Релея'}\n"
                               f"  Тест Пірсона: Статистика = {chi2_stat:.4f}, p-значення = {chi2_pval:.4f}, "
                               f"{'Релея' if chi2_pval > 0.05 else 'Не Релея'}\n")

            # T-test bootstrap
            t_results = t_test_bootstrap(values)
            results_text.append("  T-тест (середнє T-статистики та стд. відхилення):\n")
            for n, (mean_t, std_t) in t_results.items():
                results_text.append(f"    Обсяг вибірки {n}: Середнє = {mean_t:.4f}, Стд. відхилення = {std_t:.4f}\n")

    # Update plot settings
    if any_distribution_plotted or gui_objects['normal_var'].get() or gui_objects['exponential_var'].get() or \
       gui_objects['weibull_var'].get() or gui_objects['uniform_var'].get() or gui_objects['rayleigh_var'].get():
        gui_objects['ax_dist'].set_title('Гістограма та розподіли')
        gui_objects['ax_dist'].set_xlabel('Час затримки (сек)')
        gui_objects['ax_dist'].set_ylabel('Щільність')
        gui_objects['ax_dist'].legend()
        gui_objects['ax_dist'].set_xlim(x_lower, x_upper)
        gui_objects['ax_dist'].set_ylim(0, max_density * 1.1)
        gui_objects['ax_dist'].grid(True, linestyle='--', alpha=0.7)
        gui_objects['dist_canvas'].draw()

    # Update results text
    try:
        gui_objects['dist_info_text'].delete(1.0, tk.END)
        if results_text:
            gui_objects['dist_info_text'].insert(tk.END, "\n".join(results_text))
        else:
            gui_objects['dist_info_text'].insert(tk.END, "Жоден розподіл не вибрано або не підходить для даних.")
    except KeyError:
        messagebox.showerror("Помилка", "Текстове поле для результатів розподілів не знайдено. Перевірте конфігурацію GUI.")

def initialize_logic(objects):
    global gui_objects
    gui_objects = objects
    gui_objects['rayleigh_btn'].config(command=plot_rayleigh_distribution)
    gui_objects['save_btn'].config(command=save_data)
    gui_objects['data_box'].bind('<FocusOut>', update_from_data_box)
    gui_objects['load_button'].config(command=load_data)
    gui_objects['update_button'].config(command=update_histogram)
    gui_objects['standardize_btn'].config(command=standardize_data)
    gui_objects['log_btn'].config(command=log_transform)
    gui_objects['shift_btn'].config(command=shift_data)
    gui_objects['outliers_btn'].config(command=remove_outliers)
    gui_objects['reset_btn'].config(command=reset_data)
    gui_objects['plot_btn'].config(command=plot_distribution_functions)
    gui_objects['cdf_btn'].config(command=plot_exponential_distribution)
    gui_objects['call_type_btn'].config(command=analyze_call_types)
    gui_objects['refresh_graph_button'].config(command=update_histogram)
    gui_objects['update_graph_btn'].config(command=update_distribution_plot)
    update_distribution_plot()  # Ініціалізація графіка при запуску