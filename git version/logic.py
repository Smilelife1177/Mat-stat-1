import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, variation, median_abs_deviation, norm, t, chi2, sem, kstest, expon, pearsonr, weibull_min, uniform
from scipy.optimize import minimize
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
        gui_objects['rayleigh_btn'].config(state=tk.NORMAL)
        gui_objects['distributions_btn'].config(state=tk.NORMAL)

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
        # Оцінка стандартних помилок через чисельний гессіан
        std_err_c = np.sqrt(result.hess_inv[0, 0]) if hasattr(result, 'hess_inv') else np.nan
        std_err_scale = np.sqrt(result.hess_inv[1, 1]) if hasattr(result, 'hess_inv') else np.nan
        return c, scale, std_err_c, std_err_scale
    return np.nan, np.nan, np.nan, np.nan

def plot_all_distributions():
    global values
    if len(values) == 0:
        messagebox.showwarning("Попередження", "Немає даних для побудови розподілів")
        return
    
    if np.any(values < 0) and ('weibull' in [dist for dist in gui_objects['distributions_var'].get().split(',')] or 
                              'expon' in [dist for dist in gui_objects['distributions_var'].get().split(',')] or 
                              'rayleigh' in [dist for dist in gui_objects['distributions_var'].get().split(',')]):
        messagebox.showerror("Помилка", "Експоненціальний, Вейбулла та Релея розподіли можливі лише для невід’ємних значень")
        return
    
    for widget in gui_objects['tab6'].winfo_children():
        widget.destroy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_bins = gui_objects['bin_count_var'].get()
    if n_bins == 0:
        n_bins = int(np.sqrt(len(values)))
    
    bin_dt, bin_gr = np.histogram(values, bins=n_bins)
    Y = np.cumsum(bin_dt) / len(values)
    for i in range(len(Y)):
        ax.plot([bin_gr[i], bin_gr[i+1]], [Y[i], Y[i]], color='green', linewidth=2, label='Емпіричний розподіл' if i == 0 else "")
    ax.plot([bin_gr[-1], bin_gr[-1]], [Y[-1], 1], color='green', linewidth=2)
    
    confidence = gui_objects['confidence_var'].get() / 100
    n = len(values)
    epsilon = np.sqrt(-0.5 * np.log(1 - confidence) / n)
    
    x_min, x_max = np.min(values), np.max(values)
    x_range = x_max - x_min if x_max != x_min else 1
    x_lower = x_min - 0.1 * x_range
    x_margin = 0.1 * x_range
    x_upper = x_max + x_margin
    # x_upper = x_max + x_margin = 0.1 * x_range #########
    
    x_theor = np.linspace(x_lower, x_upper, 1000)
    
    distributions = [key for key, var in gui_objects['distributions'].items() if var.get()]
    colors = ['red', 'blue', 'purple', 'orange', 'cyan']
    dist_info = []
    
    for idx, dist in enumerate(distributions):
        dist = dist.strip()
        if dist == 'norm':
            mean, std = np.mean(values), np.std(values, ddof=1)
            cdf = norm.cdf(x_theor, mean, std)
            ax.plot(x_theor, cdf, color=colors[idx % len(colors)], label=f'Нормальний (μ={mean:.4f}, σ={std:.4f})')
            ks_stat, ks_p = kstest(values, 'norm', args=(mean, std))
            chi2_stat, chi2_p, df = pearson_chi2_test(values, 'norm', (mean, std), n_bins)
            dist_info.append(f"Нормальний розподіл:\nμ={mean:.4f} (±{sem(values):.4f})\nσ={std:.4f}\n"
                            f"KS: стат={ks_stat:.4f}, p={ks_p:.4f}\n"
                            f"Пірсон: χ²={chi2_stat:.4f}, p={chi2_p:.4f}, df={df}")
            ax.fill_between(x_theor, np.maximum(cdf - epsilon, 0), np.minimum(cdf + epsilon, 1), 
                            color=colors[idx % len(colors)], alpha=0.2)
        
        elif dist == 'expon':
            mean = np.mean(values)
            scale = mean
            cdf = expon.cdf(x_theor, loc=0, scale=scale)
            ax.plot(x_theor, cdf, color=colors[idx % len(colors)], label=f'Експоненціальний (λ={1/scale:.4f})')
            ks_stat, ks_p = kstest(values, 'expon', args=(0, scale))
            chi2_stat, chi2_p, df = pearson_chi2_test(values, 'expon', (0, scale), n_bins)
            dist_info.append(f"Експоненціальний розподіл:\nλ={1/scale:.4f} (±{1/(np.sqrt(n)*mean):.4f})\n"
                            f"KS: стат={ks_stat:.4f}, p={ks_p:.4f}\n"
                            f"Пірсон: χ²={chi2_stat:.4f}, p={chi2_p:.4f}, df={df}")
            ax.fill_between(x_theor, np.maximum(cdf - epsilon, 0), np.minimum(cdf + epsilon, 1), 
                            color=colors[idx % len(colors)], alpha=0.2)
        
        elif dist == 'weibull':
            c, scale, std_err_c, std_err_scale = estimate_weibull_params(values)
            if np.isnan(c) or np.isnan(scale):
                messagebox.showerror("Помилка", "Не вдалося оцінити параметри Вейбулла")
                continue
            cdf = weibull_min.cdf(x_theor, c, loc=0, scale=scale)
            ax.plot(x_theor, cdf, color=colors[idx % len(colors)], label=f'Вейбулла (c={c:.4f}, scale={scale:.4f})')
            ks_stat, ks_p = kstest(values, lambda x: weibull_min.cdf(x, c, loc=0, scale=scale))
            chi2_stat, chi2_p, df = pearson_chi2_test(values, 'weibull', (c, 0, scale), n_bins)
            dist_info.append(f"Вейбулла розподіл:\nc={c:.4f} (±{std_err_c:.4f})\nscale={scale:.4f} (±{std_err_scale:.4f})\n"
                            f"KS: стат={ks_stat:.4f}, p={ks_p:.4f}\n"
                            f"Пірсон: χ²={chi2_stat:.4f}, p={chi2_p:.4f}, df={df}")
            ax.fill_between(x_theor, np.maximum(cdf - epsilon, 0), np.minimum(cdf + epsilon, 1), 
                            color=colors[idx % len(colors)], alpha=0.2)
        
        elif dist == 'uniform':
            a, b = np.min(values), np.max(values)
            cdf = uniform.cdf(x_theor, a, b - a)
            ax.plot(x_theor, cdf, color=colors[idx % len(colors)], label=f'Рівномірний (a={a:.4f}, b={b:.4f})')
            ks_stat, ks_p = kstest(values, 'uniform', args=(a, b - a))
            chi2_stat, chi2_p, df = pearson_chi2_test(values, 'uniform', (a, b - a), n_bins)
            dist_info.append(f"Рівномірний розподіл:\na={a:.4f}\nb={b:.4f}\n"
                            f"KS: стат={ks_stat:.4f}, p={ks_p:.4f}\n"
                            f"Пірсон: χ²={chi2_stat:.4f}, p={chi2_p:.4f}, df={df}")
            ax.fill_between(x_theor, np.maximum(cdf - epsilon, 0), np.minimum(cdf + epsilon, 1), 
                            color=colors[idx % len(colors)], alpha=0.2)
        
        elif dist == 'rayleigh':
            mean = np.mean(values)
            sigma = np.sqrt(2 / np.pi) * mean
            cdf = 1 - np.exp(-x_theor**2 / (2 * sigma**2))
            ax.plot(x_theor, cdf, color=colors[idx % len(colors)], label=f'Релея (σ={sigma:.4f})')
            ks_stat, ks_p = kstest(values, lambda x: 1 - np.exp(-x**2 / (2 * sigma**2)))
            chi2_stat, chi2_p, df = pearson_chi2_test(values, 'rayleigh', (sigma,), n_bins)
            dist_info.append(f"Релея розподіл:\nσ={sigma:.4f} (±{sem(values)*np.sqrt(2/np.pi):.4f})\n"
                            f"KS: стат={ks_stat:.4f}, p={ks_p:.4f}\n"
                            f"Пірсон: χ²={chi2_stat:.4f}, p={chi2_p:.4f}, df={df}")
            ax.fill_between(x_theor, np.maximum(cdf - epsilon, 0), np.minimum(cdf + epsilon, 1), 
                            color=colors[idx % len(colors)], alpha=0.2)
    
    ax.set_title('Порівняння емпіричного та теоретичних розподілів')
    ax.set_xlabel('Час очікування (хв)')
    ax.set_ylabel('Ймовірність')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(-0.05, 1.05)
    
    canvas = FigureCanvasTkAgg(fig, master=gui_objects['tab6'])
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    info_text = "\n\n".join(dist_info)
    info_frame = tk.Frame(gui_objects['tab6'])
    info_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=10, pady=10)
    info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
    info_label.pack()

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
    ax.set_ylim(0, 1)
    
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
    
    for widget in gui_objects['tab4'].winfo_children():
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
        median = np.median(type_values)#########
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
        messagebox.showinfo("Видалення викидів", f"Видалено {removed_count} викидів")
    else:
        messagebox.showinfo("Видалення викидів", "Жодних викидів не видалено")

# def apply_bounds():
#     global values, call_types
#     if len(values) == 0:
#         messagebox.showwarning("Попередження", "Немає даних для застосування границь")
#         return
#     try:
#         lower = float(gui_objects['lower_bound_var'].get())
#         upper = float(gui_objects['upper_bound_var'].get())
#         if lower >= upper:
#             messagebox.showerror("Помилка", "Нижня границя має бути менше верхньої")
#             return
#         mask = (values >= lower) & (values <= upper)
#         if not np.any(mask):
#             messagebox.showerror("Помилка", "За вказаними границями немає даних")
#             return
#         values = values[mask]
#         call_types = call_types[mask]
#         update_statistics()
#         update_characteristics()
#         update_data_box()
#         messagebox.showinfo("Границі", f"Застосовано границі: [{lower:.4f}, {upper:.4f}]")
#     except ValueError:
#         messagebox.showerror("Помилка", "Введіть числові значення для границь")

def reset_data():
    global values, original_values, call_types, original_call_types
    if len(original_values) == 0:
        messagebox.showwarning("Попередження", "Немає початкових даних для скидання. Завантажте дані спочатку.")
        return
    
    values = original_values.copy()
    call_types = original_call_types.copy()
    # min_val, max_val = np.min(values), np.max(values)
    # gui_objects['lower_bound_var'].set(str(min_val))
    # gui_objects['upper_bound_var'].set(str(max_val))
    update_statistics()
    update_characteristics()
    update_data_box()
    for widget in gui_objects['tab2'].winfo_children():
        widget.destroy()
    messagebox.showinfo("Скидання", "Дані повернуто до початкового стану")

def plot_rayleigh_distribution():
    global values
    if len(values) == 0:
        messagebox.showwarning("Попередження", "Немає даних для побудови розподілу Релея")
        return
    
    if np.any(values < 0):
        messagebox.showerror("Помилка", "Розподіл Релея можливий лише для невід’ємних значень")
        return
    
    for widget in gui_objects['tab5'].winfo_children():
        widget.destroy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_bins = gui_objects['bin_count_var'].get()
    if n_bins == 0:
        n_bins = int(np.sqrt(len(values)))
    bin_dt, bin_gr = np.histogram(values, bins=n_bins)
    Y = np.cumsum(bin_dt) / len(values)
    
    for i in range(len(Y)):
        ax.plot([bin_gr[i], bin_gr[i+1]], [Y[i], Y[i]], color='green', linewidth=2, label='Емпіричний розподіл' if i == 0 else "")
    ax.plot([bin_gr[-1], bin_gr[-1]], [Y[-1], 1], color='green', linewidth=2)
    
    mean = np.mean(values)
    sigma = np.sqrt(2 / np.pi) * mean
    
    x_theor = np.linspace(0, np.max(values) * 1.2, 100)
    pdf_rayleigh = (x_theor / sigma**2) * np.exp(-x_theor**2 / (2 * sigma**2))
    ax.plot(x_theor, pdf_rayleigh, 'b-', label=f'Щільність Релея (σ={sigma:.4f})')
    
    cdf_rayleigh = 1 - np.exp(-x_theor**2 / (2 * sigma**2))
    ax.plot(x_theor, cdf_rayleigh, 'r-', label=f'Розподіл Релея (σ={sigma:.4f})')
    
    confidence = gui_objects['confidence_var'].get() / 100
    n = len(values)
    epsilon = np.sqrt(-0.5 * np.log(1 - confidence) / n)
    ax.fill_between(x_theor, np.maximum(cdf_rayleigh - epsilon, 0), np.minimum(cdf_rayleigh + epsilon, 1), 
                    color='red', alpha=0.2, label='Довірчий інтервал')
    
    ax.set_title('Розподіл Релея та емпіричні дані')
    ax.set_xlabel('Час очікування (хв)')
    ax.set_ylabel('Ймовірність / Щільність')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.set_xlim(0, np.max(values) * 1.2)
    ax.set_ylim(0, max(np.max(pdf_rayleigh) * 1.1, 1.05))
    
    canvas = FigureCanvasTkAgg(fig, master=gui_objects['tab5'])
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    ks_statistic, ks_pvalue = kstest(values, lambda x: 1 - np.exp(-x**2 / (2 * sigma**2)), args=(), N=n)
    chi2_stat, chi2_p, df = pearson_chi2_test(values, 'rayleigh', (sigma,), n_bins)
    critical_value = np.sqrt(-0.5 * np.log((1 - confidence) / 2)) / np.sqrt(n)
    conclusion = "Розподіл відповідає Релея" if ks_statistic < critical_value else "Розподіл не відповідає Релея"
    info_text = (f"Релея розподіл:\nσ={sigma:.4f} (±{sem(values)*np.sqrt(2/np.pi):.4f})\n"
                 f"Тест Колмогорова-Смірнова:\nСтатистика: {ks_statistic:.4f}\n"
                 f"Критичне значення: {critical_value:.4f}\np-значення: {ks_pvalue:.4f}\n"
                 f"Висновок: {conclusion}\n"
                 f"Тест Пірсона:\nχ²={chi2_stat:.4f}, p={chi2_p:.4f}, df={df}")
    info_frame = tk.Frame(gui_objects['tab5'])
    info_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=10, pady=10)
    info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
    info_label.pack()

def initialize_logic(objects):
    global gui_objects
    gui_objects = objects
    gui_objects['rayleigh_btn'].config(command=plot_rayleigh_distribution)
    gui_objects['save_btn'].config(command=save_data)
    gui_objects['data_box'].bind('<FocusOut>', update_from_data_box)
    gui_objects['load_button'].config(command=load_data)
    gui_objects['update_button'].config(command=update_histogram)
    # gui_objects['apply_bounds_btn'].config(command=apply_bounds)
    gui_objects['standardize_btn'].config(command=standardize_data)
    gui_objects['log_btn'].config(command=log_transform)
    gui_objects['shift_btn'].config(command=shift_data)
    gui_objects['outliers_btn'].config(command=remove_outliers)
    gui_objects['reset_btn'].config(command=reset_data)
    gui_objects['plot_btn'].config(command=plot_distribution_functions)
    gui_objects['cdf_btn'].config(command=plot_exponential_distribution)
    gui_objects['call_type_btn'].config(command=analyze_call_types)
    gui_objects['distributions_btn'].config(command=plot_all_distributions)
    # gui_objects['refresh_graph_button'].config(command=update_histogram) #########