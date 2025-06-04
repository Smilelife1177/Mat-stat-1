from scipy.stats import skew, kurtosis, variation, median_abs_deviation, norm, t, chi2, sem, kstest, expon, weibull_min, rayleigh, uniform, chi2_contingency, ttest_1samp
import numpy as np
from scipy.stats import norm, kstest, expon, weibull_min, rayleigh, uniform, chi2_contingency, ttest_1samp
from tkinter import messagebox
import tkinter as tk

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

def update_distribution_plot(values, gui_objects):
    if len(values) == 0:
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