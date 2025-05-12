import numpy as np
from scipy.stats import skew, kurtosis, variation, median_abs_deviation

from data_manager import values

def update_statistics():
    global values
    if len(values) == 0:
        return  # Skip calculations if array is empty
    
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)
    skewness = skew(values)
    excess_kurtosis = kurtosis(values, fisher=True)
    counter_kurtosis = 1 / (excess_kurtosis + 3) if excess_kurtosis + 3 != 0 else 0
    pearson_var = std_dev / mean if mean != 0 else 0
    nonparam_var = variation(values)
    mad_val = median_abs_deviation(values)
    med_val = np.median(values)

# Add this to the beginning of update_characteristics in statistics.py
def update_characteristics(char_table):
    global values
    
    # Очистити таблицю
    for item in char_table.get_children():
        char_table.delete(item)
    
    # Skip calculations if array is empty
    if len(values) == 0:
        return
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