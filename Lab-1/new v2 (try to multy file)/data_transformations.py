import numpy as np
from tkinter import simpledialog, messagebox

from data_manager import values, original_values

def standardize_data(update_histogram_func, update_statistics_func, 
                    update_characteristics_func, update_data_box_func):
    global values
    if len(values) == 0:
        return
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    values = (values - mean) / std
    
    update_histogram_func()
    update_statistics_func()
    update_characteristics_func()
    update_data_box_func()
    messagebox.showinfo("Стандартизація", "Дані успішно стандартизовано")

def log_transform(update_histogram_func, update_statistics_func, 
                update_characteristics_func, update_data_box_func):
    global values
    if len(values) == 0:
        return
    
    if np.min(values) <= 0:
        messagebox.showerror("Помилка", "Логарифмування можливе тільки для додатних значень")
        return
    
    values = np.log(values)
    
    update_histogram_func()
    update_statistics_func()
    update_characteristics_func()
    update_data_box_func()
    messagebox.showinfo("Логарифмування", "Дані успішно логарифмовано")

def shift_data(update_histogram_func, update_statistics_func, 
             update_characteristics_func, update_data_box_func):
    global values
    if len(values) == 0:
        return
    
    shift_value = simpledialog.askfloat("Зсув даних", "Введіть значення зсуву:")
    if shift_value is not None:
        values = values + shift_value
        
        update_histogram_func()
        update_statistics_func()
        update_characteristics_func()
        update_data_box_func()
        messagebox.showinfo("Зсув", f"Дані успішно зсунуто на {shift_value}")

def remove_outliers(update_histogram_func, update_statistics_func, 
                  update_characteristics_func, update_data_box_func):
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
    
    update_histogram_func()
    update_statistics_func()
    update_characteristics_func()
    update_data_box_func()
    messagebox.showinfo("Видалення викидів", f"Видалено {removed_count} викидів")

def reset_data(update_histogram_func, update_statistics_func, 
             update_characteristics_func, update_data_box_func,
             lower_bound_var, upper_bound_var):
    global values, original_values
    if hasattr(globals(), 'original_values'):
        values = original_values.copy()
        
        # Оновлюємо значення верхньої та нижньої границі до початкових
        min_val, max_val = np.min(values), np.max(values)
        lower_bound_var.set(min_val)
        upper_bound_var.set(max_val)
        
        update_histogram_func()
        update_statistics_func()
        update_characteristics_func()
        update_data_box_func()
        messagebox.showinfo("Скидання", "Дані повернуто до початкового стану")