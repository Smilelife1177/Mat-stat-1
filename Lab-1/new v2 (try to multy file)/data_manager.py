import numpy as np
from tkinter import filedialog, messagebox
import tkinter as tk

# Глобальні змінні для даних
values = np.array([])
original_values = np.array([])

def load_data(update_histogram_func, update_statistics_func, update_characteristics_func, 
              update_data_box_func, editing_buttons, plot_btn, cdf_btn, 
              lower_bound_var, upper_bound_var):
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    try:
        with open(file_path, 'r') as file:
            data = file.read().replace(',', '.')
        
        # Convert data to numbers
        data_list = data.split()
        if not data_list:
            messagebox.showerror("Помилка", "Файл не містить даних")
            return
            
        global values, original_values
        values = np.array(list(map(float, data_list)))
        
        # Check if the array is empty
        if values.size == 0:
            messagebox.showerror("Помилка", "Не вдалося завантажити дані: порожній масив")
            return
            
        original_values = values.copy()  # Зберігаємо оригінальні значення
        
        # Оновлюємо значення верхньої та нижньої границі
        min_val, max_val = np.min(values), np.max(values)
        lower_bound_var.set(min_val)
        upper_bound_var.set(max_val)
        
        # Update UI elements
        update_histogram_func()
        update_statistics_func()
        update_characteristics_func()
        update_data_box_func()
        
        # Активуємо кнопки редагування
        for btn in editing_buttons:
            btn.config(state=tk.NORMAL)
        plot_btn.config(state=tk.NORMAL)
        cdf_btn.config(state=tk.NORMAL)
        
    except Exception as e:
        messagebox.showerror("Помилка", f"Не вдалося завантажити дані: {str(e)}")

def apply_bounds(lower_bound_var, upper_bound_var, update_histogram_func, 
                update_statistics_func, update_characteristics_func, update_data_box_func):
    global values, original_values
    if len(original_values) == 0:
        messagebox.showerror("Помилка", "Немає даних для фільтрації")
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
        
        update_histogram_func()
        update_statistics_func()
        update_characteristics_func()
        update_data_box_func()
        messagebox.showinfo("Границі", f"Застосовано границі: [{lower:.4f}, {upper:.4f}]")
    except ValueError:
        messagebox.showerror("Помилка", "Введіть числові значення для границь")

def update_data_box(data_box):
    global values
    # Очищаємо текстове поле
    data_box.delete(1.0, tk.END)
    # Додаємо дані
    formatted_values = ', '.join([f"{val:.4f}" for val in values])
    data_box.insert(tk.END, formatted_values)