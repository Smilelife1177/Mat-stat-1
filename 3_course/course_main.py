# 3_course_main.py
import tkinter as tk
from tkinter import ttk

# Імпорт модулів для лабораторних (зауваження: назви файлів, що починаються з цифри, не дозволяють прямий імпорт.
# Рекомендую перейменувати файли на course3_01.py тощо, або використовувати sys.path для імпорту.
# Для прикладу, припустимо файл називається course3_01.py і імпортується як import course3_01
import course01 as lab1  # Замініть на актуальний імпорт після перейменування

root = tk.Tk()
root.title("Лабораторні роботи з математичної статистики")
root.geometry("800x600")

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Додавання вкладки для Lab 1
lab1.create_tab(notebook)

# Тут можна додавати інші лаби в майбутньому, наприклад:
# import course3_02 as lab2
# lab2.create_tab(notebook)

root.mainloop()