# 3_course01.py
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Frame, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_tab(notebook):
    frame = Frame(notebook)
    notebook.add(frame, text="Lab 1: Обробка вибірки")

    # Генерування нормально розподіленої послідовності (mean=0, std=1, size=100)
    data = np.random.normal(loc=0, scale=1, size=100)
    data_sorted = np.sort(data)

    # Побудова інтервального статистичного ряду (10 інтервалів)
    num_bins = 10
    hist, bin_edges = np.histogram(data, bins=num_bins)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Оцінка математичного сподівання (mean) на основі інтервального ряду
    mean_est = np.sum(midpoints * hist) / len(data)

    # Оцінка дисперсії (sample variance) на основі інтервального ряду
    var_est = np.sum(hist * (midpoints - mean_est)**2) / (len(data) - 1)

    # Створення фігури для графіків
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Графіки для лабораторної роботи 1')

    # Гістограма
    axs[0, 0].hist(data, bins=num_bins, edgecolor='black')
    axs[0, 0].set_title('Гістограма')

    # Полігон частот
    axs[0, 1].plot(midpoints, hist, 'o-', color='blue')
    axs[0, 1].set_title('Полігон частот')

    # Огіва (кумулятивна частота)
    cum_freq = np.cumsum(hist)
    axs[1, 0].plot(midpoints, cum_freq, 'o-', color='green')
    axs[1, 0].set_title('Огіва (кумулятивна частота)')

    # Кумулята (кумулятивна відносна частота)
    rel_cum_freq = cum_freq / len(data)
    axs[1, 1].plot(midpoints, rel_cum_freq, 'o-', color='red')
    axs[1, 1].set_title('Кумулята (кумулятивна відносна частота)')

    # Емпірична функція розподілу (ECDF)
    axs[2, 0].step(data_sorted, np.arange(1, len(data) + 1) / len(data), where='post', color='purple')
    axs[2, 0].set_title('Емпірична функція розподілу')
    axs[2, 0].set_ylim(0, 1)

    # Приховуємо пустий subplot
    axs[2, 1].axis('off')

    # Вбудовування фігури в Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Відображення оцінок
    results_text = f"Оцінка математичного сподівання: {mean_est:.4f}\nОцінка дисперсії: {var_est:.4f}"
    label = Label(frame, text=results_text, font=('Arial', 12))
    label.pack(pady=10)

    # Для відтворюваності, можна зафіксувати seed, але за замовчуванням random
    # np.random.seed(42)  # Розкоментувати для фіксованих результатів