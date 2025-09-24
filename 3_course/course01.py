# course3_01.py
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QTabWidget)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class Lab1Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.generate_plots()  # Початкова генерація

    def initUI(self):
        layout = QVBoxLayout()

        # Фрейм для введення параметрів
        input_layout = QHBoxLayout()
        
        # Параметри
        self.n_edit = QLineEdit("100")
        self.mean_edit = QLineEdit("0")
        self.std_edit = QLineEdit("1")
        
        labels = ["Кількість чисел (n):", "Математичне сподівання:", "Стандартне відхилення:"]
        edits = [self.n_edit, self.mean_edit, self.std_edit]
        
        for label, edit in zip(labels, edits):
            input_layout.addWidget(QLabel(label))
            input_layout.addWidget(edit)
        
        generate_btn = QPushButton("Згенерувати")
        generate_btn.clicked.connect(self.generate_plots)
        input_layout.addWidget(generate_btn)
        
        layout.addLayout(input_layout)

        # Матplotlib canvas
        self.figure = Figure(figsize=(12, 12))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Результати
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def generate_plots(self):
        try:
            # Отримання параметрів
            n = int(self.n_edit.text())
            mean = float(self.mean_edit.text())
            std = float(self.std_edit.text())

            if n <= 0 or std <= 0:
                self.result_label.setText("Помилка: n і std повинні бути додатними!")
                return

            # Генерування даних
            data = np.random.normal(loc=mean, scale=std, size=n)
            data_sorted = np.sort(data)

            # Інтервальний ряд (10 інтервалів)
            num_bins = 10
            hist, bin_edges = np.histogram(data, bins=num_bins)
            midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Оцінки
            mean_est = np.sum(midpoints * hist) / len(data)
            var_est = np.sum(hist * (midpoints - mean_est)**2) / (len(data) - 1)

            # Очищення фігури
            self.figure.clear()

            # Створення subplot'ів
            axs = self.figure.subplots(3, 2)
            self.figure.suptitle('Графіки для лабораторної роботи 1')

            # Гістограма
            axs[0, 0].hist(data, bins=num_bins, edgecolor='black')
            axs[0, 0].set_title('Гістограма')

            # Полігон частот
            axs[0, 1].plot(midpoints, hist, 'o-', color='blue')
            axs[0, 1].set_title('Полігон частот')

            # Огіва
            cum_freq = np.cumsum(hist)
            axs[1, 0].plot(midpoints, cum_freq, 'o-', color='green')
            axs[1, 0].set_title('Огіва (кумулятивна частота)')

            # Кумулята
            rel_cum_freq = cum_freq / len(data)
            axs[1, 1].plot(midpoints, rel_cum_freq, 'o-', color='red')
            axs[1, 1].set_title('Кумулята')

            # Емпірична функція розподілу
            axs[2, 0].step(data_sorted, np.arange(1, len(data) + 1) / len(data), 
                          where='post', color='purple')
            axs[2, 0].set_title('Емпірична функція розподілу')
            axs[2, 0].set_ylim(0, 1)

            # Приховуємо пустий subplot
            axs[2, 1].axis('off')

            # Оновлення canvas
            self.figure.tight_layout()
            self.canvas.draw()

            # Відображення результатів
            results_text = f"Оцінка математичного сподівання: {mean_est:.4f}\nОцінка дисперсії: {var_est:.4f}"
            self.result_label.setText(results_text)

        except ValueError:
            self.result_label.setText("Помилка: Введіть коректні числові значення!")

def create_tab(tab_widget):
    lab1_widget = Lab1Widget()
    tab_widget.addTab(lab1_widget, "Lab 1: Обробка вибірки")