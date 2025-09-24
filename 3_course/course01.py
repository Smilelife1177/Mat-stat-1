# course3_01.py
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QTabWidget, QTableWidget,
                             QTableWidgetItem, QCheckBox)
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

        # Фрейм для чекбоксів
        checkbox_layout = QHBoxLayout()
        self.hist_checkbox = QCheckBox("Гістограма")
        self.hist_checkbox.setChecked(True)  # Set "Гістограма" as checked by default
        self.polygon_checkbox = QCheckBox("Полігон частот")
        self.polygon_checkbox.setChecked(False)
        self.ogive_checkbox = QCheckBox("Огіва")
        self.ogive_checkbox.setChecked(False)
        self.cumulative_checkbox = QCheckBox("Кумулята")
        self.cumulative_checkbox.setChecked(False)
        self.ecdf_checkbox = QCheckBox("Емпірична функція розподілу")
        self.ecdf_checkbox.setChecked(False)

        checkbox_layout.addWidget(self.hist_checkbox)
        checkbox_layout.addWidget(self.polygon_checkbox)
        checkbox_layout.addWidget(self.ogive_checkbox)
        checkbox_layout.addWidget(self.cumulative_checkbox)
        checkbox_layout.addWidget(self.ecdf_checkbox)
        layout.addLayout(checkbox_layout)

        # Матplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Таблиця
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["N", "Межі інтервалу", "Середній інтервал", 
                                            "Частота", "Відноша частота", "Накопичена відноша частота"])
        # Stretch the table to fill available space
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        layout.addWidget(self.table, stretch=1)  # Add stretch factor to make table expandable

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

            # Заповнення таблиці
            self.table.setRowCount(num_bins)
            for i in range(num_bins):
                self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.table.setItem(i, 1, QTableWidgetItem(f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"))
                self.table.setItem(i, 2, QTableWidgetItem(f"{midpoints[i]:.2f}"))
                self.table.setItem(i, 3, QTableWidgetItem(str(hist[i])))
                rel_freq = hist[i] / n
                self.table.setItem(i, 4, QTableWidgetItem(f"{rel_freq:.4f}"))
                cum_rel_freq = np.sum(hist[:i+1]) / n
                self.table.setItem(i, 5, QTableWidgetItem(f"{cum_rel_freq:.4f}"))

            # Очищення фігури
            self.figure.clear()

            # Налаштування розміру фігури
            self.figure.set_size_inches(12, 6)

            # Створення одного subplot'а
            ax = self.figure.add_subplot(111)
            self.figure.suptitle('Графік для лабораторної роботи 1')

            # Визначення активного графіка
            if self.hist_checkbox.isChecked():
                ax.hist(data, bins=num_bins, edgecolor='black', color='blue')
                ax.set_title('Гістограма')
            elif self.polygon_checkbox.isChecked():
                ax.plot(midpoints, hist, 'o-', color='green')
                ax.set_title('Полігон частот')
            elif self.ogive_checkbox.isChecked():
                cum_freq = np.cumsum(hist)
                ax.plot(midpoints, cum_freq, 'o-', color='blue')
                ax.set_title('Огіва (кумулятивна частота)')
            elif self.cumulative_checkbox.isChecked():
                rel_cum_freq = np.cumsum(hist) / len(data)
                ax.plot(midpoints, rel_cum_freq, 'o-', color='red')
                ax.set_title('Кумулята')
            elif self.ecdf_checkbox.isChecked():
                ax.step(data_sorted, np.arange(1, len(data) + 1) / len(data), where='post', color='purple')
                ax.set_title('Емпірична функція розподілу')
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, 'Оберіть хоча б один графік', ha='center', va='center')
                ax.set_title('Немає активних графіків')

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