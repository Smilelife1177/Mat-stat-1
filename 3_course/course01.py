# course01.py
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
        self.generate_plots()  # Initial plot generation

    def initUI(self):
        layout = QVBoxLayout()

        # Input frame
        input_layout = QHBoxLayout()
        
        # Parameters
        self.n_edit = QLineEdit("100")
        self.mean_edit = QLineEdit("0")
        self.std_edit = QLineEdit("1")
        self.bins_edit = QLineEdit("10")  # Input for number of bins
        
        labels = ["Кількість чисел (n):", "Математичне сподівання:", 
                  "Стандартне відхилення:", "Кількість інтервалів:"]
        edits = [self.n_edit, self.mean_edit, self.std_edit, self.bins_edit]
        
        for label, edit in zip(labels, edits):
            input_layout.addWidget(QLabel(label))
            input_layout.addWidget(edit)
        
        generate_btn = QPushButton("Згенерувати нормальний розподіл")
        generate_btn.clicked.connect(self.generate_plots)
        input_layout.addWidget(generate_btn)
        
        layout.addLayout(input_layout)

        # Checkbox frame
        checkbox_layout = QHBoxLayout()
        self.hist_checkbox = QCheckBox("Гістограма")
        self.hist_checkbox.setChecked(True)  # Set "Гістограма" as checked by default
        self.polygon_checkbox = QCheckBox("Полігон частот")
        self.ogive_checkbox = QCheckBox("Огіва")
        self.cumulative_checkbox = QCheckBox("Кумулята")
        self.ecdf_checkbox = QCheckBox("Емпірична функція розподілу")
        self.ecdf_bins_checkbox = QCheckBox("Емпірична функція (інтервальний ряд)")  # New checkbox

        # Store checkboxes in a list for easier management
        self.checkboxes = [
            self.hist_checkbox,
            self.polygon_checkbox,
            self.ogive_checkbox,
            self.cumulative_checkbox,
            self.ecdf_checkbox,
            self.ecdf_bins_checkbox
        ]

        # Connect checkbox state changes to handler
        for checkbox in self.checkboxes:
            checkbox.stateChanged.connect(self.handle_checkbox_change)
            checkbox_layout.addWidget(checkbox)
        
        layout.addLayout(checkbox_layout)

        # Matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["N", "Межі інтервалу", "Середній інтервал", 
                                            "Частота", "Відноша частота", "Накопичена відноша частота"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        layout.addWidget(self.table, stretch=1)

        # Results
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def handle_checkbox_change(self):
        # Ensure only one checkbox is checked
        sender = self.sender()
        if sender.isChecked():
            for checkbox in self.checkboxes:
                if checkbox != sender:
                    checkbox.setChecked(False)
            self.generate_plots()

    def generate_plots(self):
        try:
            # Get parameters
            n = int(self.n_edit.text())
            mean = float(self.mean_edit.text())
            std = float(self.std_edit.text())
            num_bins = int(self.bins_edit.text())  # Get number of bins

            if n <= 0 or std <= 0 or num_bins <= 0:
                self.result_label.setText("Помилка: n, std і кількість інтервалів повинні бути додатними!")
                return

            # Generate data
            data = np.random.normal(loc=mean, scale=std, size=n)
            data_sorted = np.sort(data)

            # Create interval series
            hist, bin_edges = np.histogram(data, bins=num_bins)
            midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Calculate estimates based on interval series
            # Mathematical expectation: sum(midpoint * frequency) / n
            mean_est = np.sum(midpoints * hist) / len(data)
            # Variance: sum(frequency * (midpoint - mean_est)^2) / (n - 1)
            var_est = np.sum(hist * (midpoints - mean_est)**2) / (len(data) - 1)

            # Fill table with interval series data
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

            # Clear figure
            self.figure.clear()

            # Set figure size
            self.figure.set_size_inches(12, 6)

            # Create single subplot
            ax = self.figure.add_subplot(111)
            self.figure.suptitle('Графік для лабораторної роботи 1')

            # Determine active plot
            if self.hist_checkbox.isChecked():
                ax.hist(data, bins=num_bins, edgecolor='black', color='blue')
                ax.set_title('Гістограма (інтервальний статистичний ряд)')
            elif self.polygon_checkbox.isChecked():
                ax.plot(midpoints, hist, 'o-', color='green')
                ax.set_title('Полігон частот')
            elif self.ogive_checkbox.isChecked():
                rel_cum_freq = np.cumsum(hist) / len(data)
                ax.hist(bin_edges[:-1], bins=bin_edges, weights=rel_cum_freq, edgecolor='black', color='blue')
                ax.set_title('Огіва (накопичена відносна частота)')
                ax.set_ylim(0, 1)
            elif self.cumulative_checkbox.isChecked():
                rel_cum_freq = np.cumsum(hist) / len(data)
                ax.plot(midpoints, rel_cum_freq, 'o-', color='red')
                ax.set_title('Кумулята')
            elif self.ecdf_checkbox.isChecked():
                ax.step(data_sorted, np.arange(1, len(data) + 1) / len(data), where='post', color='purple')
                ax.set_title('Емпірична функція розподілу')
                ax.set_ylim(0, 1)
            elif self.ecdf_bins_checkbox.isChecked():
                rel_cum_freq = np.cumsum(hist) / len(data)
                ax.step(bin_edges[:-1], rel_cum_freq, where='post', color='orange')
                ax.set_title('Емпірична функція розподілу (інтервальний ряд)')
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, 'Оберіть хоча б один графік', ha='center', va='center')
                ax.set_title('Немає активних графіків')

            # Update canvas
            self.figure.tight_layout()
            self.canvas.draw()

            # Display results with explicit mention of interval series
            results_text = (f"Оцінка математичного сподівання (на основі інтервального ряду): {mean_est:.4f}\n"
                           f"Оцінка дисперсії (на основі інтервального ряду): {var_est:.4f}")
            self.result_label.setText(results_text)

        except ValueError:
            self.result_label.setText("Помилка: Введіть коректні числові значення!")

def create_tab(tab_widget):
    lab1_widget = Lab1Widget()
    tab_widget.addTab(lab1_widget, "Lab 1: Обробка вибірки")