# course02.py
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTabWidget, QTableWidget, QTableWidgetItem, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import chi2 as chi2_dist


class Lab2Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.data_x = None
        self.data_y = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # --- Параметри ---
        input_layout = QHBoxLayout()
        labels = ["Кількість пар (n):", "М.спод. X:", "Ст.відх. X:", "М.спод. Y:", "Ст.відх. Y:", "Кореляція ρ:"]
        defaults = ["100", "0", "1", "0", "1", "0.5"]
        self.edits = []

        for label, default in zip(labels, defaults):
            input_layout.addWidget(QLabel(label))
            edit = QLineEdit(default)
            self.edits.append(edit)
            input_layout.addWidget(edit)

        generate_btn = QPushButton("Згенерувати двовимірну вибірку")
        generate_btn.clicked.connect(self.generate_data)
        input_layout.addWidget(generate_btn)
        layout.addLayout(input_layout)

        # --- Чекбокси ---
        checkbox_layout = QHBoxLayout()
        self.scatter_cb = QCheckBox("Діаграма розсіювання")
        self.corr_field_cb = QCheckBox("Поле кореляції")
        self.hist2d_cb = QCheckBox("2D гістограма")

        self.checkboxes = [self.scatter_cb, self.corr_field_cb, self.hist2d_cb]
        for cb in self.checkboxes:
            cb.stateChanged.connect(self.handle_checkbox_change)
            checkbox_layout.addWidget(cb)
        self.scatter_cb.setChecked(True)  # За замовчуванням
        layout.addLayout(checkbox_layout)

        # --- Графік ---
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=2)

        # --- Таблиця ---
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["№", "X", "Y", "X²", "XY"])
        layout.addWidget(self.table, stretch=1)

        # --- Результати ---
        self.result_label = QLabel("Згенеруйте дані для аналізу.")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def handle_checkbox_change(self):
        sender = self.sender()
        if sender.isChecked():
            for cb in self.checkboxes:
                if cb != sender:
                    cb.setChecked(False)
            self.update_plots()

    def generate_data(self):
        try:
            n = int(self.edits[0].text())
            mux = float(self.edits[1].text())
            sigx = float(self.edits[2].text())
            muy = float(self.edits[3].text())
            sigy = float(self.edits[4].text())
            rho = float(self.edits[5].text())

            if n <= 0 or sigx <= 0 or sigy <= 0 or not (-1 <= rho <= 1):
                raise ValueError()

            mean = [mux, muy]
            cov = [[sigx**2, rho*sigx*sigy], [rho*sigx*sigy, sigy**2]]
            self.data_x, self.data_y = np.random.multivariate_normal(mean, cov, n).T

            self.fill_table()
            self.update_plots()
            self.calculate_statistics()

        except Exception:
            self.result_label.setText("Помилка: введіть коректні значення! (n>0, σ>0, |ρ|≤1)")

    def fill_table(self):
        if self.data_x is None: return
        n = len(self.data_x)
        self.table.setRowCount(n)
        for i in range(n):
            self.table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{self.data_x[i]:.4f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{self.data_y[i]:.4f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{self.data_x[i]**2:.4f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{self.data_x[i]*self.data_y[i]:.4f}"))

    def calculate_statistics(self):
        if self.data_x is None: return
        n = len(self.data_x)
        mx = np.mean(self.data_x)
        my = np.mean(self.data_y)
        vx = np.var(self.data_x, ddof=1)
        vy = np.var(self.data_y, ddof=1)
        cov_xy = np.cov(self.data_x, self.data_y, ddof=1)[0, 1]
        r = cov_xy / np.sqrt(vx * vy) if vx > 0 and vy > 0 else 0

        chi2_stat, chi2_p = self.chi_square_test()
        normality_text = f"χ²-критерій (X): {chi2_stat:.4f}, p-value = {chi2_p:.4f}"

        results = (
            f"Оцінки:\n"
            f"  M[X] = {mx:.4f},  M[Y] = {my:.4f}\n"
            f"  D[X] = {vx:.4f},  D[Y] = {vy:.4f}\n"
            f"  Cov(X,Y) = {cov_xy:.4f}\n"
            f"  Коеф. кореляції r = {r:.4f}\n"
            f"{normality_text}"
        )
        self.result_label.setText(results)

    def chi_square_test(self):
        if self.data_x is None: return 0, 1
        data = self.data_x
        hist, _ = np.histogram(data, bins=10)
        n = len(data)
        expected = n / 10
        chi2 = np.sum((hist - expected)**2 / expected)
        p_value = 1 - chi2_dist.cdf(chi2, df=9)
        return chi2, p_value

    def update_plots(self):
        if self.data_x is None or self.data_y is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.suptitle('Лабораторна робота 2: Двовимірна вибірка')

        if self.scatter_cb.isChecked():
            ax.scatter(self.data_x, self.data_y, alpha=0.6, color='blue')
            ax.set_title('Діаграма розсіювання')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)

        elif self.corr_field_cb.isChecked():
            ax.scatter(self.data_x, self.data_y, alpha=0.6, color='green')
            z = np.polyfit(self.data_x, self.data_y, 1)
            p = np.poly1d(z)
            sorted_idx = np.argsort(self.data_x)
            ax.plot(self.data_x[sorted_idx], p(self.data_x[sorted_idx]), "r--", linewidth=2)
            r = np.corrcoef(self.data_x, self.data_y)[0,1]
            ax.set_title(f'Поле кореляції (r ≈ {r:.3f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)

        elif self.hist2d_cb.isChecked():
            hb = ax.hist2d(self.data_x, self.data_y, bins=15, cmap='Blues')
            self.figure.colorbar(hb[3], ax=ax, label='Частота')
            ax.set_title('2D Гістограма')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        else:
            ax.text(0.5, 0.5, 'Оберіть графік', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)

        self.figure.tight_layout()
        self.canvas.draw()


def create_tab(tab_widget: QTabWidget):
    lab2_widget = Lab2Widget()
    tab_widget.addTab(lab2_widget, "Lab 2: Двовимірні вибірки")