# course03.py
import numpy as np
from scipy import stats
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTabWidget, QTableWidget, QTableWidgetItem, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Lab3Widget(QWidget):
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
        self.regression_cb = QCheckBox("Регресійні криві")
        self.ranks_cb = QCheckBox("Ранги (X, Y)")

        self.checkboxes = [self.scatter_cb, self.regression_cb, self.ranks_cb]
        for cb in self.checkboxes:
            cb.stateChanged.connect(self.handle_checkbox_change)
            checkbox_layout.addWidget(cb)
        self.scatter_cb.setChecked(True)
        layout.addLayout(checkbox_layout)

        # --- Графік ---
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=2)

        # --- Таблиця ---
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["№", "X", "Y", "Rang(X)", "Rang(Y)", "d_i", "d_i²"])
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

        # Обчислення рангів
        rank_x = stats.rankdata(self.data_x)
        rank_y = stats.rankdata(self.data_y)
        d_i = rank_x - rank_y
        d_i2 = d_i ** 2

        for i in range(n):
            self.table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{self.data_x[i]:.4f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{self.data_y[i]:.4f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{rank_x[i]:.1f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{rank_y[i]:.1f}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{d_i[i]:.1f}"))
            self.table.setItem(i, 6, QTableWidgetItem(f"{d_i2[i]:.1f}"))

    def calculate_statistics(self):
        if self.data_x is None: return
        n = len(self.data_x)

        # --- Коефіцієнт кореляційного відношення η(Y|X) ---
        # Групуємо Y за унікальними значеннями X (для оцінки)
        unique_x, indices = np.unique(self.data_x, return_inverse=True)
        group_means = np.array([np.mean(self.data_y[indices == i]) for i in range(len(unique_x))])
        y_pred = group_means[indices]

        ss_total = np.sum((self.data_y - np.mean(self.data_y))**2)
        ss_explained = np.sum((y_pred - np.mean(self.data_y))**2)
        eta_yx = np.sqrt(ss_explained / ss_total) if ss_total > 0 else 0

        # Значущість η²: F-статистика
        k = len(unique_x)  # кількість груп
        if k > 1 and n > k:
            F_eta = (ss_explained / (k - 1)) / ((ss_total - ss_explained) / (n - k))
            p_eta = stats.f.sf(F_eta, k-1, n-k)
        else:
            F_eta = p_eta = 0

        # --- Коефіцієнт Спірмена ---
        rho_spearman, p_spearman = stats.spearmanr(self.data_x, self.data_y)

        # --- Коефіцієнт Кендалла ---
        tau_kendall, p_kendall = stats.kendalltau(self.data_x, self.data_y)

        # --- Коефіцієнт Пірсона (для порівняння) ---
        r_pearson = np.corrcoef(self.data_x, self.data_y)[0,1]

        results = (
            f"Коефіцієнт кореляційного відношення η(Y|X) = {eta_yx:.4f}\n"
            f"  F-статистика = {F_eta:.4f}, p-value = {p_eta:.4f}\n"
            f"Коефіцієнт Спірмена ρ = {rho_spearman:.4f}, p-value = {p_spearman:.4f}\n"
            f"Коефіцієнт Кендалла τ = {tau_kendall:.4f}, p-value = {p_kendall:.4f}\n"
            f"Коефіцієнт Пірсона r = {r_pearson:.4f} (для порівняння)"
        )
        self.result_label.setText(results)

    def update_plots(self):
        if self.data_x is None or self.data_y is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.suptitle('Лабораторна робота 3: Кореляційне відношення, Спірмен, Кендалл')

        if self.scatter_cb.isChecked():
            ax.scatter(self.data_x, self.data_y, alpha=0.6, color='blue')
            ax.set_title('Діаграма розсіювання')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)

        elif self.regression_cb.isChecked():
            ax.scatter(self.data_x, self.data_y, alpha=0.6, color='lightgray', label='Точки')

            # Регресія Y ~ X
            z_yx = np.polyfit(self.data_x, self.data_y, 1)
            p_yx = np.poly1d(z_yx)
            x_sorted = np.sort(self.data_x)
            ax.plot(x_sorted, p_yx(x_sorted), "r-", linewidth=2, label='Y ~ X')

            # Регресія X ~ Y
            z_xy = np.polyfit(self.data_y, self.data_x, 1)
            p_xy = np.poly1d(z_xy)
            y_sorted = np.sort(self.data_y)
            ax.plot(p_xy(y_sorted), y_sorted, "g--", linewidth=2, label='X ~ Y')

            ax.set_title('Регресійні криві')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid(True, alpha=0.3)

        elif self.ranks_cb.isChecked():
            rank_x = stats.rankdata(self.data_x)
            rank_y = stats.rankdata(self.data_y)
            ax.scatter(rank_x, rank_y, alpha=0.7, color='purple')
            ax.set_title('Діаграма розсіювання рангів')
            ax.set_xlabel('Rang(X)')
            ax.set_ylabel('Rang(Y)')
            ax.grid(True, alpha=0.3)

        else:
            ax.text(0.5, 0.5, 'Оберіть графік', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)

        self.figure.tight_layout()
        self.canvas.draw()


def create_tab(tab_widget: QTabWidget):
    lab3_widget = Lab3Widget()
    tab_widget.addTab(lab3_widget, "Lab 3: Кореляційне відношення, Спірмен, Кендалл")