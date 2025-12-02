# course04.py
import numpy as np
from scipy import stats
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


class Lab4Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.data_x = None
        self.data_y = None
        self.beta0 = None
        self.beta1 = None
        self.s2 = None        # оцінка дисперсії σ²
        self.R2 = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # === Параметри моделі ===
        params_group = QGroupBox("Параметри моделі Y = β₀ + β₁·X + ε")
        params_layout = QHBoxLayout()

        labels = ["n (кількість точок):", "β₀ (вільний член):", "β₁ (нахил):", "σ (ст. відхилення ε):"]
        defaults = ["50", "2.0", "1.5", "1.5"]
        self.edits = []

        for label, default in zip(labels, defaults):
            params_layout.addWidget(QLabel(label))
            edit = QLineEdit(default)
            edit.setFixedWidth(80)
            self.edits.append(edit)
            params_layout.addWidget(edit)

        gen_btn = QPushButton("Згенерувати дані")
        gen_btn.clicked.connect(self.generate_data)
        params_layout.addWidget(gen_btn)
        params_layout.addStretch()
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # === Чекбокси для графіків ===
        cb_layout = QHBoxLayout()
        self.cb_scatter = QCheckBox("Діаграма розсіювання + регресія")
        self.cb_conf_int = QCheckBox("Довірчий інтервал для середнього відгуку")
        self.cb_pred_int = QCheckBox("Прогнозний інтервал")

        self.checkboxes = [self.cb_scatter, self.cb_conf_int, self.cb_pred_int]
        for cb in self.checkboxes:
            cb.stateChanged.connect(self.update_plots)
        self.cb_scatter.setChecked(True)
        self.cb_conf_int.setChecked(True)

        for cb in self.checkboxes:
            cb_layout.addWidget(cb)
        layout.addLayout(cb_layout)

        # === Графік ===
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=3)

        # === Результати ===
        self.result_label = QLabel("Згенеруйте дані для аналізу регресії.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-family: Consolas; font-size: 11pt;")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label, stretch=1)

        self.setLayout(layout)

    def generate_data(self):
        try:
            n = int(self.edits[0].text())
            beta0_true = float(self.edits[1].text())
            beta1_true = float(self.edits[2].text())
            sigma = float(self.edits[3].text())

            if n < 3 or sigma <= 0:
                raise ValueError()

            # Рівномірно розподілені X на [0, 10]
            self.data_x = np.random.uniform(0, 10, n)
            epsilon = np.random.normal(0, sigma, n)
            self.data_y = beta0_true + beta1_true * self.data_x + epsilon

            self.perform_regression()
            self.update_plots()

        except Exception as e:
            self.result_label.setText("Помилка: перевірте правильність введених даних! (n≥3, σ>0)")

    def perform_regression(self):
        x, y = self.data_x, self.data_y
        n = len(x)

        # МНК оцінки
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        self.beta1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
        self.beta0 = y_mean - self.beta1 * x_mean

        # Залишкова сума квадратів та оцінка дисперсії
        y_pred = self.beta0 + self.beta1 * x
        residuals = y - y_pred
        RSS = np.sum(residuals**2)
        self.s2 = RSS / (n - 2)  # s² — незміщена оцінка σ²
        s = np.sqrt(self.s2)      # стандартна похибка

        # Стандартні похибки коефіцієнтів
        x_var = np.var(x, ddof=1)
        se_beta1 = s / np.sqrt(np.sum((x - x_mean)**2))
        se_beta0 = s * np.sqrt(1/n + x_mean**2 / np.sum((x - x_mean)**2))

        # t-статистики та p-values
        t_beta0 = self.beta0 / se_beta0
        t_beta1 = self.beta1 / se_beta1
        p_beta0 = 2 * stats.t.sf(np.abs(t_beta0), n-2)
        p_beta1 = 2 * stats.t.sf(np.abs(t_beta1), n-2)

        # Коефіцієнти детермінації
        TSS = np.sum((y - y_mean)**2)
        self.R2 = 1 - RSS / TSS
        R2_adj = 1 - (1 - self.R2) * (n-1)/(n-2)

        # Довірчі інтервали (95%)
        alpha = 0.05
        t_crit = stats.t.ppf(1 - alpha/2, n-2)

        ci_beta0 = (self.beta0 - t_crit * se_beta0, self.beta0 + t_crit * se_beta0)
        ci_beta1 = (self.beta1 - t_crit * se_beta1, self.beta1 + t_crit * se_beta1)

        # Текст результатів
        results = f"""
<b>Результати оцінювання лінійної регресії Y = β₀ + β₁·X</b>

Оцінки параметрів (МНК):
    β₀ = {self.beta0:+.4f}    (SE = {se_beta0:.4f}, t = {t_beta0:+.3f}, p = {p_beta0:.4f})
    β₁ = {self.beta1:+.4f}    (SE = {se_beta1:.4f}, t = {t_beta1:+.3f}, p = {p_beta1:.4f})

Оцінка дисперсії шуму: σ² ≈ {self.s2:.4f}   (s = {s:.4f})

Коефіцієнт детермінації:
    R² = {self.R2:.4f}
    R² скоригований = {R2_adj:.4f}

Довірчі інтервали (95%):
    β₀ ∈ [{ci_beta0[0]:.4f}; {ci_beta0[1]:.4f}]
    β₁ ∈ [{ci_beta1[0]:.4f}; {ci_beta1[1]:.4f}]
        """.strip()

        self.result_label.setText(results)

    def update_plots(self):
        if self.data_x is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.scatter(self.data_x, self.data_y, color='steelblue', alpha=0.7, label='Спостереження')

        # Лінія регресії
        x_line = np.linspace(self.data_x.min(), self.data_x.max(), 500)
        y_line = self.beta0 + self.beta1 * x_line
        ax.plot(x_line, y_line, color='red', linewidth=2, label=f'Y = {self.beta0:+.3f} + {self.beta1:+.3f}·X')

        if self.cb_conf_int.isChecked() or self.cb_pred_int.isChecked():
            x_mean = np.mean(self.data_x)
            n = len(self.data_x)
            s = np.sqrt(self.s2)
            t_crit = stats.t.ppf(0.975, n-2)

            # Похибка для прогнозу середнього
            se_mean = s * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((self.data_x - x_mean)**2))
            # Похибка для індивідуального прогнозу
            se_pred = s * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / np.sum((self.data_x - x_mean)**2))

            lower_ci = y_line - t_crit * se_mean
            upper_ci = y_line + t_crit * se_mean
            lower_pi = y_line - t_crit * se_pred
            upper_pi = y_line + t_crit * se_pred

            if self.cb_conf_int.isChecked():
                ax.fill_between(x_line, lower_ci, upper_ci, color='orange', alpha=0.3,
                                label='95% довірчий інтервал (середнє)')

            if self.cb_pred_int.isChecked():
                ax.fill_between(x_line, lower_pi, upper_pi, color='green', alpha=0.15,
                                label='95% прогнозний інтервал')

        ax.set_title('Лінійна регресія з довірчими та прогнозними інтервалами', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()


def create_tab(tab_widget):
    tab_widget.addTab(Lab4Widget(), "Lab 4: Лінійна регресія (МНК, R², довірчі інтервали)")