# course_main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
import course01 as lab1
import course02 as lab2  # Нова лабораторна

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Лабораторні роботи з математичної статистики")
        self.setGeometry(100, 100, 1200, 1000)

        # Центральний віджет - вкладки
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Додавання лабораторних робіт
        lab1.create_tab(self.tab_widget)
        lab2.create_tab(self.tab_widget)  # Додаємо Lab 2

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())