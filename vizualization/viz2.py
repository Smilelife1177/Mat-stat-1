from graphviz import Digraph

# Створюємо об'єкт Digraph
dot = Digraph(comment='Блок-схема додатка для статистичного аналізу', format='png')

# Налаштування стилю
dot.attr(rankdir='TB', size='10,10')
dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
dot.attr('edge', color='blue')

# Додаємо вузли для модулів
dot.node('main', 'main.py\n(Точка входу)', fillcolor='lightgreen')
dot.node('gui', 'gui.py\n(Інтерфейс)', fillcolor='lightyellow')
dot.node('logic', 'logic.py\n(Логіка обробки)', fillcolor='lightcoral')

# Додаємо вузли для ключових компонентів
dot.node('root', 'Tkinter Root\n(Головне вікно)', fillcolor='lightgray')
dot.node('gui_objects', 'GUI Objects\n(Кнопки, вкладки, графіки)', fillcolor='lightcyan')
dot.node('data', 'Дані\n(values, call_types)', fillcolor='lightpink')

# Додаємо вузли для основних функцій
dot.node('create_gui', 'create_gui()\nСтворює інтерфейс', fillcolor='lightyellow')
dot.node('init_logic', 'initialize_logic()\nПрив’язує логіку', fillcolor='lightcoral')
dot.node('load_data', 'load_data()\nЗавантаження даних', fillcolor='lightcoral')
dot.node('update_hist', 'update_histogram()\nОновлення гістограми', fillcolor='lightcoral')
dot.node('char', 'update_characteristics()\nХарактеристики', fillcolor='lightcoral')
dot.node('dist', 'plot_distribution_functions()\nФункції розподілу', fillcolor='lightcoral')
dot.node('exp', 'plot_exponential_distribution()\nІмовірнісна сітка', fillcolor='lightcoral')
dot.node('call_types', 'analyze_call_types()\nАналіз за типами', fillcolor='lightcoral')
dot.node('edit', 'Редагування\n(стандартизація, аномалії тощо)', fillcolor='lightcoral')

# Додаємо зв’язки
dot.edge('main', 'root', label='Створює')
dot.edge('main', 'gui', label='Викликає create_gui()')
dot.edge('gui', 'gui_objects', label='Повертає')
dot.edge('main', 'logic', label='Передає gui_objects до\ninitialize_logic()')
dot.edge('gui_objects', 'create_gui', label='Включає')
dot.edge('logic', 'init_logic', label='Включає')
dot.edge('init_logic', 'load_data', label='Прив’язує')
dot.edge('init_logic', 'update_hist', label='Прив’язує')
dot.edge('init_logic', 'char', label='Прив’язує')
dot.edge('init_logic', 'dist', label='Прив’язує')
dot.edge('init_logic', 'exp', label='Прив’язує')
dot.edge('init_logic', 'call_types', label='Прив’язує')
dot.edge('init_logic', 'edit', label='Прив’язує')
dot.edge('load_data', 'data', label='Оновлює')
dot.edge('update_hist', 'gui_objects', label='Оновлює гістограму')
dot.edge('char', 'gui_objects', label='Оновлює таблицю')
dot.edge('dist', 'gui_objects', label='Малює графік')
dot.edge('exp', 'gui_objects', label='Малює сітку')
dot.edge('call_types', 'gui_objects', label='Оновлює таблицю\nта гістограми')
dot.edge('edit', 'data', label='Модифікує')
dot.edge('data', 'update_hist', label='Використовує')
dot.edge('data', 'char', label='Використовує')
dot.edge('data', 'dist', label='Використовує')
dot.edge('data', 'exp', label='Використовує')
dot.edge('data', 'call_types', label='Використовує')

# Зберігаємо та відображаємо схему
dot.render('block_diagram', view=True)