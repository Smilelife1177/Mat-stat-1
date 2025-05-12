import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, ConnectionPatch
import matplotlib.patches as patches

def draw_module_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Визначення модулів (прямокутники)
    modules = [
        {"name": "main.py", "x": 2, "y": 7, "color": "lightblue"},
        {"name": "gui.py\n(create_gui)", "x": 2, "y": 5, "color": "lightgreen"},
        {"name": "logic.py\n(initialize_logic)", "x": 2, "y": 3, "color": "lightcoral"},
        {"name": "gui_objects\n(словник)", "x": 5, "y": 4, "color": "lightyellow"},
        {"name": "Користувач\n(взаємодія)", "x": 5, "y": 6, "color": "lightpink"},
        {"name": "Оновлення GUI\n(таблиці, графіки)", "x": 5, "y": 2, "color": "lightgoldenrodyellow"}
    ]

    # Малювання модулів
    for module in modules:
        ax.add_patch(FancyBboxPatch(
            (module["x"] - 1, module["y"] - 0.5), 2, 1,
            boxstyle="round,pad=0.3", facecolor=module["color"], edgecolor="black"
        ))
        ax.text(
            module["x"], module["y"], module["name"],
            ha="center", va="center", fontsize=10, weight="bold"
        )

    # Малювання стрілок для взаємодій
    arrows = [
        # main.py -> create_gui
        {"start": (2, 6.5), "end": (2, 5.5), "label": "викликає"},
        # create_gui -> gui_objects
        {"start": (2, 4.5), "end": (5, 4.5), "label": "повертає"},
        # gui_objects -> initialize_logic
        {"start": (5, 3.5), "end": (2, 3.5), "label": "передається"},
        # initialize_logic -> gui_objects
        {"start": (2, 2.5), "end": (5, 2.5), "label": "прив’язує функції\n(load_data, analyze_call_types)"},
        # Користувач -> gui_objects
        {"start": (5, 5.5), "end": (5, 4.5), "label": "натискає кнопки"},
        # gui_objects -> Оновлення GUI
        {"start": (5, 3.5), "end": (5, 2.5), "label": "оновлює"}
    ]

    for arrow in arrows:
        ax.add_patch(ConnectionPatch(
            xyA=arrow["start"], xyB=arrow["end"],
            coordsA="data", coordsB="data",
            arrowstyle=ArrowStyle("->", head_length=0.4, head_width=0.2),
            linewidth=1.5, color="black"
        ))
        # Додавання тексту до стрілок
        ax.text(
            (arrow["start"][0] + arrow["end"][0]) / 2,
            (arrow["start"][1] + arrow["end"][1]) / 2 + 0.2,
            arrow["label"],
            ha="center", va="center", fontsize=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

    # Налаштування осей
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_axis_off()
    plt.title("Схема взаємодії модулів", fontsize=14, pad=20)
    plt.tight_layout()

    # Показати діаграму
    plt.show()

# Виклик функції для створення діаграми
draw_module_diagram()