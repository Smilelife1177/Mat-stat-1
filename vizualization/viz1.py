import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')  # Hide axes for a clean look

# Define square properties
square_size = 2.5
colors = {
    'main': '#A3BEF8',    # Light blue
    'gui': '#F8A3A3',     # Light red
    'logic': '#A3F8A3'    # Light green
}
font_size = 14
font_weight = 'bold'

# Positions for squares (x, y)
positions = {
    'main': (1, 3.5),
    'gui': (4, 3.5),
    'logic': (7, 3.5)
}

# Draw squares and labels
for module, (x, y) in positions.items():
    # Draw square
    square = patches.Rectangle(
        (x - square_size/2, y - square_size/2),
        square_size,
        square_size,
        facecolor=colors[module],
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(square)
    
    # Add module label
    label = module.replace('.py', '').upper()
    ax.text(
        x, y,
        label,
        ha='center',
        va='center',
        fontsize=font_size,
        fontweight=font_weight,
        color='black'
    )

# Draw arrows
arrowprops = dict(
    arrowstyle='->',
    linewidth=1.5,
    color='black',
    connectionstyle='arc3,rad=0.1'
)

# main.py -> gui.py (initializes GUI)
ax.annotate(
    '',
    xy=(positions['gui'][0] - square_size/2, positions['gui'][1]),
    xytext=(positions['main'][0] + square_size/2, positions['main'][1]),
    arrowprops=arrowprops
)
ax.text(
    (positions['main'][0] + positions['gui'][0])/2, positions['main'][1] + 0.3,
    'Ініціалізує',
    ha='center',
    fontsize=10,
    color='black'
)

# main.py -> logic.py (initializes logic)
ax.annotate(
    '',
    xy=(positions['logic'][0] - square_size/2, positions['logic'][1]),
    xytext=(positions['main'][0] + square_size/2, positions['main'][1]),
    arrowprops=arrowprops
)
ax.text(
    (positions['main'][0] + positions['logic'][0])/2, positions['main'][1] + 0.3,
    'Ініціалізує',
    ha='center',
    fontsize=10,
    color='black'
)

# gui.py -> logic.py (sends user inputs)
ax.annotate(
    '',
    xy=(positions['logic'][0] - square_size/2, positions['logic'][1] - 0.2),
    xytext=(positions['gui'][0] + square_size/2, positions['gui'][1] - 0.2),
    arrowprops=arrowprops
)
ax.text(
    (positions['gui'][0] + positions['logic'][0])/2, positions['gui'][1] - 0.5,
    'Ввід користувача',
    ha='center',
    fontsize=10,
    color='black'
)

# logic.py -> gui.py (returns results)
ax.annotate(
    '',
    xy=(positions['gui'][0] + square_size/2, positions['gui'][1] + 0.2),
    xytext=(positions['logic'][0] - square_size/2, positions['logic'][1] + 0.2),
    arrowprops=arrowprops
)
ax.text(
    (positions['gui'][0] + positions['logic'][0])/2, positions['gui'][1] + 0.7,
    'Результати',
    ha='center',
    fontsize=10,
    color='black'
)

# Title
plt.title(
    'Взаємодія модулів програми',
    fontsize=16,
    fontweight='bold',
    pad=20,
    loc='center'
)

# Save and show
plt.tight_layout()
plt.savefig('module_interaction.png', dpi=300, bbox_inches='tight')
plt.show()