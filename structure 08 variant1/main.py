import tkinter as tk
from gui import create_gui
from logic import initialize_logic

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Статистичний аналіз")
    root.state('zoomed')  
    
    gui_objects = create_gui(root)
    
    initialize_logic(gui_objects)
    
    root.mainloop()