'''
This file contains utility functions for miscellaneous tasks.
'''

import ipywidgets as widgets
import time
from typing import Callable, List, Tuple, TypeVar

import matplotlib.pyplot as plt
from IPython.display import display
from numpy.typing import NDArray

from src.utils.typing import RGBColor

# _______________________________ TIME _______________________________
class Timer:

    def __init__(self): self.start = time.time()

    def __str__ (self) -> str: 

        time = self()
        hh = int(time // 3600)
        mm = int((time % 3600) // 60)
        ss = time % 60
        
        if hh > 0: return f'{hh} hour{"s" if hh>1 else ""}, {mm} min, {int(ss)} sec'
        if mm > 0: return f'{mm} min, {int(ss)} sec'
        return f'{ss:.2f} sec'
    
    def __repr__(self) -> str: return str(self)

    def __call__(self) -> float: return time.time() - self.start

    def reset(self): self.start = time.time()

# _______________________________ COLOR _______________________________

def generate_palette(n: int, palette_type: str = "hsv") -> List[RGBColor]:
    
    if palette_type in plt.colormaps():
        # Usa le colormap predefinite di Matplotlib
        colormap = plt.cm.get_cmap(palette_type)
        palette: List[RGBColor] = [ # type: ignore - tuple has exact length three
            tuple(int(c * 255) for c in colormap(i / n)[:3]) 
            for i in range(n)
        ]
        return palette
    
    raise ValueError(
        f"Invalid palette_type '{palette_type}'. " 
        f"Available options are: {', '.join(plt.colormaps()[:10])} ... "
    )

# ------------------------------ NOTEBOOKS ------------------------------

def launch_widget(widgets_: List[widgets.Widget], update_fn: Callable):

    # Adjust the layout and style of each widget
    for widget in widgets_:
        # Set layout and style for better display
        # widget.layout = widgets.Layout(width='500px')  # type: ignore - Adjust overall width
        if hasattr(widget, 'style'):
            widget.style = {'description_width': '150px'}  # type: ignore -  Adjust label width
        
        # Attach observer
        widget.observe(update_fn, names='value')
    
    # Display widgets
    display(*widgets_)
    
    # Trigger initial update
    update_fn(change={'new': 0})

def display_frames(frames: List[Tuple[str, NDArray]], n_rows: int = 1):
    # Calculate the number of columns
    n_cols = (len(frames) + n_rows - 1) // n_rows  # Ceiling division

    # Adjust figure size dynamically
    figsize = (5 * n_cols, 5 * n_rows)
    
    plt.figure(figsize=figsize)
    
    for i, (title, frame) in enumerate(frames):
        # Determine the subplot index
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(frame, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.show()

