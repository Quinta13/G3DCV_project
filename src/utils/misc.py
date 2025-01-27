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
    ''' Timer class to measure the time elapsed since its creation. '''

    def __init__(self): self.reset()

    def __str__ (self) -> str: 

        time = self()
        hh = int(time // 3600)
        mm = int((time % 3600) // 60)
        ss = time % 60
        
        if hh > 0: return f'{hh} hour{"s" if hh>1 else ""}, {mm} min, {int(ss)} sec'
        if mm > 0: return f'{mm} min, {int(ss)} sec'
        return f'{ss:.2f} sec'
    
    def __repr__(self) -> str   : return str(self)
    def __call__(self) -> float : return time.time() - self.start

    def reset(self): self.start = time.time()

# _______________________________ COLOR _______________________________

def generate_palette(n: int, palette_type: str = "hsv") -> List[RGBColor]:
    ''' Generate a palette of n colors using the specified palette type. '''
    
    if palette_type in plt.colormaps():

        # Usa le colormap predefinite di Matplotlib
        colormap = plt.cm.get_cmap(palette_type)

        palette: List[RGBColor] = [
            tuple(int(c * 255) for c in colormap(i / n)[:3]) 
            for i in range(n)
        ]  # type: ignore - tuple has exact length three
        return palette
    
    raise ValueError(
        f"Invalid palette_type '{palette_type}'. " 
        f"Available options are: {', '.join(plt.colormaps()[:10])} ... "
    )

# ------------------------------ NOTEBOOKS ------------------------------

def launch_widget(widgets_: List[widgets.Widget], update_fn: Callable):
    ''' Launch a widget with the specified widgets and update function. '''

    # Adjust the layout and style of each widget
    for widget in widgets_:
        if hasattr(widget, 'style'): widget.style = {'description_width': '150px'}  # type: ignore -  Adjust label width
        widget.observe(update_fn, names='value')
    
    display(*widgets_)            # Display widgets
    update_fn(change={'new': 0})  # Trigger initial update

def display_frame_views(views: List[Tuple[str, NDArray]], n_rows: int = 1, figsize: Tuple[int, int] = (10, 10)):
    ''' Display a list of frames with the specified titles. '''

    # Calculate the number of columns
    n_cols = (len(views) + n_rows - 1) // n_rows  # Ceiling division    
    plt.figure(figsize=figsize)
    
    for i, (title, frame) in enumerate(views):
    
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(frame, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.show()

