import math
import time
from typing import Tuple, TypeVar

# _______________________________ TYPING _______________________________

# Type alias
Size  = Tuple[int, int]
Shape = Tuple[int, ...]

# Type generics
T = TypeVar('T')
D = TypeVar('D')

# Default for None with value
def default(var : T | None, val : D) -> T | D:
    return val if var is None else var


# _______________________________ TIME _______________________________
class Timer:

    def __init__(self): self.start = time.time()

    def __str__ (self) -> str: 

        time = self()
        hh = int(time // 3600)
        mm = int((time % 3600) // 60)
        ss = time % 60
        
        if hh > 0: return f'{hh} hour{"s" if hh>1 else ""}, {mm} min, {ss:.2f} sec'
        if mm > 0: return f'{mm} min, {int(ss)} sec'
        return f'{ss:.2f} sec'
    
    def __repr__(self) -> str: return str(self)

    def __call__(self) -> float: return time.time() - self.start

    def reset(self): self.start = time.time()



def grid_size(n: int) -> Tuple[int, int]:

    # Start with the square root of the number of images
    rows = int(math.sqrt(n))
    cols = math.ceil(n / rows)

    # Adjust rows and cols if necessary
    while rows * cols < n:
        rows += 1
        cols = math.ceil(n / rows)
    
    return rows, cols


