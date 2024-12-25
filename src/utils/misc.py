import time
from typing import TypeVar

# _______________________________ TYPING _______________________________

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
        if mm > 0: return f'{mm} min, {ss:.2f} sec'
        return f'{ss:.2f} sec'
    
    def __repr__(self) -> str: return str(self)

    def __call__(self) -> float: return time.time() - self.start

    def reset(self): self.start = time.time()

