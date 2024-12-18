import time


class Timer:

    def __init__(self): self.start = time.time()

    def __str__ (self) -> str: return f'{(self()):.3f} sec'
    def __repr__(self) -> str: return str(self)

    def __call__(self) -> float: return time.time() - self.start

    def reset(self): self.start = time.time()
