from typing import Dict, Tuple

import cv2

Size2D     = Tuple[int, int]
Shape      = Tuple[int, ...]
RGBColor   = Tuple[int, int, int] | Tuple[int, int, int, int]
Frame      = cv2.typing.MatLike
Views      = Dict[str, Frame]