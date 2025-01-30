'''
This file contains typing utils and aliases used throughout the project.
'''

from typing import Dict, Literal, Tuple, TypeVar

import cv2

# ____________________________________ GENERICS ____________________________________

_T = TypeVar('_T')
_D = TypeVar('_D')

def default(var : _T | None, val : _D) -> _T | _D: return val if var is None else var
''' Get the default value if the variable is None. '''

# ____________________________________ ALIASES ____________________________________

Size2D = Tuple[int, int]
'''
Represents a 2D size as a tuple of integers for width and height. This is used in various contexts, such as kernel size, window size, etc.
'''

Pixel = Tuple[int, int]
'''
Represents the coordinates of an image pixel as a tuple of integers (x, y).
'''

Shape = Tuple[int, ...]
'''
Represents the shape of a tensor as a tuple of integers.
'''

RGBColor = Tuple[int, int, int] | Tuple[int, int, int, int]
'''
Represents an RGB color as a tuple of three integers, each in the range [0, 255].
Optionally, it can represent an RGBA color as a tuple of four integers, where the fourth value is the alpha channel.
'''

Frame = cv2.typing.MatLike
'''
Represents a frame as a 2D matrix (grayscale) or a 3D tensor (color image) of pixels.
'''

Views = Dict[str, Frame]
'''
Represents different frame views of a video stream, indexed by string keys.
'''

CameraPoseMethod = Literal['algebraic', 'geometric']
'''
Specifies the method used to estimate the camera pose. The options are:
    - 'algebraic': Minimizes the algebraic error to orthonormalize the rotation matrix.
    - 'geometric': Uses the P2P algorithm to minimize the geometric error in the image plane.
'''

MarkerSquareMethod = Literal['border', 'descendants', 'scaled']
'''
Specifies the method for computing the black-to-white transition in marker squares (outer black and inner white squares). The options are:
    - 'border'     : Uses a mask containing only the contour pixels of each square.
    - 'descendants': Subtracts the descendant contours from the filled square contour mask, based on the hierarchy of closed contour extraction.
    - 'scaled'     : Creates an artificial child contour by scaling down the square contour and subtracting it from the original square contour mask.
'''
