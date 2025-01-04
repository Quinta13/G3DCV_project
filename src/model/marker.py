from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple
from dataclasses import dataclass

import numpy as np
import cv2 as cv
from numpy.typing import NDArray

from src.model.typing import Frame, RGBColor
from src.utils.misc   import generate_palette

class SortedVertices:

	def __init__(self, vertices: NDArray) -> None:

		self._vertices = SortedVertices._sort_point(vertices=vertices)

	@staticmethod
	def _sort_point(vertices: NDArray) -> NDArray:
    
		# Calculate the center of the contour
		center = np.mean(vertices, axis=0)
		
		# Calculate the angle of each point w.r.t. the center
		angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
		
		# Sort vertices by angle
		sorted_indices = np.argsort(angles)

		return vertices[sorted_indices]

	def __str__    (self) -> str: return f'SortedVertices[points={len(self)}]'
	def __repr__   (self) -> str: return str(self)
	def __len__    (self) -> int: return len(self._vertices)

	def __getitem__(self, key: int) -> Tuple[int, int]: 
		x, y = tuple(self._vertices[key])
		return int(x), int(y)

	@property
	def vertices(self) -> NDArray: return self._vertices

	def draw(self, frame: Frame, palette: List[RGBColor] | RGBColor = (255, 0, 0), radius: int = 5, thickness: int = 2) -> Frame:

		palette_ = palette if isinstance(palette, list) else [palette] * len(self)

		if len(palette_) != len(self): raise ValueError(f'Palette must have {len(self)} colors, got {len(palette)}.')

		# Draw lines
		for i, col in enumerate(palette_): cv.circle(img=frame, center=self[i], radius=radius, color=col, thickness=thickness)

		return frame

	def shift(self, n: int): self._vertices = np.roll(self._vertices, -n, axis=0)


class Contour:

	@dataclass
	class ContourHierarchy:

		next        : int | None
		previous    : int | None
		first_child : int | None
		parent      : int | None

		def __str__ (self) -> str: return f'ContourHierarchy[{"; ".join([f"{k}: {v}" for k, v in self.to_dict().items()])}]'
		
		def __repr__(self) -> str: return str(self)
		
		def to_dict(self) -> Dict[str, int | None]: return {
			'next'       : self.next,
			'previous'   : self.previous,
			'first_child': self.first_child,
			'parent'     : self.parent
		} 

		@classmethod
		def from_hierarchy(cls, hierarchy: NDArray) -> Contour.ContourHierarchy:

			def default_value(idx: int) -> int | None: return int(idx) if idx != -1 else None
			
			return cls(
				next        = default_value(hierarchy[0]),
				previous    = default_value(hierarchy[1]),
				first_child = default_value(hierarchy[2]),
				parent      = default_value(hierarchy[3])
			)
		
	
	# HYPERPARAMETERS
	_APPROX_FACTOR         : float = 0.01
	_CIRCULARITY_THRESHOLD : float = 0.85
    
	def __init__(self, id: int, contour: NDArray, hierarchy: NDArray):

		self._id             = id
		self._contour_orig   = contour
		self._contour_approx = Contour.approx_contour(contour=contour)
		self._hierarchy      = Contour.ContourHierarchy.from_hierarchy(hierarchy=hierarchy)

	def __str__ (self) -> str: return f'Contour(id={self.id}, points={len(self)})'
	def __repr__(self) -> str: return str(self)
	def __len__ (self) -> int: return len(self.contour)

	@staticmethod
	def approx_contour(contour: NDArray) -> NDArray:

		tollerance     = cv.arcLength   (curve=contour, closed=True) * Contour._APPROX_FACTOR
		contour_approx = cv.approxPolyDP(curve=contour, closed=True, epsilon=tollerance)

		return contour_approx
	
	@property
	def id(self) -> int: return self._id
	
	@property
	def contour_orig(self) -> NDArray: return self._contour_orig

	@property
	def contour(self) -> NDArray: return self._contour_approx

	@property
	def mean_point(self) -> Tuple[int, int]:
		mx, my = np.mean(self.contour, axis=0, dtype=np.int32)[0]
		return int(mx), int(my)

	def to_sorted_vertex(self, adjusted: bool = True) -> SortedVertices: 

		vertices = self.contour if adjusted else self.contour_orig

		return SortedVertices(vertices=vertices[:, 0, :])

	@property
	def hierarchy(self) -> Contour.ContourHierarchy: return self._hierarchy

	def draw(self, frame: Frame, color: RGBColor = (255, 0, 0), thickness: int = 2, adjusted: bool = True) -> Frame:

		contours = self.contour if adjusted else self.contour_orig

		cv.drawContours(image=frame, contours=[contours], contourIdx=-1, color=color, thickness=thickness)

		return frame
	
	def is_quadrilateral(self) -> bool: return len(self.contour) == 4

	def is_circle(self) -> bool: 
		
		# Compute circularity | 4pi * area / perimeter^2
		area      = cv.contourArea(self.contour)
		perimeter = cv.arcLength  (self.contour, closed=True)

		
		if perimeter == 0: return False  # Avoid division by zero for degenerate contours
		
		circularity = 4 * np.pi * area / (perimeter ** 2)

		return circularity > Contour._CIRCULARITY_THRESHOLD
	
	def frame_mean_value(self, frame: Frame, fill: bool = False, child_subtract: Contour | None = None) -> float:

		# Child subtraction requires filled mask
		if child_subtract is not None: 

			# Children subtraction requires filled mask
			fill = True 

			# Check children subtraction is consistent with the hierarchy
			if child_subtract.hierarchy.parent != self.id: raise ValueError(
				f'Child subtraction contour with id {child_subtract.id} '
				f'is not a child of the current contour with id {self.id}. '
			)

		# Create mask
		mask: Frame = np.zeros_like(a=frame, dtype=np.uint8)
		thickness: int = cv.FILLED if fill else 1
		cv.drawContours(image=mask, contours=[self.contour], contourIdx=-1, color=(255, ), thickness=thickness)

		# If children are to be subtracted
		if child_subtract is not None:
			cv.drawContours(image=mask, contours=[child_subtract.contour], contourIdx=-1, color=(0,), thickness=thickness)
		
		# print("DEBUG MASK"); plt.imshow(mask, cmap='gray'); plt.show()

		# Compute mean value
		mean_value = cv.mean(frame, mask=mask)[0]

		return mean_value

class Contours:

	def __init__(self, frame: Frame) :

		contours, hierarchy = cv.findContours(image=frame, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

		self._contours_dict = {
			contour_id: Contour(id=contour_id, contour=contour, hierarchy=hierarchy_line)
			for contour_id, (contour, hierarchy_line) in enumerate(zip(contours, hierarchy[0]))
		}

	def __str__     (self)           -> str              : return f'Contours[curvers: {len(self)}]'
	def __repr__    (self)           -> str              : return str(self)
	def __len__     (self)           -> int              : return len(self._contours_dict)
	def __iter__    (self)           -> Iterator[Contour]: return iter(self._contours_dict.values())
	def __getitem__ (self, key: int) -> Contour          : return self._contours_dict[key]

	def get_descendants(self, contour: Contour) -> Sequence[Contour]:
    
		def _get_descendants(id: int | None) -> List[int]:

			descendants: Set[int] = set()

			while id is not None: 
				
				descendants.add(id)

				# Child
				child = self[id].hierarchy.first_child
				if child and child not in descendants: descendants.update(_get_descendants(child))

				# Next
				id = self[id].hierarchy.next
			
			return list(descendants)

		# Start with the first child of the given node
		decendants_id = _get_descendants(id=contour.hierarchy.first_child)
		return [self[id] for id in decendants_id]
	
	def get_ancestors(self, contour: Contour) -> Sequence[Contour]:
		
		ancestors: List[Contour] = []
		current = contour

		while current.hierarchy.parent is not None:

			current = self[current.hierarchy.parent]
			ancestors.append(current)

		return ancestors

	def draw(self, frame: Frame, colors: List[RGBColor] | RGBColor = (255, 0, 0), thickness: int = 2, adjusted: bool = True) -> Frame:

		palette_ = colors if isinstance(colors, list) else [colors] * len(self)

		for contour, color in zip(self, palette_):  # type: ignore - contours is iterable
			contour.draw(frame=frame, color=color, thickness=thickness, adjusted=adjusted)

		return frame