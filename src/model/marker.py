from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple
from dataclasses import dataclass

import numpy as np
import cv2 as cv
from numpy.typing import NDArray

from src.model.typing import Frame, RGBColor, Views
from src.utils.misc   import generate_palette
from src.utils.io_ import BaseLogger, SilentLogger

Points2D = Sequence['Point2D']

@dataclass
class Point2D:

	x: int
	y: int

	def __str__(self)  -> str: return f'Point2D({self.x}, {self.y})'
	def __repr__(self) -> str: return str(self)
	def __iter__(self) -> Iterator[int]: return iter([self.x, self.y])

	@classmethod
	def from_tuple(cls, xy: Tuple[int, int]) -> Point2D: x, y = xy; return cls(x=int(x), y=int(y)) 

	def in_frame(self, img: Frame) -> bool:

		h, w, *_ = img.shape

		in_width  = 0 <= self.x < w
		in_height = 0 <= self.y < h

		return in_width and in_height
	
	def draw_circle(
		self, 
		frame     : Frame,
		radius    : int      = 3,
		color     : RGBColor = (255, 0, 0),
		thickness : int      = 5,
		fill      : bool     = False,
		**kwargs
	) -> Frame:
		
		if not self.in_frame(frame): raise ValueError(f'Point {self} is out of frame bounds.')
		
		if fill: thickness = -1
	
		cv.circle(frame, (int(self.x), int(self.y)), radius=radius, color=color, thickness=thickness, **kwargs)
		
		return frame
	
	def draw_cross(
		self, 
		frame     : Frame,
		size      : int      = 5,
		color     : RGBColor = (255, 0, 0),
		thickness : int      = 2,
		**kwargs
	) -> Frame:
		
		if not self.in_frame(frame): raise ValueError(f'Point {self} is out of frame bounds.')

		pa = Point2D(x=self.x - size, y=self.y - size)
		pb = Point2D(x=self.x + size, y=self.y + size)
		pc = Point2D(x=self.x - size, y=self.y + size)
		pd = Point2D(x=self.x + size, y=self.y - size)
		
		Point2D.draw_line(frame=frame, point1=pa, point2=pb, color=color, thickness=thickness, **kwargs)
		Point2D.draw_line(frame=frame, point1=pc, point2=pd, color=color, thickness=thickness, **kwargs)
		
		return frame
	
	@staticmethod
	def draw_line(
		frame: Frame,
		point1: Point2D,
		point2: Point2D,
		color: RGBColor = (255, 0, 0),
		thickness: int = 2,
		**kwargs
	) -> Frame:
		
		if not point1.in_frame(frame): raise ValueError(f'Point {point1} is out of frame bounds.')
		if not point2.in_frame(frame): raise ValueError(f'Point {point2} is out of frame bounds.')
		
		x1, y1 = point1
		x2, y2 = point2
		
		cv.line(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness, **kwargs)
		
		return frame

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

	def __getitem__(self, key: int) -> Point2D: return Point2D.from_tuple(self._vertices[key])

	@property
	def vertices(self) -> NDArray: return self._vertices

	def draw(self, frame: Frame, palette: List[RGBColor] | RGBColor = (255, 0, 0), radius: int = 5, thickness: int = 2) -> Frame:

		palette_ = palette if isinstance(palette, list) else [palette] * len(self)

		if len(palette_) != len(self): raise ValueError(f'Palette must have {len(self)} colors, got {len(palette)}.')

		# Draw lines
		for i, col in enumerate(palette_): self[i].draw_circle(frame=frame, radius=radius, color=col, thickness=thickness)

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
	def mean_point(self) -> Point2D: return Point2D.from_tuple(np.mean(self.contour, axis=0, dtype=np.int32)[0])

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

@dataclass 
class Marker:

	c0     : Point2D
	c1     : Point2D
	c2     : Point2D
	c3     : Point2D
	anchor : Point2D

	@classmethod
	def from_detection(
		cls,
		marker_vertices : SortedVertices,
		c0_vertex_id    : int,
		anchor_contour  : Contour
	):
		if len(marker_vertices) != 4     : raise ValueError(f'Invalid number of vertices for the marker: expected 4, got {len(marker_vertices)}. ')
		if c0_vertex_id not in range(4)  : raise ValueError(f'Invalid vertex index for the marker: expected 0-3, got {c0_vertex_id}. ')
		if not anchor_contour.is_circle(): raise ValueError(f'Invalid circle contour for the marker. ') 

		# Reorder vertices
		marker_vertices.shift(n=c0_vertex_id)

		c0, c1, c2, c3 = [marker_vertices[i] for i in range(4)]
		point = anchor_contour.mean_point

		return cls(c0=c0, c1=c1, c2=c2, c3=c3, anchor=point)
	
	def __str__(self) -> str:
		corners_str = '; '.join([f'c{i}={corner}' for i, corner in enumerate(self.corners)])
		return f'Marker[{corners_str}; anchor={self.anchor})'
	
	def __repr__(self) -> str: return str(self)

	def __getitem__(self, index: int) -> Point2D: return self.corners[index]

	@property
	def adjacent_couples(self) -> List[Tuple[Point2D, Point2D]]:
		return [(self.c0, self.c1), (self.c1, self.c2), (self.c2, self.c3), (self.c3, self.c0)]

	@property
	def corner_palette(self) -> List[RGBColor]:
		return [
			(255,  99,  71),   # Tomato Red
			( 50, 205,  50),   # Lime Green
			( 70, 130, 180),  # Steel Blue
			(255, 215,   0)    # Gold
		]

	@property
	def line_color(self) -> RGBColor:
		return (140, 240, 250)  # Medium Slate Blue

	@property
	def anchor_color(self) -> RGBColor:
		return (219, 94, 162) # Deep Pink
	
	@property
	def corners(self) -> Points2D: return [self.c0, self.c1, self.c2, self.c3]

	def draw(self, frame: Frame) -> Frame:

		for corner1, corner2 in self.adjacent_couples:
			Point2D.draw_line(frame=frame, point1=corner1, point2=corner2, color=self.line_color, thickness=10)

		for corner, color in zip(self.corners, self.corner_palette):
			corner.draw_circle(frame=frame, radius=4, color=color, thickness=15)

		self.anchor.draw_cross(frame=frame, size=12, color=self.anchor_color, thickness=10)

		return frame

class MarkerDetector:

	WHITE_THRESHOLD = 255 - 5
	BLACK_THRESHOLD =   0 + 5

	def __init__(self, logger: BaseLogger = SilentLogger(), verbose: bool = False):

		self._logger         = logger
		self._logger_verbose = logger if verbose else SilentLogger()
		self._is_verbose     = verbose
	
	def _detect_corners(self, frame: Frame, contours: Contours) -> Tuple[Tuple[Contour, Contour] | None, str]:
    
		# (child, parent)
		nested_quadrilaterls: List[Tuple[Contour, Contour]] = []

		# Loop through all contours
		for contour in contours:

			# Skip if a) not a quadrilateral or b) has no parent
			if not contour.is_quadrilateral() or contour.hierarchy.parent is None: continue

			# Get parent contour
			parent = contours[contour.hierarchy.parent]

			# Skip if parent is not a quadrilateral
			if not parent.is_quadrilateral(): continue

			# Append to list
			nested_quadrilaterls.append((contour, parent))
		
		# Check if there is only one nested quadrilateral
		if len(nested_quadrilaterls) == 0: return None, f'No nested squares found. '
		if len(nested_quadrilaterls) >  1: return None, f'Found multiple squares ({len(nested_quadrilaterls)}). '

		inner, outer = nested_quadrilaterls[0]

		# Check black to white
		inner_white = inner.frame_mean_value(frame=frame) > self.BLACK_THRESHOLD
		outer_black = outer.frame_mean_value(frame=frame) < self.WHITE_THRESHOLD

		if inner_white and outer_black:
			return (inner, outer), ''
		else: 
			return None, f'No black to white transition between squares. '
	
	def _detect_anchor(
			self,
			frame: Frame,
			contours: Contours,
			marker_vertices: Tuple[SortedVertices, SortedVertices],
		) -> Tuple[Tuple[int, Contour] | None, str]:

		def is_contour_between_points(contour: Contour, point1: Tuple[int, int], point2: Tuple[int, int]) -> bool:
	
			# Extract coordinates
			x1, y1 = point1
			x2, y2 = point2
			cx, cy, cw, ch = cv.boundingRect(contour.contour)

			# Sort points to define a region
			x_min, x_max = [op(x1, x2) for op in (min, max)]
			y_min, y_max = [op(y1, y2) for op in (min, max)]

			# Get bounding box of the contour
			cx_min, cx_max = cx, cx + cw
			cy_min, cy_max = cy, cy + ch

			# Check if the contour's bounding box is within the defined region
			x_condition = x_min <= cx_min and cx_max <= x_max
			y_condition = y_min <= cy_min and cy_max <= y_max

			return x_condition and y_condition

		# vertex index, circle contour
		marker_circles: List[Tuple[int, Contour]] = []

		for contour in contours:

			# Skip if is not a circle
			if not contour.is_circle(): continue

			# Skip if the circle is not white
			mean_value = contour.frame_mean_value(frame=frame, fill=True)
			if mean_value < self.WHITE_THRESHOLD: continue

			# For the 4 couples of vertex check if the circle is between the two points
			inner_vert, outer_vert = marker_vertices
			for i, (inner, outer) in enumerate(zip(inner_vert.vertices, outer_vert.vertices)):
				if is_contour_between_points(contour, inner, outer):
					marker_circles.append((i, contour))

		if len(marker_circles) == 0: return None, f'No white circle found within the marker. '
		if len(marker_circles) >  1: return None, f'Found multiple white circles within the marker ({len(marker_circles)}). '

		return marker_circles[0], ''

	def detect(self, frame: Frame) -> Tuple[Marker | None, str, Views]:

		frame_c = cv.cvtColor(frame.copy(), cv.COLOR_GRAY2BGR)

		views = {}
		frame_contours_orig  = frame_c.copy()
		frame_contours_adj   = frame_c.copy()
		frame_marker_corners = frame_c.copy()
		frame_marker_circle  = frame_c.copy()

		contours = Contours(frame=frame)

		palette = generate_palette(n=len(contours))
		contours.draw(frame=frame_contours_orig, colors=palette, thickness=10, adjusted=False)
		contours.draw(frame=frame_contours_adj,  colors=palette, thickness=10, adjusted=True)
		views['contours_orig'] = frame_contours_orig
		views['contours_adj' ] = frame_contours_adj

		# 1. Detect marker corners
		marker_corners, warn_message = self._detect_corners(frame=frame, contours=contours)

		if marker_corners is None: return None, warn_message, views

		inner_marker_contour, outer_marker_contour = marker_corners
		inner_marker_vertices = inner_marker_contour.to_sorted_vertex()
		outer_marker_vertices = outer_marker_contour.to_sorted_vertex()

		inner_marker_contour.draw(frame=frame_marker_corners, color=(  0, 255, 255), thickness=10)
		outer_marker_contour.draw(frame=frame_marker_corners, color=(255, 255,   0), thickness=10)
		views['marker_corners'] = frame_marker_corners

		# 2. Detect circles
		circle_detection, warning_message = self._detect_anchor(
			frame=frame,
			contours=contours, 
			marker_vertices=(inner_marker_vertices, outer_marker_vertices), 
		)

		if circle_detection is None: return None, warning_message, views

		v_id, circle = circle_detection
		circle.draw(frame=frame_marker_circle, color=(255, 0, 255), thickness=10)
		views['marker_circle'] = frame_marker_circle

		marker = Marker.from_detection(
			marker_vertices=inner_marker_vertices,
			c0_vertex_id=v_id,
			anchor_contour=circle
		)

		return marker, '', views