'''
This file contains auxiliary geometric classes used in the project.
    - Geometric primitives such as points ad a sorted set of vertices, used to handle marker corners.
	- Contours, used to represent detected contours in the frame.
	- LightDirection, used to represent the light source direction.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Set, Tuple

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from src.model.typing import Frame, RGBColor, default


# __________________________________ GEOMETRIC PRIMITIVES __________________________________ #

Points2D = Sequence['Point2D']

@dataclass
class Point2D:
	''' Class representing a 2D point and provides methods to draw it on a frame. '''

	x: int
	y: int

	def __str__ (self) -> str           : return f'{self.__class__.__name__}({self.x}, {self.y})'
	def __repr__(self) -> str           : return str(self)
	def __iter__(self) -> Iterator[int] : return iter([self.x, self.y])

	@classmethod
	def from_tuple(cls, xy: Tuple[int, int]) -> Point2D:
		x, y = xy
		return cls(x=int(x), y=int(y))
	
    # --- FRAME OPERATIONS ---

	def in_frame(self, img: Frame) -> bool:
		''' Check if the point is within the frame bounds. '''

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
		'''
		Draw the point as a circle on the frame.
		NOTE: The drawing is in-place on the frame.
		'''

		if not self.in_frame(frame): raise ValueError(f'Circle cannot be drawed. Point {self} is out of frame bounds.')

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
		'''
		Draw the point as a cross on the frame.
		NOTE: The drawing is in-place on the frame.
		'''

		if not self.in_frame(frame): raise ValueError(f'Cross cannot be drawn. Point {self} is out of frame bounds.')

		pa = Point2D(x=self.x       , y=self.y - size)
		pb = Point2D(x=self.x       , y=self.y + size)
		pc = Point2D(x=self.x - size, y=self.y    )
		pd = Point2D(x=self.x + size, y=self.y    )

		Point2D.draw_line(frame=frame, point1=pa, point2=pb, color=color, thickness=thickness, **kwargs)
		Point2D.draw_line(frame=frame, point1=pc, point2=pd, color=color, thickness=thickness, **kwargs)

		return frame

	@staticmethod
	def draw_line(
		frame     : Frame,
		point1    : Point2D,
		point2    : Point2D,
		color     : RGBColor = (255, 0, 0),
		thickness : int      = 2,
		**kwargs
	) -> Frame:
		'''
		Draw a line between two points on the frame. 
		NOTE: The drawing is in-place on the frame.
		'''

		for point in [point1, point2]:
			if not point.in_frame(frame): raise ValueError(f'Line cannot be drawn. Point {point} is out of frame bounds.')

		cv.line(img=frame, pt1=tuple(point1), pt2=tuple(point2), color=color, thickness=thickness, **kwargs)

		return frame


class SortedVertices:
	'''
	Class representing a set of vertices sorted by angle w.r.t. a center point. 
	This class it's used to model marker corners in a consisted order.
	'''

	def __init__(self, vertices: NDArray, center: Point2D | None = None) -> None:
		'''
		Sort a set of two vertices according to a center point.
		:param vertices: The set of vertices to sort as a Nx2 array.
		:param center: The center point to sort the vertices around.
			If not provided the center point is the vertices's centroid.
		'''

		# NOTE: We decide to keep points as NDArray and not as Sequence[Point2D]
		#	    because we mainly leverage on numpy functions to sort and manipulate the vertices.
		#	    We only convert the vertices to Point2D to output them.

		# Check points are 2D
		self._len, dim = vertices.shape
		if dim != 2: raise ValueError(f'Vertices must be 2D, got {dim}. ')

		# Default center is the mean of the vertices
		center_: Point2D = default(center, Point2D.from_tuple(np.mean(vertices, axis=0)))

		# Sort the vertices
		self._vertices = SortedVertices.sort_point(vertices=vertices, center=center_)

	@staticmethod
	def sort_point(vertices: NDArray, center: Point2D) -> NDArray:
		''' Sort the vertices by angle w.r.t. the center point. '''

		# Calculate the angle of each point w.r.t. the center
		angles = np.arctan2(vertices[:, 1] - center.y, vertices[:, 0] - center.x)

		# Sort vertices by angle
		sorted_indices = np.argsort(angles)

		return vertices[sorted_indices]

	# --- MAGIC METHODS ---

	def __str__    (self)           -> str     : return f'{self.__class__.__name__}[points={len(self)}]'
	def __repr__   (self)           -> str     : return str(self)
	def __len__    (self)           -> int     : return self._len
	def __getitem__(self, key: int) -> Point2D : return Point2D.from_tuple(self._vertices[key])

	# --- PROPERTIES ---

	@property
	def vertices(self) -> NDArray: return self._vertices

	# --- UTILITIES ---

	def roll(self, n: int):
		'''
		Roll the vertices array by n positions.
		NOTE: This is used to sort the vertices of marker squares to make the first one the closest to the marker anchor
		'''

		self._vertices = np.roll(self._vertices, -n, axis=0)

	def align_to(self, other: SortedVertices):
		'''
		Align the vertices to another set of vertices according to the closest point logic.
		NOTE: This is used to match the marker corners of the inner and outer squares.
		'''

		# Check the vertices have the same length
		if len(self) != len(other): raise ValueError(f'Vertices must have the same length, got {len(self)} and {len(other)}. ')

		# Compute the angle between the first point and each other point of the other set of vertices
		distances = np.linalg.norm(other.vertices - self.vertices[0], axis=1)

        # Align the vertices to match the closest point
		closest_index = int(np.argmin(distances))
		self.roll(n=-closest_index)

	def draw(
		self,
		frame     : Frame,
		palette   : List[RGBColor] | RGBColor = (255, 0, 0),
		radius    : int                       = 5,
		thickness : int                       = 2
	) -> Frame:
		'''
		Draw the vertices on the frame according to a palette of colors.
		The palette can either be a single color or a list of colors matching the number of vertices.
		'''

		# Palette
		palette_ = palette if isinstance(palette, list) else [palette] * len(self)
		if len(palette_) != len(self): raise ValueError(f'Palette must have {len(self)} colors, got {len(palette)}.')

		# Draw lines
		for i, col in enumerate(palette_): self[i].draw_circle(frame=frame, radius=radius, color=col, thickness=thickness)

		return frame

# __________________________________ LIGHT DIRECTION __________________________________

@dataclass
class LightDirection:
	'''
	Represents the light source direction as a tuple of two floats (u, v) in the range [-1, 1].
	The condition u^2 + v^2 <= 1 must hold.
	'''

	u: float
	v: float

	# --- MAGIC METHODS ---

	def __str__ (self) -> str: return f'{self.__class__.__name__}({self.u:.2f}, {self.v:.2f})'
	def __repr__(self) -> str: return str(self)
	def __iter__(self) -> Iterator[float]: return iter([self.u, self.v])

	# --- CONSTRUCTORS ---

	@classmethod
	def from_tuple(cls, uv: Tuple[float, float]) -> LightDirection:

		u, v = uv
		if u**2 + v**2 > 1: raise ValueError(f'Invalid light direction ({u}, {v}): it lies outside the unit circle. The condition u^2 + v^2 <= 1 must hold. ')

		return cls(u=u, v=v)

	@classmethod
	def from_3d_light_vector(cls, light_vector: NDArray) -> LightDirection:
		'''
		Convert an unnormalized 3D light vector in the unitary semi-sphere to a 2D light direction.
		'''

        # Normalize the light vector
		light_vector_norm = light_vector / np.linalg.norm(light_vector)

        # Extract its components
		u, v, w = light_vector_norm

        # Check the vector is above the unitary semi-sphere
		if w < 0: raise ValueError(f'Invalid vector height {w}: the vector lies below the unitary semi-sphere. ')

		return LightDirection.from_tuple((u, v))

	# --- DRAW LIGHT DIRECTION ---

	@staticmethod
	def draw_line_direction(light_direction: LightDirection, frame_side: int = 500) -> Frame:
		''' Draw the light direction as an arrow inside the unit circle. '''

		x, y = light_direction

		# Ensure x, y are within [-1, 1]
		if not (-1 <= x <= 1 and -1 <= y <= 1):
			raise ValueError("x and y must be in the range [-1, 1]")

		# Create a black background
		image: NDArray = np.zeros((frame_side, frame_side, 3), dtype=np.uint8)

		# Define the circle center and radius
		center = (frame_side // 2, frame_side // 2)
		radius = frame_side // 2  # Circle radius is half of the image size

		# Draw the white circle
		cv.circle(image, center, radius, (255, 255, 255), thickness=4)

		# Compute the arrow endpoint in pixel coordinates
		arrow_x = int(center[0] + x * radius)  # Scale x to the radius
		arrow_y = int(center[1] - y * radius)  # Scale y to the radius (inverted y-axis)

		# Draw the red arrow
		cv.arrowedLine(image, center, (arrow_x, arrow_y), (255, 0, 0), thickness=4, tipLength=0.05)

		# Add the light direction as text in the bottom-right corner, with x and y on separate lines
		text_x   = f"x: {x:+.2f}"
		text_y   = f"y: {y:+.2f}"
		font_scale  = 0.6
		font_thickness = 2
		padding  = 10
		line_spacing   = 5

		# Calculate text size to align properly
		text_x_size, _ = cv.getTextSize(text_x, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
		text_y_size, _ = cv.getTextSize(text_y, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

		# Define positions for x and y lines
		text_x_pos = (frame_side - text_x_size[0] - padding, frame_side - text_y_size[1] - padding - line_spacing)
		text_y_pos = (frame_side - text_y_size[0] - padding, frame_side - padding)

		for text, text_pos in zip([text_x, text_y], [text_x_pos, text_y_pos]):
			cv.putText(image, text, text_pos, cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)

		return image

	@staticmethod
	def draw_line_direction_history(
		light_directions : NDArray,
		img_side         : int      = 500,
		col_start        : RGBColor = (255, 255, 255),
		col_end          : RGBColor = (  0,   0, 255)
	) -> Frame:
		''' Draw the history of light directions with a color gradient. '''

		# Black image
		img = np.zeros((img_side, img_side, 3), dtype=np.uint8)

		# Compute center and radius
		center = (img_side // 2, img_side // 2)
		radius = img_side // 2

		# Draw the white circle
		cv.circle(img, center, radius, (255, 255, 255), thickness=4)

		# Normalize the points from [-1, 1] to pixel coordinates
		normalized_points = np.empty_like(light_directions, dtype=int)
		normalized_points[:, 0] = (center[0] + light_directions[:, 0] * radius).astype(int)  # Scale x
		normalized_points[:, 1] = (center[1] - light_directions[:, 1] * radius).astype(int)  # Scale y

		# Plot point with color gradient
		n_points, _ = normalized_points.shape
		for j, (x, y) in enumerate(normalized_points.astype(int)):

			# Compute the color based on the gradient
			color = tuple(
				int(col_start[i] + (j/n_points) * (col_end[i] - col_start[i]))
				for i in range(3)
			)
			cv.circle(img, (x, y), radius=3, color=color, thickness=-1)

		return img


# __________________________________ CONTOURS __________________________________

class Contour:
	''' Class representing a contour detected in the frame. '''

	@dataclass
	class ContourHierarchy:
		''' 
		Utility class to handle the hierarchy of a contour.
		The hierarchy is organized as a left-child, right-sibling tree.
		'''

		next        : int | None
		previous    : int | None
		first_child : int | None
		parent      : int | None

		def __str__ (self) -> str: return f'{self.__class__.__name__}[{"; ".join([f"{k}: {v}" for k, v in self.to_dict().items()])}]'
		def __repr__(self) -> str: return str(self)

		@classmethod
		def no_hierarchy(cls) -> Contour.ContourHierarchy: return cls(next=None, previous=None, first_child=None, parent=None)
		''' Create a no-hierarchy info with all values set to None. '''

		@classmethod
		def from_hierarchy(cls, hierarchy: NDArray) -> Contour.ContourHierarchy:
			''' Create a contour hierarchy parsing a line from the cv.findContours output. '''

			def default_value(idx: int) -> int | None: return int(idx) if idx != -1 else None

			return cls(
				next        = default_value(hierarchy[0]),
				previous    = default_value(hierarchy[1]),
				first_child = default_value(hierarchy[2]),
				parent      = default_value(hierarchy[3])
			)

		def to_dict(self) -> Dict[str, int | None]: return {
			'next'       : self.next,
			'previous'   : self.previous,
			'first_child': self.first_child,
			'parent'    : self.parent
		}

	_APPROX_FACTOR         : float = 0.01  # Approximation factor for the contour
	_CIRCULARITY_THRESHOLD : float = 0.80  # Threshold for circularity

	def __init__(self, id: int, contour: NDArray, hierarchy: Contour.ContourHierarchy):
		''' 
		Initialize the contour id in the hierarchy tree, the contour points and the hierarchy.
		It approximates the contour using the Ramer-Douglas-Peucker algorithm.

		:param id: The id of the contour in the hierarchy tree.
		:param contour: The contour points as a Nx1x2 array.
		:param hierarchy: The hierarchy of the contour.
		'''

		self._id            : int                      = id
		self._contour_orig  : NDArray                  = contour
		self._hierarchy     : Contour.ContourHierarchy = hierarchy

		# Approximate the contour using the Ramer-Douglas-Peucker algorithm
		epsilon = cv.arcLength(contour, closed=True) * Contour._APPROX_FACTOR
		self._contour_approx: NDArray= cv.approxPolyDP(curve=contour, closed=True, epsilon=epsilon)
	
    # --- MAGIC METHODS ---

	def __str__ (self) -> str: return f'{self.__class__.__name__}(id={self.id}, points={len(self)})'
	def __repr__(self) -> str: return str(self)
	def __len__ (self) -> int: return len(self.contour)

	# --- PROPERTIES ---

	@property
	def id(self) -> int: return self._id

	@property
	def hierarchy(self) -> Contour.ContourHierarchy: return self._hierarchy

	@property
	def contour_orig(self) -> NDArray: return self._contour_orig

	@property
	def contour(self) -> NDArray: return self._contour_approx

	@property
	def area(self) -> float: return cv.contourArea(self.contour)

	@property
	def perimeter(self) -> float: return cv.arcLength(self.contour, closed=True)

	@property
	def centroid(self) -> Point2D: return Point2D.from_tuple(np.mean(self.contour, axis=0, dtype=np.int32)[0])

	# --- CONTOUR SHAPE ---

	# NOTE: The following two methods are used in the marker detection logic to check if a contour is a circle or a quadrilateral.

	def is_circle(self, thresh: float | None = None) -> bool:
		''' Check if the contour is a circle based on its circularity. '''

        # Default threshold for circularity
		thresh_: float = default(thresh, Contour._CIRCULARITY_THRESHOLD)  

		# Avoid division by zero for degenerate contours
		if self.perimeter == 0: return False

		circularity = 4 * np.pi * self.area / (self.perimeter ** 2)
		return circularity > thresh_

	def is_quadrilateral(self) -> bool:
		''' Check if the contour is a quadrilateral. '''

		return len(self) == 4 and cv.isContourConvex(self.contour)

	# --- UTILITIES ---

	def to_sorted_vertex(self, center: Point2D | None = None, adjusted: bool = True) -> SortedVertices:
		''' 
		Convert the contour to a SortedVertices object.
		It can either use the original contour or the approximated one, 
			and it can be centred around a specific point.
		'''

		vertices = self.contour if adjusted else self.contour_orig

		return SortedVertices(vertices=vertices[:, 0, :], center=center)

	def draw(
		self,
		frame     : Frame,
		color     : RGBColor = (255, 0, 0),
		thickness : int      = 2,
		fill      : bool     = False,
		adjusted  : bool     = True
	) -> Frame:
		'''
		Draw the contour on the frame with a specific color and thickness.
		NOTE: The drawing is in-place on the frame.
		'''

		if fill: thickness = cv.FILLED
		contours = self.contour if adjusted else self.contour_orig

		cv.drawContours(image=frame, contours=[contours], contourIdx=-1, color=color, thickness=thickness)

		return frame


	def scale_contour(self, scale: float) -> Contour:
		''' 
		Create a new contour by scaling the current one towards its centroid.		
		A scale < 1 will shrink the contour, while a scale > 1 will expand it.
		'''

		points         = self.contour[:, 0, :]                            # Flatten the contour to a Nx2 array
		centroid       = np.mean(points, axis=0)                          # Compute the centroid of the quadrilateral
		scaled_points  = (points - centroid) * scale + centroid           # Scale each point towards the centroid
		scaled_contour = scaled_points.reshape(-1, 1, 2).astype(np.int32) # Convert back to the original contour format (Nx1x2)

		return Contour(
			id=-1, # Dummy id
			contour=scaled_contour,
			hierarchy=Contour.ContourHierarchy.no_hierarchy()
		)

	def mean_value_on_frame(
		self,
		frame: Frame,
		fill: bool = False,
		contour_subtraction: List[Contour] | None = None
	) -> Tuple[float, Frame]:
		'''
		Calculate the mean contour value on the input frame using a mask with different options.
		:param frame: The frame to compute the mean value on.
		:param fill: If True, the mask will be filled with the contour, otherwise it will only be the contour border.
		:param contour_subtraction: A list of contours to subtract from the mask before computing the mean value.
		:return: The mean value of the frame within the applied contour mask.
		'''

		# Child subtraction requires filled mask
		if contour_subtraction is not None: fill = True

		# Create mask with the contour by drawing it on a black frame
		mask: Frame = np.zeros_like(a=frame, dtype=np.uint8)
		thickness: int = cv.FILLED if fill else 2
		cv.drawContours(image=mask, contours=[self.contour], contourIdx=-1, color=(255,), thickness=thickness)

		# Subtract child contours by drawing them as black on the mask
		if contour_subtraction is not None:
			for descendant in contour_subtraction:
				cv.drawContours(image=mask, contours=[descendant.contour], contourIdx=-1, color=(0,), thickness=thickness)

		# Compute mean value
		mean_value = cv.mean(frame, mask=mask)[0]

		# Write the mean value on the bottom right corner of the mask
		text       = f'mean: {mean_value:.2f}'
		font_face  = cv.FONT_HERSHEY_SIMPLEX
		font_scale = 2.5
		thickness  = 10
		pad        = 50
		color      = (255,)

		# Calculate the text size
		(text_width, text_height), baseline = cv.getTextSize(text, font_face, font_scale, thickness)
		image_height, image_width = mask.shape[:2]

		x = image_width - text_width - pad  # pixels padding from the right edge
		y = image_height - pad              # pixels padding from the bottom edge (baseline adjustment included)

		# Put the text on the image
		mask = cv.putText(
			img=mask,
			text=text,
			org=(x, y),
			fontFace=font_face,
			fontScale=font_scale,
			color=color,
			thickness=thickness
		)

		return mean_value, mask


class Contours:
	''' Class representing a collection of contours detected in the frame. '''

	def __init__(
		self,
		frame    : Frame,
		min_area : float | None = None,
		max_area : float | None = None
	):
		'''
		Detect multiple contours in the frame, with the option to filter them by area.
		:param frame: The frame to detect the contours on.
		:param min_area: The minimum area of the contour to keep. If None, no minimum area is applied.
		:param max_area: The maximum area of the contour to keep. If None, no maximum area is applied.
		'''

		# Find contours using the RETR_TREE mode to get the hierarchy
		# NOTE: THe hierarchy is needed to get the parent-child relationship for the inner and outer marker squares
		contours, hierarchy = cv.findContours(image=frame, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
		self._contours_dict = {}

		if len(contours) == 0: return  # No contours found

		# NOTE: By filtering away some contours with the area, we may break the hierarchy tree.
		for contour_id, (contour, hierarchy_line) in enumerate(zip(contours, hierarchy[0])):

			# Area filter
			area = cv.contourArea(contour)
			if (min_area is None or area >= min_area) and (max_area is None or area <= max_area):

				self._contours_dict[contour_id] = Contour(
					id=contour_id,
					contour=contour,
					hierarchy=Contour.ContourHierarchy.from_hierarchy(hierarchy=hierarchy_line)
				)

	# --- MAGIC METHODS ---

	def __str__  (self)     -> str               : return f'{self.__class__.__name__}[curves: {len(self)}]'
	def __repr__ (self)     -> str               : return str(self)
	def __len__  (self)     -> int               : return len(self._contours_dict)
	def __iter__ (self)     -> Iterator[Contour] : return iter(self._contours_dict.values())
	
	def __getitem__ (self, key: int) -> Contour | None : return self._contours_dict.get(key, None)
	''' The getitem method returns None if the contour is not found because of the area filter. '''

	# --- DESCENDANTS and ANCESTORS ---

	def get_descendants(self, contour: Contour) -> Sequence[Contour]:
		''' Get the descendants of a contour in the hierarchy tree. '''

		def _get_descendants(id: int | None) -> List[int]:

			descendants: Set[int] = set()

			while id is not None:

					# Process current contour
					curr_contour = self[id]
					if curr_contour is None: break
					descendants.add(id)

					# Process children
					child = curr_contour.hierarchy.first_child
					if child is not None and child not in descendants: descendants.update(_get_descendants(child))
					id = curr_contour.hierarchy.next

			return list(descendants)

		descendants_id = _get_descendants(id=contour.hierarchy.first_child)

		return [self[id] for id in descendants_id if self[id] is not None]  # type: ignore - self[id] is not None check

	def get_ancestors(self, contour: Contour) -> Sequence[Contour]:
		''' Get the ancestors of a contour in the hierarchy tree. '''

		ancestors: List[Contour] = []
		current = contour

		while current.hierarchy.parent is not None:

			current = self[current.hierarchy.parent]
			if current is None: break
			ancestors.append(current)

		return ancestors

	# --- UTILITIES ---

	def draw(
		self,
		frame     : Frame,
		colors    : List[RGBColor] | RGBColor = (255, 0, 0),
		thickness : int                       = 2,
		adjusted  : bool                      = True
	) -> Frame:
		''' 
		Draw all the contours on the frame with a specific color and thickness. 
		NOTE: The drawing is in-place on the frame.
		'''

		# Palette
		palette_ = colors if isinstance(colors, list) else [colors] * len(self)
		if len(palette_) != len(self): raise ValueError(f'Palette must have {len(self)} colors, got {len(palette_)}.')

		# Draw contours
		for contour, color in zip(self, palette_):
			contour.draw(frame=frame, color=color, thickness=thickness, adjusted=adjusted)

		return frame