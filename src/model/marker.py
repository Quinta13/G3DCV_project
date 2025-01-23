from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Set, Tuple
from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
from numpy.typing import NDArray


from src.utils.calibration import CalibratedCamera
from src.model.thresholding import ThresholdedVideoStream, Thresholding
from src.utils.typing import Frame, RGBColor, Views, Size2D, LightDirectionMethod, CornerMaskMethod
from src.utils.misc   import default, generate_palette
from src.utils.io_ import BaseLogger, SilentLogger

Points2D = Sequence['Point2D']

@dataclass
class Point2D:

	x: int
	y: int

	def __str__ (self) -> str: return f'{self.__class__.__name__}({self.x}, {self.y})'
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
		
		if not self.in_frame(frame): raise ValueError(f'Cross cannot be drawed. Point {self} is out of frame bounds.')

		pa = Point2D(x=self.x       , y=self.y - size)
		pb = Point2D(x=self.x       , y=self.y + size)
		pc = Point2D(x=self.x - size, y=self.y       )
		pd = Point2D(x=self.x + size, y=self.y       )
		
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
		
		for point in [point1, point2]:
			if not point.in_frame(frame): raise ValueError(f'Line cannot be drawed. Point {point} is out of frame bounds.')

		x1, y1 = point1
		x2, y2 = point2
		
		cv.line(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness, **kwargs)
		
		return frame

class SortedVertices:

	def __init__(self, vertices: NDArray, center: Point2D | None = None) -> None:

		center_: Point2D = default(center, Point2D.from_tuple(np.mean(vertices, axis=0)))
		self._vertices = SortedVertices._sort_point(vertices=vertices, center=center_)

	@staticmethod
	def _sort_point(vertices: NDArray, center: Point2D) -> NDArray:
    
		# Calculate the angle of each point w.r.t. the center
		angles = np.arctan2(vertices[:, 1] - center.y, vertices[:, 0] - center.x)
		
		# Sort vertices by angle
		sorted_indices = np.argsort(angles)

		return vertices[sorted_indices]

	def __str__    (self) -> str: return f'{self.__class__.__name__}[points={len(self)}]'
	def __repr__   (self) -> str: return str(self)
	def __len__    (self) -> int: return len(self._vertices)

	def __getitem__(self, key: int) -> Point2D: return Point2D.from_tuple(self._vertices[key])

	@property
	def vertices(self) -> NDArray: return self._vertices

	def align_to(self, other: SortedVertices):

		if len(self) != len(other): raise ValueError(f'Vertices must have the same length, got {len(self)} and {len(other)}. ')

		# Compute the angle between the first point and each other point of the other set of vertices

		distances = np.linalg.norm(other.vertices - self.vertices[0], axis=1)

		closest_index = int(np.argmin(distances))

		self.roll(n=-closest_index)

	def draw(self, frame: Frame, palette: List[RGBColor] | RGBColor = (255, 0, 0), radius: int = 5, thickness: int = 2) -> Frame:

		palette_ = palette if isinstance(palette, list) else [palette] * len(self)

		if len(palette_) != len(self): raise ValueError(f'Palette must have {len(self)} colors, got {len(palette)}.')

		# Draw lines
		for i, col in enumerate(palette_): self[i].draw_circle(frame=frame, radius=radius, color=col, thickness=thickness)

		return frame

	def roll(self, n: int): self._vertices = np.roll(self._vertices, -n, axis=0)


class Contour:

	@dataclass
	class ContourHierarchy:

		next        : int | None
		previous    : int | None
		first_child : int | None
		parent      : int | None

		@classmethod
		def no_hierarchy(cls) -> Contour.ContourHierarchy: return cls(next=None, previous=None, first_child=None, parent=None)

		@classmethod
		def from_hierarchy(cls, hierarchy: NDArray) -> Contour.ContourHierarchy:

			def default_value(idx: int) -> int | None: return int(idx) if idx != -1 else None
			
			return cls(
				next        = default_value(hierarchy[0]),
				previous    = default_value(hierarchy[1]),
				first_child = default_value(hierarchy[2]),
				parent      = default_value(hierarchy[3])
			)
		
		def __str__ (self) -> str: return f'{self.__class__.__name__}[{"; ".join([f"{k}: {v}" for k, v in self.to_dict().items()])}]'
		def __repr__(self) -> str: return str(self)
		
		def to_dict(self) -> Dict[str, int | None]: return {
			'next'       : self.next,
			'previous'   : self.previous,
			'first_child': self.first_child,
			'parent'     : self.parent
		} 
	
	# HYPERPARAMETERS
	_APPROX_FACTOR         : float = 0.01
	_CIRCULARITY_THRESHOLD : float = 0.80
    
	def __init__(self, id: int, contour: NDArray, hierarchy: Contour.ContourHierarchy):

		self._id            : int                      = id
		self._contour_orig  : NDArray                  = contour
		self._hierarchy     : Contour.ContourHierarchy = hierarchy

		# Approximate the contour
		epsilon = cv.arcLength(contour, closed=True) * Contour._APPROX_FACTOR
		self._contour_approx: NDArray= cv.approxPolyDP(curve=contour, closed=True, epsilon=epsilon)

	def __str__ (self) -> str: return f'{self.__class__.__name__}(id={self.id}, points={len(self)})'
	def __repr__(self) -> str: return str(self)
	def __len__ (self) -> int: return len(self.contour)
	
	@property
	def id(self) -> int: return self._id
	
	@property
	def contour_orig(self) -> NDArray: return self._contour_orig

	@property
	def contour(self) -> NDArray: return self._contour_approx

	@property
	def mean_point(self) -> Point2D: return Point2D.from_tuple(np.mean(self.contour, axis=0, dtype=np.int32)[0])
	
	@property
	def area(self) -> float: return cv.contourArea(self.contour)

	@property
	def perimeter(self) -> float: return cv.arcLength(self.contour, closed=True)

	@property
	def hierarchy(self) -> Contour.ContourHierarchy: return self._hierarchy

	def to_sorted_vertex(self, center: Point2D | None = None, adjusted: bool = True) -> SortedVertices:

		vertices = self.contour if adjusted else self.contour_orig

		return SortedVertices(vertices=vertices[:, 0, :], center=center)

	def draw(self, frame: Frame, color: RGBColor = (255, 0, 0), thickness: int = 2, fill: bool = False, adjusted: bool = True) -> Frame:
	
		if fill: thickness = cv.FILLED

		contours = self.contour if adjusted else self.contour_orig

		cv.drawContours(image=frame, contours=[contours], contourIdx=-1, color=color, thickness=thickness)

		return frame
	
	def is_quadrilateral(self) -> bool:  
		return len(self) == 4 and cv.isContourConvex(self.contour)

	def is_circle(self, thresh: float | None = None) -> bool: 

		# Compute circularity | 4pi * area / perimeter^2

		thresh_: float = default(thresh, Contour._CIRCULARITY_THRESHOLD)
		
		if self.perimeter == 0: return False  # Avoid division by zero for degenerate contours

		return 4 * np.pi * self.area / (self.perimeter ** 2) > thresh_
	
	def frame_mean_value(self, frame: Frame, fill: bool = False, contour_subtraction: List[Contour] | None = None) -> Tuple[float, Frame]:

		# Child subtraction requires filled mask
		if contour_subtraction is not None: fill = True 

		# Create mask
		mask: Frame = np.zeros_like(a=frame, dtype=np.uint8)
		thickness: int = cv.FILLED if fill else 3
		cv.drawContours(image=mask, contours=[self.contour], contourIdx=-1, color=(255, ), thickness=thickness)

		# If children are to be subtracted
		if contour_subtraction is not None:
			for descendant in contour_subtraction:
				cv.drawContours(image=mask, contours=[descendant.contour], contourIdx=-1, color=(0,), thickness=thickness)
		
		# Compute mean value
		mean_value = cv.mean(frame, mask=mask)[0]

		# Write the mean value on the bottom right corner of the mask

		# Get the text size
		text = f'mean: {mean_value:.2f}'
		font_face = cv.FONT_HERSHEY_SIMPLEX
		font_scale = 2.5
		thickness = 10
		pad       = 50

		# Calculate the text size
		(text_width, text_height), baseline = cv.getTextSize(text, font_face, font_scale, thickness)

		image_height, image_width = mask.shape[:2]

		x = image_width - text_width - pad  # 10 pixels padding from the right edge
		y = image_height - pad              # 10 pixels padding from the bottom edge (baseline adjustment included)

		# Put the text on the image
		mask = cv.putText(
			img=mask,
			text=text,
			org=(x, y),
			fontFace=font_face,
			fontScale=font_scale,
			color=(255,),  # White color for grayscale image
			thickness=thickness
		)

		return mean_value, mask
	
	def scale_contour(self, scale: float) -> Contour:
		
		points = self.contour[:, 0, :]                                     # Flatten the contour to a Nx2 array
		centroid = np.mean(points, axis=0)                                 # Compute the centroid of the quadrilateral
		scaled_points = (points - centroid) * scale + centroid             # Scale each point towards the centroid
		scaled_contour = scaled_points.reshape(-1, 1, 2).astype(np.int32)  # Convert back to the original contour format (Nx1x2)
		
		return Contour(
			id=-1, 
			contour=scaled_contour, 
			hierarchy=Contour.ContourHierarchy.no_hierarchy()
		)


class Contours:

	def __init__(self, frame: Frame, min_area: float | None = None, max_area: float | None = None):

		contours, hierarchy = cv.findContours(image=frame, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
		self._contours_dict = {}

		if len(contours) == 0: return

		for contour_id, (contour, hierarchy_line) in enumerate(zip(contours, hierarchy[0])):

			area = cv.contourArea(contour)

			if (min_area is None or area >= min_area) and (max_area is None or area <= max_area):

				self._contours_dict[contour_id] = Contour(
					id=contour_id, 
					contour=contour, 
					hierarchy=Contour.ContourHierarchy.from_hierarchy(hierarchy=hierarchy_line)
				)

	def __str__     (self)           -> str              : return f'{self.__class__.__name__}[curves: {len(self)}]'
	def __repr__    (self)           -> str              : return str(self)
	def __len__     (self)           -> int              : return len(self._contours_dict)
	def __iter__    (self)           -> Iterator[Contour]: return iter(self._contours_dict.values())
	def __getitem__ (self, key: int) -> Contour | None   : return self._contours_dict.get(key, None)

	def get_descendants(self, contour: Contour) -> Sequence[Contour]:
    
		def _get_descendants(id: int | None) -> List[int]:

			descendants: Set[int] = set()

			while id is not None:
					
					curr_contour = self[id]

					if curr_contour is None: break
				
					# Add to descendants
					descendants.add(id)

					# Child
					child = curr_contour.hierarchy.first_child
					if child is not None and child not in descendants: descendants.update(_get_descendants(child))

					# Next
					id = curr_contour.hierarchy.next
			
			return list(descendants)

		# Start with the first child of the given node
		decendants_id = _get_descendants(id=contour.hierarchy.first_child)
		
		return [self[id] for id in decendants_id if self[id] is not None]  # type: ignore - self[id] is not None check
	
	def get_ancestors(self, contour: Contour) -> Sequence[Contour]:
		
		ancestors: List[Contour] = []
		current = contour

		while current.hierarchy.parent is not None:

			current = self[current.hierarchy.parent]
			if current is None: break
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
		marker_vertices.roll(n=c0_vertex_id)

		c0, c1, c2, c3 = [marker_vertices[i] for i in range(4)]
		point = anchor_contour.mean_point

		return cls(c0=c0, c1=c1, c2=c2, c3=c3, anchor=point)
	
	def __str__(self) -> str:
		corners_str = '; '.join([f'c{i}={corner}' for i, corner in enumerate(self.corners)])
		return f'{self.__class__.__name__}[{corners_str}; anchor={self.anchor})'
	
	def __repr__(self) -> str: return str(self)

	def __getitem__(self, index: int) -> Point2D: return self.corners[index]

	@property
	def adjacent_couples(self) -> List[Tuple[Point2D, Point2D]]:
		return [(self.c0, self.c1), (self.c1, self.c2), (self.c2, self.c3), (self.c3, self.c0)]

	@property
	def corner_palette(self) -> List[RGBColor]:
		return [
			( 0, 255,  255),
			( 0, 255,  255),
			( 0, 255,  255),
			( 0, 255,  255)
		]

	@property
	def line_color(self) -> RGBColor: return (255, 0, 0)

	@property
	def anchor_color(self) -> RGBColor: return (0, 255, 0)
	
	@property
	def corners(self) -> Points2D: return [self.c0, self.c1, self.c2, self.c3]

	def get_world_points(self, scale: int = 1, z: bool = False) -> NDArray: 

		unscaled_points = np.array(
			object=[
				[0, 0], 
				[0, 1], 
				[1, 1], 
				[1, 0]
			],
			dtype=np.float32
		)

		scaled_points = unscaled_points * scale

		if z: return np.hstack([scaled_points, np.zeros((4, 1))])

		return scaled_points

	def to_corners_array(self, homogeneous : bool = False) -> NDArray: 

		return np.array(
			[
				tuple(iter(corner)) + tuple([1] if homogeneous else [])
				for corner in self.corners
			], 
			dtype=np.float32
		)

	def warp(self, frame: Frame, side: int) -> Frame:

		pixel_points = self.to_corners_array()                              # Marker corner pixels in the image plane c0, c1, c2, c3
		world_points = self.get_world_points(scale=side, z=False) # World points ([0, 0, 1]; [W, 0, 1]; [W, H, 1]; [0, H, 1])

		# Compute the perspective transform matrix
		H, _ = cv.findHomography(srcPoints=pixel_points, dstPoints=world_points)

		# Apply the perspective transform
		warped = cv.warpPerspective(src=frame, M=H, dsize=(side, side))

		return warped
	
	def _camera_2d_position_geometric(
		self, 
		pixel_points: NDArray,
		world_points: NDArray,
		calibration: CalibratedCamera
	) -> Tuple[NDArray, NDArray]:

		# Homography and Calibration matrix
		H, _ = cv.findHomography(srcPoints=world_points, dstPoints=pixel_points)
		K = calibration.camera_mat

		# Compute RT = K^-1 * H
		RT = np.linalg.inv(K) @ H

		# Extract r1, r2, t
		r1, r2, t = RT.T

		# Compute scaling factor alpha = 2 / (||r1|| + ||r2||)
		alpha = 2 / (np.linalg.norm(r1) + np.linalg.norm(r2))

		# Scale r1, r2, t
		RT_norm = RT / alpha
		r1_norm, r2_norm, t_norm = RT_norm.T

		# Compute r3 = r1 x r2
		r3_norm = np.cross(r1_norm, r2_norm)

		# Construct rotation matrix, with no guarantee of orthogonality
		Q = np.column_stack((r1_norm, r2_norm, r3_norm))

		# Orthonormalize using SVD
		U, _, Vt = np.linalg.svd(Q)  # Q = U * S * V^t
		R = U @ Vt                   # R = U * V^t

		assert np.allclose(R @ R.T, np.eye(3)), 'R is not orthonormal'

		return R, t_norm


	def _camera_2d_position_algebraic(
		self, 
		pixel_points: NDArray,
		world_points: NDArray,
		calibration: CalibratedCamera
	) -> Tuple[NDArray, NDArray]:
		
		# Camera matrix
		K = calibration.camera_mat

		#Distortion coefficients
		#dist_coeffs = calibration.distortion_coeffs
		dist_coeffs = np.zeros(5)
		
		# Use p2p algorithm
		succ, r, t = cv.solvePnP(
			objectPoints=world_points,
			imagePoints=pixel_points,
			cameraMatrix=K,
			distCoeffs=dist_coeffs
		)

		if not succ: raise ValueError('PnP algorithm failed to converge. ')

		R, _ = cv.Rodrigues(r)

		return R, t[:, 0]
	
	def camera_2d_position(self, calibration: CalibratedCamera, method: LightDirectionMethod = 'algebraic', scale: int = 1):

		pixel_points = self.to_corners_array()                               # Marker corner pixels in the image plane c0, c1, c2, c3
		world_points = self.get_world_points(scale=scale, z=True) # World points ([0, 0, 1]; [W, 0, 1]; [W, H, 1]; [0, H, 1])

		match method.lower():

			case 'algebraic': get_RT_fn = self._camera_2d_position_algebraic
			case 'geometric': get_RT_fn = self._camera_2d_position_geometric
			case _: raise ValueError(f'Unknown method to compute geometric camera position: {method}. ')
				
		R, t = get_RT_fn(
			pixel_points=pixel_points,
			world_points=world_points,
			calibration=calibration
		)

		pose = -R.T @ t

		# Normalize pose
		pose_norm  = pose / np.linalg.norm(pose)

		# Decompose pose
		u, v, w = pose_norm

		assert w > 0, 'The camera is not pointing towards the marker'

		return u, v


	def draw(self, frame: Frame) -> Frame:

		for corner1, corner2 in self.adjacent_couples:
			Point2D.draw_line(frame=frame, point1=corner1, point2=corner2, color=self.line_color, thickness=8)

		for c_id, (corner, color) in enumerate(zip(self.corners, self.corner_palette)):
			corner.draw_circle(frame=frame, radius=4, color=color, thickness=12)

			# Add the name close to the point
			position = (int(corner.x) + 10, int(corner.y) - 10)  # Offset the text position slightly
			cv.putText(
				frame, f'{c_id}', position,
				fontFace=cv.FONT_HERSHEY_SIMPLEX,
				fontScale=1., color=color, thickness=4
			)

		self.anchor.draw_cross(frame=frame,  size=17, color=self.anchor_color, thickness=10)

		return frame

class MarkerDetector:

	CORNER_SCALE_FACTOR = 0.9

	def __init__(
		self, 
		white_thresh       : float        = 255 - 25,
		black_thresh       : float        =   0 + 25,
		min_contour_area   : float | None = 200,
		max_contour_area   : float | None = 1920 * 1080 * 0.5,
		corner_mask_method : CornerMaskMethod = 'scaled'
	):

		self._white_thresh       : float            = white_thresh
		self._black_thresh       : float            = black_thresh
		self._min_contour_area   : float | None     = min_contour_area
		self._max_contour_area   : float | None     = max_contour_area
		self._corner_mask_method : CornerMaskMethod = corner_mask_method
	
	def __str__(self) -> str: return f'{self.__class__.__name__}[{"; ".join([f"{k}: {v}" for k, v in self.params.items()])}]'
	
	def __repr__(self) -> str: return str(self)

	@property
	def params(self) -> Dict[str, float | CornerMaskMethod]: return {
		k: v for k, v in [
			('white_thresh'      , self._white_thresh      ),
			('black_thresh'      , self._black_thresh      ),
			('min_contour_area'  , self._min_contour_area  ),
			('max_contour_area'  , self._max_contour_area  ),
			('corner_mask_method', self._corner_mask_method)
		]
		if v is not None
	}
	
	def _detect_corners(self, frame: Frame, contours: Contours) -> Tuple[Tuple[Contour, Contour] | None, str, Views]:
    
		# (child, parent)
		nested_quadrilaterls: List[Tuple[Contour, Contour]] = []

		# Loop through all contours
		for contour in contours:

			# Skip if a) not a quadrilateral or b) has no parent
			if not contour.is_quadrilateral()         : continue
			if contour.hierarchy.parent is None       : continue

			# Get parent contour
			parent = contours[contour.hierarchy.parent]

			if parent is None: continue

			# Skip if parent is not a quadrilateral
			if not parent.is_quadrilateral(): continue

			# Append to list
			nested_quadrilaterls.append((contour, parent))
		
		# Check if there is only one nested quadrilateral
		if len(nested_quadrilaterls) == 0: return None, f'No nested squares found. ', {}
		if len(nested_quadrilaterls) >  1: return None, f'Found multiple squares ({len(nested_quadrilaterls)}). ', {}

		inner, outer = nested_quadrilaterls[0]

		# Check black to white

		match self._corner_mask_method:

			case 'border': 
				white_mean, mask1 = inner.frame_mean_value(frame=frame)
				black_mean, mask2 = outer.frame_mean_value(frame=frame)
			
			case 'descendants':
				white_mean, mask1 = inner.frame_mean_value(frame=frame, contour_subtraction=list(contours.get_descendants(contour=inner)))
				black_mean, mask2 = outer.frame_mean_value(frame=frame, contour_subtraction=[inner])
			
			case 'scaled':
				white_mean, mask1 = inner.frame_mean_value(frame=frame, contour_subtraction=[inner.scale_contour(scale=self.CORNER_SCALE_FACTOR)])
				black_mean, mask2 = outer.frame_mean_value(frame=frame, contour_subtraction=[outer.scale_contour(scale=self.CORNER_SCALE_FACTOR)])
			
			case _: raise ValueError(f'Unknown corner mask method: {self._corner_mask_method}. ')

		inner_white = self._white_thresh < white_mean
		outer_black = self._black_thresh > black_mean

		views: Views = {'inner_mask': mask1, 'outer_mask': mask2}

		if inner_white and outer_black:
			return (inner, outer), '', views
		else:
			return None, f'No black to white transition between squares. (white mean: {white_mean:.2f}, black mean: {black_mean:.2f}) ', views
	
	def _detect_anchor(
			self,
			frame: Frame,
			contours: Contours,
			marker_vertices: Tuple[SortedVertices, SortedVertices],
		) -> Tuple[Tuple[int, Contour] | None, str, Views]:

		def is_contour_between_points(contour: Contour, point1: Point2D, point2: Point2D) -> Tuple[bool, Frame]:

			mask1 = np.zeros_like(a=frame, dtype=np.uint8)
			mask2 = mask1.copy()

			color: RGBColor = (1, ) # type: ignore

			contour.draw(frame=mask1, color=color, fill=True)
			Point2D.draw_line(frame=mask2, point1=point1, point2=point2, color=color, thickness=3)

			return 2 in mask1 + mask2, (mask1 | mask2) * 255

		# vertex index, circle contour
		marker_circles: List[Tuple[int, Contour]] = []

		out_mask: Frame = np.zeros_like(a=frame, dtype=np.uint8)
		for contour in contours:

			# Skip if is not a circle
			if not contour.is_circle(): continue

			# Skip if the circle is not white
			# mean_value = contour.frame_mean_value(frame=frame, fill=True)
			# if mean_value < self._white_thresh: continue

			# For the 4 couples of vertex check if the circle is between the two points
			inner_vert, outer_vert = marker_vertices
			for i, (inner, outer) in enumerate(zip(inner_vert.vertices, outer_vert.vertices)):
				valid_contour, mask = is_contour_between_points(contour, Point2D.from_tuple(inner), Point2D.from_tuple(outer))
				if valid_contour:
					marker_circles.append((i, contour))
					out_mask: Frame = mask | out_mask  # type: ignore - or is supported for numpy arrays
		
		views: Views = {'anchor_mask': out_mask}

		if len(marker_circles) == 0: return None, f'No anchor found within the marker. ',                                views
		if len(marker_circles) >  1: return None, f'Found multiple anchors within the marker ({len(marker_circles)}). ', views

		return marker_circles[0], '', views

	def detect(self, frame: Frame) -> Tuple[Marker | None, str, Views]:

		empty_frame = np.zeros_like(frame)
		empty_views = {view: empty_frame for view in ['inner_mask', 'outer_mask', 'anchor_mask']}

		frame_c = cv.cvtColor(frame.copy(), cv.COLOR_GRAY2RGB)

		views = {}
		frame_contours_orig  = frame_c.copy()
		frame_contours_adj   = frame_c.copy()

		contours = Contours(frame=frame, min_area=self._min_contour_area, max_area=self._max_contour_area)
	
		palette = generate_palette(n=len(contours))
		contours.draw(frame=frame_contours_orig, colors=palette, thickness=10, adjusted=False)
		contours.draw(frame=frame_contours_adj,  colors=palette, thickness=10, adjusted=True)
		views['contours_orig'] = frame_contours_orig
		views['contours_adj' ] = frame_contours_adj

		# 1. Detect marker corners
		marker_corners, warn_message, corner_views = self._detect_corners(frame=frame, contours=contours)

		if marker_corners is None: return None, warn_message, views | empty_views | corner_views

		inner_marker_contour, outer_marker_contour = marker_corners
		
		center   = np.concatenate([inner_marker_contour.contour, outer_marker_contour.contour], axis=0)
		center2d = Point2D.from_tuple(np.mean(center, axis=0, dtype=np.int32)[0])

		inner_marker_vertices = inner_marker_contour.to_sorted_vertex(center=center2d)
		outer_marker_vertices = outer_marker_contour.to_sorted_vertex(center=center2d)

		inner_marker_vertices.align_to(other=outer_marker_vertices)

		# 2. Detect circles
		circle_detection, warning_message, anchor_views = self._detect_anchor(
			frame=frame,
			contours=contours, 
			marker_vertices=(inner_marker_vertices, outer_marker_vertices), 
		)

		if circle_detection is None: return None, warning_message, views | empty_views | corner_views | anchor_views

		v_id, circle = circle_detection

		marker = Marker.from_detection(
			marker_vertices=inner_marker_vertices,
			c0_vertex_id=v_id,
			anchor_contour=circle
		)

		return marker, '', views | empty_views | corner_views | anchor_views

class MarkerDetectionVideoStream(ThresholdedVideoStream):

	def __init__(
        self, 
        path            : str, 
        calibration     : CalibratedCamera,
        thresholding    : Thresholding,
		marker_detector : MarkerDetector,
        name            : str        = '',
        logger          : BaseLogger = SilentLogger(),
        verbose         : bool       = False
    ):

		super().__init__(
			path=path, 
			calibration=calibration, 
			thresholding=thresholding,
			name=name, 
			logger=logger, 
			verbose=verbose
		)

		self._marker_detector = marker_detector
		self._success: int = -1
		self._total  : int = -1

	@property
	def _str_name(self) -> str: return f'MarkerDetectionVideoStream'

	def play(
        self, 
        start        : int                               = 0,
        end          : int                        | None = None, 
        skip_frames  : int                               = 1,
        window_size  : Dict[str, Size2D] | Size2D | None = None,
        exclude_views: List[str]                         = [],
		delay        : int                               = 1
    ):
		
		self._success: int = 0
		self._total  : int = 0

		super().play(
			start=start, 
			end=end, 
			skip_frames=skip_frames, 
			window_size=window_size, 
			exclude_views=exclude_views,
			delay=delay
		)

		success, total = self.marker_detection_results

		if success == total: info_msg = f'All frames were successfully processed. '
		else:                info_msg = f'Processed {success} out of {total} frames. ({success / total:.2%}) '
		
		self._logger.info(msg=info_msg)

	@property
	def marker_detection_results(self) -> Tuple[int, int]: 

		if self._success == -1 or self._total == -1: raise ValueError(f'No results available. ')
		
		return self._success, self._total

	def _process_marker(self, views: Views, marker: Marker, frame_id: int) -> Views:

		return {'marker': marker.draw(frame=views['undistorted'].copy())}

	def _process_frame(self, frame: Frame, frame_id: int) -> Views:

		debugging = self._is_debug(frame_id=frame_id)

		if not debugging: self._total += 1
	
		views = super()._process_frame(frame=frame, frame_id=frame_id)

		# Detect marker
		marker, warning, marker_views = self._marker_detector.detect(frame=views['binary'])

		if marker is None:
			if not debugging: self._logger.warning(f'[{self.name}] Unable to process frame {frame_id} - {warning}')
			return views | marker_views | {'marker': views['undistorted']}
		
		# Process marker
		if not debugging: self._success += 1
		marker_processed_views = self._process_marker(views=views, marker=marker, frame_id=frame_id)
		
		return views | marker_views | marker_processed_views