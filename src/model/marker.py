'''
This file implement the marker detection logic. 
	- The `Marker` class provides the attributes of the marker: the square corners and the anchor point. 
		It also provides  functionalities to draw the marker on the frame, and to make operations on a frame,
		like warp the marker to a square image, and estimate the camera pose.
	- The `MarkerDetector` class detects the marker in the frame, identifying the inner and outer marker squares and the anchor point.
	- The `MarkerDetectionVideoStream` class processes the video stream to detect the marker in each frame.
'''

from __future__ import annotations

from functools import partial
from typing import Dict, List, Tuple, get_args
from dataclasses import dataclass

import numpy as np
import cv2 as cv
from numpy.typing import NDArray

from src.model.model import Contour, Contours, LightDirection, Point2D, Points2D, SortedVertices
from src.utils.misc   import generate_palette
from src.utils.io_ import BaseLogger, SilentLogger
from src.utils.calibration import CalibratedCamera
from src.utils.typing import Frame, RGBColor, Views, Size2D, CameraPoseMethod, MarkerSquareMethod
from src.model.thresholding import ThresholdedVideoStream, Thresholding

# __________________________________ MARKER __________________________________

@dataclass 
class Marker:
	''' 
	Dataclass representing a marker with its corners and anchor point in pixel coordinates.
	The dataclass can be created from the information about the marker detection.
	It provides multiple functionalities to:
		- Draw the marker on a frame.
		- Warp the marker from a frame to a square image (used for the static camera).
		- Compute the camera pose using two different methods: algebraic and geometric (used for the dynamic camera).
	'''

	# NOTE: c0 is the closest corner to the anchor point. The other corners are assigned in clockwise order.
	c0	 : Point2D
	c1	 : Point2D
	c2	 : Point2D
	c3	 : Point2D
	anchor : Point2D

	@classmethod
	def from_detection(
		cls,
		marker_vertices : SortedVertices,
		anchor_contour  : Contour
	):
		'''
		Instantiate a marker from the marker detection information: the marker vertices and the anchor contour.
		:param marker_vertices: The sorted vertices of the marker, with the closest corner to the anchor point first.
		:param anchor_contour: The contour of the anchor point.
		'''

		if len(marker_vertices) != 4	 : raise ValueError(f'Invalid number of vertices for the marker: expected 4, got {len(marker_vertices)}. ')
		if not anchor_contour.is_circle(): raise ValueError(f'Invalid circle contour for the marker. ') 

		c0, c1, c2, c3 = [marker_vertices[i] for i in range(4)]  # Corners in clockwise order
		point = anchor_contour.center_point					  # Anchor point is the mean point of the circle contour

		return cls(c0=c0, c1=c1, c2=c2, c3=c3, anchor=point)
	
	# --- MAGIC METHODS ---
	
	def __str__(self) -> str:
		corners_str = '; '.join([f'c{i}={corner}' for i, corner in enumerate(self.corners)])
		return f'{self.__class__.__name__}[{corners_str}; anchor={self.anchor})'
	
	def __repr__(self) -> str: return str(self)

	def __getitem__(self, index: int) -> Point2D: return self.corners[index]

	# --- PROPERTIES ---

	@property
	def corners(self) -> Points2D: return [self.c0, self.c1, self.c2, self.c3]

	@property
	def adjacent_couples(self) -> List[Tuple[Point2D, Point2D]]:
		return [(self.c0, self.c1), (self.c1, self.c2), (self.c2, self.c3), (self.c3, self.c0)]
	
	# --- DRAW & COLORS ---
	
	@property
	def side_color(self) -> RGBColor: return (255, 0, 0)

	@property
	def anchor_color(self) -> RGBColor: return (0, 255, 0)

	@property
	def corner_colors(self) -> List[RGBColor]:
		return [
			( 0, 255,  255),
			( 0, 255,  255),
			( 0, 255,  255),
			( 0, 255,  255)
		]

	def draw(self, frame: Frame) -> Frame:
		''' Draw the marker on the frame with the corners and the anchor point. '''

		# Draw the marker sides
		for ca, cb in self.adjacent_couples:
			Point2D.draw_line(frame=frame, point1=ca, point2=cb, color=self.side_color, thickness=8)

		# Draw the corners points and the associated order
		for c_id, (corner, color) in enumerate(zip(self.corners, self.corner_colors)):

			# Corner circle
			corner.draw_circle(frame=frame, radius=4, color=color, thickness=12)

			# Corner number
			position = (int(corner.x) + 10, int(corner.y) - 10)  # Offset the text position slightly
			cv.putText(
				frame, f'{c_id}', position,
				fontFace=cv.FONT_HERSHEY_SIMPLEX,
				fontScale=1., color=color, thickness=4
			)

		# Draw the anchor point
		self.anchor.draw_cross(frame=frame,  size=17, color=self.anchor_color, thickness=10)

		return frame

	# --- MARKER PROCESSING, WARPING and CAMERA POSE ---
	
	'''
	Following methods are used to process the marker, specifically to:
		- Warp the marker from the frame to a square image (static camera).
		- Compute the camera pose (dynamic camera).

	Both methods will work with a mapping between:
		- Word points: the marker corners in the world coordinate system.
		- Pixel points: the marker corners in the image coordinate system (pixels in the frame).
	'''

	def get_world_points(self, scale: Size2D = (1, 1), z: bool = False) -> NDArray: 
		'''
		Get the world points of the marker corners.
		The world points are up-to-scale and can be controlled with the parameter `scale` to adjust the marker size.
		Object points are planar and lie on the z=0 plane and by default the function returns 2D points.
		If `z=True`, the function returns 3D points with the z-coordinate set to 0.

		:param scale: The scale factor to adjust the marker size in height and width.
		:param z: If True, return the world points with the z-coordinate set to 0.
		:return: The world points of the marker corners (4x2 or 4x3 array).		
		'''

		# Real word points are c0: [0, 0], c1: [0, 1], c2: [1, 1], c3: [1, 0]
		# The function is flexible to scale the unitary-side square to an arbitrary rectangle
		scale_x, scale_y = scale
		points = np.array(object=[[0, 0], [0, scale_y], [scale_x, scale_y], [scale_x, 0]], dtype=np.float32)

		# Optionally add the z-coordinate as 0-vector
		if z: return np.hstack([points, np.zeros((4, 1))])

		return points

	def get_pixel_points(self, homogeneous : bool = False) -> NDArray: 
		'''
		Return the pixel points from the marker corners as a 4x2 array.
		Optionally return the pixel points with the homogeneous coordinate as a 4x3 array with the last column set to 1.

		:param homogeneous: If True, return the pixel points with the homogeneous coordinate.
		:return: The pixel points of the marker corners (4x2 or 4x3 array).
		'''

		# Stack the pixel points as a 4x2 array
		pixel_points = np.array([tuple(iter(corner)) for corner in self.corners], dtype=np.float32)

		# Optionally add the homogeneous coordinate as 1-vector
		if homogeneous: return np.hstack([pixel_points, np.ones((4, 1))])

		return pixel_points

	def warp(self, frame: Frame, size: Size2D) -> Frame:
		'''
		Warp the marker from the frame to a square image of a specific side (the number of pixels in the output frame).
		Used to process the static camera.

		:param frame: The frame to warp the marker from.
		:param size: The size of the square warped image in pixels.
		:return: The warped marker as a square image.
		'''

		# Get the pixel and world points
		pixel_points = self.get_pixel_points(homogeneous=False)   # Marker corner pixels in the image plane c0, c1, c2, c3 as a 4x2 array
		world_points = self.get_world_points(scale=size, z=False) # Real-world points ([0, 0]; [side, 0]; [side, side]; [0, side]) as a 4x2 array

		# Compute the perspective transform matrix
		H, _ = cv.findHomography(srcPoints=pixel_points, dstPoints=world_points)

		# Apply the perspective transform
		warped = cv.warpPerspective(src=frame, M=H, dsize=size)

		# Flip vertically the warped image to make (0, 0) match the top-left corner
		warped = cv.flip(warped, flipCode=0)

		return warped
	
	@staticmethod	
	def _estimate_Rt_algebraic(
		pixel_points: NDArray,
		world_points: NDArray,
		calibration: CalibratedCamera
	) -> Tuple[NDArray, NDArray]:
		'''
		Estimate the rotation matrix and the translation vector of the camera using the algebraic method:
		1. Estimate the homography from the pixel points and the world points.
		2. Obtain the first two columns of the rotation matrix r1 and r2 and the translation vector t from the homography.
		3. Normalize by the scaling factor alpha = 2 / (||r1|| + ||r2||).
		4. Compute the third column of the rotation matrix r3 = r1 x r2 and build the rotation matrix Q = [r1, r2, r3].
		5. Orthonormalize the rotation matrix using the SVD decomposition.

		:param pixel_points: The pixel points of the marker corners as a 4x2 array.
		:param world_points: The world points of the marker corners as a 4x3 array.
		:param calibration: The calibrated camera object with the intrinsic matrix.
		:return: The rotation matrix and the translation vector of the camera.
		'''

		# Homography H and Intrinsic Camera Matrix K
		H, _ = cv.findHomography(srcPoints=world_points, dstPoints=pixel_points)
		K = calibration.camera_mat

		# Compute RT = K^-1 @ H
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
		R = U @ Vt				   # R = U * V^t

		# Check if the rotation matrix is orthonormal
		if not np.allclose(R @ R.T, np.eye(3)): raise ValueError('The estimated rotation matrix is not orthonormal. ')

		return R, t_norm

	@staticmethod
	def _estimate_RT_geometric(
		pixel_points : NDArray,
		world_points : NDArray,
		calibration  : CalibratedCamera,
		undistort	: bool = True
	) -> Tuple[NDArray, NDArray]:
		'''
		Estimate the rotation matrix and the translation vector of the camera using the geometric method.
		The solvePnP algorithm is used to retrieve the orthonormal rotation matrix and the translation vector.

		:param pixel_points: The pixel points of the marker corners as a 4x2 array.
		:param world_points: The world points of the marker corners as a 4x3 array.
		:param calibration: The calibrated camera object with the intrinsic matrix.
		:param undistort: If True, undistort the pixel points using the camera matrix.
		:return: The rotation matrix and the translation vector of the camera.
		'''
		
		# Camera matrix
		K = calibration.camera_mat

		# Distortion coefficients
		# NOTE: When `undistort` is False we expect to work with an already undistorted image
		dist_coeffs = calibration.distortion_coeffs if undistort else np.zeros(5)
		
		# Use p2p algorithm
		succ, r, t = cv.solvePnP(
			objectPoints=world_points,
			imagePoints=pixel_points,
			cameraMatrix=K,
			distCoeffs=dist_coeffs
		)

		if not succ: raise ValueError('PnP algorithm failed to converge. ')

		# Convert the rotation vector to the rotation matrix using Rodrigues formula
		R, _ = cv.Rodrigues(r)

		return R, t[:, 0]
	
	def estimate_camera_pose(self, calibration: CalibratedCamera, method: CameraPoseMethod = 'algebraic', size: Size2D = (1, 1)) -> LightDirection:
		'''
		Estimate camera pose using marker pixel points and the intrinsic camera matrix.
		It can use two different methods:
			1. Algebraic: retrieves the orthonormal rotation matrix and the translation vector from the estimated homography.
			2. Geometric: uses the solvePnP algorithm.
		
		:param calibration: The calibrated camera object with the intrinsic matrix.
		:param method: The method to estimate the camera pose: `algebraic` or `geometric`.
		:param size: The size to adjust the real-world marker size.
		'''

		pixel_points = self.get_pixel_points(homogeneous=False)   # Marker corner pixels in the image plane c0, c1, c2, c3 as a 4x2 array
		world_points = self.get_world_points(scale=size, z=True) # Real-world points ([0, 0, 0]; [side, 0, 0]; [side, side, 0]; [0, side, 0]) as a 4x3 array

		# Use the method to select which of the two function to use to estimate Rotation Matrix and Translation Vector
		match method.lower():
			case 'algebraic': get_RT_fn = self._estimate_Rt_algebraic
			case 'geometric': get_RT_fn = partial(self._estimate_RT_geometric, undistort=False)		   # NOTE: We expect to work with an already undistorted image
			case _: raise ValueError(f'Unknown method to compute geometric camera position: {method}. ')
				
		# Retrieve
		# - R: 3x3 rotation matrix;
		# - T: 3x1 translation vector 
		# from the pixel points and the world points using the selected method
		R, t = get_RT_fn(
			pixel_points=pixel_points,
			world_points=world_points,
			calibration=calibration
		)

		# Estimate the camera pose using the inverse homography
		# R^T @ t is the position of the center of the camera in the world reference system
		pose = -R.T @ t

		return LightDirection.from_3d_light_vector(light_vector=pose)
	
# __________________________________ DETECTION __________________________________

class MarkerDetector:
	''' 
	Class to detect the marker in a specific frame: it identifies specific contours as the inner and outer marker squares and the anchor point.	
	. It is parametrized with different options to:
		- filter out the contours based on their area (min and max area).
		- detect the black to white transition between the inner and outer marker squares (color thresholds, corner mask methods).
		- detect the anchor point (circularity threshold).
	'''

	def __init__(
		self, 
		min_contour_area	: float | None       = 200,
		max_contour_area	: float | None       = 1920 * 1080 * 0.5,
		white_thresh		: float		         = 255 - 25,
		black_thresh		: float		         =   0 + 25,
		corner_mask_method  : MarkerSquareMethod = 'scaled',
		corner_scale_factor : float              = 0.9,
		circularity_thresh  : float              = 0.80,
	):
		'''
		Initialize the marker detector with the different parameters to process contours.
		1. Area filter by area: minimum and maximum area bounds are given to filter out the contours.
		2. For the black to white transition of the marker squares:
			- We establish white and black thresholds to detect the transition.
			- We define the method to mask the corners: {`border`, `descendants`, `scaled`}.
			- In the case of `scaled` method, we define the scale factor to apply to the inner and outer marker squares.
		3. For the anchor point detection, we set the circularity threshold to filter out the non-circle contours.

		:param min_contour_area: The minimum area of the contour to keep. If None, no minimum area is applied.
		:param max_contour_area: The maximum area of the contour to keep. If None, no maximum area is applied.
		:param white_thresh: The white threshold to detect the black to white transition between the inner and outer marker squares.
		:param black_thresh: The black threshold to detect the black to white transition between the inner and outer marker squares.
		:param corner_mask_method: The method to mask the corners: {`border`, `descendants`, `scaled`}.
		:param corner_scale_factor: The scale factor to apply to the inner and outer marker squares in the `scaled` method (not used for the other methods).
		:param circularity_thresh: The circularity threshold to detect the anchor point.	
		'''

		self._white_thresh		  : float			   = white_thresh
		self._black_thresh		  : float			   = black_thresh
		self._min_contour_area	  : float | None	   = min_contour_area
		self._max_contour_area	  : float | None	   = max_contour_area
		self._corner_mask_method  : MarkerSquareMethod = corner_mask_method
		self._corner_scale_factor : float			   = corner_scale_factor
		self._circularity_thresh  : float			   = circularity_thresh

	# --- MAGIC METHODS ---
	
	def __str__ (self) -> str : return f'{self.__class__.__name__}[{"; ".join([f"{k}: {v}" for k, v in self.params.items()])}]'
	def __repr__(self) -> str : return str(self)

	@property
	def params(self) -> Dict[str, float | MarkerSquareMethod]: return {
		k: v for k, v in [
			('white thresh'	  , self._white_thresh	  ),
			('black thresh'	  , self._black_thresh	  ),
			('min contour area'  , self._min_contour_area  ),
			('max contour area'  , self._max_contour_area  ),
			('corner mask method', self._corner_mask_method),
			('circularity thresh', self._circularity_thresh)
		] + [('corner scale factor', self._corner_scale_factor) if self._corner_mask_method == 'scaled' else None]
		if v is not None
	}

	# --- DETECTION ---

	'''
	We break up the detection process in two steps: 
		1) Detect the marker squares.
		2) Detect the anchor point.
	If the first fails, the second is not executed.

	Each of the two step returns a triple with:
	- The detected contours if detected, otherwise None.
	- The string with the warning message if not detected, otherwise an empty string.
	- The views dictionary with the intermediate processing steps of the detection process (masks, etc...)
	'''
	
	def _detect_marker_squares(self, frame: Frame, contours: Contours) -> Tuple[Tuple[Contour, Contour] | None, str, Views]:
		'''
		Detect the inner and outer marker squares in the frame. 
		1. It uses the contours detected in the frame and their hierarchy to find the nested quadrilaterals.
			Any couple of nested quadrilaterals are valid candidates for the inner and outer marker squares.
		2. If a single couple of nested quadrilaterals is found, it checks the black-to-white transition between the squares.

		:param frame: The binary frame to detect the marker squares on (used to check the black-to-white transition).
		:param contours: The contours detected in the frame as candidate marker squares.
		'''
	
		# Couple of valid nested quadrilaterals found. 
		# The tuple is composed by the inner and outer marker squares.
		nested_quadrilaterals: List[Tuple[Contour, Contour]] = []

		for contour in contours:

			if not contour.is_quadrilateral()   : continue  # Skip if contour is not a quadrilateral
			if contour.hierarchy.parent is None : continue  # Skip if contour has no parent

			parent = contours[contour.hierarchy.parent]

			if parent is None				 : continue  # Skip if parent was removed by the area filter
			if not parent.is_quadrilateral() : continue  # Skip if parent is not a quadrilateral

			nested_quadrilaterals.append((contour, parent)) # Valid nested quadrilateral found
		
		# If not exactly one couple of nested quadrilaterals is found, return None
		if len(nested_quadrilaterals) == 0: return None, f'No nested squares found. ',							   {}
		if len(nested_quadrilaterals) >  1: return None, f'Found multiple squares ({len(nested_quadrilaterals)}). ', {}

		# Get the inner and outer marker squares
		inner, outer = nested_quadrilaterals[0]

		# Check the black-to-white transition between the inner and outer marker squares using the selected corner mask method
		match self._corner_mask_method:

			# A) Border method: Compute the mean value only along the borders of the squares.
			# Pros: Fast and simple.
			# Cons: The border lies at the edge between the black and white squares, making it prone to noise. Typically requires more flexible thresholds.
			case 'border': 
				white_mean, mask1 = inner.mean_value_on_frame(frame=frame)
				black_mean, mask2 = outer.mean_value_on_frame(frame=frame)
			
			# B) Descendants method: Subtract the area of descendant contours from the contour's inner area.
			# Pros: More robust as it explicitly uses the full surface of the inner and outer squares not covered by other squares.
			# Cons: More computationally expensive.
			case 'descendants':
				white_mean, mask1 = inner.mean_value_on_frame(frame=frame, contour_subtraction=list(contours.get_descendants(contour=inner)))
				black_mean, mask2 = outer.mean_value_on_frame(frame=frame, contour_subtraction=[inner]) # NOTE: For the outer square, we subtract only the inner square.
			
			# C) Scaled method: A middle ground between the other two methods. 
			#				    It computes an artificial child contour by scaling the inner and outer squares by a factor.
			# Pros: A direct and simple method to create a reasonable child contour and compute a good inner area of the squares. 
			case 'scaled':
				white_mean, mask1 = inner.mean_value_on_frame(frame=frame, contour_subtraction=[inner.scale_contour(scale=self._corner_scale_factor)])
				black_mean, mask2 = outer.mean_value_on_frame(frame=frame, contour_subtraction=[outer.scale_contour(scale=self._corner_scale_factor)])
			
			case _: raise ValueError(f'Unknown corner mask method: {self._corner_mask_method}. Use one of {list(get_args(CornerMaskMethod))}. ')

		# Processing views
		views: Views = {'inner_mask': mask1, 'outer_mask': mask2}

		# Use thresholds to check the black-to-white transition
		is_inner_white = self._white_thresh < white_mean
		is_outer_black = self._black_thresh > black_mean

		if is_inner_white and is_outer_black: 
			return (inner, outer), '', views
		else:
			return None, f'No black to white transition between squares. (white mean: {white_mean:.2f}, black mean: {black_mean:.2f}) ', views
	
	def _detect_anchor(
			self,
			frame           : Frame,
			contours        : Contours,
			marker_vertices : Tuple[SortedVertices, SortedVertices],
		) -> Tuple[Tuple[int, Contour] | None, str, Views]:
		'''
		It detects the anchor point within the marker contours.
		1. It check if the candidate anchor contour is a circle using the circularity threshold.
		2. Given the inner and outer marker squares, it checks if the anchor contour is between exactly one
			of two corresponding vertices of the inner and outer marker squares.

					:param frame: The binary frame to detect the anchor point on.
		:param contours: The contours detected in the frame as candidate anchor points.
		:param marker_vertices: The sorted vertices of the detected inner and outer marker squares.
		:return: It returns a triple of
			- A tuple with the index of the corner where the anchor is found and the anchor contour itself.
			- The warning message if the anchor is not found.
			- The views dictionary with the mask of the anchor detection.
		'''

		def is_anchor_between_marker_corners(anchor_contour: Contour, inner_corner: Point2D, outer_corner: Point2D) -> Tuple[bool, Frame]:
			'''
			It checks if a candidate anchor contour is between two corresponding vertices of the inner and outer marker squares. 
			It creates two masks: one for the anchor contour and one for the line between the two points. If the two overlap, the anchor is between the points.

			:param anchor_contour: The candidate anchor contour to check.
			:param inner_corner: The inner corner vertex.
			:param outer_corner: The outer corner vertex, corresponding to the inner corner.
			:return: A tuple with a boolean indicating if the anchor is between the two points and the mask of the overlapping area.
			'''

			color: RGBColor = (1, ) # type: ignore

			# Empty masks
			mask1 = np.zeros_like(a=frame, dtype=np.uint8)
			mask2 = mask1.copy()

			anchor_contour.draw(frame=mask1, color=color, fill=True)										   # First mask:  anchor contour
			Point2D.draw_line(frame=mask2, point1=inner_corner, point2=outer_corner, color=color, thickness=3) # Second mask: line between the two points

			overlap = 2 in mask1 + mask2		  # Check if the two masks overlap
			overlap_mask = (mask1 | mask2) * 255  # Create a unique view of the overlapping area
			
			return overlap, overlap_mask

		# Tuple of candidates contours (vertex index, circle contour)
		marker_circles: List[Tuple[int, Contour]] = []

		# Mask to show every anchor found
		out_mask: Frame = np.zeros_like(a=frame, dtype=np.uint8)

		for contour in contours:

			if not contour.is_circle(thresh=self._circularity_thresh): continue  # Skip if contour is not a circle

			# NOTE: To check if the contour is white is not necessary as the anchor is always white if found in a binary image inside the black square
			# anchor_mean, anchor_mask = contour.mean_value_on_frame(frame=frame, fill=True)
			# if anchor_mean < self._white_thresh: continue  # Skip if the circle is not white

			# For the 4 couples of corresponding corners check if the circle is between the two points
			inner_vert, outer_vert = marker_vertices

			for corner_id, (inner_corner, outer_corner) in enumerate(zip(inner_vert.vertices, outer_vert.vertices)):

				# Check if the anchor is between the two points
				is_anchor, anchor_mask = is_anchor_between_marker_corners(
					anchor_contour=contour, 
					inner_corner=Point2D.from_tuple(inner_corner), 
					outer_corner=Point2D.from_tuple(outer_corner)
				)
				
				# If the anchor is between the two points, 
				# add it to the list of marker circles
				if is_anchor:
					marker_circles.append((corner_id, contour))
					out_mask: Frame = anchor_mask | out_mask # Update the mask with the anchor found # type: ignore
		
		anchor_view = {'anchor_mask': out_mask}

		# If not a single anchor is found, return None
		if len(marker_circles) == 0: return None, f'No anchor found within the marker. ',								anchor_view
		if len(marker_circles) >  1: return None, f'Found multiple anchors within the marker ({len(marker_circles)}). ', anchor_view

		# Return the single anchor found
		return marker_circles[0], '', anchor_view
	
	def __call__(self, frame: Frame) -> Tuple[Marker | None, str, Views]:
		'''
		Detect the marker in the frame. First it detects the marker squares and then the anchor point.
		If the marker is detected, it returns the marker object, otherwise it returns None.

		:param frame: The binary frame to detect the marker on.
		:return: A triple of
			- The detected marker if found, None otherwise.
			- The warning message if the marker is not detected, empty string otherwise.
			- The views dictionary with the intermediate processing steps.
		'''

		# Create empty views if intermediate processing steps do not succeed
		empty_frame = np.zeros_like(frame)
		empty_views = {view: empty_frame for view in ['inner_mask', 'outer_mask', 'anchor_mask']}

		# Create copies of the frame to draw contours
		views = {}
		frame_c = cv.cvtColor(frame.copy(), cv.COLOR_GRAY2RGB)
		frame_contours_orig = frame_c.copy()
		frame_contours_adj  = frame_c.copy()

		# Detect contours
		contours = Contours(frame=frame, min_area=self._min_contour_area, max_area=self._max_contour_area)
	
		# Draw original and adjusted detected contours
		palette = generate_palette(n=len(contours))
		contours.draw(frame=frame_contours_orig, colors=palette, thickness=10, adjusted=False)
		contours.draw(frame=frame_contours_adj,  colors=palette, thickness=10, adjusted=True)
		views['contours_orig'] = frame_contours_orig
		views['contours_adj' ] = frame_contours_adj

		# 1. Detect marker corners
		marker_corners, warn_message, corner_views = self._detect_marker_squares(frame=frame, contours=contours)

		# Failed to detect marker corners
		if marker_corners is None: return None, warn_message, views | empty_views | corner_views

		# Sort marker inner and outer corners and align them
		inner_marker_contour, outer_marker_contour = marker_corners
		inner_marker_vertices = inner_marker_contour.to_sorted_vertex()
		outer_marker_vertices = outer_marker_contour.to_sorted_vertex()
		inner_marker_vertices.align_to(other=outer_marker_vertices)	 # NOTE: This is needed for certain angles where the two squares are not aligned

		# 2. Detect anchor
		anchor_detection, warning_message, anchor_views = self._detect_anchor(
			frame=frame,
			contours=contours, 
			marker_vertices=(inner_marker_vertices, outer_marker_vertices), 
		)

		# Failed to detect the anchor
		if anchor_detection is None: return None, warning_message, views | empty_views | corner_views | anchor_views

		# Extract the corner-id where the anchor is found and the anchor contour
		corner_id, circle = anchor_detection

		# Roll the inner marker vertices to align the anchor with the first corner
		inner_marker_vertices.roll(n=corner_id)

		# Create the marker object
		marker = Marker.from_detection(
			marker_vertices=inner_marker_vertices,
			anchor_contour=circle
		)

		return marker, '', views | empty_views | corner_views | anchor_views

# _____________________________________________________________

class MarkerDetectionVideoStream(ThresholdedVideoStream):
	'''
	Video stream to detect the marker in each frame using the marker detector.
	'''

	def __init__(
		self, 
		path			: str, 
		calibration	 : CalibratedCamera,
		thresholding	: Thresholding,
		marker_detector : MarkerDetector,
		name			: str | None = None,
		logger		  : BaseLogger = SilentLogger()
	):
		'''
		The class requires the thresholding method and the marker detector to apply to each frame.
		'''

		super().__init__(
			path=path, 
			calibration=calibration, 
			thresholding=thresholding,
			name=name, 
			logger=logger
		)

		self._marker_detector = marker_detector

		# We save the number of frames successfully processed and the total number of frames
		self._success: int = -1
		self._total  : int = -1

	@property
	def marker_detection_results(self) -> Tuple[int, int]: 

		if self._success == -1 or self._total == -1: raise ValueError(f'No results available. ')

		return self._success, self._total

	def play(
		self, 
		start		: int							   = 0,
		end		  : int						| None = None, 
		skip_frames  : int							   = 1,
		window_size  : Dict[str, Size2D] | Size2D | None = None,
		exclude_views: List[str]						 = [],
		delay		: int							   = 1
	):
		
		# Reset the number of frames successfully processed and the total number of frames
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

		# Log the results of the marker detection
		success, total = self.marker_detection_results
		if success == total: info_msg = f'All frames were successfully processed. '
		else:			 	 info_msg = f'Processed {success} out of {total} frames. ({success / total:.2%}) '
		self._logger.info(msg=info_msg)


	def _process_marker(self, views: Views, marker: Marker, frame_id: int) -> Views:
		'''
		The logic of marker processing is separated from the frame processing.
		This allow for more custom operation in subclasses.

		NOTE: For instance, this method is used in the to warp the marker and estimate the camera pose.
		'''

		return {'marker': marker.draw(frame=views['undistorted'].copy())}

	def _process_frame(self, frame: Frame, frame_id: int) -> Views:
		'''
		Process the frame to detect the marker.
		'''

		# Check if the frame is in debug mode
		debugging = self._is_debug(frame_id=frame_id)
		if not debugging: self._total += 1
	
		views = super()._process_frame(frame=frame, frame_id=frame_id)

		# Detect marker on the binary frame
		binary_frame = views['binary']
		marker, warning, marker_views = self._marker_detector(frame=binary_frame)

		# If the marker is not detected, return the views and log the warning message
		if marker is None:
			if not debugging: self._logger.warning(f'[{self.name}] Unable to process frame {frame_id} - {warning}')
			return views | marker_views | {'marker': views['undistorted']}
		
		# Otherwise process the marker and return the views
		if not debugging: self._success += 1
		marker_processed_views = self._process_marker(views=views, marker=marker, frame_id=frame_id)
		
		return views | marker_views | marker_processed_views