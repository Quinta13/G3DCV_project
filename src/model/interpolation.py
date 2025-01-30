'''
This module contains classes to interpolate a basis from a set of coordinates and values.
It provides two main classes:
- The Basis class to represent and manipulate a basis tensor, representing a precomputed discretized function;
- The Interpolators classes to interpolate a basis from a set of coordinates and values. Two interpolators are implemented:
	1. The Radial Basis Interpolator;
	2. The Polynomial Texture Map Interpolator.
'''

import itertools
import pickle
from abc import ABC, abstractmethod
from functools import partial
from statistics import mean
from typing import Sequence, Tuple, Callable, Type, cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.interpolate import Rbf

from src.model.geom import LightDirection
from src.model.typing import Shape, Frame, Pixel, Size2D
from src.model.mlic import MultiLightImageCollection
from src.utils.io_ import BaseLogger, PathUtils, SilentLogger, InputSanitizationUtils as ISUtils, Timer

# __________________________________ BASIS __________________________________

class Basis:
	''' 
	Class to represent a basis: a discretized representation of an arbitrary function.
	The basis is represented as a tensor of arbitrary dimension, 
		with each dimension associated to a specific range of coordinates.

	The class provides methods to access the basis values by mapping the coordinates to the basis indices.
	'''
    
	def __init__(
		self, 
		basis: NDArray,
		coords_range : Tuple[Tuple[float, float], ...],
		values_range : Tuple[float, float]
	):
		'''
		Initialize the basis with the basis tensor of M dimensions,
			the range of coordinates and the range of values.
		The constructor performs some sanity checks on the basis tensor 
			to ensure that the values are within the specified range.
		
		:param basis: The basis tensor with M dimensions.
		:param coords_range: A tuple of size M, associating each dimension with the minimum and maximum value of the coordinates.
		:param values_range: The minimum and maximum value of the basis.
		'''
		
		self._basis : NDArray = basis

		# Check if basis is within the range of values
		min_val, max_val = values_range
		if not np.all((min_val <= self._basis) & (self._basis <= max_val)):
			raise ValueError(f'The minimum value of the basis is not within the minimum and maximum values of the basis [{min_val}, {max_val}]. ')
		
		self._values_range: Tuple[float, float] = values_range
		
		# Check if the number of dimensions match the length of the coordinates range
		if not (self.n_dims == len(coords_range)):
			raise ValueError(f'The number of basis dimensions ({self.n_dims}) don\'t match the length of min and max values {len(coords_range)}. ')
		
		# Check if each dimension has a valid range
		for i, (min_val, max_val) in enumerate(coords_range):
			if not (min_val < max_val): raise ValueError(f'The {i+1} coordinate range [{min_val}, {max_val}] is not valid. ')
		
		self._coords_range: Tuple[Tuple[float, float], ...] = coords_range
	
	# --- MAGIC METHODS ---

	def __str__ (self): return f'{self.__class__.__name__}[shape: {"x".join([str(i) for i in self.shape])}]'
	def __repr__(self): return str(self)

	def __getitem__(self, coord: Tuple[float, ...]) -> float:
		''' Get the basis value at the specified coordinate. '''

		idx = self.discretize_coordinate(coord=coord, min_max_coords=self.coords_range, shape=self.shape)
		return self._basis[idx]
	
	# --- PROPERTIES ---

	@property
	def basis(self) -> NDArray: return self._basis
	
	@property
	def coords_range(self): return self._coords_range

	@property
	def values_range(self): return self._values_range

	@property
	def n_dims(self): return len(self.shape)

	@property
	def shape(self): return self._basis.shape

	# --- COORDINATE MAPPING  ---

	'''
	NOTE:   The mapping and discretization methods from coordinate system to basis indexes 
			are static to allow the user to use them without instantiating the class.
			The class implements corresponding methods as STUB using proper class attributes.
	'''

	@staticmethod
	def map_coordinate(
		coord        : Tuple[float, ...],
		coords_range : Tuple[Tuple[float, float], ...],
		shape        : Shape
	) -> Tuple[float, ...]:
		'''
		Map a point from the coordinate space to the basis space given the shape of the basis and the range of its coordinates.
		NOTE: The mapping maps to the basis space, but not to the basis indices.

		:param coord: The coordinate to map.
		:param coords_range: The range of the coordinates of each dimension in the basis.
		:param shape: The shape of the basis tensor.
		:return: The mapped coordinate in the basis space as continuous values.
		'''

		out_idx = []

		# Perform the mapping for each coordinate and dimension
		for i, (c, (min_val, max_val), dim) in enumerate(zip(coord, coords_range, shape)):

			# Check consistency between coordinate to map and the coordinate range
			if not (min_val <= c <= max_val): raise ValueError( f'The {i+1} coordinate {c} is not within the minimum '
																f'and maximum values of the basis [{min_val}, {max_val}]. ')
			
			# Map the coordinate to the basis space
			step = (max_val - min_val) / (dim-1) 
			idx = (c - min_val) / step

			out_idx.append(idx)

		return tuple(out_idx)
	
	@staticmethod
	def discretize_coordinate(
		coord		  : Tuple[float, ...],
		min_max_coords: Tuple[Tuple[float, float], ...],
		shape         : Shape
	) -> Tuple[int, ...]:
		'''
		Map a coordinate from the coordinate space to the basis indices given the shape of the basis and the range of its coordinates.

		:param coord: The coordinate to map.
		:param coords_range: The range of the coordinates of each dimension in the basis.
		:param shape: The shape of the basis tensor.
		:return: The mapped coordinate in the basis indices.
		'''
		
		# 1. Perform mapping
		mapped_coord = Basis.map_coordinate(
			coord=coord, 
			coords_range=min_max_coords, 
			shape=shape
		)

		# 2. Clip the mapped coordinate to the basis indices
		out_idx = []
		for c, dim in zip(mapped_coord, shape):
			idx = int(np.clip(c, 0, dim-1))
			out_idx.append(idx)
			
		return tuple(out_idx)

	def _map_coordinate(self, coord: Tuple[float, ...]) -> Tuple[float, ...]: 
		return self.map_coordinate(coord=coord, coords_range=self.coords_range, shape=self.shape)
	
	def _discretize_coordinate(self, coord: Tuple[float, ...]) -> Tuple[int,   ...]: 
		return self.discretize_coordinate(coord=coord, min_max_coords=self.coords_range, shape=self.shape)

	# --- PLOTTING ---
	
	def plot_interpolation(
		self,
		min_max_colorbar : bool = False,
		points           : Tuple[NDArray, NDArray] | None = None,
		ax			     : Axes | None = None
	):
		'''
		Plot a 2D basis matrix as an image with the option to add scatter points with values.

		:param min_max_colorbar: If True, the colorbar will be set to the minimum and maximum values of the basis (actual values it takes),
			otherwise, it will be set to the minimum and maximum range of the values (values it can take).
		:param points: A tuple of two arrays: the first array contains the values of the points, the second array contains the coordinates of the points.
			If not given, no points will be plotted.
		:param ax: The axis to plot the basis. If None, a new figure will be created.
		'''
		
		TICK_STEP = 5

		if self.n_dims != 2: raise ValueError(f'The basis must be 2D to plot. ')

		if ax is None: _, ax = plt.subplots(figsize=(12, 10))

		ax.set_xlabel('U Light')
		ax.set_ylabel('V Light')

		(min_x, max_x), (min_y, max_y) = self.coords_range
		dim_x, dim_y = self.shape

		# Set the axis limits and ticks
		ax.set_xlim(0, dim_x-1); 
		ax.set_xticks(np.linspace(0, dim_x-1, TICK_STEP)); 
		ax.set_xticklabels(-np.linspace(min_x, max_x, TICK_STEP))
		ax.set_ylim(0, dim_y-1); 
		ax.set_yticks(np.linspace(0, dim_y-1, TICK_STEP)); 
		ax.set_yticklabels(-np.linspace(min_y, max_y, TICK_STEP))

		# Set the colorbar limits
		if min_max_colorbar: vmin, vmax = self.values_range
		else:                vmin, vmax = self.basis.min(), self.basis.max()

		# Plot points on the basis if given
		if points is not None:
			
			values, coords = points

			# Move the coordinates to the pixel space
			coords_ = np.array([self._map_coordinate((u, v)) for (u, v) in coords])

			if not min_max_colorbar:
				vmin = min(vmin, values.min())
				vmax = max(vmax, values.max())

			# Swap the points to match matplotlib coordinates			
			ax.scatter(
				coords_[:, 0], coords_[:, 1],
				c=values,       # Values for coloring
				cmap='viridis',       # Use the same colormap
				edgecolor='k',        # Optional: edge for better visibility
				s=25,                 # Size of scatter points
				linewidths=0.5,       # Edge width for scatter points
				vmin=vmin, vmax=vmax, # Set the colorbar limits
			)
		
		# Plot the basis as an image
		img = ax.imshow(
			self.basis,
			cmap='viridis', 
			vmin=vmin, 
			vmax=vmax,
			aspect='auto',  # Allow the image to stretch to fill the axis
			origin='lower'  # Set the origin to the lower left corner
		)

		ax.figure.colorbar(img, ax=ax, label='Pixel Luminance')  # type: ignore - figure has attribute colorbar

class BasisCollection:
	'''
	Represents a collection of basis, each with the same shape and range of coordinates and values.
	The single Basis are not stored as single objects but stacked in a NumPy tensor for faster access.

	The object can be pickled and unpickled to save and load the basis collection.
	'''
    
	def __init__(self, basis_collection: Sequence[Basis]):

		# Check all have the same min and max coordinates
		self._coords_range = basis_collection[0].coords_range
		if not all(bi.coords_range == self._coords_range for bi in basis_collection):
			raise ValueError('All basis must have the same minimum and maximum coordinates. ')
		
		# Check all have the same min and max values
		self._values_range = basis_collection[0].values_range
		if not all(bi.values_range == self._values_range for bi in basis_collection):
			raise ValueError('All basis  must have the same minimum and maximum values. ')
		
		# Stack all basis 
		# NOTE Raises an error if the shapes are not the same
		self._basis_collection: NDArray = np.stack([bi.basis for bi in basis_collection])

	# --- MAGIC METHODS ---

	def __str__ (self): return f'{self.__class__.__name__}['\
		f'basis: {len(self)}; '\
		f'basis shape: {"x".join([str(i) for i in self.basis_shape])}]'
	
	def __repr__(self): return str(self)

	def __len__(self): return self._basis_collection.shape[0]

	def __getitem__(self, index: int) -> Basis: 
		''' Instantiate a Basis object corresponding to the index in the stacked tensor basis. '''
		
		return Basis(basis=self._basis_collection[index], coords_range=self._coords_range, values_range=self._values_range)
	
	# --- PROPERTIES ---

	@property
	def basis_shape(self) -> Shape:
		n_object, *shape = self._basis_collection.shape
		return tuple(shape)
	
	# --- METHODS ---

	def discretize_coordinate(self, coord: Tuple[float, ...]) -> Tuple[int, ...]:
		''' Discretize a coordinate to a basis index. '''
		
		return Basis.discretize_coordinate(coord=coord, min_max_coords=self._coords_range, shape=self.basis_shape)

	def get_vector(self, coord: Tuple[float, ...]) -> NDArray:
		''' Given a coordinate, stack in a vector all the basis values at that coordinate. '''
		
		idx = self.discretize_coordinate(coord=coord)
		vector = self._basis_collection[:, *idx[::-1]].astype('uint8')  # NOTE: Invert the index to match the column-major order
	
		return vector
	
	# --- PICKLE ---

	@classmethod
	def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'BasisCollection':
		''' Load the Basis Collection from a pickle object. '''

		ISUtils.check_input(path=path, logger=logger)

		logger.info(msg=f"Loading basis collection from {path}")
		timer = Timer()
		with open(path, 'rb') as f: data = pickle.load(f)
		logger.info(msg=f"Completed in {timer}. ")

		return data

	def dump(
        self,
        path   : str,
        logger : BaseLogger = SilentLogger()
    ) -> None:
		''' Save the Basis Collection to a pickle file. '''

		ISUtils.check_output(path=PathUtils.get_folder_path(path=path), logger=logger)

		logger.info(msg=f"Saving basis collection to {path} ...")

		timer = Timer()
		with open(path, 'wb') as f: pickle.dump(self, f)
		logger.info(msg=f"Completed in {timer}. ")


class MLICPixelsBasisCollection(BasisCollection):
	'''
	Class to represent a collection of basis for a Multi-Light Image Collection:
	- each basis represent a pixel luminance;
	- each basis is 2D, representing the light direction.

	The method allows to reconstruct the frame given a light direction given an output shape.

	It also allows to compute the mean squared of the basis with a ground truth MLIC object.
	'''

	def __init__(self, basis_collection: Sequence[Basis], out_shape: Shape):
		'''
		Perform sanity checks on the basis collection and the output shape.
		The class requires the output shape to reconstruct the frame.
		'''

		if not all(bi.n_dims == 2 for bi in basis_collection):
			raise ValueError('All basis must be 2D, representing the light direction. ')
		
		if not len(out_shape) == 2:
			raise ValueError('The output shape must be 2D, to reconstruct the object frame. ')

		super().__init__(basis_collection=basis_collection)

		self._out_shape = out_shape

	# --- MAGIC METHODS ---

	def __str__(self) -> str:
		
		super_str = super().__str__()[:-1]
		w, h = self._out_shape

		return f'{super_str}; out shape: {w}x{h}]'\
		
	# --- PROPERTIES ---
	
	@property
	def out_shape(self) -> Shape: return self._out_shape

	# --- METHODS ---

	@classmethod
	def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'MLICPixelsBasisCollection':
		return cast('MLICPixelsBasisCollection', super().from_pickle(path=path, logger=logger))

	def get_frame(self, light_direction: LightDirection) -> Frame:
		''' Reconstruct the frame by reshaping the basis vector for a given light direction. '''

		vector = super().get_vector(coord=tuple(light_direction))
		frame  = vector.reshape(self.out_shape, order='F')

		return frame
	
	def mse_error(self, mlic: MultiLightImageCollection) -> float:
		''' Compute the MSE error between the basis collection and a ground truth MLIC object. '''

		errors = []

		# Loop on the MLIC objects
		for true_frame, light_direction in mlic:
			
			# Get the predicted frame
			predicted_frame = self.get_frame(light_direction=light_direction)

			# Compute the per pixel-squared error
			frame_pixel_distance = (predicted_frame - true_frame)**2

			# Append the frame mean pixel error
			errors.append(frame_pixel_distance.mean())

		# Compute the mean squared error across objects
		mse = mean(errors)

		return float(mse)

# __________________________________ INTERPOLATORS __________________________________

class BasisInterpolator(ABC):
	'''
	Abstract class to interpolate a basis from a set of coordinates and values.

	The coordinates, their range and values are attributes of the class, while the specific values are passed as of the interpolation function.
	NOTE: This strategy is the most suitable for the MLIC context, where light direction coordinates are fixed for each pixel and what changes is the pixel luminance.

	The subclasses must implement the fit logic to interpolate the basis.
	'''
    
	def __init__(
		self, 
		coords             : NDArray,
		interpolation_size : Shape,
		range_coordinates  : Tuple[Tuple[float, float], ...],
		range_values       : Tuple[float, float]
	):
		'''
		Initialize the interpolator with its coordinates, their range and their values.
		The interpolation size is the size of the basis tensor to interpolate.

		:param coords: Coordinates of points to interpolate.
		:param interpolation_size: The size of the basis tensor to interpolate.
		:param range_coordinates: The range of the coordinates of each dimension in the basis.
			The number of coordinate must match the dimensions of `coords` and `interpolation_size`.
		:param range_values: The minimum and maximum value of the basis.
		'''
	
		self._len, self._dims = coords.shape

		if self._dims != len(range_coordinates) != len(interpolation_size): 
			raise ValueError(
				f'The number of coordinates dimension ({self._dims}), coordinates ranges ({len(range_coordinates)}) '
				f'and interpolation size ({len(interpolation_size)}) must be the same. '
			)

		self._coords             : NDArray                         = coords
		self._interpolation_size : Shape                           = interpolation_size
		self._range_coords       : Tuple[Tuple[float, float], ...] = range_coordinates
		self._range_values       : Tuple[float, float]             = range_values

		# Precompute interpolation grid - NOTE: This is to save time when interpolating
		# The interpolation grid is a meshgrid of the coordinates to interpolate 
		# based on the range of the coordinates and the interpolation size
		self._interpolation_grid = np.meshgrid(*[
			np.linspace(min_val, max_val, dim) 
			for (min_val, max_val), dim in zip(self._range_coords, self._interpolation_size)
		])

	# --- MAGIC METHODS ---
	
	def __str__(self) -> str : return f'{self.__class__.__name__}['\
		f'points: {len(self)}; '\
		f'dimensions: {self.dims}; '\
		f'interpolation size: {"x".join([str(i) for i in self._interpolation_size])}]'
	
	def __repr__(self): return str(self)

	def __len__(self) -> int: return self._len
	''' Number of points to interpolate. '''

	# --- PROPERTIES ---

	@property
	def dims(self) -> int: return self._dims

	# --- INTERPOLATION ---

	@abstractmethod
	def _fit_interpolation_function_on_values(self, values: NDArray) -> Callable[[NDArray], NDArray]: pass
	''' 
	The abstract method requires to fit an interpolation function on the values associated to the coordinates. 
	It must return a function mapping the interpolation grid to the interpolated basis.
	'''

	def __call__(self, values: NDArray) -> Basis:
		'''
		Return the interpolated basis given the values associated to the coordinates.
		It uses the interpolation function fitted on the values to interpolate the basis.
		'''

		# Get interpolation function
		interpolation_function = self._fit_interpolation_function_on_values(values=values)

		# Interpolate
		basis = interpolation_function(*self._interpolation_grid)

		# Clip the basis to the range of values
		basis = np.clip(basis, *self._range_values)

		return Basis(
			basis=basis, 
			coords_range=self._range_coords,
			values_range=self._range_values
		)

class MLICBasisInterpolator(BasisInterpolator):
	'''
	Basis interpolator for a Multi-Light Image Collection, where
	
	- the basis is 2D, representing the light direction in [-1, 1]x[-1, 1];
	- the values represent pixel luminance in [0, 255].
	'''

	COORD_RANGE  = (-1,   1)
	VALUES_RANGE = ( 0, 255)

	def __init__(
		self,
		coordinates : NDArray,
		interpolation_size: Size2D
	):
		super().__init__(
			coords=coordinates,
			interpolation_size=interpolation_size,
			range_coordinates=(self.COORD_RANGE, self.COORD_RANGE),
			range_values=self.VALUES_RANGE
		)

class MLICRadialBasisInterpolator(MLICBasisInterpolator):
	''' Uses Radial Basis Function interpolation to interpolate the basis. '''
	
	def __init__(self, coordinates: NDArray, interpolation_size: Size2D):
		super().__init__(coordinates=coordinates, interpolation_size=interpolation_size)

	def _fit_interpolation_function_on_values(self, values: NDArray):
		return Rbf(*self._coords.T, values, function='linear', smooth=0.1)

class RTIPolynomialTextureMapInterpolator(MLICBasisInterpolator):
	''' Uses Polynomial Texture Maps to interpolate the basis. '''

	def __init__(self, coordinates: NDArray, interpolation_size: Size2D):
		super().__init__(coordinates=coordinates, interpolation_size=interpolation_size)

	def _fit_interpolation_function_on_values(self, values: NDArray):

		# Extract light coordinates (u, v)
		u, v = self._coords.T

		# Construct the matrix A for polynomial regression
		# Given the light l = Lx, Ly
		# I(x, y, l) = a0 + a1*Lx + a2*Ly + a3*Lx^2 + a4*Ly^2 + a5*Lx*Ly
		A = np.column_stack([
			np.ones_like(u),  # a0
			u,                # a1
			v,                # a2
			u**2,             # a3
			v**2,             # a4
			u * v             # a5
		])

		# Solve the linear system to get the coefficients
		coefficients, _, _, _ = np.linalg.lstsq(A, values, rcond=None)

		# Create the interpolation function using the coefficients
		def polynomial_texture_map(u: NDArray, v: NDArray, coefficients: NDArray) -> NDArray:
			''' Compute the PTM interpolation given light direction and coefficients. '''

			return (
				coefficients[0] * 1     +
				coefficients[1] * u     +
				coefficients[2] * v     +
				coefficients[3] * u**2  +
				coefficients[4] * v**2  +
				coefficients[5] * u * v
			)
        
		# Return the interpolation function with fixed coefficients computed in the linear system
		return partial(polynomial_texture_map, coefficients=coefficients)

class MLICBasisCollectionInterpolator:
	'''
	Class representing an interpolator from a Multi-Light Image Collection to a collection of basis, one per pixel.
	'''
    
	def __init__(
		self,
		mlic: MultiLightImageCollection,
		C_rti_interpolator: Type[MLICBasisInterpolator],
		interpolation_size: Size2D,
		logger: BaseLogger = SilentLogger()
	):
		'''
		The class requires the Multi-Light Image Collection, the interpolator class and the interpolation size.

		:param mlic: The Multi-Light Image Collection to interpolate.
		:param C_rti_interpolator: The class of the interpolator to use.
		:param interpolation_size: The size of the basis tensor to interpolate.
		:param logger: The logger to use for logging. Default is a silent logger.
		'''
		
		self._logger             : BaseLogger = logger
		self._mlic               : MultiLightImageCollection = mlic
		self._interpolation_size : Size2D           = interpolation_size
		self._rti_interpolator   : MLICBasisInterpolator     = C_rti_interpolator(coordinates=self._mlic.light_directions, interpolation_size=self._interpolation_size)
	
	# --- MAGIC METHODS ---
	
	def __str__(self) -> str:
		
		sx, sy = self._interpolation_size
		mx, my = self._mlic.size
		
		return f'{self.__class__.__name__}['\
            f'interpolation size: {sx}x{sy}; '\
            f'MLIC objects: {len(self._mlic)}; '\
			f'MLIC size: {mx}x{my}; '\
            f'interpolator: {self._rti_interpolator.__class__.__name__}]'
	
	def __repr__(self) -> str: return str(self)

	# --- METHODS ---

	def get_pixel_interpolation(self, pixel: Pixel) -> Basis:
		''' Get the interpolated basis for a specific pixel. '''

		return self._rti_interpolator(values=self._mlic.get_pixel_values(pixel=pixel))
	
	def get_interpolation_collection(self, progress: int | None = None) -> MLICPixelsBasisCollection:
		'''
		Interpolate all the pixels in the Multi-Light Image Collection and return a collection of basis, one per pixel.
		The method logs the progress every `progress` pixels, if given.
		'''

		rows, cols = self._mlic.size
		
		tot = rows * cols
		self._logger.info(msg=f'Starting interpolation for all pixels ({tot}). ')

		interpolated_basis = []

		timer = Timer()

		for i, (px, py) in enumerate(itertools.product(range(rows), range(cols))):

			# Log progress
			if progress and i % progress == 0:
				self._logger.info(msg=f' > Interpolating pixel {i} of {tot} ({i/tot:.2%}) - Elapsed time: {timer}. ')
			
			# Interpolate the pixel basis
			bi = self.get_pixel_interpolation(pixel=(px, py))
			interpolated_basis.append(bi)
		
		self._logger.info(msg=f'Interpolation completed in {timer}. \n')

		self._logger.info("Creating basis collection ...")
		timer.reset()
		mlic_basis_collection = MLICPixelsBasisCollection(basis_collection=interpolated_basis, out_shape=self._mlic.size)
		self._logger.info(msg=f"Completed in {timer}. ")
		self._logger.info(msg=f'{mlic_basis_collection}. \n')

		return mlic_basis_collection