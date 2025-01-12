import itertools
import pickle
import os
from typing import Tuple, List, Callable, Type
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.figure import Figure
from scipy.interpolate import Rbf

from src.model.typing import Shape
from src.model.mlic import MLIC
from src.utils.misc import default
from src.utils.io_ import BaseLogger, PathUtils, SilentLogger, InputSanitizationUtils as ISUtils, Timer

class BasisInterpolation:
    
	def __init__(
		self, 
		basis: NDArray,
		min_max_coors : Tuple[Tuple[float, float], ...],
		min_max_values: Tuple[float, float] | None = None
	):
		
		self._basis : NDArray = basis

		min_val, max_val = default(min_max_values, (float(basis.min()), float(basis.max())))

		# Check if basis is withing mix and max values
		if not  np.all((min_val <= self._basis) & (self._basis <= max_val)):
			raise ValueError(f'The minimum value of the basis is not within the minimum and maximum values of the basis. ')
		
		self._min_max_values: Tuple[float, float] = min_val, max_val
		
		if not (self.n_dims == len(min_max_coors)):
			raise ValueError(f'The number of basis dimensions ({self.n_dims}) don\'t match the length of min and max values {len(min_max_coors)}. ')
		
		self._min_max_coords: Tuple[Tuple[float, float], ...] = min_max_coors

	@property
	def basis(self) -> NDArray: return self._basis

	def __str__(self): return f'{self.__class__.__name__}[shape: {"x".join([str(i) for i in self.shape])}]'
	def __repr__(self): return str(self)

	def _map_coordinate       (self, coord: Tuple[float, ...]) -> Tuple[float, ...]: return self.map_coordinate       (coord=coord, min_max_coords=self.min_max_coords, shape=self.shape)
	def _discretize_coordinate(self, coord: Tuple[float, ...]) -> Tuple[int,   ...]: return self.discretize_coordinate(coord=coord, min_max_coords=self.min_max_coords, shape=self.shape)

	def __getitem__(self, coord: Tuple[float, ...]) -> float:

		idx = self.discretize_coordinate(coord=coord, min_max_coords=self.min_max_coords, shape=self.shape)

		return self._basis[idx]
	
	@property
	def shape(self): return self._basis.shape

	@property
	def n_dims(self): return len(self.shape)

	@property
	def min_max_coords(self): return self._min_max_coords

	@property
	def min_max_values(self): return self._min_max_values

	@staticmethod
	def map_coordinate(
		coord          : Tuple[float, ...],
		min_max_coords : Tuple[Tuple[float, float], ...],
		shape          : Shape
	) -> Tuple[float, ...]:

		out_idx = []

		for i, (c, (min_val, max_val), dim) in enumerate(zip(coord, min_max_coords, shape)):

			if not (min_val <= c <= max_val): raise ValueError(f'The {i+1} coordinate {c} is not within the minimum and maximum values of the basis [{min_val}, {max_val}]. ')
			
			step = (max_val - min_val) / (dim - 1)
			idx = (c - min_val) / step
			out_idx.append(idx)
		
		return tuple(out_idx)
	
	@staticmethod
	def discretize_coordinate(
		coord		  : Tuple[float, ...],
		min_max_coords: Tuple[Tuple[float, float], ...],
		shape         : Shape
	) -> Tuple[int, ...]:
		
		return tuple(map(int, BasisInterpolation.map_coordinate(coord, min_max_coords, shape)))
	
	def plot_interpolation(
		self,
		title: str = 'Basis Interpolation',
		min_max_colorbar: bool = False,
		points_coord: Tuple[NDArray, NDArray] | None = None
	) -> Figure:
		
		if self.n_dims != 2: raise ValueError(f'The basis must be 2D to plot. ')

		fig, ax = plt.subplots(figsize=(8, 6))

		ax.set_title(title, fontsize=14)
		ax.set_xlabel('U Light')
		ax.set_ylabel('V Light')

		(min_x, max_x), (min_y, max_y) = self.min_max_coords

		STEP = 5
		dim_x, dim_y = self.shape
		ax.set_xlim(0, dim_x-1); ax.set_xticks(np.linspace(0, dim_x-1, STEP)); ax.set_xticklabels(np.linspace(min_x, max_x, STEP))
		ax.set_ylim(0, dim_y-1); ax.set_yticks(np.linspace(0, dim_y-1, STEP)); ax.set_yticklabels(np.linspace(min_y, max_y, STEP))

		if min_max_colorbar:
			vmin, vmax = self.min_max_values
		else:
			vmin, vmax = self.basis.min(), self.basis.max()
	

		if points_coord is not None:
			
			values, coords = points_coord

			# Move the coordinates to the pixel space
			coords_ = np.array([self._map_coordinate(c) for c in coords])

			if not min_max_colorbar:
				vmin = min(vmin, values.min())
				vmax = max(vmax, values.max())
			
			ax.scatter(
				*coords_.T,        # Coordinates for scatter points
				c=values,       # Values for coloring
				cmap='viridis', # Use the same colormap
				edgecolor='k',  # Optional: edge for better visibility
				s=25,           # Size of scatter points
				linewidths=0.5,  # Edge width for scatter points
				vmin=vmin, 
				vmax=vmax
			)
		
		img = ax.imshow(self.basis, cmap='viridis', vmin=vmin, vmax=vmax)  # type: ignore
		fig.colorbar(img, ax=ax, label='Pixel Luminance')
		
		return fig
	
class BasisInterpolationCollection:
    
	def __init__(self, basis_interpolations: List[BasisInterpolation], out_shape: Shape | None = None):

		# Check all have the same min and max coordinates
		self._min_max_coords = basis_interpolations[0].min_max_coords
		if not all(bi.min_max_coords == self._min_max_coords for bi in basis_interpolations):
			raise ValueError('All basis interpolations must have the same minimum and maximum coordinates. ')
		
		# Check all have the same min and max values
		self._min_max_values = basis_interpolations[0].min_max_values
		if not all(bi.min_max_values == self._min_max_values for bi in basis_interpolations):
			raise ValueError('All basis interpolations must have the same minimum and maximum values. ')
		
		# Stack all basis interpolations
		# NOTE Raises an error if the shapes are not the same
		self._basis_interpolations: NDArray = np.stack([bi.basis for bi in basis_interpolations])

		self._out_shape: Shape | None = out_shape

	def __str__ (self): return f'{self.__class__.__name__}[basis: {len(self)}; shape: {"x".join([str(i) for i in self.shape])}]'
	def __repr__(self): return str(self)

	def __len__(self): return self._basis_interpolations.shape[0]

	def __getitem__(self, coord: Tuple[float, ...]) -> NDArray:

		idx = BasisInterpolation.discretize_coordinate(coord=coord, min_max_coords=self._min_max_coords, shape=self.shape)

		bi = self._basis_interpolations[:, *idx]

		if self._out_shape is not None:
			return bi.reshape(self._out_shape)
		
		return bi
	
	@property
	def shape(self) -> Shape:
		n_object, *shape = self._basis_interpolations.shape
		return tuple(shape)

	@classmethod
	def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'MLIC':
		''' Load camera calibration from a pickle file. '''

		logger.info(msg=f"Loading camera calibration from {path}")
		with open(path, 'rb') as f: return pickle.load(f)

	def dump(
        self,
        path   : str,
        logger : BaseLogger = SilentLogger(),
        verbose: bool       = False
    ) -> None:
		''' Save the camera calibration to a pickle file. '''

		logger_verbose = logger if verbose else SilentLogger()

		ISUtils.check_output(path=PathUtils.get_folder_path(path=path), logger=logger_verbose)

		logger.info(msg=f"Saving basis interpolation collection to {path} ...")

		timer = Timer()
		with open(path, 'wb') as f: pickle.dump(self, f)
		logger.info(msg=f"Completed in {timer}")

class BasisInterpolator(ABC):
    
	def __init__(
		self, 
		values      : NDArray,
		coordinates : NDArray,
		min_max_coordinates: Tuple[Tuple[float, float], ...],
		min_max_values: Tuple[float, float] | None = None
	):
		
		self._values      : NDArray = values

		if not len(self._values.shape) == 1:
			raise ValueError(f'The values must be one dimensional. Got {self._values.shape} dimensions. ')

		self._n_points = self._values.shape[0]
		coord_points, self._dims = coordinates.shape

		if coord_points != self._n_points: 
			raise ValueError(f'The number of points {self._n_points} must match the one of given coordinates ({coord_points}). ')

		if self._dims != len(min_max_coordinates): 
			raise ValueError(f'The number of dimensions of the coordinates {self._dims} must match the length of min and max coordinates ({len(min_max_coordinates)}). ')

		self._coordinates         : NDArray                         = coordinates
		self._min_max_coordinates : Tuple[Tuple[float, float], ...] = min_max_coordinates
		self._minmax_values       : Tuple[float, float]             = default(min_max_values, (float(self._values.min()), float(self._values.max())))
	
	def __str__(self) : return f'{self.__class__.__name__}[{len(self)} points; {self.dims} dimensions]'
	def __repr__(self): return str(self)

	def __len__(self) -> int: return self._n_points

	@property
	def dims(self) -> int: return self._dims

	@property
	@abstractmethod
	def interpolate_function(self) -> Callable[[NDArray], NDArray]: pass

	def interpolate(self, size: Tuple[int, ...]) -> BasisInterpolation:

		# Create interpolation grid
		grid = np.meshgrid(*[
			np.linspace(min_val, max_val, dim) 
			for (min_val, max_val), dim in zip(self._min_max_coordinates, size)
		])

		# Interpolate
		basis = self.interpolate_function(*grid)

		min_val, max_val = self._minmax_values
		basis = np.clip(basis, min_val, max_val)

		return BasisInterpolation(
			basis=basis, 
			min_max_coors=self._min_max_coordinates,
			min_max_values=self._minmax_values
		)

class RTIBasisInterpolator(BasisInterpolator):

	COORD_RANGE  = (-1,   1)
	VALUES_RANGE = ( 0, 255)

	def __init__(
		self,
		values      : NDArray,
		coordinates : NDArray,
	):
		super().__init__(
			values=values,
			coordinates=coordinates,
			min_max_coordinates=(self.COORD_RANGE, self.COORD_RANGE),
			min_max_values=self.VALUES_RANGE
		)

class RTIRadialBasisInterpolator(RTIBasisInterpolator):
	
	def __init__(self, values: NDArray, coordinates : NDArray):
		
		super().__init__(values=values, coordinates=coordinates)

	@property
	def interpolate_function(self) -> Callable[[NDArray], NDArray]:

		return Rbf(*self._coordinates.T, self._values, function='linear', smooth=0.1)
	


class MLICBasisInterpolator:
    
	def __init__(
		self,
		mlic: MLIC,
		C_rti_interpolator: Type[RTIBasisInterpolator],
		interpolation_size: Tuple[int, int],
		logger: BaseLogger = SilentLogger(),
		verbose: bool = False
	):
		
		self._logger         : BaseLogger = logger
		self._logger_verbose : BaseLogger = logger if verbose else SilentLogger()
		self._is_verbose     : bool       = verbose

		self._mlic               : MLIC                       = mlic
		self._C_rti_interpolator : Type[RTIBasisInterpolator] = C_rti_interpolator
		self._interpolation_size : Tuple[int, int]            = interpolation_size 
	
	def __str__(self) -> str:
		
		sx, sy = self._interpolation_size
		
		return f'{self.__class__.__name__}['\
            f'size: {sx}x{sy}; '\
            f'MLIC objects: {len(self._mlic)}; '\
            f'interpolator: {self._C_rti_interpolator.__name__}]'
	
	def __repr__(self) -> str: return str(self)

	def get_pixel_interpolation(self, pixel: Tuple[int, int]) -> BasisInterpolation:

		ri_interpolator = self._C_rti_interpolator(
			values=self._mlic.get_pixel_values(pixel=pixel), 
			coordinates=self._mlic.light_directions
		)

		return ri_interpolator.interpolate(size=self._interpolation_size)
	
	def get_interpolation_collection(self, progress: int | None = None, save_dir: str = '') -> BasisInterpolationCollection:

		row, cols = self._mlic.size
		interpolated_basis = []
		
		i = 0

		self._logger.info(msg=f'Starting interpolation for all pixels ({np.prod(self._mlic.size)}). ')

		if save_dir:
			ISUtils.check_output(path=save_dir, logger=self._logger_verbose)
			self._logger.info(msg=f'Saving plots to {save_dir}. ')

		for pixel in itertools.product(range(row), range(cols)):
			
			i += 1
	
			if i == 10: break
			
			bi = self.get_pixel_interpolation(pixel=pixel)

			interpolated_basis.append(bi)

			if save_dir:
				px, py = pixel
				fig = bi.plot_interpolation(
					title=f'Pixel {pixel}', 
					points_coord=(self._mlic.get_pixel_values(pixel=pixel), self._mlic.light_directions)
				)

				save_path = os.path.join(save_dir, f'pixel_{px}_{py}.png')
				self._logger.info(msg=f'Saving plot for pixel {pixel} to {save_path} . ')
				fig.savefig(os.path.join(save_dir, f'pixel_{pixel}.png'))
				plt.close(fig)
		
		self._logger.info

		return BasisInterpolationCollection(basis_interpolations=interpolated_basis)