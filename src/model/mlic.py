from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
import cv2 as cv

from src.utils.io_ import InputSanitizationUtils as ISUtils, PathUtils
from src.model.typing import LightDirection
from src.model.stream import Stream, SynchronizedVideoStream
from src.model.calibration import CameraCalibration
from src.model.marker import MarkerDetectionVideoStream, Marker, MarkerDetector
from src.model.thresholding import Thresholding
from src.model.typing import Frame, Size2D, Views, Pixel
from src.utils.io_ import BaseLogger, SilentLogger
from src.utils.misc import Timer, default

class MLIC:

    def __init__(self, object_frames: NDArray, light_directions: NDArray, uv_means: Tuple[NDArray, NDArray]):

        u_mean, v_mean = uv_means

        n_obj, *obj_shape = object_frames   .shape
        n_ld,  *ld_shape  = light_directions.shape

        if n_obj != n_ld:
            raise ValueError(f"The number of frames must be equal to the number of light sources, got {n_obj} (frames) and {n_ld} (light sources). ")
        
        if n_obj == 0:
            raise ValueError("Cannot create a MLIC with no frames. ")
        
        if tuple(ld_shape) != (2,):
            raise ValueError(f"Light source is expected to be bidimensional, got {ld_shape}. ")
        
        if not np.all((-1 <= light_directions) & (light_directions <= 1)): 
            raise ValueError("Light source values must be in the range [-1, 1]")

        if not (u_mean.shape == v_mean.shape == tuple(obj_shape)):
            raise ValueError(f"Frame shape and UV means shape must be equal, got {obj_shape} (frame shape) {u_mean.shape} (U-mean), {v_mean.shape} (V-mean). ")
        
        if not (len(obj_shape) == 2):
            raise ValueError(f"Frame shape must be bidimensional, got {obj_shape}. ")
        
        for name, array in zip(['frame', 'U mean', 'V mean'], [object_frames[1:], u_mean, v_mean]):
            if not np.all((0 <= array) & (array <= 255)): raise ValueError(f"{name} values must be in the range [0, 255]")
        
        self._n_frames         : int                     = n_obj
        self._shape            : Size2D                  = tuple(obj_shape)
        self._obj_frames       : NDArray                 = object_frames
        self._light_directions : NDArray                 = light_directions
        self._uv_means         : Tuple[NDArray, NDArray] = uv_means
    
    @classmethod
    def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'MLIC':
        ''' Load camera calibration from a pickle file. '''

        logger.info(msg=f"Loading camera calibration from {path}")

        with open(path, 'rb') as f: return pickle.load(f)

    def __str__(self)  -> str: return f"MLIC[shape: {'x'.join([str(s) for s in self._shape])}; items: {self._n_frames}]"
    def __repr__(self) -> str: return str(self)
    def __len__(self)  -> int: return self._n_frames

    def __iter__(self) -> Iterator[Tuple[Frame, LightDirection]]: return iter(zip(self._obj_frames, self._light_directions))

    def __getitem__(self, index: int) -> Tuple[Frame, LightDirection]: return self._obj_frames[index], self._light_directions[index]

    @property
    def obj_frames(self) -> NDArray: return self._obj_frames

    @property
    def light_directions(self) -> NDArray: return self._light_directions

    def add_uv_channels(self, y_frame: Frame) -> Frame:

        if y_frame.shape != self._shape:
            raise ValueError(f"Frame shape must be equal to the MLIC shape of {self.shape}, got {y_frame.shape} (frame). ")

        yuv_frame = np.stack([y_frame, *self._uv_means], axis=-1, dtype=np.uint8)
        rgb_frame = cv.cvtColor(yuv_frame, cv.COLOR_YUV2RGB)

        return rgb_frame
    
    def get_pixel_values(self, pixel: Pixel) -> NDArray:

        px, py = pixel
        w, h = self.shape

        if not (0 <= px < w and 0 <= py < h): raise ValueError(f"Pixel coordinates must be within the frame shape {self.shape}, got {pixel}. ")

        return self._obj_frames[:, px, py]


    def get_views(self, index: int) -> Views:

        obj_frame, light_source = self[index]

        return {
            'object_frame'    : self.add_uv_channels(y_frame=obj_frame),
            'light_direction' :  MLICDynamicCameraVideoStream._draw_arrow_in_circle(light_direction=light_source)
        }

    @property
    def shape(self) -> Size2D: return self._shape

    def to_stream(
        self, name: str = 'mlic',
        logger: BaseLogger = SilentLogger(),
        verbose: bool = False
    ) -> MLICStream: 
        return MLICStream(mlic=self, name=name, logger=logger, verbose=verbose)
    
    def dump(
        self,
        path   : str,
        logger : BaseLogger = SilentLogger(),
        verbose: bool       = False
    ) -> None:
        ''' Save the camera calibration to a pickle file. '''

        logger_verbose = logger if verbose else SilentLogger()

        ISUtils.check_output(path=PathUtils.get_folder_path(path=path), logger=logger_verbose)

        logger.info(msg=f"Saving MLIC to {path} ...")

        timer = Timer()

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(msg=f"Completed in {timer}")

class MLICStream(Stream):

    def __init__(
        self, 
        mlic    : MLIC,
        name    : str = 'mlic', 
        logger  : BaseLogger = SilentLogger(), 
        verbose : bool = False
    ):

        super().__init__(name=name, logger=logger, verbose=verbose)

        self._mlic: MLIC = mlic

    def __len__(self) -> int: return len(self._mlic)

    def iter_range(self, start: int, end: int, step: int = 1) -> Iterator[Tuple[int, Views]]: 

        for idx in range(start, end, step): yield idx, self[idx]

    def __getitem__(self, idx: int) -> Views: return self._mlic.get_views(index=idx)

    @property
    def _default_window_size(self) -> Size2D: return self._mlic.shape

    @property
    def views(self) -> List[str]: return list(self[0].keys()) # NOTE: at least one frame is guaranteed


class MLICAccumulator:

    def __init__(self):

        self._object_frames    : List[Frame]          = []
        self._light_directions : List[LightDirection] = []

        self._tot_processed_frames : int = 0
        self._object_frames_succ   : int = 0
        self._light_directions_succ: int = 0

    def __str__ (self) -> str: return f"MLICAccumulator[items: {len(self)}]"
    def __repr__(self) -> str: return str(self)
    def __len__ (self) -> int: return len(self._object_frames)

    @property
    def processed_info(self) -> Tuple[int, int, int]: return self._object_frames_succ, self._light_directions_succ, self._tot_processed_frames
    
    def add(self, obj_frame: Frame | None, light_direction: LightDirection | None):

        self._tot_processed_frames += 1

        if obj_frame       is not None: self._object_frames_succ    += 1
        if light_direction is not None: self._light_directions_succ += 1

        if obj_frame is not None and light_direction is not None:

            self._object_frames    .append(obj_frame)
            self._light_directions .append(light_direction)
    
    def filter(self, indices: Sequence[int]):

        self._object_frames    = [self._object_frames   [i] for i in indices]
        self._light_directions = [self._light_directions[i] for i in indices]

    def to_mlic(self) -> MLIC:

        light_directions = np.array(self._light_directions)

        obj_frames_yuv: NDArray = np.array([cv.cvtColor(obj_frame, cv.COLOR_RGB2YUV) for obj_frame in self._object_frames])

        obj_frames_y      = obj_frames_yuv[:, :, :, 0]
        obj_frames_u_mean = np.mean(obj_frames_yuv[:, :, :, 1], axis=0).astype(np.uint8)
        obj_frames_v_mean = np.mean(obj_frames_yuv[:, :, :, 2], axis=0).astype(np.uint8)

        return MLIC(
            object_frames=obj_frames_y,
            light_directions=light_directions,
            uv_means=(obj_frames_u_mean, obj_frames_v_mean)
        )

class MLICStaticCameraVideoStream(MarkerDetectionVideoStream):

    def __init__(
        self, 
        path            : str, 
        calibration     : CameraCalibration,
        thresholding    : Thresholding,
		marker_detector : MarkerDetector,
        mlic_side       : int,
        name            : str        = '',
        logger          : BaseLogger = SilentLogger(),
        verbose         : bool       = False
    ):
        
        super().__init__(
            path=path,
            calibration=calibration,
            thresholding=thresholding,
            marker_detector=marker_detector,
            name=name,
            logger=logger,
            verbose=verbose
        )

        self._mlic_side            : int = mlic_side
        self._last_processed_frame : int = -1

    def __str__(self)  -> str: return f"MLICStaticCameraVideoStream[{self.name}, frames: {len(self)}]"
    def __repr__(self) -> str: return str(self)

    @property
    def last_processed_frame(self) -> int: return self._last_processed_frame

    def _process_marker(self, views: Views, marker: Marker, frame_id: int) -> Views:

        self._last_processed_frame = frame_id

        super_views = super()._process_marker(views=views, marker=marker, frame_id=frame_id)

        warped = marker.warp(frame=views['calibrated'].copy(), side=self._mlic_side)

        # Flip vertically the warped image to make 0,0 the top-left corner
        warped = cv.flip(warped, flipCode=0)

        return super_views | {'warped': warped}

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        super_views = super()._process_frame(frame=frame, frame_id=frame_id)

        return super_views | ({'warped': np.zeros_like(frame)} if 'warped' not in super_views else super_views)

class MLICDynamicCameraVideoStream(MarkerDetectionVideoStream):

    def __init__(
        self, 
        path            : str, 
        calibration     : CameraCalibration,
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
            marker_detector=marker_detector,
            name=name,
            logger=logger,
            verbose=verbose
        )

        self._last_processed_frame: int = -1

    def __str__(self)  -> str: return f"MLICDynamicCameraVideoStream[{self.name}, frames: {len(self)}]"
    def __repr__(self) -> str: return str(self)

    @property
    def last_processed_frame(self) -> int: return self._last_processed_frame
    
    @property
    def last_processed_direction(self) -> LightDirection: 

        if hasattr(self, '_light_direction'): return self._light_direction
        else: raise AttributeError("Still no light direction processed")

    @staticmethod
    def _draw_arrow_in_circle(light_direction: LightDirection, frame_side: int = 500) -> Frame:

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

        return image

    def _process_marker(self, views: Views, marker: Marker, frame_id: int) -> Views:

        self._last_processed_frame = frame_id

        super_views = super()._process_marker(views=views, marker=marker, frame_id=frame_id)

        self._light_direction = marker.camera_2d_position(calibration=self._calibration)

        direction_frame = self._draw_arrow_in_circle(light_direction=self._light_direction)

        return super_views | {'light_direction': direction_frame}

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        super_views = super()._process_frame(frame=frame, frame_id=frame_id)

        return super_views | ({'light_direction': np.zeros_like(frame)} if 'light_direction' not in super_views else super_views)

class MLICCollector(SynchronizedVideoStream):

    def __init__(
        self,
        mlic_static  : MLICStaticCameraVideoStream,
        mlic_dynamic : MLICDynamicCameraVideoStream,
        logger       : BaseLogger = SilentLogger(),
        verbose      : bool       = False
    ):

        return super().__init__(streams=[mlic_static, mlic_dynamic], logger=logger, verbose=verbose)
    
    def _play_iter_progress(self, frames: Dict[str, Dict[str, Frame]], frame_id: int):

        mlic_static  : MLICStaticCameraVideoStream
        mlic_dynamic : MLICDynamicCameraVideoStream
        mlic_static, mlic_dynamic = self._streams.values()  # type: ignore - check made in play method

        static_success  = frame_id == mlic_static. last_processed_frame
        dynamic_success = frame_id == mlic_dynamic.last_processed_frame

        obj_frame = frames[mlic_static.name]['warped']    if static_success  else None
        light_dir = mlic_dynamic.last_processed_direction if dynamic_success else None

        self._accumulator.add(obj_frame=obj_frame, light_direction=light_dir)

    def collect(
        self, 
        start           : int         = 0, 
        end             : int | None = None, 
        skip_frames     : int        = 1,
        delay           : int        = 1,
        win_rect_scale  : float      = 0.5,
        win_square_side : int        = 250,
    ) -> MLIC:
        
        assert len(self._streams) == 2, "The synchronized mlic video stream must have exactly two streams"
        
        mlic_static, mlic_dynamic = self._streams.values()

        assert isinstance(mlic_static,  MLICStaticCameraVideoStream),  "The first stream  must be a MLICStaticCameraVideoStream"
        assert isinstance(mlic_dynamic, MLICDynamicCameraVideoStream), "The second stream must be a MLICDynamicCameraVideoStream"
    
        # Window size
        static_win_size  : Size2D = tuple(int(s * win_rect_scale) for s in mlic_static .metadata.size)  # type: ignore - they have size 2
        dynamic_win_size : Size2D = tuple(int(s * win_rect_scale) for s in mlic_dynamic.metadata.size)  # type: ignore - they have size 2
        win_square_size  : Size2D = (win_square_side, win_square_side)

        window_size: Dict[str, Dict[str, Size2D]] = {
            mlic_static.name :  {view: static_win_size  if view != 'warped'          else win_square_size for view in mlic_static .views},
            mlic_dynamic.name:  {view: dynamic_win_size if view != 'light_direction' else win_square_size for view in mlic_dynamic.views}
        }

        # Exclude views
        exclude_views = {
            mlic_static .name : [view for view in mlic_static.views  if view not in ['marker', 'warped']],
            mlic_dynamic.name : [view for view in mlic_dynamic.views if view not in ['marker', 'light_direction']]
        }

        # Initialize the accumulator
        self._accumulator = MLICAccumulator()

        # Play the synchronized video stream
        self._logger.info(msg=f'Playing dynamic and static video streams for MLIC collection...\n')
        timer = Timer()

        super().play(
            start=start,
            end=end,
            skip_frames=skip_frames,
            window_size=window_size,      # type: ignore - TODO check why it is not working
            exclude_views=exclude_views,
            delay=delay
        )

        self._logger.info(msg=f'')
        self._logger.info(msg=f'Completed in {timer}\n')
        self._logger.info(msg=f'Collected MLIC frames: {self._accumulator}')
        
        obj_frames, light_dirs, tot_frames = self._accumulator.processed_info
        mlic_tot = len(self._accumulator)

        self._logger.info(f' - Static  frames processed: {obj_frames}/{tot_frames} ({obj_frames/tot_frames:.2%})')
        self._logger.info(f' - Dynamic frames processed: {light_dirs}/{tot_frames} ({light_dirs/tot_frames:.2%})')
        self._logger.info(f' - Total MLIC processed:     {mlic_tot  }/{tot_frames} ({mlic_tot  /tot_frames:.2%})')
        self._logger.info(f'')

        self._logger.info('Converting accumulator to MLIC...')
        timer.reset()
        mlic = self._accumulator.to_mlic()
        self._logger.info(f'Completed in {timer}')
        self._logger.info(f'Multi-Light Image Collection: {mlic}')
        self._logger.info('')

        return mlic

    def to_mlic(self) -> MLIC: 

        self._logger.info(msg='Converting accumulator to MLIC...')

        timer = Timer()
        accumulator = self._accumulator.to_mlic()
        self._logger.info(msg=f'Completed in {timer}')

        return accumulator

