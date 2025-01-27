from __future__ import annotations

import pickle
from typing import Dict, Iterator, List, Sequence, Tuple

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from src.model.thresholding import Thresholding
from src.model.marker import MarkerDetectionVideoStream, Marker, MarkerDetector
from src.utils.typing import LightDirection, Frame, Size2D, Views, Pixel, RGBColor, CameraPoseMethod
from src.utils.stream import Stream, SynchronizedVideoStream
from src.utils.calibration import CalibratedCamera
from src.utils.io_ import (
    BaseLogger, SilentLogger,
    InputSanitizationUtils as ISUtils, PathUtils,
)
from src.utils.misc import Timer

# --- MLIC ---

class MultiLightImageCollection:
    '''
    Class representing a Multi-Light Image Collection (MLIC) composed of a set of object frames
        and their corresponding light directions.
    
    The collection of N frames of images with size W x H are stored in a tensor of shape (N, W, H) in range [0, 255].
    The light directions are stored as a tensor of shape (N, 2) where each row is a light direction (u, v), each in the range [-1, 1].

    The image is represented in the YUV, where the Y channels representing the luminance is the one stored in the tensor.
    
    To reconstruct a color image, the U and V channels must be added. 
    This is implemented by keeping them fixed as the per-pixel mean across all frames. 
    The U V means are stored as tensors of shape (W, H, 2) in range [0, 255].

    The MLIC is pickle-able to save and load the collection to/from a file.

    The MLIC can be splitted in train and test sets, to allow for the two phases of fitting and evaluating.
    '''

    def __init__(
            self, 
            object_frames    : NDArray, 
            light_directions : NDArray, 
            uv_means         : Tuple[NDArray, NDArray]
        ):
        '''
        The method initializes the Multi-Light Image Collection with the object frames, light directions, and UV means.
        It performs some sanity checks on the input tensors to ensure they have the correct shape and values.

        :param object_frames: The object frames tensor of shape (N, W, H) representing the luminance of the images.
        :param light_directions: The light directions tensor of shape (N, 2) representing the light source direction.
        :param uv_means: A tuple of the U and V means tensors of shape (W, H) representing the per-pixel mean of the U and V channels.
        '''

        u_mean, v_mean = uv_means

        n_obj, *obj_size = object_frames   .shape
        n_ld,  *ld_shape = light_directions.shape

        # Sanity checks
        if n_obj != n_ld:
            raise ValueError(f"The number of frames must be equal to the number of light sources, got {n_obj} (frames) and {n_ld} (light sources). ")
                
        if tuple(ld_shape) != (2,):
            raise ValueError(f"Light source is expected to be bidimensional, got {ld_shape}. ")
        
        if not np.all((-1 <= light_directions) & (light_directions <= 1)): 
            raise ValueError("Light source values must be in the range [-1, 1]. ")

        if not (u_mean.shape == v_mean.shape == tuple(obj_size)):
            raise ValueError(f"Frame shape and UV means shape must be equal, got {obj_size} (frame shape) {u_mean.shape} (U-mean), {v_mean.shape} (V-mean). ")
        
        if not (len(obj_size) == 2):
            raise ValueError(f"Frame shape must be bidimensional, got {obj_size}. ")
        
        for name, array in zip(['Frame', 'U mean', 'V mean'], [object_frames[1:], u_mean, v_mean]):
            if not np.all((0 <= array) & (array <= 255)): raise ValueError(f"{name} values must be in the range [0, 255]. ")
        
        self._n_frames         : int                     = n_obj
        self._size             : Size2D                  = tuple(obj_size)
        self._obj_frames       : NDArray                 = object_frames
        self._light_directions : NDArray                 = light_directions
        self._uv_means         : Tuple[NDArray, NDArray] = uv_means
    
    # --- PICKLE METHODS ---
    
    @classmethod
    def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'MultiLightImageCollection':
        ''' Load MLIC from a pickle file. '''

        logger.info(msg=f"Loading Multi-Light Image Collection from {path}")

        timer = Timer()
        with open(path, 'rb') as f: data = pickle.load(f)
        logger.info(msg=f"Completed in {timer}. ")

        return data
    
    def dump(
        self,
        path   : str,
        logger : BaseLogger = SilentLogger()
    ) -> None:
        ''' Save the camera calibration to a pickle file. '''

        ISUtils.check_output(path=PathUtils.get_folder_path(path=path), logger=logger)

        logger.info(msg=f"Saving MLIC to {path} ...")

        timer = Timer()
        with open(path, 'wb') as f: pickle.dump(self, f)
        logger.info(msg=f"Completed in {timer}. ")

    
    # --- MAGIC METHODS ---

    def __str__(self)  -> str: return f"{self.__class__.__name__}[shape: {'x'.join([str(s) for s in self._size])}; items: {self._n_frames}]"
    def __repr__(self) -> str: return str(self)
    def __len__(self)  -> int: return self._n_frames

    def __iter__(self) -> Iterator[Tuple[Frame, LightDirection]]: return iter(zip(self._obj_frames, self._light_directions))
    ''' Iterate on the couple of (object frame, light direction) for each frame. '''

    def __getitem__(self, index: int) -> Tuple[Frame, LightDirection]: return self._obj_frames[index], self._light_directions[index]
    ''' Return the object frame and light direction at the specified index. '''

    # --- PROPERTIES ---

    @property
    def obj_frames(self) -> NDArray: return self._obj_frames

    @property
    def light_directions(self) -> NDArray: return self._light_directions

    @property
    def size(self) -> Size2D: return self._size

    # --- UTILITY METHODS ---

    def add_uv_channels(self, y_frame: Frame) -> Frame:
        '''
        Reconstruct a color image from a luminance frame by adding the U and V channels.
        It returns a color image in RGB format.

        :param y_frame: The luminance frame to convert to color.
        :return: The color image in RGB format.
        '''

        if y_frame.shape != self._size:
            raise ValueError(f"Frame shape must be equal to the MLIC shape of {self.size}, got {y_frame.shape} (frame). ")

        yuv_frame = np.stack([y_frame, *self._uv_means], axis=-1, dtype=np.uint8)
        rgb_frame = cv.cvtColor(yuv_frame, cv.COLOR_YUV2RGB)

        return rgb_frame
    
    def get_pixel_values(self, pixel: Pixel) -> NDArray:
        ''' Returns a vector of pixel values across all the MLIC frames at the specified pixel coordinates. '''

        px, py = pixel
        w, h = self.size

        if not (0 <= px < w and 0 <= py < h): raise ValueError(f"Pixel coordinates must be within the frame shape {self.size}, got {pixel}. ")

        return self._obj_frames[:, px, py]
    
    def train_test_split(self, test_size: float = 0.1) -> Tuple[MultiLightImageCollection, MultiLightImageCollection]:
        ''' 
        Split the MLIC into train and test sets according to the specified test size.
        The split is performed by sampling test indices at uniform distances from the start to the end.

        :param test_size: The fraction of the dataset to include in the test split.
        :return: A tuple of the train and test Multi-Light Image Collections.        
        '''

        # Sample test indices at uniform distances from the start to the end
        test_indices = np.linspace(start=0, stop=len(self)-1, num=int(len(self)*test_size), dtype=int)

        # Create complementary train indices
        train_indices = np.setdiff1d(np.arange(len(self)), test_indices)

        return\
            MultiLightImageCollection(object_frames=self._obj_frames[train_indices], light_directions=self._light_directions[train_indices], uv_means=self._uv_means),\
            MultiLightImageCollection(object_frames=self._obj_frames[test_indices],  light_directions=self._light_directions[test_indices],  uv_means=self._uv_means)
    
    # --- STREAMING ---

    def get_views(self, index: int) -> Views:
        ''' 
        Return the object frame views for a specified index in the collection, specifically:
            - the colored object frame;
            - the current light direction;
            - the history of light directions.
        '''

        obj_frame, light_source = self[index]

        return {
            'object_frame'         : self.add_uv_channels(y_frame=obj_frame),
            'light_direction'      : DynamicCameraVideoStream.draw_line_direction        (light_direction=light_source),
            'light_direction_hist' : DynamicCameraVideoStream.draw_line_direction_history(points=self._light_directions[:index+1])
        }

    def to_stream(self, name: str = 'mlic', logger: BaseLogger = SilentLogger()) -> MultiLightImageCollectionStream: 
        ''' Convert the MLIC to a stream. '''

        return MultiLightImageCollectionStream(mlic=self, name=name, logger=logger)


class MultiLightImageCollectionStream(Stream):
    ''' Class for streaming the Multi-Light Image Collection, providing the synchronized view of the object frames and the source light directions. '''

    def __init__(
        self, 
        mlic    : MultiLightImageCollection,
        name    : str = 'mlic', 
        logger  : BaseLogger = SilentLogger()
    ):
        ''' Initialize the Multi-Light Image Collection Stream with the MLIC to stream. '''

        super().__init__(name=name, logger=logger)

        self._mlic: MultiLightImageCollection = mlic

    def __len__    (self)           -> int   : return len(self._mlic)
    def __getitem__(self, idx: int) -> Views : return self._mlic.get_views(index=idx)

    def iter_range(self, start: int, end: int, step: int = 1) -> Iterator[Tuple[int, Views]]: 
        ''' The iterator simply yields the views for the specified range of indices. '''
        for idx in range(start, end, step): yield idx, self[idx]

    @property
    def _default_window_size(self) -> Size2D: return self._mlic.size

    @property
    def views(self) -> List[str]: return list(self[0].keys()) # NOTE: at least one frame is guaranteed


class MultiLightImageCollectionAccumulator:
    ''' 
    Class used to accumulate the object frames and light directions processed by the video streams,
        and to convert them into a Multi-Light Image Collection.
    
    It also updates some statistics about the processed frames and light directions.
    '''

    def __init__(self):
        '''
        The object frames and light directions are stored in separate lists.
        '''

        self._object_frames    : List[Frame]          = []
        self._light_directions : List[LightDirection] = []

        self._tot_processed_frames : int = 0
        self._object_frames_succ   : int = 0
        self._light_directions_succ: int = 0

    def __str__ (self) -> str: return f"{self.__class__.__name__}[items: {len(self)}]"
    def __repr__(self) -> str: return str(self)
    def __len__ (self) -> int: return len(self._object_frames)

    # --- PROPERTIES ---

    @property
    def object_frames(self) -> List[Frame] : return self._object_frames

    @property
    def light_directions(self) -> List[LightDirection] : return self._light_directions

    @property
    def processed_info(self) -> Tuple[int, int, int]: return self._object_frames_succ, self._light_directions_succ, self._tot_processed_frames
    
    # --- METHODS ---
    
    def add(self, obj_frame: Frame | None, light_direction: LightDirection | None):
        ''' 
        Updates the accumulator with the processed object frame and light direction.
        In the case one was not processed, it is given as None and statistics are updated accordingly.

        The accumulator stores the object frames and light directions in separate lists when both were processed correctly.
        '''

        # Update statistics
        self._tot_processed_frames += 1
        if obj_frame       is not None: self._object_frames_succ    += 1
        if light_direction is not None: self._light_directions_succ += 1

        # Store the object frame and light direction if both were correctly processed
        if obj_frame is not None and light_direction is not None:

            self._object_frames    .append(obj_frame)
            self._light_directions .append(light_direction)

    def to_mlic(self) -> MultiLightImageCollection:
        ''' Convert the accumulated object frames and light directions to a Multi-Light Image Collection. '''

        # Convert the accumulated object frames and light directions to numpy arrays
        # The frames are converted to YUV format
        light_directions : NDArray = np.array(self._light_directions)
        obj_frames_yuv   : NDArray = np.array([cv.cvtColor(obj_frame, cv.COLOR_RGB2YUV) for obj_frame in self._object_frames])

        # Extract the Y channel for the MLIC
        # Store the mean of the U and V channels across all frames
        obj_frames_y      = obj_frames_yuv[:, :, :, 0]
        obj_frames_u_mean = np.mean(obj_frames_yuv[:, :, :, 1], axis=0).astype(np.uint8)
        obj_frames_v_mean = np.mean(obj_frames_yuv[:, :, :, 2], axis=0).astype(np.uint8)

        return MultiLightImageCollection(
            object_frames=obj_frames_y,
            light_directions=light_directions,
            uv_means=(obj_frames_u_mean, obj_frames_v_mean)
        )

# --- CAMERA STREAMS ---
class StaticCameraVideoStream(MarkerDetectionVideoStream):
    '''
    The Static Camera Processing is responsible to process the video stream from a static camera,
        detect the marker and warp the frame to a square image.
    '''

    def __init__(
        self, 
        path            : str, 
        calibration     : CalibratedCamera,
        thresholding    : Thresholding,
		marker_detector : MarkerDetector,
        mlic_size       : Size2D,
        name            : str | None = None,
        logger          : BaseLogger = SilentLogger()
    ):
        '''
        The Static Camera Video Stream requires the size of the image to be warped in pixels.
        '''

        super().__init__(
            path=path,
            calibration=calibration,
            thresholding=thresholding,
            marker_detector=marker_detector,
            name=name,
            logger=logger
        )

        self._mlic_size               : Size2D = mlic_size
        self._last_processed_frame_id : int = -1
    
    # --- MAGIC METHODS ---

    def __str__ (self) -> str: return f"{self.__class__.__name__}[{self.name}, frames: {len(self)}]"
    def __repr__(self) -> str: return str(self)

    # --- PROPERTIES ---

    @property
    def last_processed_frame_id(self) -> int: 

        if self._last_processed_frame_id == -1:
            self._logger.handle_error(msg="No frame processed yet", exception=AttributeError)
        
        return self._last_processed_frame_id
    
    # --- PROCESSING ---

    def _process_marker(self, views: Views, marker: Marker, frame_id: int) -> Views:
        '''
        Use marker to warp the undistorted frame to an image of size `mlic_size`.
        '''

        # Update last processed frame
        self._last_processed_frame_id = frame_id

        # Get previously processed views
        super_views = super()._process_marker(views=views, marker=marker, frame_id=frame_id)
        frame_undistorted = views['undistorted'].copy()

        # Warp the frame to the MLIC size
        warped = marker.warp(frame=frame_undistorted, size=self._mlic_size)

        return super_views | {'warped': warped}

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        super_views = super()._process_frame(frame=frame, frame_id=frame_id)

        # Add to the views the warped frame
        return super_views | ({'warped': np.zeros_like(frame)} if 'warped' not in super_views else super_views)

class DynamicCameraVideoStream(MarkerDetectionVideoStream):
    '''
    The Dynamic Camera Processing is responsible to process the video stream from a dynamic camera,
        detect the marker and estimate the camera pose, that is the approximately the same of the light direction.
    '''

    def __init__(
        self, 
        path            : str, 
        calibration     : CalibratedCamera,
        thresholding    : Thresholding,
        marker_detector : MarkerDetector,
        name            : str | None       = None,
        method          : CameraPoseMethod = 'algebraic',
        plot_history    : bool             = False,
        logger          : BaseLogger       = SilentLogger()
    ):
        '''
        The Dynamic Camera Video Stream requires the method to estimate the camera pose (either 'algebraic' or 'geometric').
        It also has a flag to stream the history of the light directions.
        '''
        
        super().__init__(
            path=path,
            calibration=calibration,
            thresholding=thresholding,
            marker_detector=marker_detector,
            name=name,
            logger=logger
        )

        self._light_direction_method  : CameraPoseMethod     = method
        self._plot_history            : bool                 = plot_history
        self._last_processed_frame_id : int                  = -1
        self._light_directions        : List[LightDirection] = [] # We keep the list of processed light direction to plot the history

    # --- MAGIC METHODS ---

    def __str__ (self) -> str: return f"{self.__class__.__name__}[{self.name}, frames: {len(self)}]"
    def __repr__(self) -> str: return str(self)

    # --- PROPERTIES ---

    @property
    def last_processed_frame_id(self) -> int: 

        if self._last_processed_frame_id == -1:
            self._logger.handle_error(msg="No frame processed yet", exception=AttributeError)
        
        return self._last_processed_frame_id
    
    @property
    def last_processed_direction(self) -> LightDirection: 

        try: return self._light_directions[-1]
        except IndexError: self._logger.handle_error(msg="Still no light direction processed", exception=AttributeError)

    # --- PROCESSING ---

    def _process_marker(self, views: Views, marker: Marker, frame_id: int) -> Views:
        '''
        Use the detected marker to estimate the camera pose using the specified method.
        '''

        # Save the last processed frame
        self._last_processed_frame_id = frame_id 

        # Get previously processed views of marker processing
        super_views = super()._process_marker(views=views, marker=marker, frame_id=frame_id)

        # Estimate the light direction from the marker
        light_direction = marker.estimate_camera_pose(calibration=self._calibration, method=self._light_direction_method)

        # Store the light direction
        self._light_directions.append(light_direction)

        # Draw light direction and history (if flag is set)
        direction_frame = self.draw_line_direction(light_direction=light_direction)
        views_out = super_views | {'light_direction': direction_frame}

        if self._plot_history:
            direction_hist_frame = self.draw_line_direction_history(points=np.array(self._light_directions))
            views_out |= {'light_direction_hist': direction_hist_frame}
        
        return views_out

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        if frame_id == 0: self._light_directions = []  # Reset the light directions at the beginning of the stream

        super_views = super()._process_frame(frame=frame, frame_id=frame_id)

        # Add to the views the light direction and history
        return super_views |\
            ( {'light_direction'     : np.zeros_like(frame)} if 'light_direction'      not in super_views else super_views) |\
            (({'light_direction_hist': np.zeros_like(frame)} if 'light_direction_hist' not in super_views else super_views) if self._plot_history else {})

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
        text_x         = f"x: {x:+.2f}"
        text_y         = f"y: {y:+.2f}"
        font_scale     = 0.6
        font_thickness = 2
        padding        = 10
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
        points    : NDArray,
        img_side  : int      = 500,
        col_start : RGBColor = (255, 255, 255),
        col_end   : RGBColor = (  0,   0, 255)
    ) -> Frame:
        ''' Draw the history of light directions as points with a color gradient. '''
        
        # Blank image
        img = np.zeros((img_side, img_side, 3), dtype=np.uint8)
        n_points = points.shape[0]
        
        # Compute center and radius
        center = (img_side // 2, img_side // 2)
        radius = img_side // 2
        
        # Draw the white circle
        cv.circle(img, center, radius, (255, 255, 255), thickness=4)

        # Normalize the points from [-1, 1] to pixel coordinates
        normalized_points = np.empty_like(points, dtype=int)
        normalized_points[:, 0] = (center[0] + points[:, 0] * radius).astype(int)  # Scale x
        normalized_points[:, 1] = (center[1] - points[:, 1] * radius).astype(int)  # Scale y

        # Plot point with color gradient
        for j, (x, y) in enumerate(normalized_points.astype(int)):
            
            # Compute the color based on the gradient
            color = tuple(
                int(col_start[i] + (j/n_points) * (col_end[i] - col_start[i]))
                for i in range(3)
            )
            cv.circle(img, (x, y), radius=3, color=color, thickness=-1)

        return img

class MLICCollector(SynchronizedVideoStream):

    def __init__(
        self,
        mlic_static  : StaticCameraVideoStream,
        mlic_dynamic : DynamicCameraVideoStream,
        logger       : BaseLogger = SilentLogger()
    ):
        ''' 
        Initialize the MLIC Collector with the static and dynamic video streams. 
        '''

        return super().__init__(streams=[mlic_static, mlic_dynamic], logger=logger)
    
    def _play_iter_progress(self, views: Dict[str, Dict[str, Frame]], frame_id: int):
        '''
        Updates the accumulator with the object frame and light direction processed by the video streams
        '''

        mlic_static  : StaticCameraVideoStream
        mlic_dynamic : DynamicCameraVideoStream
        mlic_static, mlic_dynamic = self._streams.values()  # type: ignore - check made in play method

        # If the last processed frame id is the same as the current one, the marker was successfully processed
        static_success  = frame_id == mlic_static. last_processed_frame_id
        dynamic_success = frame_id == mlic_dynamic.last_processed_frame_id

        # If the detection was successful, use the processed result, otherwise use None to signal the failure
        obj_frame = views[mlic_static.name]['warped']     if static_success  else None
        light_dir = mlic_dynamic.last_processed_direction if dynamic_success else None

        # Update the accumulator
        self._accumulator.add(obj_frame=obj_frame, light_direction=light_dir)

    def collect(
        self, 
        start       : int                   = 0, 
        end         : int | None            = None, 
        skip_frames : int                   = 1,
        delay       : int                   = 1,
        win_rect    : Tuple[Size2D, Size2D] = ((216, 384), (384, 216)),
        win_square  : Size2D                = (256, 256)
    ) -> MultiLightImageCollection:
        '''
        The collect method wraps the play method to synchronize the video streams and collect the object frames and light directions in the accumulator.
        It then converts the accumulator to a Multi-Light Image Collection.
        '''
        
        assert len(self._streams) == 2, "The synchronized mlic video stream must have exactly two streams"
        
        mlic_static, mlic_dynamic = self._streams.values()

        assert isinstance(mlic_static,  StaticCameraVideoStream),  "The first stream  must be a MLICStaticCameraVideoStream"
        assert isinstance(mlic_dynamic, DynamicCameraVideoStream), "The second stream must be a MLICDynamicCameraVideoStream"
    
        # Window size
        static_win, dynamic_win = win_rect

        window_size: Dict[str, Dict[str, Size2D]] = {
            mlic_static.name :  {view: static_win  if view not in ['warped']                                  else win_square for view in mlic_static .views},
            mlic_dynamic.name:  {view: dynamic_win if view not in ['light_direction', 'light_direction_hist'] else win_square for view in mlic_dynamic.views}
        }

        # Exclude views
        exclude_views = {
            mlic_static .name : [view for view in mlic_static.views  if view not in ['marker', 'warped']],
            mlic_dynamic.name : [view for view in mlic_dynamic.views if view not in ['marker', 'light_direction', 'light_direction_hist']]
        }

        # Initialize the accumulator
        self._accumulator = MultiLightImageCollectionAccumulator()

        # Play the synchronized video stream
        self._logger.info(msg=f'Playing dynamic and static video streams for MLIC collection...\n')
        timer = Timer()

        super().play(
            start=start,
            end=end,
            skip_frames=skip_frames,
            window_size=window_size,     # type: ignore
            exclude_views=exclude_views,
            delay=delay
        )

        # Log the results
        self._logger.info(msg=f'\nCompleted in {timer}\n')
        self._logger.info(msg=f'Collected MLIC frames: {self._accumulator}')
        
        obj_frames, light_dirs, tot_frames = self._accumulator.processed_info
        mlic_tot = len(self._accumulator)

        self._logger.info(f' - Static  frames processed: {obj_frames}/{tot_frames} ({obj_frames/tot_frames:.2%})')
        self._logger.info(f' - Dynamic frames processed: {light_dirs}/{tot_frames} ({light_dirs/tot_frames:.2%})')
        self._logger.info(f' - Total MLIC processed:     {mlic_tot  }/{tot_frames} ({mlic_tot  /tot_frames:.2%})')
        self._logger.info(f'')

        # Convert the accumulator to a Multi-Light Image Collection
        self._logger.info('Converting accumulator to MLIC...')
        timer.reset()
        mlic = self._accumulator.to_mlic()
        self._logger.info(f'Completed in {timer}. ')
        self._logger.info(f'Multi-Light Image Collection: {mlic}')
        self._logger.info('')

        return mlic
