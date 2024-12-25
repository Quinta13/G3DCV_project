import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Iterable

from numpy.typing import NDArray
import cv2 as cv

from src.utils.io_ import (
    BaseLogger, SilentLogger,
    PathUtils, InputSanitizationUtils as ISUtils,
    VideoFile
)
from src.utils.misc import Timer, default
from src.utils.io_ import BaseLogger, InputSanitizationUtils as ISUtils, PathUtils, SilentLogger


class VideoStream:

    EXIT_KEY = ord('q')

    def __init__(self, path: str, logger: BaseLogger = SilentLogger(), verbose: bool = False):
        ''' Initialize a video stream object from a video file. '''

        self._logger        : BaseLogger = logger
        self._logger_verbose: BaseLogger = logger if verbose else SilentLogger()
        
        ISUtils.check_input    (path=path, logger=self._logger_verbose)
        ISUtils.check_extension(path=path, logger=self._logger_verbose, ext=VideoFile.VIDEO_EXT)

        self._path: str = path
        self._metadata: VideoFile.VideoMetadata = VideoFile.VideoMetadata.from_video_path(path=self.path, logger=self._logger, verbose=verbose)

        self._video_capture = cv.VideoCapture(self.path)

        if not self._video_capture.isOpened():
            self._logger.handle_error(msg=f"Failed to open video stream at {self.path}", exception=FileNotFoundError)
    
    # --- PROPERTIES ---

    @property    
    def path(self) -> str: return self._path

    @property
    def metadata(self) -> VideoFile.VideoMetadata: return self._metadata

    @property
    def name(self) -> str: return PathUtils.get_folder_and_file(path=self.path)

    # --- MAGIC METHODS ---

    def __str__(self) -> str:

        w, h = self.metadata.size
        
        return f"VideoStream[{self.name}; "\
            f"frames: {len(self)}; "\
            f"size: {w}x{h} pixels]"
    
    def __repr__(self) -> str: return str(self)

    def __len__(self) -> int: return self.metadata.frames
    """ Return the number of frames in the video. """

    def __iter__(self) -> Iterable[NDArray]: return self.iter_range(start=0, end=len(self))
    """ Iterate over all frames in the video. """
        
    def __getitem__(self, idx: int) -> NDArray:
        """ Get a specific frame from the video. """

        # Read the frame
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._video_capture.read()

        if ret: return frame
        else:   self._logger.handle_error(msg=f"Error: Unable to read frame at index {idx}.", exception=IndexError)
    
    # --- STREAMING ---

    def iter_range(self, start: int, end: int, step: int = 1) -> Iterable[NDArray]:
        """ Iterate over a specific range of frames in the video. """
        
        self._video_capture.set(cv.CAP_PROP_POS_FRAMES, start)
        
        for frame in range(start, end, step):

            ret, frame = self._video_capture.read()
            if not ret: break
            yield frame
    
    def play(
        self, 
        start      : int                    = 0,
        end        : int             | None = None, 
        skip_frames: int                    = 1,
        window_size: Tuple[int, int] | None = None
    ):
        ''' 
        Stream the video from start to end frame. 
        It is possible to resize the window by specifying the `window_size` parameter.
        '''
        
        end_: int = default(end, len(self))

        # Resizable windows
        if window_size is not None:
            w, h = window_size
            cv.namedWindow(self.name, cv.WINDOW_NORMAL)
            cv.resizeWindow(self.name, w, h)

        # Compute delay based on video FPS (1000 ms = 1 s)
        delay = int(1000 / self.metadata.fps)
        
        for frame_id, frame in enumerate(self.iter_range(start=start, end=end_, step=skip_frames)):

            frame_: NDArray = self._process_frame(frame=frame, frame_id=frame_id)

            cv.imshow(self.name, frame_)
            if cv.waitKey(delay) & 0xFF == self.EXIT_KEY: break

        cv.destroyAllWindows()
    
    def _process_frame(self, frame: NDArray, frame_id: int) -> NDArray:
        """
        Process a frame before displaying it.
        NOTE: This method can be overridden by subclasses to apply custom processing.
        """

        return frame


@dataclass
class CameraCalibration:
    ''' Dataclass to store camera calibration coefficients. '''

    K                : NDArray        # 3x3 Intrinsic Camera Matrix
    distortion_coeffs: NDArray        # 1x5 Distortion Coefficients
    params           : Dict[str, Any] # Camera Calibration Hyperparameters

    @classmethod
    def from_points(
        cls,
        obj_points: List[NDArray],
        img_points: List[NDArray],
        size      : Tuple[int, int],
        params    : Dict[str, Any] | None = None,
        logger    : BaseLogger            = SilentLogger()
    ) -> 'CameraCalibration':
        ''' Calibrates the camera using object and image points. '''

        params_ : Dict = default(params, {})

        # Check if the number of object and image points are equal
        if len(obj_points) != len(img_points):
            logger.handle_error(
                msg=f"Number of object points and image points must be equal. Got {len(obj_points)} and {len(img_points)}",
                exception=ValueError
            )

        logger.info(msg=f"Starting camera calibration for {len(obj_points)} samples ...")
        timer = Timer()

        # Calibrate the camera
        ret, K, distortion_coeffs, _, _ = cv.calibrateCamera( # type: ignore
            obj_points, img_points, size, None, None                         # type: ignore
        )

        if not ret: logger.handle_error(msg="Camera calibration failed. ", exception=RuntimeError)
        else:
            logger.info(msg=f"Camera calibration completed in {timer} with calibration error: {ret} pixels.")
            if ret > 1: logger.warning(msg="Calibration error is too high. Consider recalibrating the camera.")

        return cls(
            K=K,
            distortion_coeffs=distortion_coeffs,
            params=params_ | {"reprojection_error": ret},
        )

    @classmethod
    def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'CameraCalibration':
        ''' Load camera calibration from a pickle file. '''

        with open(path, 'rb') as f: return pickle.load(f)

    def __str__(self) -> str:

        # Camera Matrix
        K_str = "Intrisic Camera Matrix\n"
        col_widths = [max(len(f"{row[i]:.6f}") for row in self.K) for i in range(len(self.K[0]))]
        for row in self.K:
            K_str += " | ".join(f"{val:>{col_widths[i]}.6f}" for i, val in enumerate(row)) + "\n"

        # Distortion Coefficients
        dist_str = "Distortion Coefficients\n"
        dist_str += " | ".join(f"{val:.6f}" for val in self.distortion_coeffs[0]) + "\n"

        # Mean Reprojection Error
        error_str = f"Mean Pixel Error: {self.params.get('reprojection_error', None)}\n"

        return f"{K_str}\n{dist_str}\n{error_str}"

    def __repr__(self) -> str: return str(self)

    def dump(
        self,
        path   : str,
        logger : BaseLogger = SilentLogger(),
        verbose: bool       = False
    ) -> None:
        ''' Save the camera calibration to a pickle file. '''

        logger_verbose = logger if verbose else SilentLogger()

        ISUtils.check_output(path=PathUtils.get_folder_path(path=path), logger=logger_verbose)

        logger.info(msg=f"Saving camera calibration to {path}")

        with open(path, 'wb') as f:
            pickle.dump(self, f)
