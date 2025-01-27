'''
This file contains utilities for camera calibration and undistortion.
It provides 
    - A dataclass to store camera calibration data (Intrinsic Camera Matrix and Distortion Coefficients).
    - A video stream class that uses the calibration data to undistort the frames. 
'''

import pickle
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import cv2 as cv
from numpy.typing import NDArray

from src.utils.misc import Timer
from src.utils.io_ import (
    SilentLogger, BaseLogger,
    InputSanitizationUtils as ISUtils, PathUtils
)
from src.utils.stream import VideoStream
from src.utils.typing import Views, Frame, Size2D, default

@dataclass
class CalibratedCamera:
    ''' 
    Dataclass to store camera calibration data (intrinsic camera matrix, distortion coefficient, info). 

    The camera calibration can be instantiated in three ways:
        - Using object and image points.
        - Loaded from a pickle file.
        - Trivial calibration using camera size with no distortion.
    '''

    camera_mat       : NDArray         # 3x3 Intrinsic Camera Matrix
    distortion_coeffs: NDArray         # 1x5 Distortion Coefficients
    info             : Dict[str, Any]  # General purpose info dictionary to store additional data about the calibration
    white_mask       : bool = False    # Whether to fill empty pixels with white after undistortion

    # --- INITIALIZERS ---

    @classmethod
    def from_points(
        cls,
        obj_points: List[NDArray],
        img_points: List[NDArray],
        size      : Size2D,
        info      : Dict[str, Any] | None = None,
        logger    : BaseLogger            = SilentLogger()
    ) -> 'CalibratedCamera':
        '''
        Calibrates the camera using object and image points. 

        :param obj_points: List of object points in 3D space.
        :param img_points: List of corresponding image points in 2D space.
        :param size: Size of the image (width, height).
        :param params: Additional information for the calibration.
        :param logger: Logger to display messages.
        '''

        WARN_THRESH = 1  # Warning threshold for the reprojection error

        info_ : Dict = default(info, {})

        # Check if the number of object and image points are equal
        if len(obj_points) != len(img_points):
            logger.handle_error(
                msg=f"Number of object points and image points must be equal."\
                    f"Got {len(obj_points)} object points and {len(img_points)} image points.",
                exception=ValueError
            )

        logger.info(msg=f"Starting camera calibration for {len(obj_points)} samples ...")
        timer = Timer()

        # Calibrate the camera
        ret, camera_mat, distortion_coeffs, _, _ = cv.calibrateCamera(
            obj_points, img_points, size, None, None                    # type: ignore
        )                                                               # type: ignore

        # Process the calibration results
        if not ret: 
            logger.handle_error(msg="Camera calibration failed. ", exception=RuntimeError)
        else:
            logger.info(msg=f"Camera calibration completed in {timer} with calibration error: {ret} pixels. ")
            if ret > WARN_THRESH: 
                logger.warning(msg=f"Calibration error is too high (> {WARN_THRESH}). Consider recalibrating the camera.")

        return cls(
            camera_mat=camera_mat,
            distortion_coeffs=distortion_coeffs,
            info=info_ | {"reprojection_error": ret},
        )
    
    @classmethod
    def trivial_calibration(cls, size: Size2D) -> 'CalibratedCamera':
        '''
        Create a trivial camera calibration using camera size with no distortion.
        
        :param size: Size of the image (width, height).
        '''

        w, h = size
        max_ = max(*size)

        camera_mat = np.array([
            [max_,     0, w // 2],
            [   0,  max_, h // 2],
            [   0,     0,      1]
        ])

        distortion_coeffs = np.zeros((1, 5))

        info = {
            "reprojection_error": None,
            "from_trivial_size" : size
        }

        return cls(camera_mat=camera_mat, distortion_coeffs=distortion_coeffs, info=info)

    @classmethod
    def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'CalibratedCamera':
        ''' Load camera calibration from a pickle file. '''

        ISUtils.check_input(path=path, logger=logger)
        logger.info(msg=f"Loading camera calibration from {path}")

        with open(path, 'rb') as f: return pickle.load(f)
    
    # --- MAGIC METHODS ---

    def __str__(self) -> str:

        # Camera Matrix
        K_str = "Intrisic Camera Matrix\n"
        col_widths = [max(len(f"{row[i]:.6f}") for row in self.camera_mat) for i in range(len(self.camera_mat[0]))]
        for row in self.camera_mat:
            K_str += " | ".join(f"{val:>{col_widths[i]}.6f}" for i, val in enumerate(row)) + "\n"

        # Distortion Coefficients
        dist_str = "Distortion Coefficients\n"
        dist_str += " | ".join(f"{val:.6f}" for val in self.distortion_coeffs[0]) + "\n"

        # Mean Reprojection Error
        error_str = f"Mean Pixel Error: {self.info.get('reprojection_error', None)}\n"

        return f"{K_str}\n{dist_str}\n{error_str}"

    def __repr__(self) -> str: return str(self)

    # --- METHODS ---

    def undistort(self, frame: Frame) -> Frame:
        ''' Undistort an image using the camera calibration coefficients. '''

        # Perform undistortion
        undistorted = cv.undistort(frame, self.camera_mat, self.distortion_coeffs)

        # White masking
        if self.white_mask:

            # Create a white probe image
            probe_img = np.full_like(frame, 255)
            probe_img_undistorted = cv.undistort(probe_img, self.camera_mat, self.distortion_coeffs)

            # Create a mask for empty pixels
            if len(probe_img.shape) == 3: mask = (probe_img_undistorted == 0).all(axis=2)  # For color images
            else:                         mask = (probe_img_undistorted == 0)              # For grayscale images

            # Fill empty pixels with white
            undistorted[mask] = 255

        return undistorted

    def dump(
        self,
        path   : str,
        logger : BaseLogger = SilentLogger()
    ):
        ''' Save the camera calibration to a pickle file. '''

        ISUtils.check_output(path=PathUtils.get_folder_path(path=path), logger=logger)
        logger.info(msg=f"Saving camera calibration to {path}")

        with open(path, 'wb') as f: pickle.dump(self, f)


class CalibratedVideoStream(VideoStream):
    '''
    Video stream using camera calibration distortion parameters to undistort the frames.
    '''

    def __init__(
        self, 
        path        : str, 
        calibration : CalibratedCamera, 
        name        : str | None        = None,
        logger      : BaseLogger        = SilentLogger()
    ):

        super().__init__(path=path, name=name, logger=logger)

        self._calibration: CalibratedCamera = calibration

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        # Process the frame using the parent class
        views = super()._process_frame(frame=frame, frame_id=frame_id)

        # Undistort the frame
        raw_frame = views['raw'].copy()
        frame_undistorted = self._calibration.undistort(frame=raw_frame)

        # Add the undistorted frame to the views
        return views | {'undistorted': frame_undistorted}