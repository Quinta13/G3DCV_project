from dataclasses import dataclass
from typing import List

import numpy as np
import cv2 as cv

from src.model.typing import LightDirection
from src.model.calibration import CameraCalibration
from src.model.marker import MarkerDetectionVideoStream, Marker, MarkerDetector
from src.model.thresholding import Thresholding
from src.model.typing import Frame, Size2D, Views
from src.utils.io_ import BaseLogger, SilentLogger

class MLIC:

    def __init__(self, frames: List[Frame], light_source: List[LightDirection]):

        self._frames       : List[Frame]          = frames
        self._light_source : List[LightDirection] = light_source


class RTIStaticCameraVideoStream(MarkerDetectionVideoStream):

    def __init__(
        self, 
        path            : str, 
        calibration     : CameraCalibration,
        thresholding    : Thresholding,
		marker_detector : MarkerDetector,
        mlic_size       : Size2D,
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

        self._mlic_size: Size2D = mlic_size

    def _process_marker(self, views: Views, marker: Marker) -> Views:

        super_views = super()._process_marker(views=views, marker=marker)

        warped = marker.warp(frame=views['calibrated'].copy(), size=self._mlic_size)

        return super_views | {'warped': warped}

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        super_views = super()._process_frame(frame=frame, frame_id=frame_id)

        return super_views | ({'warped': np.zeros_like(frame)} if 'warped' not in super_views else super_views)


class RTIDynamicCameraVideoStream(MarkerDetectionVideoStream):

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

    @staticmethod
    def _draw_arrow_in_circle(light_direction: LightDirection, image_size: int = 500) -> Frame:

        x, y = light_direction

        # Ensure x, y are within [-1, 1]
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            raise ValueError("x and y must be in the range [-1, 1]")
        
        # Create a black background
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        # Define the circle center and radius
        center = (image_size // 2, image_size // 2)
        radius = image_size // 2  # Circle radius is half of the image size

        # Draw the white circle
        cv.circle(image, center, radius, (255, 255, 255), thickness=4)

        # Compute the arrow endpoint in pixel coordinates
        arrow_x = int(center[0] + x * radius)  # Scale x to the radius
        arrow_y = int(center[1] - y * radius)  # Scale y to the radius (inverted y-axis)

        # Draw the red arrow
        cv.arrowedLine(image, center, (arrow_x, arrow_y), (255, 0, 0), thickness=4, tipLength=0.05)

        return image

    def _process_marker(self, views: Views, marker: Marker) -> Views:

        super_views = super()._process_marker(views=views, marker=marker)

        light_direction = marker.camera_2d_position(calibration=self._calibration)

        direction_frame = self._draw_arrow_in_circle(light_direction=light_direction)

        return super_views | {'light_direction': direction_frame}

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        super_views = super()._process_frame(frame=frame, frame_id=frame_id)

        return super_views | ({'light_direction': np.zeros_like(frame)} if 'light_direction' not in super_views else super_views)