from typing import Tuple

import cv2 as cv
import numpy as np

from src.model.interpolation import BasisInterpolationCollection as BICollection
from src.model.mlic import MultiLightImageCollection, DynamicCameraVideoStream
from src.utils.typing import Size2D, Frame
from src.utils.io_    import BaseLogger, SilentLogger

class RTIObjectViewer:

    EXIT_KEY = 'q'

    def __init__(self, 
        bi_collection : BICollection,
        mlic          : MultiLightImageCollection,
        frame_size    : int                 = 500, 
        initial_coord : Tuple[float, float] = (0, 0),
        logger        : BaseLogger          = SilentLogger()
    ):
        
        if bi_collection.out_shape != mlic.size:
            self._logger.handle_error(
                msg=f'Invalid collection shape {bi_collection.out_shape} for MLIC of size {mlic.size}. ',
                exception=ValueError
            )

        self._bi_collection : BICollection        = bi_collection
        self._mlic          : MultiLightImageCollection                = mlic
        self._frame_size    : int                 = frame_size
        self._coord         : Tuple[float, float] = initial_coord
        self._draw_arrow    : bool                = True
        self._logger        : BaseLogger          = logger

    def draw_arrow(self) -> Tuple[Frame, Frame]:

        cx, cy = self.coord
        light_direction_frame = DynamicCameraVideoStream.draw_line_direction(light_direction=(cx, cy), frame_side=self._frame_size)
        object_frame_y = self._bi_collection.get_interpolation_frame(coord=self.coord)
        object_frame = self._mlic.add_uv_channels(y_frame=object_frame_y)

        return light_direction_frame, object_frame

    @property
    def coord(self) -> Tuple[float, float]: return self._coord

    @coord.setter
    def coord(self, value: Tuple[float, float]):
        self._coord = value
        self._draw_arrow = True

    def mouse_callback(self, event, x: float, y: float, flags, param) :

        if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_MOUSEMOVE] and flags & cv.EVENT_FLAG_LBUTTON:

            half = self._frame_size // 2
            norm_x = (x - half) / half
            norm_y = (y - half) / half

            distance = np.sqrt(norm_x**2 + norm_y**2)
            if distance > 1.0:
                norm_x /= distance
                norm_y /= distance
            
            self.coord = (norm_x, -norm_y)

    def play(self, window_size: Size2D = (500, 500)) -> None:

        self._logger.info("Starting RTI object viewer...")

        try: 

            cv.namedWindow('light_direction', cv.WINDOW_NORMAL); cv.resizeWindow('light_direction', *window_size)
            cv.namedWindow('rti_object',      cv.WINDOW_NORMAL); cv.resizeWindow('rti_object',      *window_size)

            cv.setMouseCallback('light_direction', self.mouse_callback)

            while True:
            
                if self._draw_arrow:
                    
                    # Draw the frame with the updated arrow
                    ld_frame, obj_frame = self.draw_arrow()
                    self._draw_arrow = False

                # Show the frame
                cv.imshow('light_direction', cv.cvtColor(ld_frame,  cv.COLOR_RGB2BGR))
                cv.imshow('rti_object',      cv.cvtColor(obj_frame, cv.COLOR_RGB2BGR))

                # Check for the 'q' key to quit
                key = cv.waitKey(1)
                if key == ord(self.EXIT_KEY):
                    break

            # Cleanup
            self._logger.info("Exiting RTI object viewer...")
            cv.destroyAllWindows()
        
        except Exception as e:
            cv.destroyAllWindows()
            raise e