'''
This file contains the class to perform Reflectance Transformation Imaging (RTI) of an object using 
fitted interpolation basis images and a Multi-Light Image Collection (MLIC).
'''

from typing import Tuple

import cv2 as cv
import numpy as np

from src.model.geom import LightDirection
from src.model.interpolation import MLICPixelsBasisCollection
from src.model.mlic import MultiLightImageCollection
from src.model.typing import Size2D, Frame
from src.utils.io_    import BaseLogger, SilentLogger

class InteractiveReflectanceTransformationImaging:
    '''
    Class to play a demo of reflectance transformation imaging of an object using a Multi-Light Image Collection (MLIC) and a collection of basis images.
    The basis collection is used to interpolate the object's reflectance under different lighting conditions and reconstruct the frame luminance (Y channel).
    The Multi-Light Image Collection is used to add the UV channels to the frame, so that it can be displayed in color.
    '''

    EXIT_KEY = 'q'

    def __init__(self, 
        bi_collection           : MLICPixelsBasisCollection,
        mlic                    : MultiLightImageCollection,
        frame_size              : int                 = 500, 
        initial_light_direction : Tuple[float, float] = (0, 0),
        logger                  : BaseLogger          = SilentLogger()
    ):
        '''
        Function to initialize the RealTimeIllumination object.

        :param bi_collection: The collection of basis images used to interpolate the object's reflectance under different lighting conditions.
        :param mlic: The Multi-Light Image Collection used to add the UV channels to the frame, so that it can be displayed in color.
        :param frame_size: The size of the frame to display the object and the light direction arrow.
        :param initial_light_direction: The initial light direction arrow coordinates.
        :param logger: The logger object to log messages.
        '''
        
        # Check if the basis collection output shape match the MLIC size
        if bi_collection.out_shape != mlic.size:
            self._logger.handle_error(
                msg=f'Invalid collection shape {bi_collection.out_shape} for MLIC of size {mlic.size}. ',
                exception=ValueError
            )

        self._bi_collection   : MLICPixelsBasisCollection = bi_collection
        self._mlic            : MultiLightImageCollection = mlic
        self._frame_size      : int                       = frame_size
        self._light_direction : LightDirection            = LightDirection.from_tuple(initial_light_direction)
        self._update          : bool                      = True
        self._logger          : BaseLogger                = logger

    def get_frames(self) -> Tuple[Frame, Frame]:
        '''
        Uses the current light directions to draw the light direction arrow and reconstruct the object frame.
        '''

        # Draw the light direction arrow and the object frame
        light_direction_frame = LightDirection.draw_line_direction(light_direction=self.light_direction, frame_side=self._frame_size)

        # Reconstruct the object frame
        object_frame_y = self._bi_collection.get_frame(light_direction=self.light_direction)
        object_frame = self._mlic.add_uv_channels(y_frame=object_frame_y)

        return light_direction_frame, object_frame
    
    # --- PROPERTIES ---

    @property
    def light_direction(self) -> LightDirection: return self._light_direction

    @light_direction.setter
    def light_direction(self, value: Tuple[float, float]):
        self._light_direction = LightDirection(*value)
        self._update = True  # When a new light direction is set, set the update flag on

    # --- STREAM METHODS ---

    def mouse_callback(self, event, x: float, y: float, flags, param) :
        ''' Get the mouse position and update the light direction arrow. '''

        if event in [cv.EVENT_LBUTTONDOWN, cv.EVENT_MOUSEMOVE] and flags & cv.EVENT_FLAG_LBUTTON:

            # Normalize the mouse coordinates in [-1, 1]
            half = self._frame_size // 2
            norm_x = (x - half) / half
            norm_y = (y - half) / half

            # Normalize the vector in the unit circle
            distance = np.sqrt(norm_x**2 + norm_y**2)
            if distance > 1.0:
                norm_x /= distance
                norm_y /= distance
            
            self.light_direction = (norm_x, -norm_y)  # NOTE: -y to invert the y-axis and match the image coordinates

    def play(self, window_size: Size2D = (500, 500)) -> None:
        ''' Play the real time illumination demo. '''

        self._logger.info("Starting RTI object viewer...")

        try: 

            cv.namedWindow('light_direction', cv.WINDOW_NORMAL); cv.resizeWindow('light_direction', *window_size)
            cv.namedWindow('rti_object',      cv.WINDOW_NORMAL); cv.resizeWindow('rti_object',      *window_size)
            cv.setMouseCallback('light_direction', self.mouse_callback)

            while True:
            
                # Update frames is light direction has changed
                if self._update:
                    ld_frame, obj_frame = self.get_frames()
                    self._update = False

                # Display the frames
                cv.imshow('light_direction', cv.cvtColor(ld_frame,  cv.COLOR_RGB2BGR))
                cv.imshow('rti_object',      cv.cvtColor(obj_frame, cv.COLOR_RGB2BGR))

                # Check for the exit key
                key = cv.waitKey(1)
                if key == ord(self.EXIT_KEY):
                    break

            # Cleanup
            self._logger.info("Exiting RTI object viewer...")
            cv.destroyAllWindows()
        
        except Exception as e:
            cv.destroyAllWindows()
            raise e