'''
This file contains the main classes to implement thresholding logic on video streams
'''

from abc import ABC, abstractmethod
from typing import Any, Dict
import cv2 as cv

from src.utils.calibration import UndistortedVideoStream, CalibratedCamera
from src.utils.io_ import SilentLogger, BaseLogger
from src.model.typing import Frame, Views, Size2D


class Thresholding(ABC):
    ''' 
    Abstract class to implement a thresholding on a frame. 
    It contains an abstract method `__call__` that first converts the frame to grayscale 
        and then requires the implementation of the thresholding logic in the subclasses.
    '''

    def __init__(self): pass

    def __str__ (self) -> str: return f'{self.__class__.__name__}[{"; ".join([f"{k}: {v}" for k, v in self.params.items()])}]'
    def __repr__(self) -> str: return str(self)

    @property
    def params(self) -> Dict[str, Any]: return {}
    ''' Dictionary of thresholding method parameters. '''

    @abstractmethod
    def __call__(self, frame: Frame) -> Views: 
        '''
        Apply the thresholding logic to the frame and return the views of thresholding processing steps.
        '''

        if len(frame.shape) == 3: frame_g = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        else:                     frame_g = frame

        return {'grayscale': frame_g}

class BaseThresholding(Thresholding):
    ''' 
    Class to implement a simple thresholding on a frame given a value. 
    It optionally applies a Gaussian blur before thresholding.
    '''

    def __init__(self, t: int, gaussian_kernel_size: Size2D | None = None):
        '''
        The class requires a threshold value and optionally a Gaussian kernel size. 
        If not provided, no Gaussian blur is applied.
        '''

        self._t           : int           = t
        self._kernel_size : Size2D | None = gaussian_kernel_size

    @property
    def params(self) -> Dict[str, Any]: 

        if self._kernel_size: 
            w, h = self._kernel_size
            gaussian_param = {'gaussian kernel size': f'{w}x{h}'}
        else: gaussian_param = {}

        return super().params | {'t': self._t} | gaussian_param
    
    def __call__(self, frame: Frame) -> Views:
        ''' Apply thresholding to the frame. '''

        views = super().__call__(frame=frame)
        gray = views['grayscale']

        # Optionally apply Gaussian blur
        if self._kernel_size is None:
            frame_blur = gray
            blur_dict = {}
        else:
            frame_blur = cv.GaussianBlur(gray, self._kernel_size, 0)
            blur_dict = {'blurred': frame_blur}
        
        # Apply thresholding
        _, frame_b = cv.threshold(src=frame_blur, thresh=self._t, maxval=255, type=cv.THRESH_BINARY)

        return views | blur_dict | {'binary': frame_b}

class OtsuThresholding(Thresholding):
    ''' Apply Otsu thresholding to a frame, optionally applying a Gaussian blur before thresholding. '''

    def __init__(self, kernel_size: Size2D | None = None):
        ''' The class optionally applies a Gaussian blur before thresholding. '''

        self._kernel_size = kernel_size

    @property
    def params(self) -> Dict[str, Any]:
        
        if self._kernel_size: 
            w, h = self._kernel_size
            gaussian_param = {'gaussian kernel size': f'{w}x{h}'}
        else: gaussian_param = {}

        return super().params | gaussian_param
    
    def __call__(self, frame: Frame) -> Views:
        ''' Apply Otsu thresholding to the frame. '''

        views = super().__call__(frame=frame)
        gray = views['grayscale']

        # Optionally apply Gaussian blur
        if self._kernel_size is None:
            frame_blur = gray
            blur_dict = {}
        else:
            frame_blur = cv.GaussianBlur(gray, self._kernel_size, 0)
            blur_dict = {'blurred': frame_blur}
        
        # Apply Otsu thresholding
        # NOTE: thresh = 0 is ignored in Otsu's method
        _, frame_b = cv.threshold(src=frame_blur, thresh=0, maxval=255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)

        return views | blur_dict | {'binary': frame_b} 

class TopHatOtsuThresholding(Thresholding):
    ''' Apply top hat transform and Otsu thresholding to a frame. '''

    def __init__(self, se_size: Size2D, se_shape: int = cv.MORPH_ELLIPSE):
        ''' 
        The class requires a structuring element size and shape for the top hat transform.
        The default shape is an ellipse.
        '''

        self._se_size  = se_size
        self._se_shape = se_shape

    @property
    def params(self) -> Dict[str, Any]: 

        w, h = self._se_size
        
        morph_shape: str = {
            cv.MORPH_RECT    : 'rectangle',
            cv.MORPH_CROSS   : 'cross',
            cv.MORPH_ELLIPSE : 'ellipse'
        }[self._se_shape]
        
        return {
            'structuring element shape': morph_shape,
            'structuring element size' : f'{w}x{h}'
        }

    def __call__(self, frame: Frame) -> Views:
        ''' Apply top hat transform and Otsu thresholding to the frame. '''

        views = super().__call__(frame=frame)
        gray = views['grayscale']

        # Apply top hat transform
        kernel   = cv.getStructuringElement(shape=self._se_shape, ksize=self._se_size)
        frame_th = cv.morphologyEx(src=gray, op=cv.MORPH_TOPHAT, kernel=kernel, iterations=1)

        # Otsu thresholding
        # NOTE: thresh = 0 is ignored in Otsu's method
        _, frame_b = cv.threshold(src=frame_th, thresh=0, maxval=255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)

        return views | {'top-hat': frame_th, 'binary': frame_b}

class AdaptiveThresholding(Thresholding):
    ''' Apply adaptive thresholding to a frame. '''

    def __init__(self, block_size: int, c: int):
        ''' The class requires a block size and a constant value for adaptive thresholding. '''

        self._block_size = block_size
        self._c = c

    @property
    def params(self) -> Dict[str, Any]: return {'block size': self._block_size, 'c': self._c}

    def __call__(self, frame: Frame) -> Views:
        ''' Apply adaptive thresholding to the frame. '''

        views = super().__call__(frame=frame)
        gray = views['grayscale']

        # Apply adaptive thresholding
        frame_b = cv.adaptiveThreshold(
            src=gray,
            maxValue=255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv.THRESH_BINARY,
            C=self._c,
            blockSize=self._block_size
        )

        return views | {'binary': frame_b}

class ThresholdedVideoStream(UndistortedVideoStream):
    ''' Video stream with a thresholding method applied to the frames. '''

    def __init__(
        self, 
        path        : str, 
        calibration : CalibratedCamera,
        thresholding: Thresholding,
        name        : str | None = None,
        logger      : BaseLogger = SilentLogger()
    ):
        ''' The class requires the thresholding method to apply to the frames. '''

        super().__init__(path=path, calibration=calibration, name=name, logger=logger)

        self._thresholding: Thresholding = thresholding

    @property
    def _str_name(self) -> str: return f'{self._thresholding.__class__.__name__}VideoStream'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | self._thresholding.params

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:
        ''' Process the frame using the thresholding method. '''

        # Get previous processing steps
        views = super()._process_frame(frame=frame, frame_id=frame_id)
        calibrated_frame = views['undistorted']
        
        # Apply thresholding to the undistorted frame
        thresh_views = self._thresholding(frame=calibrated_frame)

        # Return old views and new thresholding views
        return views | thresh_views
