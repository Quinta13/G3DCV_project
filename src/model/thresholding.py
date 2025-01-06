from abc import ABC, abstractmethod
from typing import Any, Dict
import cv2 as cv
from cv2.typing import Size

from src.model.calibration import CalibratedVideoStream, CameraCalibration
from src.utils.io_ import SilentLogger
from src.model.typing import Frame, Views
from src.utils.io_ import BaseLogger

class Thresholding(ABC):

    def __init__(self): pass

    @abstractmethod
    def __call__(self, frame: Frame) -> Views: 

        if len(frame.shape) == 3: frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:                     frame_g = frame

        return {'grayscale': frame_g}
    
    def __str__(self)  -> str: return f'{self.name}[{"; ".join([f"{k}: {v}" for k, v in self.params.items()])}]'
    def __repr__(self) -> str: return str(self)
    
    @property
    def name(self) -> str: return 'Thresholding'

    @property
    def params(self) -> Dict[str, Any]: return {}
    
class BaseThresholding(Thresholding):

    def __init__(self, t: int, kernel_size: Size | None = None):
        self._t = t
        self._kernel_size = kernel_size

    @property
    def name(self) -> str: return 'BaseThresholding'

    @property
    def params(self) -> Dict[str, Any]: 

        if self._kernel_size: 
            w, h = self._kernel_size
            gaussian_param = {'gaussian kernel size': f'{w}x{h}'}
        else: gaussian_param = {}

        return super().params | {'t': self._t} | gaussian_param
    
    def __call__(self, frame: Frame) -> Views:

        frame_dict = super().__call__(frame=frame)
        gray = frame_dict['grayscale']

        # Apply Gaussian blur
        if self._kernel_size is None:
            frame_blur = gray
            blur_dict = {}
        else:
            frame_blur = cv.GaussianBlur(gray, self._kernel_size, 0)
            blur_dict = {'blurred': frame_blur}
        
        # Apply thresholding
        _, frame_b = cv.threshold(src=frame_blur, thresh=self._t, maxval=255, type=cv.THRESH_BINARY)

        return blur_dict | {'binary': frame_b} | frame_dict

class OtsuThresholding(Thresholding):

    def __init__(self, kernel_size: Size | None = None):
        self._kernel_size = kernel_size

    @property
    def name(self) -> str: return 'OtsuThresholding'

    @property
    def params(self) -> Dict[str, Any]:
        
        if self._kernel_size: 
            w, h = self._kernel_size
            gaussian_param = {'gaussian kernel size': f'{w}x{h}'}
        else: gaussian_param = {}

        return super().params | gaussian_param
    
    def __call__(self, frame: Frame) -> Views:

        frame_dict = super().__call__(frame=frame)
        gray = frame_dict['grayscale']

        # Apply Gaussian blur
        if self._kernel_size is None:
            frame_blur = gray
            blur_dict = {}
        else:
            frame_blur = cv.GaussianBlur(gray, self._kernel_size, 0)
            blur_dict = {'blurred': frame_blur}
        
        # Apply Otsu thresholding
        # NOTE: thresh = 0 is ignored in Otsu's method
        _, frame_b = cv.threshold(src=frame_blur, thresh=0, maxval=255, type=cv.THRESH_BINARY + cv.THRESH_OTSU)

        return blur_dict | {'binary': frame_b} | frame_dict

class TopHatOtsuThresholding(Thresholding):

    def __init__(self, kernel_size: Size, kernel_shape: int = cv.MORPH_ELLIPSE):
        self._kernel_size  = kernel_size
        self._kernel_shape = kernel_shape

    @property
    def name(self) -> str: return 'TopHatOtsuThresholding'

    @property
    def params(self) -> Dict[str, Any]: return {'top hat kernel size': f'{self._kernel_size[0]}x{self._kernel_size[1]}'}

    def __call__(self, frame: Frame) -> Views:

        frame_dict = super().__call__(frame=frame)
        gray = frame_dict['grayscale']

        # Apply top hat transform
        kernel   = cv.getStructuringElement(shape=self._kernel_shape, ksize=self._kernel_size)
        frame_th = cv.morphologyEx(src=gray, op=cv.MORPH_TOPHAT, kernel=kernel, iterations=1)

        # Otsu thresholding
        tresh, frame_b = cv.threshold(frame_th, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        return {'top-hat': frame_th, 'binary': frame_b} | frame_dict
    
class AdaptiveThresholding(Thresholding):

    def __init__(self, block_size: int, c: int):
        self._block_size = block_size
        self._c = c

    @property
    def name(self) -> str: return 'AdaptiveThresholding'

    @property
    def params(self) -> Dict[str, Any]: return {'block size': self._block_size, 'c': self._c}

    def __call__(self, frame: Frame) -> Views:

        frame_dict = super().__call__(frame=frame)
        gray = frame_dict['grayscale']

        # Apply adaptive thresholding
        frame_b = cv.adaptiveThreshold(
            src=gray,
            maxValue=255,
            adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv.THRESH_BINARY,
            C=self._c,
            blockSize=self._block_size
        )

        return {'binary': frame_b} | frame_dict

class AdaptiveThresholdingPlusClosing(AdaptiveThresholding):

    def __init__(self, block_size: int, c: int, kernel_size: Size, kernel_shape: int = cv.MORPH_ELLIPSE):
        super().__init__(block_size=block_size, c=c)
        self._kernel_size  = kernel_size
        self._kernel_shape = kernel_shape

    @property
    def name(self) -> str: return 'AdaptiveThresholdingPlusClosing'

    @property
    def params(self) -> Dict[str, Any]: return super().params | {'closing kernel size': f'{self._kernel_size[0]}x{self._kernel_size[1]}'}

    def __call__(self, frame: Frame) -> Views:
        
        # 1. Apply adaptive Thresholded
        frame_dict = super().__call__(frame=frame)
        adaptive = frame_dict['binary']

        # 2. Apply morphological closing
        kernel        = cv.getStructuringElement(shape=self._kernel_shape, ksize=self._kernel_size)
        frame_closing = cv.morphologyEx(src=adaptive, op=cv.MORPH_CLOSE, kernel=kernel, iterations=1)

        # NOTE: The 'binary' key is overwriting the previous value
        return {'adaptive': adaptive} | frame_dict | {'binary': frame_closing}

class ThresholdedVideoStream(CalibratedVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration,
        thresholding: Thresholding,
        name        : str        = '',
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool       = False
    ):

        super().__init__(path=path, calibration=calibration, name=name, logger=logger, verbose=verbose)

        self._thresholding: Thresholding = thresholding

    @property
    def _str_name(self) -> str: return f'{self._thresholding.name}VideoStream'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | self._thresholding.params

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        frame_dict = super()._process_frame(frame=frame, frame_id=frame_id)
        calibrated_frame = frame_dict['calibrated']
        threhsolding_dict = self._thresholding(frame=calibrated_frame)

        return threhsolding_dict | frame_dict
