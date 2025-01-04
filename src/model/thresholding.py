from typing import Any, Dict
import cv2 as cv
from cv2.typing import Size

from src.model.calibration import CalibratedVideoStream, CameraCalibration
from src.utils.io_ import SilentLogger
from src.model.typing import Frame
from src.utils.io_ import BaseLogger

class ThresholdedVideoStream(CalibratedVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration, 
        name        : str        = '',
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool       = False
    ):

        super().__init__(path=path, calibration=calibration, name=name, logger=logger, verbose=verbose)

    @property
    def _str_name(self) -> str: return 'ThresholdedVideoStream'

    def _process_frame(self, frame: Frame, frame_id: int) -> Views:

        frame_dict = super()._process_frame(frame=frame, frame_id=frame_id)

        if len(frame.shape) == 3: frame_g = cv.cvtColor(frame_dict['calibrated'], cv.COLOR_BGR2GRAY)
        else:                     frame_g = frame_dict['calibrated']

        frame_b = self._binarize(frame_g=frame_g)

        return {'grayscale': frame_g} | frame_b | frame_dict

    def _binarize(self, frame_g: Frame) -> Views:
        ''' Binarize a grayscale frame. '''

        raise NotImplementedError('Method _binarize must be implemented in derived classes.')

class BaseThresholdedVideoStream(ThresholdedVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration, 
        t           : int,
        kernel_size : Size | None = None,
        name        : str         = '', 
        logger      : BaseLogger  = SilentLogger(),
        verbose     : bool        = False
    ):

        super().__init__(path=path, calibration=calibration, name=name, logger=logger, verbose=verbose)
        self._t = t
        self._kernel_size = kernel_size

    @property
    def _str_name(self) -> str: return 'BaseThresholding'

    @property
    def _str_params(self) -> Dict[str, Any]: 

        if self._kernel_size: w, h = self._kernel_size

        return super()._str_params |\
            {'t': self._t} |\
            {'gaussian kernel size': f'{w}x{h}' if self._kernel_size is not None else {}}

    def _binarize(self, frame_g: Frame) -> Views:

        # Apply Gaussian blur
        if self._kernel_size is None: 
            frame_blur = frame_g
            blur_dict  = {}
        else:
            frame_blur = cv.GaussianBlur(frame_g, self._kernel_size, 0)
            blur_dict  = {'blurred': frame_blur}

        # Apply thresholding
        _, frame_b = cv.threshold(
            frame_blur,       # source grayscale input image
            self._t,          # threshold value
            255,              # value to assign to pixels exceeding the threshold
            cv.THRESH_BINARY  # binary threshold
        )

        return blur_dict | {'binary': frame_b}

class OtsuThresholdedVideoStream(ThresholdedVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration, 
        kernel_size : Size | None = None,
        name        : str        = '', 
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool       = False
    ):

        super().__init__(path=path, calibration=calibration, name=name, logger=logger, verbose=verbose)

        self._kernel_size = kernel_size

    @property
    def _str_name(self) -> str: return 'GaussianOtsuThresholding'

    @property
    def _str_params(self) -> Dict[str, Any]: 
        
        if self._kernel_size: w, h = self._kernel_size

        return super()._str_params |\
            {'gaussian kernel size': f'{w}x{h}' if self._kernel_size is not None else {}}

    def _binarize(self, frame_g: Frame) -> Views:

        # Apply Gaussian blur
        if self._kernel_size is None: 
            frame_blur = frame_g
            blur_dict  = {}
        else:
            frame_blur = cv.GaussianBlur(frame_g, self._kernel_size, 0)
            blur_dict  = {'blurred': frame_blur}

        # Otsu thresholding
        thresh, frame_otsu = cv.threshold(
            frame_blur,                        # source grayscale input image
            0,                                 # ignored in Otsu's method
            255,                               # value to assign to pixels exceeding the threshold
            cv.THRESH_BINARY + cv.THRESH_OTSU  # binary threshold
        )

        return blur_dict | {'binary': frame_otsu}


class TopHatOtsuThresholdedVideoStream(ThresholdedVideoStream):

    def __init__(
        self, 
        path         : str, 
        calibration  : CameraCalibration,
        kernel_size  : Size,
        kernel_shape : int = cv.MORPH_ELLIPSE,
        name         : str = '',
        logger       : BaseLogger = SilentLogger(),
        verbose      : bool = False
    ):
        super().__init__(path, calibration, name, logger, verbose)
        self._kernel_size  = kernel_size
        self._kernel_shape = kernel_shape
    
    @property
    def _str_name(self) -> str: return 'TopHatOtsuThresholding'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | {'top hat kernel size': f'{self._kernel_size[0]}x{self._kernel_size[1]}'}
    
    def _binarize(self, frame_g: Frame) -> Views:

        # Apply top hat transform
        kernel   = cv.getStructuringElement(shape=self._kernel_shape, ksize=self._kernel_size)
        frame_th = cv.morphologyEx(src=frame_g, op=cv.MORPH_TOPHAT, kernel=kernel, iterations=1)

        # Otsu thresholding
        tresh, frame_otsu = cv.threshold(
            frame_th,                          # source grayscale input image
            0,                                 # ignored in Otsu's method
            255,                               # value to assign to pixels exceeding the threshold
            cv.THRESH_BINARY + cv.THRESH_OTSU  # binary threshold
        )

        return {'top-hat': frame_th, 'binary': frame_otsu}


class AdaptiveThresholdedVideoStream(ThresholdedVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration, 
        block_size  : int,
        c           : int,
        name        : str        = '', 
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool       = False
    ):

        super().__init__(path=path, calibration=calibration, name=name, logger=logger, verbose=verbose)

        self._block_size = block_size # size of the local region
        self._c          = c          # constant subtracted from the mean

    @property
    def _str_name(self) -> str: return 'AdaptiveThresholding'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | {'block size': self._block_size, 'c': self._c}

    def _binarize(self, frame_g: Frame) -> Views:

        # Apply adaptive Thresholded
        frame_adaptive = cv.adaptiveThreshold(
            frame_g,                    # source grayscale input image
            255,                        # value to assign to pixels exceeding the threshold
            cv.ADAPTIVE_THRESH_MEAN_C,  # adaptive method
            cv.THRESH_BINARY,           # binary threshold
            self._block_size,           # size of the local region
            self._c                     # constant subtracted from the mean
        )

        return {'binary': frame_adaptive}


class AdaptiveThresholdedPlusClosingVideoStream(AdaptiveThresholdedVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration, 
        block_size  : int,
        c           : int,
        kernel_size : Size,
        kernel_shape: int = cv.MORPH_ELLIPSE,
        name        : str        = '', 
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool       = False
    ):
        
        super().__init__(path=path, calibration=calibration, block_size=block_size, c=c, name=name, logger=logger, verbose=verbose)

        self._kernel_size  = kernel_size
        self._kernel_shape = kernel_shape
    
    @property
    def _str_name(self) -> str: return 'AdaptiveThresholdingPlusClosing'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | {'closing kernel size': f'{self._kernel_size[0]}x{self._kernel_size[1]}'}

    def _binarize(self, frame_g: Frame) -> Views:

        # 1. Apply adaptive Thresholded
        frame_dict = super()._binarize(frame_g=frame_g)

        # 2. Apply morphological closing
        kernel        = cv.getStructuringElement(shape=self._kernel_shape, ksize=self._kernel_size)
        frame_closing = cv.morphologyEx(src=frame_dict['binary'], op=cv.MORPH_CLOSE, kernel=kernel, iterations=1)

        return {'binary': frame_closing} | {'adaptive': frame_dict['binary']}
