from typing import Any, Dict
import cv2 as cv
from cv2.typing import Size
from numpy.typing import NDArray

from src.model.calibration import CalibratedVideoStream
from src.model.calibration import CameraCalibration
from src.utils.io_ import BaseLogger, SilentLogger

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

    def _process_frame(self, frame: NDArray, frame_id: int) -> NDArray:

        if len(frame.shape) == 3: frame_g = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:                     frame_g = frame

        frame_b = self._binarize(frame_g=frame_g)
    
        return super()._process_frame(frame=frame_b, frame_id=frame_id)

    def _binarize(self, frame_g: NDArray) -> NDArray:
        ''' Binarize a grayscale frame. '''

        raise NotImplementedError('Method _binarize must be implemented in derived classes.')

class OtsuThresholdingVideoStream(ThresholdedVideoStream):

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
    def _str_name(self) -> str: return 'OtsuTresholding'
    
    def _binarize(self, frame_g: NDArray) -> NDArray:
        
        # Apply Otsu thresholding
        tresh, frame_b = cv.threshold(
            frame_g,                           # source grayscale input image
            0,                                 # ignored in Otsu's method
            255,                               # value to assign to pixels exceeding the threshold
            cv.THRESH_BINARY + cv.THRESH_OTSU  # binary threshold combined with Otsu's automatic calculation
        )

        return frame_b

class GaussianOtsuTresholdingVideoStream(OtsuThresholdingVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration, 
        kernel_size : Size,
        name        : str        = '', 
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool       = False
    ):

        super().__init__(path=path, calibration=calibration, name=name, logger=logger, verbose=verbose)

        self._kernel_size = kernel_size

    @property
    def _str_name(self) -> str: return 'GaussianOtsuTresholding'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | {'gaussian kernel size': f'{self._kernel_size[0]}x{self._kernel_size[1]}'}
    
    def _binarize(self, frame_g: NDArray) -> NDArray:
        
        # Apply Gaussian blur
        frame_b = cv.GaussianBlur(
            frame_g,             # source grayscale input image
            self._kernel_size, # kernel size
            0                  # standard deviation in X direction, 0 means it is calculated from kernel size
        )

        # Apply Otsu thresholding
        return super()._binarize(frame_g=frame_b)

class TopHatOtsuTresholdingVideoStream(OtsuThresholdingVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration,
        kernel_size : Size,
        name        : str = '',
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool = False
    ):
        super().__init__(path, calibration, name, logger, verbose)
        self._kernel_size = kernel_size
    
    @property
    def _str_name(self) -> str: return 'TopHatOtsuTresholding'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | {'top hat kernel size': f'{self._kernel_size[0]}x{self._kernel_size[1]}'}

    def _binarize(self, frame_g: NDArray) -> NDArray:

        # Apply top hat transform
        kernel  = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=self._kernel_size)
        frame_b = cv.morphologyEx(src=frame_g, op=cv.MORPH_TOPHAT, kernel=kernel)

        # Apply Otsu thresholding
        return super()._binarize(frame_g=frame_b)

class AdaptiveThresholdingVideoStream(ThresholdedVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration, 
        block_size  : int,
        c           : int        = 2,
        name        : str        = '', 
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool       = False
    ):

        super().__init__(path=path, calibration=calibration, name=name, logger=logger, verbose=verbose)

        self._block_size = block_size # size of the local region
        self._c         = c           # constant subtracted from the mean

    @property
    def _str_name(self) -> str: return 'AdaptiveThresholding'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | {'block size': self._block_size, 'c': self._c}

    def _binarize(self, frame_g: NDArray) -> NDArray:

        # Apply adaptive thresholding
        frame_b = cv.adaptiveThreshold(
            frame_g,                   # source grayscale input image
            255,                       # value to assign to pixels exceeding the threshold
            cv.ADAPTIVE_THRESH_MEAN_C, # adaptive method
            cv.THRESH_BINARY,          # binary threshold
            self._block_size,          # size of the local region
            self._c                    # constant subtracted from the mean
        )

        return frame_b

class AdaptiveThresholdingPlusOpeningVideoStream(AdaptiveThresholdingVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration, 
        block_size  : int,
        kernel_size : Size,
        c           : int        = 2,
        name        : str        = '', 
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool       = False
    ):
        
        super().__init__(path=path, calibration=calibration, block_size=block_size, c=c, name=name, logger=logger, verbose=verbose)

        self._kernel_size = kernel_size
    
    @property
    def _str_name(self) -> str: return 'AdaptiveThresholdingPlusOpening'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | {'opening kernel size': f'{self._kernel_size[0]}x{self._kernel_size[1]}'}

    def _binarize(self, frame_g: NDArray) -> NDArray:

        # 1. Apply adaptive thresholding
        frame_b = super()._binarize(frame_g=frame_g)

        # 2. Apply morphological dilation
        kernel  = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=self._kernel_size)
        frame_d = cv.dilate(src=frame_b, kernel=kernel, iterations=1)

        return frame_d

class AdaptiveThresholdingPlusMedianFilterVideoStream(AdaptiveThresholdingVideoStream):

    def __init__(
        self, 
        path        : str, 
        calibration : CameraCalibration, 
        block_size  : int,
        kernel_side : int,
        c           : int        = 2,
        name        : str        = '', 
        logger      : BaseLogger = SilentLogger(),
        verbose     : bool       = False
    ):
        
        super().__init__(path=path, calibration=calibration, block_size=block_size, c=c, name=name, logger=logger, verbose=verbose)

        self._kernel_side = kernel_side
    
    @property
    def _str_name(self) -> str: return 'AdaptiveThresholdingPlusMedianFilter'

    @property
    def _str_params(self) -> Dict[str, Any]: return super()._str_params | {'kernel side': self._kernel_side}

    def _binarize(self, frame_g: NDArray) -> NDArray:

        # 1. Apply adaptive thresholding
        frame_b = super()._binarize(frame_g=frame_g)

        # 2. Apply median filter
        frame_d = cv.medianBlur(src=frame_b, ksize=self._kernel_side)

        return frame_d