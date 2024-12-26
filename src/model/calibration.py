import pickle
import cv2 as cv
from dataclasses import dataclass
from typing import Any, Dict, List
from src.model.stream import VideoStream
from src.utils.io_ import BaseLogger, InputSanitizationUtils as ISUtils, PathUtils, SilentLogger


from numpy.typing import NDArray

from src.utils.misc import Size, Timer, default


@dataclass
class CameraCalibration:
    ''' Dataclass to store camera calibration coefficients. '''

    camera_mat       : NDArray        # 3x3 Intrinsic Camera Matrix
    distortion_coeffs: NDArray        # 1x5 Distortion Coefficients
    params           : Dict[str, Any] # Camera Calibration Hyperparameters

    @classmethod
    def from_points(
        cls,
        obj_points: List[NDArray],
        img_points: List[NDArray],
        size      : Size,
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
        ret, camera_mat, distortion_coeffs, _, _ = cv.calibrateCamera( # type: ignore
            obj_points, img_points, size, None, None                         # type: ignore
        )

        if not ret: logger.handle_error(msg="Camera calibration failed. ", exception=RuntimeError)
        else:
            logger.info(msg=f"Camera calibration completed in {timer} with calibration error: {ret} pixels.")
            if ret > 1: logger.warning(msg="Calibration error is too high. Consider recalibrating the camera.")

        return cls(
            camera_mat=camera_mat,
            distortion_coeffs=distortion_coeffs,
            params=params_ | {"reprojection_error": ret},
        )

    @classmethod
    def from_pickle(cls, path: str, logger: BaseLogger = SilentLogger()) -> 'CameraCalibration':
        ''' Load camera calibration from a pickle file. '''

        logger.info(msg=f"Loading camera calibration from {path}")

        with open(path, 'rb') as f: return pickle.load(f)

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
        error_str = f"Mean Pixel Error: {self.params.get('reprojection_error', None)}\n"

        return f"{K_str}\n{dist_str}\n{error_str}"

    def __repr__(self) -> str: return str(self)

    def undistort(self, img: NDArray) -> NDArray:
        ''' Undistort an image using the camera calibration coefficients. '''

        return cv.undistort(img, self.camera_mat, self.distortion_coeffs)

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


class CalibratedVideoStream(VideoStream):

    def __init__(self, path: str, calibration: CameraCalibration, name: str = '', logger: BaseLogger = SilentLogger(), verbose: bool = False):

        super().__init__(path=path, name=name, logger=logger, verbose=verbose)
        self._calibration: CameraCalibration = calibration

    @property
    def _str_name(self) -> str: return 'CalibratedVideoStream'

    def _process_frame(self, frame: NDArray, frame_id: int) -> NDArray:

        # Undistort the frame
        frame_ =  self._calibration.undistort(frame)

        return super()._process_frame(frame=frame_, frame_id=frame_id)