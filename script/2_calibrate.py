'''
This script calibrates a camera using a chessboard pattern captured in a video.

The calibration process requires 
    - A specified number of samples, which are extracted from the video at equidistant intervals (parameter `SAMPLES`).
    - The chessboard size, which is the number of internal corners in the chessboard pattern (parameter `CHESSBOARD_SIZE`).

The calibration results include:

    1. The 3x3 intrinsic camera matrix.
    2. The 5 distortion coefficients.
    3. Additional information about the calibration process.

The calibration data is saved in a .pkl file within the `calibration` directory.
'''

import os

from src.model.preprocessing import ChessboardCameraCalibrator
from src.utils.calibration import UndistortedVideoStream
from src.utils.misc import Timer
from src.utils.io_ import IOUtils, FileLogger
from src.utils.settings import (
    CALIBRATION_DIR, CALIBRATION_FILE, CALIBRATION_PATH, 
    CAMERA_2, CAMERA_2_RAW_PATH, EXP_NAME, 
    SAMPLES, CHESSBOARD_SIZE
)

CAMERA_2_WINSIZE = (576, 324)

def main():

    # Output directory
    IOUtils.make_dir(path=CALIBRATION_DIR)
    logger = FileLogger(file=os.path.join(CALIBRATION_DIR, f'{CAMERA_2}_calibration.log'))
    logger.info(
        msg=f'Saving calibration data for experiment {EXP_NAME} '
            f'of camera {CAMERA_2} to {CALIBRATION_DIR} . \n'
    )

    # Calibration object
    timer = Timer()
    logger.info(msg='PREPARING CALIBRATION')
    logger.info(msg=f'Creating video camera calibrator from chessboard video. ')
    calibrator = ChessboardCameraCalibrator(
        path            = CALIBRATION_PATH,
        chessboard_size = CHESSBOARD_SIZE,
        samples         = SAMPLES,
        logger          = logger,
    )
    logger.info(msg=f'{calibrator}\n')

    # Calibration
    logger.info(msg='PERFORMING CAMERA CALIBRATION')
    camera_calibration = calibrator.calibrate(window_size=CAMERA_2_WINSIZE)
    logger.info(msg='\nCamera calibration results: ')
    logger.info(msg=f'{camera_calibration}')
    logger.info(msg=f'Mean Pixel Error: {camera_calibration.info["reprojection_error"]}\n')


    # Saving results
    logger.info(msg='SAVING CAMERA CALIBRATION')
    camera_calibration.dump(path=CALIBRATION_FILE, logger=logger)
    logger.info(msg=f'\nCalibration completed in {timer}. \n')

    # Playing Distorted VS Undistorted video
    logger.info(msg='PLAYING DISTORTED VS UNDISTORTED VIDEO')
    timer.reset()
    undistorted_video = UndistortedVideoStream(path=CAMERA_2_RAW_PATH, calibration=camera_calibration, logger=logger)
    undistorted_video.play(window_size=CAMERA_2_WINSIZE)
    logger.info(msg=f'Video played in {timer}. \n')

if __name__ == '__main__': main()