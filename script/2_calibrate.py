import os

from src.model.preprocessing import ChessboardCameraCalibrator
from src.utils.calibration import CalibratedVideoStream
from src.utils.misc import Timer
from src.utils.io_ import IOUtils, FileLogger
from src.utils.settings import CALIBRATION_DIR, CALIBRATION_FILE, CALIBRATION_PATH, CAMERA_2, CAMERA_2_RAW_PATH, EXP_NAME, SAMPLES, CHESSBOARD_SIZE

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
    logger.info(msg=f'{camera_calibration}\n')

    # Saving results
    logger.info(msg='SAVING CAMERA CALIBRATION')
    camera_calibration.dump(path=CALIBRATION_FILE, logger=logger)
    logger.info(msg=f'\nCalibration completed in {timer}. \n')

    # Playing Distorted VS Undistorted video
    logger.info(msg='PLAYING DISTORTED VS UNDISTORTED VIDEO')
    timer.reset()
    undistorted_video = CalibratedVideoStream(path=CAMERA_2_RAW_PATH, calibration=camera_calibration, logger=logger)
    undistorted_video.play(window_size=CAMERA_2_WINSIZE)
    logger.info(msg=f'Video played in {timer}. \n')

def __main__(): main()