import os
from dotenv import load_dotenv

from src.model.preprocessing import ChessboardCameraCalibrator
from src.model.calibration import CalibratedVideoStream
from src.utils.misc import Timer
from src.utils.io_ import IOUtils, FileLogger

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', '.')
OUT_DIR  = os.getenv('OUT_DIR', '.')

C_ID = 2

match C_ID:

    case 1:

        CAMERA          = 'cam1-static'
        WINDOW_SIZE     = (324, 576)
        CALIBRATION_EXT = 'mov'
    
    case 2:

        CAMERA          = 'cam2-moving_light'
        WINDOW_SIZE     = (576, 324)
        CALIBRATION_EXT = 'mp4'

    case _:

        raise ValueError(f'Invalid camera id {C_ID}. ')

CALIBRATION_FILE = 'calibration'
EXP_NAME         = 'coin1'

VIDEO_PATH       = os.path.join(DATA_DIR, CAMERA,  f'{CALIBRATION_FILE}.{CALIBRATION_EXT}')
CALIBRATION_DIR  = os.path.join(OUT_DIR, EXP_NAME, f'calibration')
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR,   f'{CAMERA}.pkl')

CHESSBOARD_SIZE = (9, 6)
SAMPLES         = 50

if __name__ == "__main__":

    # Output directory
    IOUtils.make_dir(path=CALIBRATION_DIR)
    logger = FileLogger(file=os.path.join(CALIBRATION_DIR, f'{CAMERA}_calibration.log'))
    logger.info(
        msg=f'Saving synchronization data for experiment {EXP_NAME} '
            f'of camera {CAMERA} to {CALIBRATION_DIR} . '
    )
    logger.info(msg=f'')

    # Calibration object
    timer = Timer()
    logger.info(msg='PREPARING CALIBRATION')
    logger.info(msg=f'Creating video camera calibrator from chessboard video. ')
    calibrator = ChessboardCameraCalibrator(
        path            = VIDEO_PATH,
        chessboard_size = CHESSBOARD_SIZE,
        samples         = SAMPLES,
        logger          = logger,
    )
    logger.info(msg=str(calibrator))
    logger.info(msg='')

    # Calibration
    logger.info(msg='PERFORMING CAMERA CALIBRATION')
    camera_calibration = calibrator.calibrate(window_size=WINDOW_SIZE)
    logger.info(msg='')

    # Saving results
    logger.info(msg='SAVING CAMERA CALIBRATION')
    logger.info(msg=str(camera_calibration))
    camera_calibration.dump(path=CALIBRATION_FILE, logger=logger)
    logger.info(msg='')

    logger.info(msg=f'Calibration completed in {timer}. ')
    logger.info(msg='')

    # Playing Distorted VS Undistorted video
    logger.info(msg='PLAYING DISTORTED VS UNDISTORTED VIDEO')
    timer.reset()
    undistorted_video = CalibratedVideoStream(path=VIDEO_PATH, calibration=camera_calibration, logger=logger)
    undistorted_video.play(window_size=WINDOW_SIZE)
    logger.info(msg=f'Video played in {timer}. ')
    logger.info(msg='')