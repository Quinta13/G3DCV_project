import os
from dotenv import load_dotenv

from src.model.calibration import CalibratedVideoStream
from src.model.stream import VideoStream, SynchronizedVideoStream
from src.utils.misc import Timer
from src.utils.io_ import FileLogger, IOUtils
from src.preprocessing import ChessboardCameraCalibrator

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', '.')
OUT_DIR  = os.getenv('OUT_DIR', '.')

CAMERA           = 'cam2-moving_light'
CALIBRATION_FILE = 'calibration'
CALIBRATION_EXT  = 'mp4'
EXP_NAME         = 'coin1'

VIDEO_PATH       = os.path.join(DATA_DIR, CAMERA,   f'{CALIBRATION_FILE}.{CALIBRATION_EXT}')
CALIBRATION_DIR  = os.path.join(OUT_DIR, EXP_NAME, f'calibration')
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR,    f'{CAMERA}.pkl')

CHESSBOARD_SIZE = (9, 6)
SAMPLES         = 50

WINDOW_SIZE = (576, 324)

if __name__ == "__main__":

    # Output directory
    IOUtils.make_dir(path=CALIBRATION_DIR)
    logger = FileLogger(file=os.path.join(CALIBRATION_DIR, f'{CAMERA}_calibration.log'))
    logger.info(msg=f'Saving synchronization data for experiment {EXP_NAME} '
                f'of camera {CAMERA} to {CALIBRATION_DIR} . ')
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
    distorted_video   = VideoStream          (path=VIDEO_PATH, name='uncalibrated')
    undistorted_video = CalibratedVideoStream(path=VIDEO_PATH, name='calibrated', calibration=camera_calibration)
    sync_video = SynchronizedVideoStream(streams=[distorted_video, undistorted_video], logger=logger)
    sync_video.play(window_size=WINDOW_SIZE)
    logger.info(msg=f'Video played in {timer}. ')
    logger.info(msg='')