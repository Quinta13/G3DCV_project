import os

from dotenv import load_dotenv

import numpy as np

from src.model.marker import MarkerDetector
from src.model.mlic import DynamicCameraVideoStream, StaticCameraVideoStream, MLICCollector
from src.model.thresholding import AdaptiveThresholding, OtsuThresholding
from src.utils.io_ import FileLogger, IOUtils, VideoFile
from src.utils.calibration import CalibratedCamera
from src.utils.settings import CALIBRATION_FILE, CAMERA_1_PATH, CAMERA_2_PATH, MLIC_DIR, MLIC_SIZE, LIGHT_POSITION_METHOD


CALIBRATED_1  = CalibratedCamera.trivial_calibration(size=VideoFile(path=CAMERA_1_PATH).metadata.size)
CALIBRATED_2  = CalibratedCamera.from_pickle(path=CALIBRATION_FILE)

THRESHOLD_1 =  OtsuThresholding()
THRESHOLD_2 =  AdaptiveThresholding(block_size=161, c=15)

DETECTOR = MarkerDetector(
    white_thresh=230,
    black_thresh=25,
    min_contour_area=200,
    max_contour_area=float(np.prod(VideoFile(path=CAMERA_1_PATH).metadata.size) * 0.5) 
)

SKIP_FRAMES      = 1
PLOT_HISTORY     = False
SCALE            = 0.5
CAMERA_1_WINSIZE = (360, 640)
CAMERA_2_WINSIZE = (640, 360)
SQUARE_WINSIZE   = (256, 256)

def main():

    suffix = f'{LIGHT_POSITION_METHOD}_{MLIC_SIZE}'

    logger = FileLogger(file=os.path.join(MLIC_DIR, f'mlic_{suffix}.log'))

    # Output directory
    IOUtils.make_dir(path=MLIC_DIR)
    logger.info(msg=f'Saving Multi-Light Image Collection for experiment {MLIC_DIR} to {MLIC_DIR} .')
    logger.info(msg=f'')

    # Load video streams
    logger.info(msg=f'LOADING VIDEO STREAMS.')
    logger.info(f'Using Marker Detector: {DETECTOR}')
    logger.info('')

    mlic_static = StaticCameraVideoStream(
        path=CAMERA_1_PATH, 
        calibration=CALIBRATED_1, 
        thresholding=THRESHOLD_1, 
        marker_detector=DETECTOR, 
        mlic_size=MLIC_SIZE, 
        logger=logger
    )
    logger.info(f'Static video stream: {mlic_static}\n')
	
    mlic_dynamic = DynamicCameraVideoStream(
        path=CAMERA_2_PATH, 
        calibration=CALIBRATED_2, 
        thresholding=THRESHOLD_2,
        marker_detector=DETECTOR, 
        method=LIGHT_POSITION_METHOD,
        logger=logger,
		plot_history=PLOT_HISTORY
    )
    logger.info(f'Dynamic video stream: {mlic_dynamic}\n')

    # Synchronize streams
    mlic_collector = MLICCollector(mlic_static=mlic_static, mlic_dynamic=mlic_dynamic, logger=logger)
	
    logger.info('STARTING MLIC COLLECTION.')
    mlic = mlic_collector.collect(skip_frames=SKIP_FRAMES, win_rect=(CAMERA_1_WINSIZE, CAMERA_2_WINSIZE), win_square=SQUARE_WINSIZE)
    
    logger.info('SAVING MLIC')
    out_mlic = os.path.join(MLIC_DIR, f'mlic_{suffix}.pkl')
    mlic.dump(out_mlic, logger=logger)
	
if __name__ == '__main__': main()