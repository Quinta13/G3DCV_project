import os

from dotenv import load_dotenv

import cv2 as cv

from src.model.marker import MarkerDetector
from src.model.mlic import MLICDynamicCameraVideoStream, MLICStaticCameraVideoStream, MLICCollector
from src.model.thresholding import AdaptiveThresholding, BaseThresholding, OtsuThresholding, Thresholding, TopHatOtsuThresholding
from src.utils.io_ import FileLogger, IOUtils, VideoFile
from src.model.calibration import CalibratedCamera
from src.utils.misc import Timer

load_dotenv()

OUT_DIR  = os.getenv('OUT_DIR',  '.')

EXP_NAME   = 'coin1'

CAMERA_1 = 'cam1-static'
CAMERA_2 = 'cam2-moving_light'

CALIBRATION_1 = None
CALIBRATION_2 = os.path.join(OUT_DIR, EXP_NAME, 'calibration', f'{CAMERA_2}.pkl')

BINARY_TYPE_1 = 'otsu'
BINARY_TYPE_2 = 'adaptive'

CAMERA_EXT = 'mp4'

VIDEO_1 = os.path.join(OUT_DIR, EXP_NAME, 'sync', f'{CAMERA_1}.{CAMERA_EXT}')
VIDEO_2 = os.path.join(OUT_DIR, EXP_NAME, 'sync', f'{CAMERA_2}.{CAMERA_EXT}')

C_1 = CalibratedCamera.from_pickle(CALIBRATION_1) if CALIBRATION_1 else CalibratedCamera.trivial_calibration(size=VideoFile(VIDEO_1).metadata.size)
C_2 = CalibratedCamera.from_pickle(CALIBRATION_2) if CALIBRATION_2 else CalibratedCamera.trivial_calibration(size=VideoFile(VIDEO_2).metadata.size)

def get_binary_thresholding(type_: str) -> Thresholding:

	match type_:

		case 'threshold' : return BaseThresholding      (t=65, kernel_size=(27, 27))
		case 'otsu'      : return OtsuThresholding      ()
		case 'tophat'    : return TopHatOtsuThresholding(kernel_size=(191, 191), kernel_shape=cv.MORPH_ELLIPSE)
		case 'adaptive'  : return AdaptiveThresholding  (block_size=117, c=10)
		case _           : raise ValueError(f'Invalid binary type: {type_}')
		

MLIC_SIDE = 256

SKIP_FRAMES = 1
WIN_SCALE   = 0.25
WIN_SQUARE  = 250

DETECTOR_PARAMS = {
	'white_thresh'  : 230,
    'black_thresh'  : 25,
    'min_area'      : 200,
    'max_area_prop' : 0.5
}

MLIC_DIR = os.path.join(OUT_DIR, EXP_NAME, 'mlic')

def main():

    logger = FileLogger(file=os.path.join(MLIC_DIR, f'mlic.log'))

    # Output directory
    IOUtils.make_dir(path=OUT_DIR)
    logger.info(msg=f'Saving Multi-Light Image Collection for experiment {EXP_NAME} to {MLIC_DIR} .')
    logger.info(msg=f'')

    # Load video streams
    logger.info(msg=f'LOADING VIDEO STREAMS.')
    marker_detector = MarkerDetector(**DETECTOR_PARAMS)
    logger.info(f'Using Marker Detector: {marker_detector}')
    logger.info('')

    static_thresholding = get_binary_thresholding(BINARY_TYPE_1)
    mlic_static = MLICStaticCameraVideoStream(
        path=VIDEO_1, 
        calibration=C_1, 
        thresholding=static_thresholding, 
        marker_detector=marker_detector, 
        mlic_side=MLIC_SIDE, 
        logger=logger
    )
    logger.info(f'Static video stream: {mlic_static}')
    logger.info(f' - Using thresholding: {static_thresholding}')
    logger.info('')
	
    dynamic_thresholding = get_binary_thresholding(BINARY_TYPE_2)
    mlic_dynamic = MLICDynamicCameraVideoStream(
        path=VIDEO_2, 
        calibration=C_2, 
        thresholding=dynamic_thresholding,
        marker_detector=marker_detector, 
        logger=logger
    )
    logger.info(f'Dynamic video stream: {mlic_dynamic}')
    logger.info(f' - Using thresholding: {dynamic_thresholding}')
    logger.info('')

    # Synchronize streams
    mlic_collector = MLICCollector(mlic_static=mlic_static, mlic_dynamic=mlic_dynamic, logger=logger)
	
    logger.info('STARTING MLIC COLLECTION.')
    mlic = mlic_collector.collect(skip_frames=SKIP_FRAMES, win_rect_scale=WIN_SCALE, win_square_side=WIN_SQUARE)
    
    logger.info('SAVING MLIC')
    out_mlic = os.path.join(MLIC_DIR, f'mlic.pkl')
    mlic.dump(out_mlic, logger=logger)
	
if __name__ == '__main__': main()