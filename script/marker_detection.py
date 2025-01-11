from dataclasses import dataclass
import os
import pickle
from dotenv import load_dotenv

from src.model.marker import MarkerDetectionVideoStream, MarkerDetector
from src.model.thresholding import AdaptiveThresholding, OtsuThresholding, Thresholding, TopHatOtsuThresholding
from src.model.calibration import CameraCalibration
from src.utils.misc import Timer
from src.utils.io_ import IOUtils, FileLogger, VideoFile

@dataclass
class MarkerDetectionResult:

    camera_name  : str
    thresholding : Thresholding
    detector     : MarkerDetector
    tot_frame    : int
    succ_frame   : int

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', '.')
OUT_DIR  = os.getenv('OUT_DIR', '.')

CAMERA_ID = 2
EXP_NAME = 'coin1'

match CAMERA_ID:

    case 1:

        CAMERA          = 'cam1-static'
        WINDOW_SIZE     = (324, 576)
        CALIBRATION_EXT = 'mov'
        CALIBRATION     = None
        THRESHOLDING    = 'otsu'
    
    case 2:

        CAMERA          = 'cam2-moving_light'
        WINDOW_SIZE     = (576, 324)
        CALIBRATION_EXT = 'mp4'
        CALIBRATION     = os.path.join(OUT_DIR, EXP_NAME, 'calibration', f'{CAMERA}.pkl')
        THRESHOLDING    = 'adaptive'

    case _:

        raise ValueError(f'Invalid camera id {CAMERA_ID}. ')


EXP_NAME     = 'coin1'

CAMERA_EXT   = 'mp4'
VIDEO        = os.path.join(OUT_DIR, EXP_NAME, 'sync', f'{CAMERA}.{CAMERA_EXT}')
MARKER_DIR   = os.path.join(OUT_DIR, EXP_NAME, 'marker_detection')
RESULTS_FILE = os.path.join(MARKER_DIR, f'results.pkl')

BLACK_THRESH, WHITE_THRESH, MIN_AREA = 25, 230, 200

SKIP_FRAMES   = 1

if __name__ == "__main__":

    # Output directory
    IOUtils.make_dir(path=MARKER_DIR)
    logger = FileLogger(file=os.path.join(MARKER_DIR, f'{CAMERA}_{THRESHOLDING}_marker_detection.log'))
    logger.info(
        msg=f'Saving synchronization data for experiment {EXP_NAME} '
            f'of camera {CAMERA} to {MARKER_DIR} . '
    )
    logger.info(msg=f'')

    # Thresholding
    match THRESHOLDING:

        case 'threshold' : thresholding = BaseThresholding      (t=65, kernel_size=(27, 27))
        case 'otsu'      : thresholding = OtsuThresholding      ()
        case 'tophat'    : thresholding = TopHatOtsuThresholding(kernel_size=(191, 191), kernel_shape=cv.MORPH_ELLIPSE)
        case 'adaptive'  : thresholding = AdaptiveThresholding  (block_size=117, c=10)
        case _           : raise ValueError(f'Invalid binary type: {THRESHOLDING}')

    logger.info(msg=f'LOADING DETECTION OBJECTS ')
    logger.info(msg=f'Using thresholding method: {thresholding}. ')

    # Marker detection
    detector = MarkerDetector(white_thresh=WHITE_THRESH, black_thresh=BLACK_THRESH, min_area=MIN_AREA)
    logger.info(msg=f'Using marker detector: {detector}. ')
    logger.info(msg='')
    
    # Stream
    logger.info(msg='PREPARING MARKER DETECTION STREAM')
    logger.info(msg='Using camera calibration data. ')
    calibration = CameraCalibration.from_pickle(path=CALIBRATION, logger=logger) if CALIBRATION else CameraCalibration.trivial_calibration(size=VideoFile(path=VIDEO).metadata.size)
    # calibration.white_mask = True
    logger.info(msg=str(calibration))
    logger.info(msg='')
    
    logger.info(msg='Preparing video stream')
    detection_stream = MarkerDetectionVideoStream(
        path=VIDEO,
        calibration=calibration,
        thresholding=thresholding,
        marker_detector=detector,
        logger=logger
    )
    logger.info(msg=str(detection_stream))
    logger.info(msg='')

    # Performing detection
    timer = Timer()
    logger.info(msg='RUNNING MARKER DETECTION STREAM')
    detection_stream.play(
        window_size=WINDOW_SIZE, 
        skip_frames=SKIP_FRAMES,
        exclude_views=[view for view in detection_stream.views if view != 'marker']
    )
    logger.info(msg=f'Completed in {timer}. ')
    logger.info(msg='')

    # Saving results
    if os.path.exists(RESULTS_FILE):
        logger.info(f'Loading previous marker detection results from {RESULTS_FILE}. ')
        with open(RESULTS_FILE, 'rb') as file: results = pickle.load(file)
    else:
        logger.info('No previous results found. Creating new results object. ')
        results = []

    succ_frame, tot_frame = detection_stream.marker_detection_results
    
    result = MarkerDetectionResult(
        camera_name  = CAMERA,
        thresholding = thresholding,
        detector     = detector,
        tot_frame    = tot_frame,
        succ_frame   = succ_frame
    )

    results.append(result)

    logger.info(f'Saving results to {RESULTS_FILE}. ')

    with open(RESULTS_FILE, 'wb') as file: pickle.dump(results, file)

    logger.info(msg='')