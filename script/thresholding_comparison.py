import math
import os
from typing import Any, Dict, List, Tuple, Type
from dotenv import load_dotenv

import cv2 as cv
from matplotlib import pyplot as plt

from src.model.stream import SynchronizedVideoStream, VideoStream
from src.model.calibration import CalibratedVideoStream, CameraCalibration
from src.utils.io_ import IOUtils
from src.utils.misc import Timer, grid_size
from src.model.thresholding import (
    ThresholdedVideoStream,
    BaseThresholdedVideoStream,
    OtsuThresholdedVideoStream,
    TopHatOtsuThresholdedVideoStream,
    AdaptiveThresholdedVideoStream, 
)
from src.utils.io_ import FileLogger
from src.utils.io_ import PrintLogger

load_dotenv()

OUT_DIR  = os.getenv('OUT_DIR',  '.')

EXP_NAME   = 'coin1'
C_ID = 2

match C_ID:

    case 1:

        CAMERA      = 'cam1-static'
        CALIBRATION = None
        WINDOW_SIZE = (216, 384)
    
    case 2:

        CAMERA      = 'cam2-moving_light'
        CALIBRATION = os.path.join(OUT_DIR, EXP_NAME, 'calibration', f'{CAMERA}.pkl')
        WINDOW_SIZE = (384, 216)

    case _:

        raise ValueError(f'Invalid camera id {C_ID}. ')

CAMERA_EXT = 'mp4'
VIDEO      = os.path.join(OUT_DIR, EXP_NAME, 'sync', f'{CAMERA}.{CAMERA_EXT}')

COMPARISON_DIR = os.path.join(OUT_DIR, EXP_NAME, 'thresholding_comparison')

SKIP_FRAMES   = 50
EXAMPLES_PLOT = 5

# name: (threshold class, arguments)
STREAMS_INFO: Dict[str, Tuple[Type[ThresholdedVideoStream], Dict[str, Any]]] = {
    'simple_threshold' : (BaseThresholdedVideoStream,       {'t': 50}),
    'otsu'             : (OtsuThresholdedVideoStream,       {}),
    'top-hat + otsu'   : (TopHatOtsuThresholdedVideoStream, {'kernel_size': (265, 265), 'kernel_shape': cv.MORPH_CROSS}),
    'adaptive'         : (AdaptiveThresholdedVideoStream,   {'block_size': 117, 'c': 5})
}

if __name__ == "__main__":

    logger = PrintLogger()

    # Output directory
    IOUtils.make_dir(path=OUT_DIR)
    logger = FileLogger(file=os.path.join(COMPARISON_DIR, f'{CAMERA}_Thresholded.log'))
    logger.info(msg=f'Saving synchronization data for experiment {EXP_NAME} to {COMPARISON_DIR} .')
    logger.info(msg=f'')

    # Read videos
    logger.info(msg=f'CREATING STREAMS')

    logger.info(msg='Using camera calibration data. ')
    calibration = CameraCalibration.from_pickle(path=CALIBRATION, logger=logger) if CALIBRATION else CameraCalibration.trivial_calibration()

    logger.info(msg='Creating video streams. ')

    streams: List[VideoStream] = []
    for name, (C_stream, args) in STREAMS_INFO.items():
        stream = C_stream(path=VIDEO, calibration=calibration, name=name, **args)
        streams.append(stream)
        logger.info(msg=f' - {stream}')
    logger.info(msg=f'Adding original stream as reference. ')
    streams.append(CalibratedVideoStream(path=VIDEO, calibration=calibration, name='reference'))
    logger.info(msg='')

    # Streaming
    logger.info(msg=f"PLAYING STREAMS")
    sync_stream = SynchronizedVideoStream(
        streams=streams,
        logger=logger,
    )
    timer = Timer()

    exclude_views = {
        stream.name: [
            view_name for view_name in stream.views
            if not (view_name == 'binary' or (stream.name=='reference' and view_name=='calibrated'))# or (stream.name == 'reference' and view_name == 'raw'))
        ]
        for stream in streams
    }

    print(exclude_views)

    sync_stream.play(
        window_size=WINDOW_SIZE,
        skip_frames=SKIP_FRAMES,
        exclude_views=exclude_views
    )
    logger.info(msg=f'Streaming completed in {timer}. ')
    logger.info(msg='')

    # Saving some example frames
    logger.info(msg=f'SAVING EXAMPLE FRAMES')

    r, c = grid_size(len(sync_stream))

    step = sync_stream.num_frames // EXAMPLES_PLOT

    for plot_id, frame_id in enumerate(range(0, sync_stream.num_frames, step)):

        fig, axes = plt.subplots(r, c, figsize=(9, 7))
        axes = axes.flatten()

        for i, stream in enumerate(streams):

            frame = stream[frame_id]['binary'] if stream.name != 'reference' else stream[frame_id]['raw']
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(stream.name)
            axes[i].axis('off')
        
        for i in range(len(streams), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'Frame {frame_id}')

        out_fp = os.path.join(COMPARISON_DIR, f'{CAMERA}_example_{plot_id}.png')
        logger.info(msg=f' > Saving example frames n.{plot_id} to {out_fp}. ')
        plt.tight_layout()
        fig.savefig(out_fp)
