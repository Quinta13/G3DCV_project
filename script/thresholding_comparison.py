import math
import os
from typing import Any, Dict, Tuple, Type
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from src.model.calibration import CameraCalibration
from src.model.stream import SynchronizedVideoStream
from src.utils.io_ import FileLogger, IOUtils, PrintLogger
from src.utils.misc import Timer, grid_size
from src.model.thresholding import (
    ThresholdedVideoStream,
    OtsuThresholdingVideoStream, 
    GaussianOtsuTresholdingVideoStream, 
    TopHatOtsuTresholdingVideoStream,
    AdaptiveThresholdingVideoStream, 
    AdaptiveThresholdingPlusOpeningVideoStream,
    AdaptiveThresholdingPlusMedianFilterVideoStream
)

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', '.')
OUT_DIR  = os.getenv('OUT_DIR',  '.')

CAMERA     = 'cam2-moving_light'
CAMERA_EXT = 'mp4'
EXP_NAME   = 'coin1'

VIDEO       = os.path.join(OUT_DIR, EXP_NAME, 'sync', f'{CAMERA}.{CAMERA_EXT}')
CALIBRATION = os.path.join(OUT_DIR, EXP_NAME, 'calibration', f'{CAMERA}.pkl')

COMPARISON_DIR = os.path.join(OUT_DIR, EXP_NAME, 'thresholding_comparison')

WINDOW_SIZE = (384, 216)
SKIP_FRAMES = 50

EXAMPLES_PLOT = 5

# name: (threshold class, arguments)
STREAMS_INFO: Dict[str, Tuple[Type[ThresholdedVideoStream], Dict[str, Any]]] = {
    'otsu'                     : (OtsuThresholdingVideoStream,                     {}),
    'gaussian otsu'            : (GaussianOtsuTresholdingVideoStream,              {'kernel_size': (33, 33)}),
    'top-hat otsu'             : (TopHatOtsuTresholdingVideoStream,                {'kernel_size': (151, 151)}),
    'adaptive'                 : (AdaptiveThresholdingVideoStream,                 {'block_size': 17, 'c': 2}),
    'adaptive + opening'       : (AdaptiveThresholdingPlusOpeningVideoStream,      {'block_size': 17, 'c': 2, 'kernel_size': (5, 5)}),
    'adaptive + median filter' : (AdaptiveThresholdingPlusMedianFilterVideoStream, {'block_size': 17, 'c': 2, 'kernel_side': 11}),
}

if __name__ == "__main__":

    logger = PrintLogger()

    # Output directory
    IOUtils.make_dir(path=OUT_DIR)
    logger = FileLogger(file=os.path.join(COMPARISON_DIR, f'{CAMERA}_thresholding.log'))
    logger.info(msg=f'Saving synchronization data for experiment {EXP_NAME} to {COMPARISON_DIR} .')
    logger.info(msg=f'')

    # Read videos
    logger.info(msg=f'CREATING STREAMS')

    logger.info(msg='Using camera calibration data. ')
    calibration = CameraCalibration.from_pickle(path=CALIBRATION, logger=logger)

    logger.info(msg='Creating video streams. ')

    streams = []
    for name, (C_stream, args) in STREAMS_INFO.items():
        stream = C_stream(path=VIDEO, calibration=calibration, name=name, **args)
        streams.append(stream)
        logger.info(msg=f' - {stream}')
    logger.info(msg='')

    # Streaming
    logger.info(msg=f"PLAYING STREAMS")
    sync_stream = SynchronizedVideoStream(
        streams=streams,
        logger=logger,
    )
    timer = Timer()
    sync_stream.play(
        window_size=WINDOW_SIZE,
        skip_frames=SKIP_FRAMES
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

            frame = stream[frame_id]
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(stream.name)
            axes[i].axis('off')
        
        for i in range(len(streams), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'Frame {frame_id}')

        out_fp = os.path.join(COMPARISON_DIR, f'{plot_id}_example_frames.png')
        logger.info(msg=f' > Saving example frames n.{plot_id} to {out_fp}. ')
        plt.tight_layout()
        fig.savefig(out_fp)
