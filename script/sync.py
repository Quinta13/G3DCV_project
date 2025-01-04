import os
from dotenv import load_dotenv

from model.preprocessing import VideoSync
from src.model.stream import SynchronizedVideoStream
from src.utils.io_ import VideoFile, IOUtils
from src.utils.misc import Timer
from src.utils.io_ import FileLogger

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', '.')
OUT_DIR  = os.getenv('OUT_DIR',  '.')

CAMERA_1     = 'cam1-static'
CAMERA_2     = 'cam2-moving_light'
CAMERA_1_EXT = 'mov'
CAMERA_2_EXT = 'mp4'
EXP_NAME     = 'coin1'

INPUT_1  = os.path.join(DATA_DIR, CAMERA_1, f'{EXP_NAME}.{CAMERA_1_EXT}')
INPUT_2  = os.path.join(DATA_DIR, CAMERA_2, f'{EXP_NAME}.{CAMERA_2_EXT}')

SYNC_DIR = os.path.join(OUT_DIR, EXP_NAME, 'sync')

WINDOW_SIZE = [(324, 576), (576, 324)]

if __name__ == "__main__":

    # Output directory
    IOUtils.make_dir(path=SYNC_DIR)
    logger = FileLogger(file=os.path.join(SYNC_DIR, 'sync.log'))
    logger.info(msg=f'Saving synchronization data for experiment {EXP_NAME} to {SYNC_DIR} .')
    logger.info(msg=f'')

    # Read videos
    logger.info(msg=f'READING VIDEOS')
    video1 = VideoFile(path=INPUT_1, logger=logger)
    video2 = VideoFile(path=INPUT_2, logger=logger)
    logger.info(msg=f'')

    # Synchronization
    logger.info(msg=f"CREATING SYNCHRONIZATION OBJECT ")
    video_syncer = VideoSync(
        video1=video1,
        video2=video2,
        out_dir=SYNC_DIR,
        logger=logger
    )
    logger.info(msg=str(video_syncer))
    logger.info(msg='')

    stream1, stream2 = video_syncer.sync()

    # Playing synced videos
    logger.info(msg=f'PLAYING SYNCED VIDEOS')
    timer = Timer()
    sync_stream = SynchronizedVideoStream(
        streams=[stream1, stream2],
        logger=logger,
    )
    sync_stream.play(
        window_size=WINDOW_SIZE
    )
    logger.info(msg=f'Streaming completed in {timer}. ')
    logger.info(msg='')