import os

from dotenv import load_dotenv

from src.model.preprocessing import VideoSync
from src.model.stream import SynchronizedVideoStream
from src.utils.io_ import VideoFile, IOUtils, FileLogger
from src.utils.misc import Timer

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', '.')
OUT_DIR  = os.getenv('OUT_DIR',  '.')

CAMERA_1, CAMERA_1_EXT, CAMERA_1_WINSIZE = 'cam1-static',       'mov', (324, 576)
CAMERA_2, CAMERA_2_EXT, CAMERA_2_WINSIZE = 'cam2-moving_light', 'mp4', (576, 324)

EXP_NAME     = 'coin1'

INPUT_1  = os.path.join(DATA_DIR, CAMERA_1, f'{EXP_NAME}.{CAMERA_1_EXT}')
INPUT_2  = os.path.join(DATA_DIR, CAMERA_2, f'{EXP_NAME}.{CAMERA_2_EXT}')

SYNC_DIR = os.path.join(OUT_DIR, EXP_NAME, 'sync')

def main():

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
    sync_stream.play(window_size={stream1.name: CAMERA_1_WINSIZE, stream2.name: CAMERA_2_WINSIZE})
    logger.info(msg=f'Streaming completed in {timer}. ')
    logger.info(msg='')

if __name__ == '__main__': main()