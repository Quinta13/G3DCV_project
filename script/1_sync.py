import os

from src.model.preprocessing import VideoSync
from src.utils.settings import (
    EXP_NAME, SYNC_DIR,
    CAMERA_1_RAW_PATH, CAMERA_2_RAW_PATH, SYNC_EXT
)
from src.utils.stream import SynchronizedVideoStream
from src.utils.io_ import VideoFile, IOUtils, FileLogger
from src.utils.misc import Timer

CAMERA_1_WINSIZE = (324, 576)
CAMERA_2_WINSIZE = (576, 324)

def main():

    # Create output directory
    IOUtils.make_dir(path=SYNC_DIR)
    logger = FileLogger(file=os.path.join(SYNC_DIR, 'sync.log'))
    logger.info(msg=f'Saving synchronization data for experiment {EXP_NAME} to {SYNC_DIR} .\n')

    # Read video files
    logger.info(msg=f'READING VIDEOS')
    video1 = VideoFile(path=CAMERA_1_RAW_PATH, logger=logger)
    video2 = VideoFile(path=CAMERA_2_RAW_PATH, logger=logger)
    logger.info(msg=f'')

    # Synchronization
    logger.info(msg=f"CREATING SYNCHRONIZATION OBJECT")
    video_syncer = VideoSync(
        video1=video1,
        video2=video2,
        out_dir=SYNC_DIR,
        out_video_ext=SYNC_EXT,
        logger=logger
    )
    logger.info(msg=str(video_syncer))
    logger.info(msg='')

    stream1, stream2 = video_syncer()

    # Playing synced videos
    logger.info(msg=f'PLAYING SYNCED VIDEOS')
    timer = Timer()
    sync_stream = SynchronizedVideoStream(
        streams=[stream1, stream2],
        logger=logger,
    )
    sync_stream.play(
        window_size={
            stream1.name: CAMERA_1_WINSIZE, 
            stream2.name: CAMERA_2_WINSIZE
        }
    )
    logger.info(msg=f'Streaming completed in {timer}. ')
    logger.info(msg='')

if __name__ == '__main__': main()