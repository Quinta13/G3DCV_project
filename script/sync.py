import os
from src.utils.io_ import FileLogger, VideoFile, IOUtils
from src.preprocessing import VideoSync

DATA_DIR = r'C:\Users\user.LAPTOP-G27BJ7JO\Documents\GitHub\g3dcv\data'

CAMERA_1     = 'cam1-static'
CAMERA_2     = 'cam2-moving_light'
CAMERA_1_EXT = 'mov'
CAMERA_2_EXT = 'mp4'
EXP_NAME     = 'coin1'

INPUT_1  = os.path.join(DATA_DIR, CAMERA_1, f'{EXP_NAME}.{CAMERA_1_EXT}')
INPUT_2  = os.path.join(DATA_DIR, CAMERA_2, f'{EXP_NAME}.{CAMERA_2_EXT}')
SYNC_DIR = os.path.join(DATA_DIR, EXP_NAME, 'sync')

PREFIX = lambda x: f' > {x}'

if __name__ == "__main__":

    IOUtils.make_dir(path=SYNC_DIR)

    logger = FileLogger(file=os.path.join(SYNC_DIR, 'sync.log'))

    def start(msg: str): logger.info(msg);         logger.formatter = PREFIX
    def end()          : logger.reset_formatter(); logger.info('') 

    start(f'READING VIDEOS')
    video1 = VideoFile(path=INPUT_1, logger=logger)
    video2 = VideoFile(path=INPUT_2, logger=logger)
    end()

    start(f"CREATING SYNCHRONIZATION")
    video_syncer = VideoSync(
        video1=video1,
        video2=video2,
        out_dir=SYNC_DIR,
        logger=logger
    )  
    end()

    video_syncer.sync()





