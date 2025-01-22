import os
from typing import Tuple
from dotenv import load_dotenv
from src.utils.typing import Size2D

def parse_winsize(win_size_str: str) -> Size2D:

    winsize = tuple(map(int, win_size_str.strip('()').split(',')))
    if len(winsize) != 2: 
        raise ValueError(f'Invalid 2D window size: {winsize}. Got {len(winsize)} dimensions, 2 expected.')
    return winsize

load_dotenv(r'.env', override=True)

DATA_DIR     : str = os.getenv('DATA_DIR',     '')
OUT_DIR      : str = os.getenv('OUT_DIR',      '')
EXP_NAME     : str = os.getenv('EXP_NAME',     '')
CAMERA_1     : str = os.getenv('CAMERA_1',     '')
CAMERA_2     : str = os.getenv('CAMERA_2',     '')
CAMERA_1_EXT : str = os.getenv('CAMERA_1_EXT', '')
CAMERA_2_EXT : str = os.getenv('CAMERA_2_EXT', '')
SYNC_EXT     : str = os.getenv('SYNC_EXT',     '')

CAMERA_1_WINSIZE : Size2D = parse_winsize(win_size_str=os.getenv('CAMERA_1_WINSIZE', ''))
CAMERA_2_WINSIZE : Size2D = parse_winsize(win_size_str=os.getenv('CAMERA_2_WINSIZE', ''))
WARPED_WINSIZE   : Size2D = parse_winsize(win_size_str=os.getenv('WARPED_WINSIZE',   ''))

CAMERA_1_RAW_PATH = os.path.join(DATA_DIR, CAMERA_1, f'{EXP_NAME }.{CAMERA_1_EXT}')
CAMERA_2_RAW_PATH = os.path.join(DATA_DIR, CAMERA_2, f'{EXP_NAME }.{CAMERA_2_EXT}')
CALIBRATION_PATH  = os.path.join(DATA_DIR, CAMERA_2, f'calibration.{CAMERA_2_EXT}')


SYNC_DIR         = os.path.join(OUT_DIR, EXP_NAME, 'sync')
CAMERA_1_PATH    = os.path.join(SYNC_DIR, f'{CAMERA_1}.{SYNC_EXT}')
CAMERA_2_PATH    = os.path.join(SYNC_DIR, f'{CAMERA_2}.{SYNC_EXT}')


CALIBRATION_DIR  = os.path.join(OUT_DIR, EXP_NAME, 'calibration')
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, f'{CAMERA_2}.pkl')
