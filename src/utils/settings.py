'''
This file loads the environment variables from the .env file as macros,
and creates the file paths for directories and files used in the project.
'''

import os
from typing import Tuple
from dotenv import load_dotenv

from src.model.typing import CameraPoseMethod

load_dotenv(r'.env', override=True)

def parse_tuple(tuple_str: str) -> Tuple[int, ...]:
    return tuple(map(int, tuple_str.strip('()').split(',')))

# ________________ Environment Variables from .env ________________

DATA_DIR              : str              = os.getenv('DATA_DIR',              '')
OUT_DIR               : str              = os.getenv('OUT_DIR',               '')
EXP_NAME              : str              = os.getenv('EXP_NAME',              '')
CAMERA_1              : str              = os.getenv('CAMERA_1',              '')
CAMERA_2              : str              = os.getenv('CAMERA_2',              '')
CAMERA_1_EXT          : str              = os.getenv('CAMERA_1_EXT',          '')
CAMERA_2_EXT          : str              = os.getenv('CAMERA_2_EXT',          '')
SYNC_EXT              : str              = os.getenv('SYNC_EXT',              '')
MLIC_FILE_NAME        : str              = os.getenv('MLIC_FILE',             '')
BASIS_FILE_NAME       : str              = os.getenv('BASIS_FILE',            '')
INTERPOLATION_ALGO    : str              = os.getenv('INTERPOLATION_ALGO',    '')
LIGHT_POSITION_METHOD : CameraPoseMethod = os.getenv('LIGHT_POSITION_METHOD', '')  # type: ignore

MLIC_SIZE   : int   = int  (os.getenv('MLIC_SIZE',    0))
SAMPLES     : int   = int  (os.getenv('SAMPLES',      0))
SPLIT_RATIO : float = float(os.getenv('SPLIT_SIZE',   0))

_CHESSBOARD_SIZE: Tuple[int, ...] = parse_tuple(os.getenv('CHESSBOARD_SIZE', ''))
assert len(_CHESSBOARD_SIZE) == 2, 'CHESSBOARD_SIZE must be a tuple of 2 integers.'
CHESSBOARD_SIZE: Tuple[int, int] = _CHESSBOARD_SIZE

_INTERPOLATION_SIZE: Tuple[int, ...] = parse_tuple(os.getenv('INTERPOLATION_SIZE', ''))
assert len(_INTERPOLATION_SIZE) == 2, 'INTERPOLATION_SIZE must be a tuple of 2 integers.'
INTERPOLATION_SIZE: Tuple[int, int] = _INTERPOLATION_SIZE

# ___________________________________ File Paths ___________________________________

# Raw data paths
CAMERA_1_RAW_PATH  = os.path.join(DATA_DIR, CAMERA_1, f'{EXP_NAME }.{CAMERA_1_EXT}')
CAMERA_2_RAW_PATH  = os.path.join(DATA_DIR, CAMERA_2, f'{EXP_NAME }.{CAMERA_2_EXT}')
CALIBRATION_PATH   = os.path.join(DATA_DIR, CAMERA_2, f'calibration.{CAMERA_2_EXT}')

# Synced data paths - `scripts/1_sync.py`
SYNC_DIR           = os.path.join(OUT_DIR, EXP_NAME, 'sync')
CAMERA_1_PATH      = os.path.join(SYNC_DIR, f'{CAMERA_1}.{SYNC_EXT}')
CAMERA_2_PATH      = os.path.join(SYNC_DIR, f'{CAMERA_2}.{SYNC_EXT}')

# Calibration paths - `scripts/2_calibrate.py`
CALIBRATION_DIR    = os.path.join(OUT_DIR, EXP_NAME, 'calibration')
CALIBRATION_FILE   = os.path.join(CALIBRATION_DIR, f'{CAMERA_2}.pkl')

# MLIC paths - `scripts/3_collect_mlic.py`
MLIC_DIR           = os.path.join(OUT_DIR, EXP_NAME, 'mlic')
MLIC_FILE_PATH     = os.path.join(MLIC_DIR, f'{MLIC_FILE_NAME}.pkl')

# Interpolation paths - `scripts/4_interpolate.py`
INTERPOLATION_DIR  = os.path.join(OUT_DIR, EXP_NAME, 'interpolation')
BASIS_FILE_PATH    = os.path.join(INTERPOLATION_DIR, f'{BASIS_FILE_NAME}.pkl')