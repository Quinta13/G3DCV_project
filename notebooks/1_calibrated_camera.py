# %% [markdown]
# # Camera Calibration

# %% [markdown]
# ## Camera 1

# %% [markdown]
# Using simple calibration derived by video size.

# %%
from src.utils.calibration import CalibratedCamera
from src.utils.io_ import VideoFile
from src.utils.settings import CAMERA_1_PATH

camera_size = VideoFile(path=CAMERA_1_PATH).metadata.size

calibrated_camera_1 = CalibratedCamera.trivial_calibration(size=camera_size)

calibrated_camera_1 

# %%
calibrated_camera_1

# %%
calibrated_camera_1.info

# %% [markdown]
# ## Camera 2

# %% [markdown]
# Loading camera calibration from pickle file, computed in `script/2_calibrate.py`.

# %%
from src.utils.calibration import CalibratedCamera
from src.utils.io_ import PrintLogger
from src.utils.settings import CALIBRATION_FILE

calibrated_camera_2 = CalibratedCamera.from_pickle(path=CALIBRATION_FILE, logger=PrintLogger())

calibrated_camera_2

# %%
calibrated_camera_2.info

# %% [markdown]
# ### Example of undistortion

# %%
from typing import Dict, List, Tuple

import ipywidgets as widgets
from src.utils.misc import display_frames, launch_widget
from src.utils.calibration import CalibratedVideoStream
from src.utils.settings import CAMERA_2_PATH

CAMERA_INFO: Dict[str, Tuple[str, CalibratedCamera]] = {
    'static ': (CAMERA_1_PATH, calibrated_camera_1),
    'dynamic': (CAMERA_2_PATH, calibrated_camera_2)
}

WHITE_MASK: Dict[str, bool] = {
    'black': False,
    'white': True,
}

CAMERA_NAMES : List[str] = list(CAMERA_INFO.keys())
MASK_NAMES   : List[str] = list(WHITE_MASK.keys())

min_frame = min([VideoFile(path=path).metadata.frames for path, _ in CAMERA_INFO.values()])

frame_w  = widgets.IntSlider(value=0, min=0, max=min_frame-1, step=1, description='Frame ID:', continuous_update=False)
camera_w = widgets.Dropdown(options=CAMERA_NAMES, value=CAMERA_NAMES[0], description='Camera:',              continuous_update=False)
mask_w   = widgets.Dropdown(options=MASK_NAMES,   value=MASK_NAMES  [0], description='Undistortion mask:',   continuous_update=False)
output_w = widgets.Output()

def update(change):

    with output_w:

        # Widget values
        path, calibrated = CAMERA_INFO[camera_w.value]  # type: ignore
        white_mask       = WHITE_MASK [  mask_w.value]  # type: ignore
        frame            = frame_w.value

		# Create calibrated stream
        calibrated.white_mask = white_mask
        stream = CalibratedVideoStream(path=path, calibration=calibrated)
        
        # Clear previous content
        output_w.clear_output(wait=True)

        # Display frames
        display_frames(frames=[*stream[frame].items()])

# %%
launch_widget(
    widgets_=[frame_w, camera_w, mask_w, output_w],
    update_fn=update
)


