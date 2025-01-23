# %% [markdown]
# # Thresholding Methods

# %%
from typing import Dict, List, Tuple

import ipywidgets as widgets
import cv2 as cv

from src.utils.settings    import CALIBRATION_FILE, CAMERA_1_PATH, CAMERA_2_PATH
from src.utils.calibration import CalibratedCamera
from src.utils.io_         import VideoFile
from src.utils.misc        import display_frames, launch_widget

calibrated_camera_1 = CalibratedCamera.trivial_calibration(size=VideoFile(path=CAMERA_1_PATH).metadata.size)
calibrated_camera_2 = CalibratedCamera.from_pickle(path=CALIBRATION_FILE)

CAMERA_INFO: Dict[str, Tuple[str, CalibratedCamera]] = {
    'static' : (CAMERA_1_PATH, calibrated_camera_1),
    'dynamic': (CAMERA_2_PATH, calibrated_camera_2)
}

WHITE_MASK: Dict[str, bool] = {
    'black': False,
    'white': True,
}

MORPH_OPTIONS = {
    'Rect'    : cv.MORPH_RECT,
    'Cross'   : cv.MORPH_CROSS,
    'Ellipse' : cv.MORPH_ELLIPSE
}

CAMERA_NAMES : List[str] = list(CAMERA_INFO  .keys())
MASK_NAMES   : List[str] = list(WHITE_MASK   .keys())
MORPH_NAMES  : List[str] = list(MORPH_OPTIONS.keys())

min_frame = min([VideoFile(path=path).metadata.frames for path, _ in CAMERA_INFO.values()])

def get_widgets():

	return\
		widgets.Dropdown(options=CAMERA_NAMES, value=CAMERA_NAMES[1], description='Camera:',            continuous_update=False),\
		widgets.IntSlider(value=0, min=0, max=min_frame-1, step=1, description='Frame ID:',             continuous_update=False),\
		widgets.Dropdown(options=MASK_NAMES,   value=MASK_NAMES  [0], description='Undistortion mask:', continuous_update=False),\
		widgets.Output()


# %% [markdown]
# ## Base Thresholding

# %%
from src.model.thresholding import ThresholdedVideoStream, BaseThresholding

camera_w, frame_w, mask_w, output_w = get_widgets()

threhsold_w = widgets.IntSlider(value=50,    min=0, max=255, step=1, description='T',                continuous_update=False)
kernel_w    = widgets.IntSlider(value=3,     min=1, max=121, step=2, description='Blur Kernel Side', continuous_update=False)
blur_w      = widgets.Checkbox (value=True,                          description='Apply Blur',       continuous_update=False)

def update_base_threshold(change):

    with output_w:

        # Widget values
        path, calibrated = CAMERA_INFO[camera_w.value]  # type: ignore
        white_mask       = WHITE_MASK [  mask_w.value]  # type: ignore
        frame            = frame_w.value
        t                = threhsold_w.value
        kernel_side      = kernel_w   .value
        apply_blur       = blur_w     .value
        
		# Base thresholding
        calibrated.white_mask = white_mask
        thresholding = BaseThresholding(t=t, kernel_size=(kernel_side, kernel_side) if apply_blur else None)
        stream = ThresholdedVideoStream(path=path, calibration=calibrated, thresholding=thresholding)
        
		# Clear previous content
        output_w.clear_output(wait=True)
        
		# Display frames
        display_frames(frames=[*stream[frame].items()])


launch_widget(
    widgets_=[camera_w, frame_w, mask_w, threhsold_w, kernel_w, blur_w, output_w],
    update_fn=update_base_threshold
)

# %% [markdown]
# # Otsu Thresholding

# %%
from src.model.thresholding import OtsuThresholding, ThresholdedVideoStream

camera_w, frame_w, mask_w, output_w = get_widgets()
threhsold_w = widgets.IntSlider(value=50,    min=0, max=255, step=1, description='T',                continuous_update=False)
kernel_w    = widgets.IntSlider(value=3,     min=1, max=121, step=2, description='Blur Kernel Side', continuous_update=False)
blur_w      = widgets.Checkbox (value=True,                          description='Apply Blur',       continuous_update=False)

def update_otsu(change):

    with output_w:

        # Widget values
        path, calibrated = CAMERA_INFO[camera_w.value]  # type: ignore
        white_mask       = WHITE_MASK [  mask_w.value]  # type: ignore
        frame            = frame_w.value
        kernel_side      = kernel_w   .value
        apply_blur       = blur_w     .value

        # Otsu thresholding
        calibrated.white_mask = white_mask
        thresholding = OtsuThresholding(kernel_size=(kernel_side, kernel_side) if apply_blur else None)
        stream       = ThresholdedVideoStream (path=path, calibration=calibrated, thresholding=thresholding)

        # Clear previous content
        output_w.clear_output(wait=True)
        
        # Display frames
        display_frames(frames=[*stream[frame].items()])

launch_widget(
	widgets_=[camera_w, frame_w, mask_w, kernel_w, blur_w, output_w],
	update_fn=update_otsu
)

# %% [markdown]
# # Top Hat + Otsu

# %%
from src.model.thresholding import TopHatOtsuThresholding, ThresholdedVideoStream

camera_w, frame_w, mask_w, output_w = get_widgets()

kernel_w = widgets.IntSlider(value=111, min=1, max=1001, step=2, description='Structuring element side', continuous_update=False)
se_w     = widgets.Dropdown(options=MORPH_NAMES, value=MORPH_NAMES[0], description='Structuring element', continuous_update=False)

def update_tophat_otsu(change):

    with output_w:

        # Widget values
        path, calibrated = CAMERA_INFO  [camera_w.value] # type: ignore
        white_mask       = WHITE_MASK   [  mask_w.value] # type: ignore
        se_shape         = MORPH_OPTIONS[    se_w.value]  # type: ignore
        frame            = frame_w.value
        side             = kernel_w.value

        # Otsu + TopHat thresholding
        calibrated.white_mask = white_mask
        thresholding = TopHatOtsuThresholding(kernel_size=(side, side), kernel_shape=se_shape)
        stream       = ThresholdedVideoStream(path=path, calibration=calibrated, thresholding=thresholding)
        
        # Clear previous content
        output_w.clear_output(wait=True)

        # Display frames
        display_frames(frames=[*stream[frame].items()])

launch_widget(
	widgets_=[camera_w, frame_w, mask_w, kernel_w, se_w, output_w],
	update_fn=update_tophat_otsu
)

# %% [markdown]
# # Adaptive Thresholding

# %%
from src.model.thresholding import AdaptiveThresholding, ThresholdedVideoStream

camera_w, frame_w, mask_w, output_w = get_widgets()

block_w = widgets.IntSlider(value=161,  min= 3,  max=199, step=2, description='Block Side', continuous_update=False)
c_w     = widgets.IntSlider(value=15,   min=-50, max= 50, step=1, description='C',          continuous_update=False)

def update_adaptive(change):

    with output_w:

        # Widget values
        path, calibrated = CAMERA_INFO  [camera_w.value] # type: ignore
        white_mask       = WHITE_MASK   [  mask_w.value] # type: ignore
        frame            = frame_w.value
        block_side       = block_w.value
        c			     = c_w.value 

        # Adaptive thresholding
        calibrated.white_mask = white_mask
        threhsolding = AdaptiveThresholding(block_size=block_side, c=c)
        stream       = ThresholdedVideoStream(path=path, calibration=calibrated, thresholding=threhsolding)
        
        # Clear previous content
        output_w.clear_output(wait=True)
        
		# Display frames
        display_frames(frames=[*stream[frame].items()])

launch_widget(
    widgets_=[camera_w, frame_w, mask_w, block_w, c_w, output_w],
    update_fn=update_adaptive
)

# %% [markdown]
# ## Comparison Stream

# %%
from typing import Any, Type

from src.model.thresholding import Thresholding
from src.utils.calibration import CalibratedVideoStream
from src.utils.stream import SynchronizedVideoStream
from src.utils.io_ import PrintLogger

CAMERA_1_WINSIZE = (216, 384)
CAMERA_2_WINSIZE = (384, 216)


STREAMS_INFO: Dict[str, Tuple[Type[Thresholding], Dict[str, Any]]] = {
    'simple_threshold' : (BaseThresholding,       {'t': 50}),
    'otsu'             : (OtsuThresholding,       {}),
    'top-hat + otsu'   : (TopHatOtsuThresholding, {'kernel_size': (265, 265), 'kernel_shape': cv.MORPH_CROSS}),
    'adaptive'         : (AdaptiveThresholding,   {'block_size': 161, 'c': 15})
}

CAMERA      = 'dynamic'
SKIP_FRAMES = 25
UNDISTORT_MASK_WHITE = False

streams = []
path, calibration = CAMERA_INFO[CAMERA]
calibration.white_mask = UNDISTORT_MASK_WHITE

match CAMERA:
	case 'static': window_size = CAMERA_1_WINSIZE
	case 'dynamic': window_size = CAMERA_2_WINSIZE
	case _ 	   : raise ValueError(f'Invalid camera: {CAMERA}')

window_size = tuple([size * 4 // 5 for size in window_size])
logger = PrintLogger()

for name, (C_threshold, args) in STREAMS_INFO.items():
    
	stream = ThresholdedVideoStream(
		path=path,
		calibration=calibration,
		thresholding=C_threshold(**args),
		name=name
	)

	streams.append(stream)
	logger.info(msg=f' - {stream}')

streams.append(CalibratedVideoStream(path=path, calibration=calibration, name='reference'))

sync_stream = SynchronizedVideoStream(
	streams=streams,
	logger=logger,
)

exclude_views = {
	stream.name: [
		view_name for view_name in stream.views
		if not (view_name == 'binary' or (stream.name=='reference' and view_name=='undistorted'))
	]
	for stream in streams
}

sync_stream

# %%
sync_stream.play(
	window_size=window_size,
	skip_frames=SKIP_FRAMES,
	exclude_views=exclude_views
)

# %%



