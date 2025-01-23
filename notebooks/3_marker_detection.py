# %% [markdown]
# # Marker Detection

# %%
from typing import Dict, List, Tuple, get_args

import ipywidgets as widgets
import cv2 as cv
import numpy as np

from src.model.thresholding import AdaptiveThresholding, OtsuThresholding, Thresholding
from src.utils.settings    import CALIBRATION_FILE, CAMERA_1_PATH, CAMERA_2_PATH
from src.utils.calibration import CalibratedCamera
from src.utils.io_         import VideoFile, PrintLogger
from src.utils.misc        import display_frames, launch_widget
from src.utils.typing      import CornerMaskMethod

calibrated_camera_1  = CalibratedCamera.trivial_calibration(size=VideoFile(path=CAMERA_1_PATH).metadata.size)
calibrated_camera_2  = CalibratedCamera.from_pickle(path=CALIBRATION_FILE)
thresh_camera_1      = OtsuThresholding()
thresh_camera_2      = AdaptiveThresholding(block_size=161, c=15)

CAMERA_INFO: Dict[str, Tuple[str, CalibratedCamera, Thresholding]] = {
    'static' : (CAMERA_1_PATH, calibrated_camera_1, thresh_camera_1),
    'dynamic': (CAMERA_2_PATH, calibrated_camera_2, thresh_camera_2),
}

CAMERA_NAMES : List[str] = list(CAMERA_INFO  .keys())
CORNER_MASK_NAMES : List[str] = list(get_args(CornerMaskMethod))

min_frame = min([VideoFile(path=path).metadata.frames for path, _, _ in CAMERA_INFO.values()])
max_area  = np.prod(VideoFile(path=CAMERA_1_PATH).metadata.size)

camera_w      = widgets.Dropdown (options=CAMERA_NAMES,      value=CAMERA_NAMES[1],       description='Camera:',      continuous_update=False)
corner_mask_w = widgets.Dropdown (options=CORNER_MASK_NAMES, value=CORNER_MASK_NAMES[-1], description='Corner Mask:', continuous_update=False)

frame_w       = widgets.IntSlider  (value=          0, min=0,   max=min_frame-1,    step=1,   description='Frame ID:',     continuous_update=False)
black_w       = widgets.IntSlider  (value=         25, min=0,   max=255,            step=1,   description='Black thresh:', continuous_update=False)
white_w       = widgets.IntSlider  (value=        230, min=0,   max=255,            step=1,   description='White thresh:', continuous_update=False)
min_area_w    = widgets.IntSlider  (value=        200, min=0,   max=10000,          step=1,   description='Min Area:',     continuous_update=False)
max_area_w    = widgets.IntSlider  (value=max_area//2, min=0,   max=max_area,       step=1,   description='Max Area:',     continuous_update=False)
scale_w       = widgets.FloatSlider(value=      0.9,   min=0.1, max=1.0,            step=0.1, description='Mask Scale:',   continuous_update=False)

output_w      = widgets.Output()

# %% [markdown]
# ## Hyperparameters

# %%
from src.model.marker import MarkerDetectionVideoStream, MarkerDetector

def update(change):

	with output_w:

		# Values
		camera      = camera_w     .value
		corner_mask = corner_mask_w.value
		frame       = frame_w      .value
		black       = black_w      .value
		white       = white_w      .value
		min_area    = min_area_w   .value
		max_area    = max_area_w   .value
		scale       = scale_w      .value

		path, calibration, thresholding = CAMERA_INFO[camera]

		# Detector
		detector = MarkerDetector(
			white_thresh=white,
			black_thresh=black,
			min_contour_area=min_area,
			max_contour_area=max_area,
			corner_mask_method=corner_mask,
		)
		detector.CORNER_SCALE_FACTOR = scale

		stream = MarkerDetectionVideoStream(
			path=path,
			calibration=calibration,
			thresholding=thresholding,
			marker_detector=detector,
			logger=PrintLogger()
		)
		
		# Clear output
		output_w.clear_output(wait=True)

		# Display frames
		display_frames([*stream[frame].items()], n_rows=2)

launch_widget(
    widgets_=[camera_w, corner_mask_w, frame_w, black_w, white_w, min_area_w, max_area_w, scale_w, output_w],
    update_fn=update
)

# %% [markdown]
# ## Corner Mask Methods Example

# %%
from src.utils.stream import SynchronizedVideoStream

CAMERA_1_WINSIZE = (216, 384)
CAMERA_2_WINSIZE = (384, 216)

sync_stream = SynchronizedVideoStream(
    streams = [
        MarkerDetectionVideoStream(
			name='border',
			path=CAMERA_2_PATH,
			calibration=calibrated_camera_2,
			thresholding=thresh_camera_2,
			marker_detector=MarkerDetector(corner_mask_method='border'),
		),
		MarkerDetectionVideoStream(
			name='descendants',
			path=CAMERA_2_PATH,
			calibration=calibrated_camera_2,
			thresholding=thresh_camera_2,
			marker_detector=MarkerDetector(corner_mask_method='descendants', min_contour_area=None),
		),
        MarkerDetectionVideoStream(
			name='scaled',
			path=CAMERA_2_PATH,
			calibration=calibrated_camera_2,
			thresholding=thresh_camera_2,
			marker_detector=MarkerDetector(corner_mask_method='scaled'),
		)
	],
    logger=PrintLogger()
)

exclude_views = {
	stream.name: [
		view_name for view_name in stream.views
		if view_name not in ['inner_mask', 'outer_mask']
	]
	for stream in sync_stream.streams
}

sync_stream

# %%
SKIP_FRAMES = 25

sync_stream.play(
    skip_frames=SKIP_FRAMES,
    exclude_views=exclude_views,
    window_size=tuple([size * 2 // 3 for size in CAMERA_2_WINSIZE]),
)

# %% [markdown]
# ## Corner Detection for the two Cameras

# %%
from src.utils.stream import SynchronizedVideoStream

detector = MarkerDetector()

sync_stream = SynchronizedVideoStream(
    streams = [
        MarkerDetectionVideoStream(
			name='static',
			path=CAMERA_1_PATH,
			calibration=calibrated_camera_1,
			thresholding=thresh_camera_1,
			marker_detector=detector,
		),
		MarkerDetectionVideoStream(
			name='dynamic',
			path=CAMERA_2_PATH,
			calibration=calibrated_camera_2,
			thresholding=thresh_camera_2,
			marker_detector=detector,
		),
	],
    logger=PrintLogger()
)

exclude_views = {
	stream.name: [
		view_name for view_name in stream.views
		if view_name not in ['marker']
	]
	for stream in sync_stream.streams
}

sync_stream

# %%
SKIP_FRAMES = 25

sync_stream.play(
    skip_frames=SKIP_FRAMES,
    exclude_views=exclude_views,
    window_size={'static': CAMERA_1_WINSIZE, 'dynamic': CAMERA_2_WINSIZE}
)

# %% [markdown]
# 


