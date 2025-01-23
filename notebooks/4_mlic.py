# %% [markdown]
# # Multi-Light Image Collection

# %% [markdown]
# ## Dataset

# %%
from src.model.mlic import MLIC
from src.utils.settings import MLIC_FILE_PATH
from src.utils.io_ import PrintLogger

WINDOW_SIZE = (350, 350)
DELAY       = 50

mlic = MLIC.from_pickle(path=MLIC_FILE_PATH, logger=PrintLogger())
mlic

# %%
mlic_stream = mlic.to_stream()
mlic_stream

# %%
mlic_stream.play(window_size=WINDOW_SIZE, delay=DELAY)

# %%
import ipywidgets as widgets
from src.utils.misc import display_frames, launch_widget

frame_w  = widgets.IntSlider(value=0, min=0, max=(len(mlic_stream))-1, step=1, description='Frame')
output_w = widgets.Output()

def update_fn(change):

	with output_w:

		frame = frame_w.value
		output_w.clear_output(wait=True)
		display_frames(mlic_stream[frame].items())

launch_widget(
	widgets_=[frame_w, output_w],
	update_fn=update_fn
)


# %% [markdown]
# ## Algebraic vs Geometric Camera Pose Estimation

# %%
import os
from src.utils.settings import MLIC_DIR
from src.utils.stream import SynchronizedVideoStream

MLIC_GEOM = os.path.join(MLIC_DIR, 'mlic_geometric_256.pkl')
MLIC_ALG  = os.path.join(MLIC_DIR, 'mlic_algebraic_256.pkl')

stream = SynchronizedVideoStream(
    streams=[
        MLIC.from_pickle(path=MLIC_GEOM, logger=PrintLogger()).to_stream(name='Geometric'),
        MLIC.from_pickle(path=MLIC_ALG, logger=PrintLogger()).to_stream(name='Algebraic')
	]
)

stream

# %%
exlude_views = {
    'Geometric' : ['light_direction_hist', 'object_frame']
}
stream.play(window_size=WINDOW_SIZE, delay=DELAY, exclude_views=exlude_views)

# %% [markdown]
# ## Train vs Test Split

# %%
mlic_train, mlic_test = mlic.train_test_split(test_size=0.05)

print(f'Original: {mlic}')
print(f'Train:    {mlic_train}')
print(f'Test:     {mlic_test}')


