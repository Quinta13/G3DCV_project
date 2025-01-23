# %% [markdown]
# # Interpolation Function

# %%
from src.model.mlic import MLIC
from src.utils.settings import MLIC_FILE_PATH

mlic = MLIC.from_pickle(MLIC_FILE_PATH)
mlic

# %%
import ipywidgets as widgets

from src.model.interpolation import RTIPolynomialTextureMapInterpolator, RTIRadialBasisInterpolator, MLICBasisInterpolator
from src.utils.misc import launch_widget

intepolation_function_algos = {
    'RBF': RTIRadialBasisInterpolator,
    'PTM': RTIPolynomialTextureMapInterpolator,
}
function_names = list(intepolation_function_algos.keys())

pixel_i_w                = widgets.IntSlider(value= 0, min=0, max=mlic.size[0]-1, step=1, description='Pixel X:',              continuous_update=False)
pixel_j_w                = widgets.IntSlider(value= 0, min=0, max=mlic.size[1]-1, step=1, description='Pixel Y:',              continuous_update=False)
interpolation_size_w     = widgets.IntSlider(value=48, min=10, max=100,            step=1, description='Interpolation size X:', continuous_update=False)
interpolation_function_w = widgets.Dropdown(options=function_names, value=function_names[0], description='Interpolation Function:', continous_update=False)
points_w                 = widgets.Checkbox(value=True, description='Points:',       continuous_update=False)
min_max_cbar_w           = widgets.Checkbox(value=False, description='Min Max CBAR:', continuous_update=False)
output_w                 = widgets.Output()

# %%
import matplotlib.pyplot as plt

def update(change):
    
	with output_w:

		pixel_i                 = pixel_i_w               .value
		pixel_j                 = pixel_j_w               .value
		interpolation_size      = interpolation_size_w    .value
		interpolation_function  = interpolation_function_w.value
		points                  = points_w			      .value
		min_max_cbar            = min_max_cbar_w          .value

		algo = intepolation_function_algos[interpolation_function] # type: ignore

		output_w.clear_output(wait=True)

		# print(f'Interpolating pixel ({px}, {py}) with {algo.__name__} and size {i_size}')

		mlic_bi = MLICBasisInterpolator(
			mlic=mlic,
			C_rti_interpolator=algo,
			interpolation_size=(interpolation_size, interpolation_size),
		)

		interpolated = mlic_bi.get_pixel_interpolation(pixel=(pixel_i, pixel_j))

		poits_coord = (mlic.get_pixel_values(pixel=(pixel_i, pixel_j)), mlic.light_directions) if points else None

		fig = interpolated.plot_interpolation(
			points_coord=poits_coord,
			min_max_colorbar=min_max_cbar, 
		)

		plt.show(fig)

launch_widget(widgets_=[pixel_i_w, pixel_j_w, interpolation_size_w, interpolation_function_w, points_w, min_max_cbar_w, output_w], update_fn=update)

# %%



