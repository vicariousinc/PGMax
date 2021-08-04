# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: 'Python 3.8.5 64-bit (''pgmax-JcKb81GE-py3.8'': poetry)'
#     name: python3
# ---

# %%
# %matplotlib inline
# fmt: off

# Standard Package Imports
import matplotlib.pyplot as plt  # isort:skip
import numpy as np  # isort:skip
import jax  # isort:skip
import jax.numpy as jnp  # isort:skip
from typing import Any, Tuple, List  # isort:skip
from timeit import default_timer as timer  # isort:skip
from dataclasses import dataclass  # isort:skip

# Custom Imports
import pgmax.interface.datatypes as interface_datatypes  # isort:skip
import pgmax.fg.graph as graph  # isort:skip

# fmt: on

# %%
# Instantiate all the Variables in the factor graph via VariableGroups
im_size = (30, 30)

pixel_vars = interface_datatypes.NDVariableArray(3, im_size)
hidden_vars = interface_datatypes.NDVariableArray(
    17, (im_size[0] - 2, im_size[1] - 2)
)  # Each hidden var is connected to a 3x3 patch of pixel vars
composite_vargroup = interface_datatypes.CompositeVariableGroup(
    ((0, pixel_vars), (1, hidden_vars))
)  # The 0 vs 1 key refers to the level of the VariableGroup in the hierarchy


# %%
@dataclass
class BinaryFactorGroup(interface_datatypes.FactorGroup):
    num_hidden_rows: int
    num_hidden_cols: int
    kernel_row: int
    kernel_col: int

    def connected_variables(self) -> List[List[Tuple[Any, ...]]]:
        ret_list: List[List[Tuple[Any, ...]]] = []
        for h_row in range(self.num_hidden_rows):
            for h_col in range(self.num_hidden_cols):
                ret_list.append(
                    [
                        (1, h_row, h_col),
                        (0, h_row + self.kernel_row, h_col + self.kernel_col),
                    ]
                )
        return ret_list


# %%
crbm_weights = np.load("crbm_mnist_weights_surfaces_pmap002.npz")
W, _, _ = crbm_weights["W"], crbm_weights["bX"], crbm_weights["bH"]
W = W.swapaxes(0, 1)
print(W.shape)

# %%
# We know there are 17 states for every hidden var and 3 for every pixel var, so we just need to get a list of their inner product
factor_valid_configs = np.array([[h_s, p_s] for h_s in range(17) for p_s in range(3)])
# We make 1 BinaryFactorGroup for every index in the 3x3 convolutional kernel grid
binary_factor_group_list = [
    BinaryFactorGroup(
        factor_valid_configs,
        W[..., k_row, k_col],
        composite_vargroup,
        28,
        28,
        k_row,
        k_col,
    )
    for k_row in range(3)
    for k_col in range(3)
]

# %%
# Generate tuples of all the factors and variables
vars_tuple = composite_vargroup.get_all_vars()
facs_tuple = sum([fac_group.factors for fac_group in binary_factor_group_list], ())


# %%
class ConcreteHereticGraph(graph.FactorGraph):
    def get_evidence(self, data: Any = None, context: Any = None) -> jnp.ndarray:
        """Function to generate evidence array. Need to be overwritten for concrete factor graphs

        Args:
            data: Data for generating evidence
            context: Optional context for generating evidence

        Returns:
            Array of shape (num_var_states,) representing the flattened evidence for each variable
        """
        prng_key = jax.random.PRNGKey(42)
        return jax.random.gumbel(prng_key, (self.num_var_states,))


# %%
# Create the factor graph
fg_creation_start_time = timer()
fg = ConcreteHereticGraph(vars_tuple, facs_tuple)
fg_creation_end_time = timer()
print(f"fg Creation time = {fg_creation_end_time - fg_creation_start_time}")

# Run BP
bp_start_time = timer()
final_msgs = fg.run_bp(500, 1.0)
bp_end_time = timer()
print(f"time taken for bp {bp_end_time - bp_start_time}")

# Run inference and convert result to human-readable data structure
data_writeback_start_time = timer()
map_message_dict = fg.decode_map_states(final_msgs)
data_writeback_end_time = timer()
print(
    f"time taken for data conversion of inference result {data_writeback_end_time - data_writeback_start_time}"
)


# %%
# Viz function from @lazarox's code
def plot_images(images, zoom_times=0, filename=None, display=True, return_image=False):
    n_images, H, W = images.shape
    images = images - images.min()
    images /= images.max() + 1e-10

    nr = nc = np.ceil(np.sqrt(n_images)).astype(int)
    big_image = np.ones(((H + 1) * nr + 1, (W + 1) * nc + 1, 3))
    big_image[..., :2] = 0
    im = 0
    for r in range(nr):
        for c in range(nc):
            if im < n_images:
                big_image[
                    (H + 1) * r + 1 : (H + 1) * r + 1 + H,
                    (W + 1) * c + 1 : (W + 1) * c + 1 + W,
                    :,
                ] = images[im, :, :, None]
            im += 1

    if display and filename is None:
        plt.figure(figsize=(10, 10))
        plt.imshow(big_image, interpolation="none")
    for i in range(zoom_times):
        big_image = ndimage.zoom(big_image, [2, 2, 1], order=0)
    if filename:
        imwrite(filename, img_as_ubyte(big_image))
    if return_image:
        return big_image


# %%
img_arr = np.zeros((1, im_size[0], im_size[1]))

for row in range(im_size[0]):
    for col in range(im_size[1]):
        img_arr[0, row, col] = map_message_dict[composite_vargroup[0, row, col]]  # type: ignore

plot_images(img_arr)
