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
prng_key = jax.random.PRNGKey(42)

pixel_vars = interface_datatypes.NDVariableArray(3, im_size)
hidden_vars = interface_datatypes.NDVariableArray(
    17, (im_size[0] - 2, im_size[1] - 2)
)  # Each hidden var is connected to a 3x3 patch of pixel vars
composite_vargroup = interface_datatypes.CompositeVariableGroup(
    ((1, hidden_vars), (0, pixel_vars))
)  # The 0 vs 1 key refers to the level of the VariableGroup in the hierarchy


# %%
@dataclass
class BinaryFactorGroup(interface_datatypes.PairwiseFactorGroup):
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
# Load weights and create evidence (taken directly from @lazarox's code)
crbm_weights = np.load("crbm_mnist_weights_surfaces_pmap002.npz")
W_orig, bX, bH = crbm_weights["W"], crbm_weights["bX"], crbm_weights["bH"]
n_samples = 1
T = 1

im_height, im_width = im_size
n_cat_X, n_cat_H, f_s = W_orig.shape[:3]
W = W_orig.reshape(1, n_cat_X, n_cat_H, f_s, f_s, 1, 1)
bXn = jnp.zeros((n_samples, n_cat_X, 1, 1, 1, im_height, im_width))
border = jnp.zeros((1, n_cat_X, 1, 1, 1) + im_size)
border = border.at[:, 1:, :, :, :, :1, :].set(-10)
border = border.at[:, 1:, :, :, :, -1:, :].set(-10)
border = border.at[:, 1:, :, :, :, :, :1].set(-10)
border = border.at[:, 1:, :, :, :, :, -1:].set(-10)
bXn = bXn + border
rng, rng_input = jax.random.split(prng_key)
rnX = jax.random.gumbel(
    rng_input, shape=(n_samples, n_cat_X, 1, 1, 1, im_height, im_width)
)
bXn = bXn + bX[None, :, :, :, :, None, None] + T * rnX
rng, rng_input = jax.random.split(rng)
rnH = jax.random.gumbel(
    rng_input,
    shape=(n_samples, 1, n_cat_H, 1, 1, im_height - f_s + 1, im_width - f_s + 1),
)
bHn = bH[None, :, :, :, :, None, None] + T * rnH

# Create the evidence array by concatenating bXn and bHn
# bXn_concat = bXn.reshape((3, 30, 30)).flatten("F")
# bHn_concat = bHn.reshape((17, 28, 28)).flatten("F")
# evidence = jnp.concatenate((bXn_concat, bHn_concat))
# print(W_orig.shape)

# %%


def custom_flatten_ordering(Mdown, Mup):
    flat_idx = 0
    flat_Mdown = Mdown.flatten()
    flat_Mup = Mup.flatten()
    flattened_arr = np.zeros(
        (flat_Mdown.shape[0] + flat_Mup.shape[0]),
    )
    for kernel_row in range(Mdown.shape[1]):
        for kernel_col in range(Mdown.shape[2]):
            for row in range(Mdown.shape[3]):
                for col in range(Mdown.shape[4]):
                    flattened_arr[flat_idx : flat_idx + Mup.shape[0]] = Mup[
                        :, kernel_row, kernel_col, row, col
                    ]
                    flat_idx += Mup.shape[0]
                    flattened_arr[flat_idx : flat_idx + Mdown.shape[0]] = Mdown[
                        :, kernel_row, kernel_col, row, col
                    ]
                    flat_idx += Mdown.shape[0]
    return flattened_arr


# Create initial messages using bXn and bHn messages from
# features to pixels (taken directly from @lazarox's code)
rng, rng_input = jax.random.split(rng)
Mdown = jnp.zeros(
    (n_samples, n_cat_X, 1, f_s, f_s, im_height - f_s + 1, im_width - f_s + 1)
)
Mup = jnp.zeros(
    (n_samples, 1, n_cat_H, f_s, f_s, im_height - f_s + 1, im_width - f_s + 1)
)
# Make beliefs zero initially (seems critical for low energy solution)
Mdown = Mdown - bXn[:, :, :, :, :, 1:-1, 1:-1] / f_s ** 2
Mup = Mup - bHn / f_s ** 2

# init_weights = np.load("init_weights_mnist_surfaces_pmap002.npz")
# Mdown, Mup = init_weights["Mdown"], init_weights["Mup"]
# reshaped_Mdown = Mdown.reshape(3, 3, 3, 30, 30)
# reshaped_Mdown = reshaped_Mdown[:,:,:,1:-1, 1:-1]
reshaped_Mdown = Mdown.reshape(3, 3, 3, 28, 28)
reshaped_Mup = Mup.reshape(17, 3, 3, 28, 28)

init_msgs = jax.device_put(
    custom_flatten_ordering(np.array(reshaped_Mdown), np.array(reshaped_Mup))
)

# %%
W_pot = W_orig.swapaxes(0, 1)
binary_factor_group_list: List[BinaryFactorGroup] = []
for k_row in range(3):
    for k_col in range(3):
        binary_factor_group_list.append(
            BinaryFactorGroup(
                var_group=composite_vargroup,
                num_hidden_rows=28,
                num_hidden_cols=28,
                kernel_row=k_row,
                kernel_col=k_col,
                log_potential_matrix=W_pot[:, :, k_row, k_col],
            )
        )


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
        bXn, bHn = data  # type: ignore
        evidence_arr = np.zeros(
            self.num_var_states,
        )
        pixel_grid_shape, hidden_grid_shape, composite_vargroup = context  # type: ignore
        for row in range(pixel_grid_shape[0]):
            for col in range(pixel_grid_shape[1]):
                curr_var = composite_vargroup[0, row, col]
                evidence_arr[
                    self._vars_to_starts[curr_var] : self._vars_to_starts[curr_var]
                    + curr_var.num_states
                ] = bXn[0, :, 0, 0, 0, row, col]
        for row in range(hidden_grid_shape[0]):
            for col in range(hidden_grid_shape[1]):
                curr_var = composite_vargroup[1, row, col]
                evidence_arr[
                    self._vars_to_starts[curr_var] : self._vars_to_starts[curr_var]
                    + curr_var.num_states
                ] = bHn[0, 0, :, 0, 0, row, col]

        return jax.device_put(evidence_arr)


# %%
# Create the factor graph
fg_creation_start_time = timer()
fg = ConcreteHereticGraph(tuple(binary_factor_group_list))
fg_creation_end_time = timer()
print(f"fg Creation time = {fg_creation_end_time - fg_creation_start_time}")

# Run BP
bp_start_time = timer()
final_msgs = fg.run_bp(
    500,
    0.5,
    evidence_data=(np.array(bXn), np.array(bHn)),
    evidence_context=(im_size, (im_size[0] - 2, im_size[1] - 2), composite_vargroup),
    init_msgs=init_msgs,
)
bp_end_time = timer()
print(f"time taken for bp {bp_end_time - bp_start_time}")

# Run inference and convert result to human-readable data structure
data_writeback_start_time = timer()
map_message_dict = fg.decode_map_states(
    final_msgs,
    evidence_data=(np.array(bXn), np.array(bHn)),
    evidence_context=(im_size, (im_size[0] - 2, im_size[1] - 2), composite_vargroup),
)
data_writeback_end_time = timer()
print(
    f"time taken for data conversion of inference result {data_writeback_end_time - data_writeback_start_time}"
)


# %%
# Viz function from @lazarox's code
def plot_images(images):
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

    plt.figure(figsize=(10, 10))
    plt.imshow(big_image, interpolation="none")


# %%
img_arr = np.zeros((1, im_size[0], im_size[1]))

for row in range(im_size[0]):
    for col in range(im_size[1]):
        img_val = float(map_message_dict[composite_vargroup[0, row, col]])  # type: ignore
        if img_val == 2.0:
            img_val = 0.4
        img_arr[0, row, col] = img_val * 1.0

plot_images(img_arr)
