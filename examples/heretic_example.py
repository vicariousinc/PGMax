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
#     display_name: 'Python 3.7.11 64-bit (''pgmax-zIh0MZVc-py3.7'': venv)'
#     name: python371164bitpgmaxzih0mzvcpy37venve540bb1b5cdf4292a3f5a12c4904cc40
# ---

from timeit import default_timer as timer
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp

# %%
# %matplotlib inline
# Standard Package Imports
import matplotlib.pyplot as plt
import numpy as np

import pgmax.fg.graph as graph

# Custom Imports
import pgmax.fg.groups as groups

# %% [markdown]
# # Setup Variables

# %% tags=[]
# Define some global constants
im_size = (30, 30)
prng_key = jax.random.PRNGKey(42)

# Instantiate all the Variables in the factor graph via VariableGroups
pixel_vars = groups.NDVariableArray(3, im_size)
hidden_vars = groups.NDVariableArray(
    17, (im_size[0] - 2, im_size[1] - 2)
)  # Each hidden var is connected to a 3x3 patch of pixel vars
composite_vargroup = groups.CompositeVariableGroup((pixel_vars, hidden_vars))

# %% [markdown]
# # Load Trained Weights And Setup Evidence

# %%
# Load weights and create evidence (taken directly from @lazarox's code)
crbm_weights = np.load("example_data/crbm_mnist_weights_surfaces_pmap002.npz")
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

bXn_evidence = bXn.reshape((3, 30, 30))
bXn_evidence = bXn_evidence.swapaxes(0, 1)
bXn_evidence = bXn_evidence.swapaxes(1, 2)
bHn_evidence = bHn.reshape((17, 28, 28))
bHn_evidence = bHn_evidence.swapaxes(0, 1)
bHn_evidence = bHn_evidence.swapaxes(1, 2)


# %% [markdown]
# # Create FactorGraph and Assign Evidence

# %%
# Create the factor graph
fg = graph.FactorGraph((pixel_vars, hidden_vars))

# %% [markdown]
# # Add all Factors to graph via constructing FactorGroups


# %% tags=[]
def binary_connected_variables(
    num_hidden_rows, num_hidden_cols, kernel_row, kernel_col
):
    ret_list: List[List[Tuple[Any, ...]]] = []
    for h_row in range(num_hidden_rows):
        for h_col in range(num_hidden_cols):
            ret_list.append(
                [
                    (1, h_row, h_col),
                    (0, h_row + kernel_row, h_col + kernel_col),
                ]
            )
    return ret_list


W_pot = W_orig.swapaxes(0, 1)
for k_row in range(3):
    for k_col in range(3):
        fg.add_factor(
            factor_factory=groups.PairwiseFactorGroup,
            connected_var_keys=binary_connected_variables(28, 28, k_row, k_col),
            log_potential_matrix=W_pot[:, :, k_row, k_col],
        )

# %% [markdown]
# # Construct Initial Messages

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


# NOTE: This block only works because it exploits knowledge about the order in which the flat message array is constructed within PGMax.
# Normal users won't have this...

# Create initial messages using bXn and bHn messages from
# features to pixels (taken directly from @lazarox's code)
rng, rng_input = jax.random.split(rng)
Mdown = jnp.zeros(
    (n_samples, n_cat_X, 1, f_s, f_s, im_height - f_s + 1, im_width - f_s + 1)
)
Mup = jnp.zeros(
    (n_samples, 1, n_cat_H, f_s, f_s, im_height - f_s + 1, im_width - f_s + 1)
)
Mdown = Mdown - bXn[:, :, :, :, :, 1:-1, 1:-1] / f_s ** 2
Mup = Mup - bHn / f_s ** 2

# init_weights = np.load("init_weights_mnist_surfaces_pmap002.npz")
# Mdown, Mup = init_weights["Mdown"], init_weights["Mup"]
# reshaped_Mdown = Mdown.reshape(3, 3, 3, 30, 30)
# reshaped_Mdown = reshaped_Mdown[:,:,:,1:-1, 1:-1]
reshaped_Mdown = Mdown.reshape(3, 3, 3, 28, 28)
reshaped_Mup = Mup.reshape(17, 3, 3, 28, 28)

# %% [markdown]
# # Run Belief Propagation and Retrieve MAP Estimate

# %% tags=[]
# Run BP
init_msgs = fg.get_init_msgs()
init_msgs.ftov = graph.FToVMessages(
    factor_graph=fg,
    init_value=jax.device_put(
        custom_flatten_ordering(np.array(reshaped_Mdown), np.array(reshaped_Mup))
    ),
)
init_msgs.evidence[0] = np.array(bXn_evidence)
init_msgs.evidence[1] = np.array(bHn_evidence)
bp_start_time = timer()
# Assign evidence to pixel vars
final_msgs = fg.run_bp(
    500,
    0.5,
    init_msgs=init_msgs,
)
bp_end_time = timer()
print(f"time taken for bp {bp_end_time - bp_start_time}")

# Run inference and convert result to human-readable data structure
data_writeback_start_time = timer()
map_message_dict = fg.decode_map_states(
    final_msgs,
)
data_writeback_end_time = timer()
print(
    f"time taken for data conversion of inference result {data_writeback_end_time - data_writeback_start_time}"
)


# %% [markdown]
# # Plot Results

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
        img_val = float(map_message_dict[0, row, col])
        if img_val == 2.0:
            img_val = 0.4
        img_arr[0, row, col] = img_val * 1.0

plot_images(img_arr)
