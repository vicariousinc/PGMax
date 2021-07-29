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
import os

import pgmax.fg.graph as graph

# Custom Imports
import pgmax.fg.nodes as nodes  # isort:skip
import pgmax.interface.datatypes as interface_datatypes  # isort:skip

# Standard Package Imports
import matplotlib.pyplot as plt  # isort:skip
import numpy as np  # isort:skip
import jax  # isort:skip
import jax.numpy as jnp  # isort:skip
from numpy.random import default_rng  # isort:skip
from scipy import sparse  # isort:skip
from scipy.ndimage import gaussian_filter  # isort:skip
from typing import Any, Dict, Tuple, List  # isort:skip
from timeit import default_timer as timer  # isort:skip
from dataclasses import dataclass  # isort:skip

# fmt: on

# %% [markdown]
# ## Setting up Image and Factor Graph

# %%
# Set random seed for rng
rng = default_rng(23)

# Make sure these environment variables are set correctly to get an accurate picture of memory usage
os.environ["XLA_PYTHON_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
print(os.getenv("XLA_PYTHON_ALLOCATOR", "default").lower())
print(os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE"))

# Create a synthetic depth image for testing purposes
im_size = 32
depth_img = 5.0 * np.ones((im_size, im_size))
depth_img[
    np.tril_indices(im_size, 0)
] = 1.0  # This sets the lower triangle of the image to 1's
depth_img = gaussian_filter(
    depth_img, sigma=0.5
)  # Filter the depth image for realistic noise simulation?
labels_img = np.zeros((im_size, im_size), dtype=np.int32)
labels_img[np.tril_indices(im_size, 0)] = 1

# Plot the depth and label images
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(depth_img)
ax[0].set_title("Depth Observation Image (yellow is higher depth than purple)")
ax[1].imshow(labels_img)
ax[1].set_title("Label Image (yellow is higher depth than purple)")

# %%
M: int = depth_img.shape[0]
N: int = depth_img.shape[1]
# Compute dI/dx (horizontal derivative)
horizontal_depth_differences = depth_img[:-1] - depth_img[1:]
# Compute dI/dy (vertical derivative)
vertical_depth_differences = depth_img[:, :-1] - depth_img[:, 1:]

# The below code block assigns values for the orientation of every horizontally-oriented cut
# It creates a matrix to represent each of the horizontal cuts in the image graph (there will be (M-1, N) possible cuts)
# Assigning a cut of 1 points to the left and a cut of 2 points to the right. A cut of 0 indicates an 'off; state
horizontal_oriented_cuts = np.zeros((M - 1, N))
horizontal_oriented_cuts[horizontal_depth_differences < 0] = 1
horizontal_oriented_cuts[horizontal_depth_differences > 0] = 2

# The below code block assigns values for the orientation of every vertically-oriented cut
# It creates a matrix to represent each of the vertical cuts in the image graph (there will be (M-1, N) possible cuts)
# Assigning a cut of 1 points up and a cut of 2 points down. A cut of 0 indicates an 'off' state
vertical_oriented_cuts = np.zeros((M, N - 1))
vertical_oriented_cuts[vertical_depth_differences < 0] = 1
vertical_oriented_cuts[vertical_depth_differences > 0] = 2
gt_has_cuts = np.zeros((2, M, N))
gt_has_cuts[0, :-1] = horizontal_oriented_cuts
gt_has_cuts[1, :, :-1] = vertical_oriented_cuts

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(horizontal_oriented_cuts)
ax[1].imshow(vertical_oriented_cuts)


# %%
# Helper function to easily generate a list of valid configurations for a given suppression diameter
def create_valid_suppression_config_arr(suppression_diameter):
    valid_suppressions_list = []
    base_list = [0] * suppression_diameter
    valid_suppressions_list.append(base_list)
    for idx in range(suppression_diameter):
        new_valid_list1 = base_list[:]
        new_valid_list2 = base_list[:]
        new_valid_list1[idx] = 1
        new_valid_list2[idx] = 2
        valid_suppressions_list.append(new_valid_list1)
        valid_suppressions_list.append(new_valid_list2)
    ret_arr = np.array(valid_suppressions_list)
    ret_arr.flags.writeable = False
    return ret_arr


# %%
# Before constructing a Factor Graph, we specify all the valid configurations for the non-suppression factors
"""
      1v
0h  factor  2h
      3v
"""
valid_configs_non_supp = np.array(
    [
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [2, 0, 2, 0],
        [0, 0, 1, 1],
        [0, 0, 2, 2],
        [2, 0, 0, 1],
        [1, 0, 0, 2],
        [1, 0, 1, 1],
        [2, 0, 2, 1],
        [1, 0, 1, 2],
        [2, 0, 2, 2],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 2, 0],
        [1, 1, 0, 1],
        [2, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 2, 1],
        [1, 1, 1, 0],
        [2, 1, 2, 0],
        [0, 2, 0, 2],
        [2, 2, 0, 0],
        [0, 2, 1, 0],
        [2, 2, 0, 2],
        [1, 2, 0, 2],
        [0, 2, 2, 2],
        [0, 2, 1, 2],
        [2, 2, 2, 0],
        [1, 2, 1, 0],
    ]
)
valid_configs_non_supp.flags.writeable = False
# Now, we specify the valid configurations for all the suppression factors
SUPPRESSION_DIAMETER = 9
valid_configs_supp = create_valid_suppression_config_arr(SUPPRESSION_DIAMETER)


# %%
# Subclass FactorGroup into the 3 different groups that appear in this problem


@dataclass
class FourFactorGroup(interface_datatypes.FactorGroup):
    num_rows: int
    num_cols: int
    grid_vars_group: interface_datatypes.NDVariableArray
    additional_keys_group: interface_datatypes.KeyTupleVariableGroup

    def _get_connected_var_keys_for_factors(
        self,
    ) -> List[List[Tuple[Tuple[int, ...], interface_datatypes.VariableGroup]]]:
        ret_list: List[
            List[Tuple[Tuple[int, ...], interface_datatypes.VariableGroup]]
        ] = []
        for row in range(self.num_rows - 1):
            for col in range(self.num_cols - 1):
                if row != self.num_rows - 2 and col != self.num_cols - 2:
                    ret_list.append(
                        [
                            ((0, row, col), self.grid_vars_group),
                            ((1, row, col), self.grid_vars_group),
                            ((0, row, col + 1), self.grid_vars_group),
                            ((1, row + 1, col), self.grid_vars_group),
                        ]
                    )
                elif row != self.num_rows - 2:
                    ret_list.append(
                        [
                            ((0, row, col), self.grid_vars_group),
                            ((1, row, col), self.grid_vars_group),
                            ((0, row, col + 1), self.additional_keys_group),
                            ((1, row + 1, col), self.grid_vars_group),
                        ]
                    )
                elif col != self.num_cols - 2:
                    ret_list.append(
                        [
                            ((0, row, col), self.grid_vars_group),
                            ((1, row, col), self.grid_vars_group),
                            ((0, row, col + 1), self.grid_vars_group),
                            ((1, row + 1, col), self.additional_keys_group),
                        ]
                    )
                else:
                    ret_list.append(
                        [
                            ((0, row, col), self.grid_vars_group),
                            ((1, row, col), self.grid_vars_group),
                            ((0, row, col + 1), self.additional_keys_group),
                            ((1, row + 1, col), self.additional_keys_group),
                        ]
                    )

        return ret_list


@dataclass
class VertSuppressionFactorGroup(interface_datatypes.FactorGroup):
    num_rows: int
    num_cols: int
    suppression_diameter: int
    grid_vars_group: interface_datatypes.NDVariableArray
    additional_keys_group: interface_datatypes.KeyTupleVariableGroup

    def _get_connected_var_keys_for_factors(
        self,
    ) -> List[List[Tuple[Tuple[int, ...], interface_datatypes.VariableGroup]]]:
        ret_list: List[
            List[Tuple[Tuple[int, ...], interface_datatypes.VariableGroup]]
        ] = []
        for col in range(self.num_cols):
            for start_row in range(self.num_rows - self.suppression_diameter):
                if col != self.num_cols - 1:
                    ret_list.append(
                        [
                            ((0, r, col), self.grid_vars_group)
                            for r in range(
                                start_row, start_row + self.suppression_diameter
                            )
                        ]
                    )
                else:
                    ret_list.append(
                        [
                            ((0, r, col), self.additional_keys_group)
                            for r in range(
                                start_row, start_row + self.suppression_diameter
                            )
                        ]
                    )

        return ret_list


@dataclass
class HorzSuppressionFactorGroup(interface_datatypes.FactorGroup):
    num_rows: int
    num_cols: int
    suppression_diameter: int
    grid_vars_group: interface_datatypes.NDVariableArray
    additional_keys_group: interface_datatypes.KeyTupleVariableGroup

    def _get_connected_var_keys_for_factors(
        self,
    ) -> List[List[Tuple[Tuple[int, ...], interface_datatypes.VariableGroup]]]:
        ret_list: List[
            List[Tuple[Tuple[int, ...], interface_datatypes.VariableGroup]]
        ] = []
        for row in range(self.num_rows):
            for start_col in range(self.num_cols - self.suppression_diameter):
                if row != self.num_rows - 1:
                    ret_list.append(
                        [
                            ((1, row, c), self.grid_vars_group)
                            for c in range(
                                start_col, start_col + self.suppression_diameter
                            )
                        ]
                    )
                else:
                    ret_list.append(
                        [
                            ((1, row, c), self.additional_keys_group)
                            for c in range(
                                start_col, start_col + self.suppression_diameter
                            )
                        ]
                    )
        return ret_list


# %%
# We create a NDVariableArray such that the [0,i,j] entry corresponds to the vertical cut variable (i.e, the one
# attached horizontally to the factor) that's at that location in the image, and the [1,i,j] entry corresponds to
# the horizontal cut variable (i.e, the one attached vertically to the factor) that's at that location
grid_vars_group = interface_datatypes.NDVariableArray(3, (2, M - 1, N - 1))

extra_row_keys: List[Tuple[Any, ...]] = [(0, row, N - 1) for row in range(M - 1)]
extra_col_keys: List[Tuple[Any, ...]] = [(1, M - 1, col) for col in range(N - 1)]
additional_keys = tuple(extra_row_keys + extra_col_keys)
additional_keys_group = interface_datatypes.KeyTupleVariableGroup(3, additional_keys)

# %%
# Now, we instantiate the four factors
four_factors_group = FourFactorGroup(
    valid_configs_non_supp, M, N, grid_vars_group, additional_keys_group
)
# Next, we instantiate all the vertical suppression variables
vert_suppression_group = VertSuppressionFactorGroup(
    valid_configs_supp,
    M,
    N,
    SUPPRESSION_DIAMETER,
    grid_vars_group,
    additional_keys_group,
)
# Next, we instantiate all the horizontal suppression variables
horz_suppression_group = HorzSuppressionFactorGroup(
    valid_configs_supp,
    M,
    N,
    SUPPRESSION_DIAMETER,
    grid_vars_group,
    additional_keys_group,
)

# Finally, we construct the tuple of all the factors and variables involved in the problem.
facs_tuple = tuple(
    list(four_factors_group.factors)
    + list(vert_suppression_group.factors)
    + list(horz_suppression_group.factors)
)
vars_tuple = tuple(
    list(grid_vars_group.get_all_vars()) + list(additional_keys_group.get_all_vars())
)


# %%
# Override and define a concrete FactorGraph Class with the get_evidence function implemented
class ConcreteFactorGraph(graph.FactorGraph):
    def get_evidence(
        self, data: Dict[nodes.Variable, np.array], context: Any = None
    ) -> jnp.ndarray:
        """Function to generate evidence array. Need to be overwritten for concrete factor graphs

        Args:
            data: Data for generating evidence
            context: Optional context for generating evidence

        Returns:
            None, but must set the self._evidence attribute to a jnp.array of shape (num_var_states,)
        """
        evidence = np.zeros(self.num_var_states)
        for var in self.variables:
            start_index = self._vars_to_starts[var]
            evidence[start_index : start_index + var.num_states] = data[var]
        return jax.device_put(evidence)

    def output_inference(
        self, final_var_states: jnp.ndarray, context: Any = None
    ) -> Any:
        """Function to take the result of message passing and output the inference result for
            each variable

        Args:
            final_var_states: an array of shape (num_var_states,) that is the result of belief
                propagation
            context: Optional context for using this array

        Returns:
            An evidence array of shape (num_var_states,)
        """
        # NOTE: An argument can be passed here to do different inferences for sum-product and
        # max-product respectively
        var_to_map_dict = {}
        final_var_states_np = np.array(final_var_states)
        for var in self.variables:
            start_index = self._vars_to_starts[var]
            var_to_map_dict[var] = np.argmax(
                final_var_states_np[start_index : start_index + var.num_states]
            )

        return var_to_map_dict


# %%
gt_has_cuts = gt_has_cuts.astype(np.int32)

# Now, we use this array along with the gt_has_cuts array computed earlier using the image in order to derive the evidence values
var_evidence_dict = {}
for i in range(2):
    for row in range(M):
        for col in range(N):
            # The dictionary key is in grid_vars_group at loc [i,row,call]
            evidence_arr = np.zeros(
                3
            )  # Note that we know num states for each variable is 3, so we can do this
            evidence_arr[
                gt_has_cuts[i, row, col]
            ] = 2.0  # This assigns belief value 2.0 to the correct index in the evidence vector
            evidence_arr = (
                evidence_arr - evidence_arr[0]
            )  # This normalizes the evidence by subtracting away the 0th index value
            evidence_arr[1:] += 0.1 * rng.logistic(
                size=evidence_arr[1:].shape
            )  # This adds logistic noise for every evidence entry
            try:
                var_evidence_dict[grid_vars_group[i, row, col]] = evidence_arr
            except ValueError:
                try:
                    var_evidence_dict[additional_keys_group[i, row, col]] = evidence_arr
                except ValueError:
                    pass


# %% [markdown]
# ## Belief Propagation

# %%
# Create the factor graph
fg_creation_start_time = timer()
fg = ConcreteFactorGraph(vars_tuple, facs_tuple)
fg_creation_end_time = timer()
print(f"fg Creation time = {fg_creation_end_time - fg_creation_start_time}")

# Run BP
bp_start_time = timer()
final_msgs = fg.run_bp(1000, 0.5, evidence_data=var_evidence_dict)
bp_end_time = timer()
print(f"time taken for bp {bp_end_time - bp_start_time}")

# Run inference and convert result to human-readable data structure
data_writeback_start_time = timer()
map_message_dict = fg.decode_map_states(final_msgs, evidence_data=var_evidence_dict)
data_writeback_end_time = timer()
print(
    f"time taken for data conversion of inference result {data_writeback_end_time - data_writeback_start_time}"
)

# %% [markdown]
# ## Visualization of Results

# %% tags=[]
# Place the variable values derived from BP onto an image-sized array so they can be visualized. Do the same for bottom-up evidences that are just GT + logistic noise
bp_values = np.zeros((2, M, N))
bu_evidence = np.zeros((2, M, N, 3))
for i in range(2):
    for row in range(M):
        for col in range(N):
            try:
                bp_values[i, row, col] = map_message_dict[grid_vars_group[i, row, col]]  # type: ignore
                bu_evidence[i, row, col, :] = var_evidence_dict[
                    grid_vars_group[i, row, col]
                ]
            except ValueError:
                try:
                    bp_values[i, row, col] = map_message_dict[additional_keys_group[i, row, col]]  # type: ignore
                    bu_evidence[i, row, col, :] = var_evidence_dict[
                        additional_keys_group[i, row, col]
                    ]
                except ValueError:
                    pass


# %%
# Helpful function for viz
def get_color_mask(image, nc=None):
    image = image.astype(int)
    n_colors = image.max() + 1

    cm = plt.get_cmap("gist_rainbow")
    colors = [cm(1.0 * i / n_colors) for i in np.random.permutation(n_colors)]

    color_mask = np.zeros(image.shape + (3,)).astype(np.uint8)
    for i in np.unique(image):
        color_mask[image == i, :] = np.array(colors[i][:3]) * 255
    return color_mask


# %%
def get_surface_labels_from_cuts(has_cuts):
    """get_surface_labels_from_cuts

    Parameters
    ----------
    has_cuts : np.array
        Array of shape (2, M, N)
    Returns
    -------
    surface_labels : np.array
        Array of shape (M, N)
        Surface labels of each pixel
    """
    M, N = has_cuts.shape[1:]
    # Indices for 4-connected grid
    nodes_indices0 = (np.arange(1, M) - 1)[:, None] * N + np.arange(N)
    nodes_indices1 = (np.arange(M - 1) + 1)[:, None] * N + np.arange(N)
    nodes_indices2 = np.arange(M)[:, None] * N + np.arange(1, N) - 1
    nodes_indices3 = np.arange(M)[:, None] * N + np.arange(N - 1) + 1
    row_indices_for_grid = np.concatenate(
        [nodes_indices0.ravel(), nodes_indices2.ravel()]
    )
    col_indices_for_grid = np.concatenate(
        [nodes_indices1.ravel(), nodes_indices3.ravel()]
    )
    # Indices for cuts
    horizontal_row_indices_for_cuts, horizontal_col_indices_for_cuts = np.nonzero(
        has_cuts[0, :-1]
    )
    vertical_row_indices_for_cuts, vertical_col_indices_for_cuts = np.nonzero(
        has_cuts[1, :, :-1]
    )
    row_indices_for_cuts = np.concatenate(
        [
            horizontal_row_indices_for_cuts * N + horizontal_col_indices_for_cuts,
            vertical_row_indices_for_cuts * N + vertical_col_indices_for_cuts,
        ]
    )
    col_indices_for_cuts = np.concatenate(
        [
            (horizontal_row_indices_for_cuts + 1) * N + horizontal_col_indices_for_cuts,
            vertical_row_indices_for_cuts * N + (vertical_col_indices_for_cuts + 1),
        ]
    )
    csgraph = sparse.lil_matrix((M * N, M * N), dtype=np.int32)
    csgraph[row_indices_for_grid, col_indices_for_grid] = 1
    csgraph[col_indices_for_grid, row_indices_for_grid] = 1
    csgraph[row_indices_for_cuts, col_indices_for_cuts] = 0
    csgraph[col_indices_for_cuts, row_indices_for_cuts] = 0
    n_connected_components, surface_labels = sparse.csgraph.connected_components(
        csgraph.tocsr(), directed=False, return_labels=True
    )
    surface_labels = np.random.permutation(n_connected_components)[
        surface_labels.reshape((M, N))
    ]
    return surface_labels


# %%
# Ground truth cuts
gt_cuts_img = np.zeros((2 * M, 2 * N))
gt_cuts_img[
    np.arange(1, 2 * M, 2).reshape((-1, 1)), np.arange(0, 2 * N, 2).reshape((1, -1))
] = gt_has_cuts[0]
gt_cuts_img[
    np.arange(0, 2 * M, 2).reshape((-1, 1)), np.arange(1, 2 * N, 2).reshape((1, -1))
] = gt_has_cuts[1]

# Bottom-up evidences for cuts
bu_has_cuts = np.argmax(bu_evidence, axis=-1)
bu_cuts_img = np.zeros((2 * M, 2 * N))
bu_cuts_img[
    np.arange(1, (2 * M), 2).reshape((-1, 1)), np.arange(0, (2 * N), 2).reshape((1, -1))
] = bu_has_cuts[0]
bu_cuts_img[
    np.arange(0, (2 * M), 2).reshape((-1, 1)), np.arange(1, (2 * N), 2).reshape((1, -1))
] = bu_has_cuts[1]

# Predicted cuts
cuts_img = np.zeros((2 * M, 2 * N))
cuts_img[
    np.arange(1, 2 * M, 2).reshape((-1, 1)), np.arange(0, 2 * N, 2).reshape((1, -1))
] = bp_values[0]
cuts_img[
    np.arange(0, 2 * M, 2).reshape((-1, 1)), np.arange(1, 2 * N, 2).reshape((1, -1))
] = bp_values[1]

# Plot ground-truth cuts
fig, ax = plt.subplots(2, 3, figsize=(30, 20))
ax[0, 0].imshow(gt_cuts_img)
ax[0, 0].set_title("Ground truth", fontsize=40)
ax[0, 0].axis("off")
ax[1, 0].imshow(get_color_mask(labels_img))
ax[1, 0].axis("off")

# Plot bottom-up evidences for cuts
ax[0, 1].imshow(bu_cuts_img)
ax[0, 1].axis("off")
ax[0, 1].set_title("Using bottom-up evidences", fontsize=40)
ax[1, 1].imshow(get_color_mask(get_surface_labels_from_cuts(bu_has_cuts > 0)))
ax[1, 1].axis("off")

# Plot predicted cuts
ax[0, 2].imshow(cuts_img)
ax[0, 2].axis("off")
ax[0, 2].set_title("Using surface model", fontsize=40)
ax[1, 2].imshow(get_color_mask(get_surface_labels_from_cuts(bp_values > 0)))
ax[1, 2].axis("off")
fig.tight_layout()
