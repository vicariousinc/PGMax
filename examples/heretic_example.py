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
from numpy.random import default_rng  # isort:skip
from scipy import sparse  # isort:skip
from scipy.ndimage import gaussian_filter  # isort:skip
from typing import Any, Dict, Tuple, List  # isort:skip
from timeit import default_timer as timer  # isort:skip
from dataclasses import dataclass  # isort:skip
import itertools  # isort:skip

# Custom Imports
import pgmax.fg.nodes as nodes  # isort:skip
import pgmax.interface.datatypes as interface_datatypes  # isort:skip

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

# TODO: Instantiate a concrete Factor Graph for these data.
