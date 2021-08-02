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

# Custom Imports
import pgmax.fg.nodes as nodes  # isort:skip
import pgmax.interface.datatypes as interface_datatypes  # isort:skip

# fmt: on

# %%
# Instantiate all the Variables in the factor graph via VariableGroups
im_size = (30, 30)

pixel_vars = interface_datatypes.NDVariableArray(3, im_size)
hidden_vars = interface_datatypes.NDVariableArray(
    3, (im_size[0] - 2, im_size[1] - 2)
)  # Each hidden var is connected to a 3x3 patch of pixel vars
composite_vargroup = interface_datatypes.CompositeVariableGroup(
    ((0, pixel_vars), (1, hidden_vars))
)  # The 0 vs 1 key refers to the level of the VariableGroup in the hierarchy


# %%
@dataclass
class BinaryFactorGroup(interface_datatypes.FactorGroup):
    num_hidden_rows: int
    num_hidden_cols: int
    num_kernel_rows: int
    num_kernel_cols: int

    def connected_variables(self) -> List[List[Tuple[Any, ...]]]:
        ret_list: List[List[Tuple[Any, ...]]] = []
        for h_row in range(self.num_hidden_rows):
            for h_col in range(self.num_hidden_cols):
                for k_row in range(self.num_kernel_rows):
                    for k_col in range(self.num_kernel_cols):
                        ret_list.append(
                            [
                                (
                                    self.var_group[1, h_row, h_col],
                                    self.var_group[0, h_row + k_row, h_col + k_col],
                                )
                            ]
                        )
        return ret_list


# %%
# TODO: Figure out all the configurations and then instantiate the BinaryFactorGroup.
# TODO: Figure out how to set the potential function correctly from the given weights, then make it so that BP actually uses a potential function!

# %%
crbm_weights = np.load("crbm_mnist_weights_surfaces_pmap002.npz")
W, bX, bH = crbm_weights["W"], crbm_weights["bX"], crbm_weights["bH"]

print(W.shape)
print(bX.shape)
print(bH.shape)
