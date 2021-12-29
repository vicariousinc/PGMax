# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import itertools

import jax
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm

from pgmax.fg import graph, groups

# %%
# Load parameters
params = np.load("example_data/rbm_mnist.npz")
bv = params["bv"]
bh = params["bh"]
W = params["W"]

# %%
# Initialize factor graph
hidden_variables = groups.NDVariableArray(num_states=2, shape=bh.shape)
visible_variables = groups.NDVariableArray(num_states=2, shape=bv.shape)
fg = graph.FactorGraph(
    variables=dict(hidden=hidden_variables, visible=visible_variables),
)

# %%
# Add unary factors
for ii in range(bh.shape[0]):
    fg.add_factor(
        variable_names=[("hidden", ii)],
        factor_configs=np.arange(2)[:, None],
        log_potentials=np.array([0, bh[ii]]),
    )

for jj in range(bv.shape[0]):
    fg.add_factor(
        variable_names=[("visible", jj)],
        factor_configs=np.arange(2)[:, None],
        log_potentials=np.array([0, bv[jj]]),
    )


# Add binary factors
factor_configs = np.array(list(itertools.product(np.arange(2), repeat=2)))
for ii in tqdm(range(bh.shape[0])):
    for jj in range(bv.shape[0]):
        fg.add_factor(
            variable_names=[("hidden", ii), ("visible", jj)],
            factor_configs=factor_configs,
            log_potentials=np.array([0, 0, 0, W[ii, jj]]),
        )

# %%
# # Add unary factors
# fg.add_factor_group(
#     factory=groups.EnumerationFactorGroup,
#     variable_names_for_factors=[[("hidden", ii)] for ii in range(bh.shape[0])],
#     factor_configs=np.arange(2)[:, None],
#     log_potentials=np.stack([np.zeros_like(bh), bh], axis=1),
# )
# fg.add_factor_group(
#     factory=groups.EnumerationFactorGroup,
#     variable_names_for_factors=[[("visible", jj)] for jj in range(bv.shape[0])],
#     factor_configs=np.arange(2)[:, None],
#     log_potentials=np.stack([np.zeros_like(bv), bv], axis=1),
# )
#
# # Add binary factors
# log_potential_matrix = np.zeros(W.shape + (2, 2)).reshape((-1, 2, 2))
# log_potential_matrix[:, 1, 1] = W.ravel()
# fg.add_factor_group(
#     factory=groups.PairwiseFactorGroup,
#     variable_names_for_factors=[
#         [("hidden", ii), ("visible", jj)]
#         for ii in range(bh.shape[0])
#         for jj in range(bv.shape[0])
#     ],
#     log_potential_matrix=log_potential_matrix,
# )

# %%
run_bp, _, get_beliefs = graph.BP(fg.bp_state, 100)

# %%
# Run inference and decode using vmap
n_samples = 16
bp_arrays = jax.vmap(run_bp, in_axes=0, out_axes=0)(
    evidence_updates={
        "hidden": np.stack(
            [
                np.zeros((n_samples,) + bh.shape),
                np.random.logistic(size=(n_samples,) + bh.shape),
            ],
            axis=-1,
        ),
        "visible": np.stack(
            [
                np.zeros((n_samples,) + bv.shape),
                np.random.logistic(size=(n_samples,) + bv.shape),
            ],
            axis=-1,
        ),
    }
)
map_states = graph.decode_map_states(
    jax.vmap(get_beliefs, in_axes=0, out_axes=0)(bp_arrays)
)

# %%
# Visualize decodings
fig, ax = plt.subplots(4, 4, figsize=(10, 10))
for ii in range(16):
    ax[np.unravel_index(ii, (4, 4))].imshow(
        map_states["visible"][ii].copy().reshape((28, 28))
    )
    ax[np.unravel_index(ii, (4, 4))].axis("off")

fig.tight_layout()
