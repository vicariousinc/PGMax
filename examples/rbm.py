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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import itertools

import jax
import matplotlib.pyplot as plt
import numpy as np

from pgmax.fg import graph, groups

# %%
# Load parameters
params = np.load("example_data/rbm_mnist.npz")
bv = params["bv"]
bh = params["bh"]
W = params["W"]
nv = bv.shape[0]
nh = bh.shape[0]

# %%
# Build factor graph
visible_variables = groups.NDVariableArray(variable_size=2, shape=(nv,))
hidden_variables = groups.NDVariableArray(variable_size=2, shape=(nh,))
fg = graph.FactorGraph(
    variables=dict(visible=visible_variables, hidden=hidden_variables),
)
for ii in range(nh):
    for jj in range(nv):
        fg.add_factor(
            [("hidden", ii), ("visible", jj)],
            np.array(list(itertools.product(np.arange(2), repeat=2))),
            np.array([0, 0, 0, W[ii, jj]]),
        )

# %%
run_bp, _, _, decode_map_states = graph.BP(fg.bp_state, 100)

# %%
# Run inference and decode using vmap
n_samples = 16
bp_arrays = jax.vmap(run_bp, in_axes=0, out_axes=0)(
    evidence_updates={
        "hidden": np.stack(
            [
                np.zeros((n_samples,) + bh.shape),
                bh + np.random.logistic(size=(n_samples,) + bh.shape),
            ],
            axis=-1,
        ),
        "visible": np.stack(
            [
                np.zeros((n_samples,) + bv.shape),
                bv + np.random.logistic(size=(n_samples,) + bv.shape),
            ],
            axis=-1,
        ),
    }
)
map_states = jax.vmap(decode_map_states, in_axes=0, out_axes=0)(bp_arrays)

# %%
# Visualize decodings
fig, ax = plt.subplots(4, 4, figsize=(10, 10))
for ii in range(16):
    ax[np.unravel_index(ii, (4, 4))].imshow(
        map_states["visible"][ii].copy().reshape((28, 28))
    )
    ax[np.unravel_index(ii, (4, 4))].axis("off")

fig.tight_layout()
