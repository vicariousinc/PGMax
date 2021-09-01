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
import matplotlib.pyplot as plt
import numpy as np

from pgmax.fg import graph, groups

# %% [markdown]
# ### Construct variable grid and initialize factor graph

# %%
variables = groups.NDVariableArray(variable_size=2, shape=(50, 50))
fg = graph.FactorGraph(variables=variables, evidence_default_mode="random")

# %% [markdown]
# ### Add factors to the factor graph

# %%
connected_var_keys = []
for ii in range(50):
    for jj in range(50):
        kk = (ii + 1) % 50
        ll = (jj + 1) % 50
        connected_var_keys.append([(ii, jj), (kk, jj)])
        connected_var_keys.append([(ii, jj), (ii, ll)])

fg.add_factor(
    factor_factory=groups.PairwiseFactorGroup,
    connected_var_keys=connected_var_keys,
    log_potential_matrix=0.8 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
)

# %% [markdown]
# ### Run inference and visualize results

# %%
msgs = fg.run_bp(3000, 0.5)
map_states = fg.decode_map_states(msgs)
img = np.zeros((50, 50))
for key in map_states:
    img[key] = map_states[key]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img)
