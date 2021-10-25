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
import jax
import matplotlib.pyplot as plt
import numpy as np

from pgmax.fg import graph, groups, transforms

# %% [markdown]
# ### Construct variable grid, initialize factor graph, and add factors

# %%
variables = groups.NDVariableArray(variable_size=2, shape=(50, 50))
fg = graph.FactorGraph(variables=variables)
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
    name="factors",
)

# %% [markdown]
# ### Run inference and visualize results

# %%
run_bp, get_bp_state = transforms.BP(fg.bp_state, 3000)

# %%
ftov_msgs = run_bp(
    evidence_updates={None: jax.device_put(np.random.gumbel(size=(50, 50, 2)))}
)
bp_state = get_bp_state(ftov_msgs)

# %%
decode_map_states = transforms.DecodeMAPStates(bp_state)
map_states = decode_map_states()
img = np.zeros((50, 50))
for key in map_states:
    img[key] = map_states[key]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img)

# %% [markdown]
# ### Message and evidence manipulation

# %%
# Query evidence for variable (0, 0)
bp_state.evidence[0, 0]

# %%
# Set evidence for variable (0, 0)
bp_state.evidence[0, 0] = np.array([1.0, 1.0])
bp_state.evidence[0, 0]

# %%
# Set evidence for all variables using an array
evidence = np.random.randn(50, 50, 2)
bp_state.evidence[None] = evidence
bp_state.evidence[10, 10] == evidence[10, 10]

# %%
# Query messages from the factor involving (0, 0), (0, 1) in factor group "factors" to variable (0, 0)
bp_state.ftov_msgs[[(0, 0), (0, 1)], (0, 0)]

# %%
# Set messages from the factor involving (0, 0), (0, 1) in factor group "factors" to variable (0, 0)
bp_state.ftov_msgs[[(0, 0), (0, 1)], (0, 0)] = np.array([1.0, 1.0])
bp_state.ftov_msgs[[(0, 0), (0, 1)], (0, 0)]

# %%
# Uniformly spread expected belief at a variable to all connected factors
bp_state.ftov_msgs[0, 0] = np.array([1.0, 1.0])
bp_state.ftov_msgs[[(0, 0), (0, 1)], (0, 0)]
