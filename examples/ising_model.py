# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pgmax.fg import graph
from pgmax.groups import enumeration
from pgmax.groups import variables as vgroup

# %% [markdown]
# ### Construct variable grid, initialize factor graph, and add factors

# %%
variables = vgroup.NDVariableArray(num_states=2, shape=(50, 50))
fg = graph.FactorGraph(variables=variables)

variables_for_factors = []
for ii in range(50):
    for jj in range(50):
        kk = (ii + 1) % 50
        ll = (jj + 1) % 50
        variables_for_factors.append([variables[ii, jj], variables[kk, jj]])
        variables_for_factors.append([variables[ii, jj], variables[kk, ll]])

fg.add_factor_group(
    factory=enumeration.PairwiseFactorGroup,
    variables_for_factors=variables_for_factors,
    log_potential_matrix=0.8 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
    name="factors",
)

# %% [markdown]
# ### Run inference and visualize results

# %%
bp = graph.BP(fg.bp_state, temperature=0)

# %%
# TODO: check\ time for before BP vs time for BP
# TODO: time PGMAX vs PMP
bp_arrays = bp.init(
    evidence_updates={variables: jax.device_put(np.random.gumbel(size=(50, 50, 2)))}
)
bp_arrays = bp.run_bp(bp_arrays, num_iters=3000)
beliefs = bp.get_beliefs(bp_arrays)

# %%
img = graph.decode_map_states(beliefs)[variables]
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img)


# %% [markdown]
# ### Gradients and batching

# %%
def loss(log_potentials_updates, evidence_updates):
    bp_arrays = bp.init(
        log_potentials_updates=log_potentials_updates, evidence_updates=evidence_updates
    )
    bp_arrays = bp.run_bp(bp_arrays, num_iters=3000)
    beliefs = bp.get_beliefs(bp_arrays)
    loss = -jnp.sum(beliefs[variables])
    return loss


batch_loss = jax.jit(jax.vmap(loss, in_axes=(None, {variables: 0}), out_axes=0))
log_potentials_grads = jax.jit(jax.grad(loss, argnums=0))

# %%
batch_loss(None, {variables: jax.device_put(np.random.gumbel(size=(10, 50, 50, 2)))})

# %%
grads = log_potentials_grads(
    {"factors": jnp.eye(2)},
    {variables: jax.device_put(np.random.gumbel(size=(50, 50, 2)))},
)

# %% [markdown]
# ### Message and evidence manipulation

# %%
bp_state = bp.to_bp_state(bp_arrays)

# Query evidence for variable (0, 0)
bp_state.evidence[variables[0, 0]]

# %%
# Set evidence for variable (0, 0)
bp_state.evidence[variables[0, 0]] = np.array([1.0, 1.0])
bp_state.evidence[variables[0, 0]]

# %%
# Set evidence for all variables using an array
evidence = np.random.randn(50, 50, 2)
bp_state.evidence[variables] = evidence
np.allclose(bp_state.evidence[variables[10, 10]], evidence[10, 10])
