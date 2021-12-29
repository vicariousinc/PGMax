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
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pgmax.fg import graph, groups

# %% [markdown]
# ### Construct variable grid, initialize factor graph, and add factors

# %%
variables = groups.NDVariableArray(num_states=2, shape=(50, 50))
fg = graph.FactorGraph(variables=variables)
variable_names = []
for ii in range(50):
    for jj in range(50):
        kk = (ii + 1) % 50
        ll = (jj + 1) % 50
        variable_names.append([(ii, jj), (kk, jj)])
        variable_names.append([(ii, jj), (ii, ll)])

fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    variable_names=variable_names,
    log_potential_matrix=0.8 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
    name="factors",
)

# %% [markdown]
# ### Run inference and visualize results

# %%
bp_state = fg.bp_state
run_bp, _, get_beliefs = graph.BP(bp_state, 3000)

# %%
bp_arrays = run_bp(
    evidence_updates={None: jax.device_put(np.random.gumbel(size=(50, 50, 2)))}
)

# %%
img = graph.decode_map_states(get_beliefs(bp_arrays))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img)


# %% [markdown]
# ### Gradients and batching

# %%
def loss(log_potentials_updates, evidence_updates):
    bp_arrays = run_bp(
        log_potentials_updates=log_potentials_updates, evidence_updates=evidence_updates
    )
    beliefs = get_beliefs(bp_arrays)
    loss = -jnp.sum(beliefs)
    return loss


batch_loss = jax.jit(jax.vmap(loss, in_axes=(None, {None: 0}), out_axes=0))
log_potentials_grads = jax.jit(jax.grad(loss, argnums=0))

# %%
batch_loss(None, {None: jax.device_put(np.random.gumbel(size=(10, 50, 50, 2)))})

# %%
grads = log_potentials_grads(
    {"factors": jnp.eye(2)}, {None: jax.device_put(np.random.gumbel(size=(50, 50, 2)))}
)

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
