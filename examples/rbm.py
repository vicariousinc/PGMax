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

# %% [markdown]
# [Restricted Boltzmann Machine (RBM)](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) is a well-known and widely used PGM for learning probabilistic distributions over binary data. We demonstrate how we can easily implement [perturb-and-max-product (PMP)](https://proceedings.neurips.cc/paper/2021/hash/07b1c04a30f798b5506c1ec5acfb9031-Abstract.html) sampling from an RBM trained on MNIST digits using PGMax. PMP is a recently proposed method for approximately sampling from a PGM by computing the maximum-a-posteriori (MAP) configuration (using max-product LBP) of a perturbed version of the model.
#
# We start by making some necessary imports.

# %%
# %matplotlib inline
import functools

import jax
import matplotlib.pyplot as plt
import numpy as np

from pgmax import fgraph, fgroup, infer, vgroup

# %% [markdown]
# The [`pgmax.fgraph`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fgraph.html#module-pgmax.fgraph) module contains classes for specifying factor graphs, the [`pgmax.fgroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fgroup.html#module-pgmax.vgroup) module contains classes for specifying groups of variables, the [`pgmax.vgroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fgroup.html#module-pgmax.fgroup) module contains classes for specifying groups of factors and the [`pgmax.infer`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.infer.html#module-pgmax.infer) module containing core functions to perform LBP.
#
# We next load the RBM trained in Sec. 5.5 of the [PMP paper](https://proceedings.neurips.cc/paper/2021/hash/07b1c04a30f798b5506c1ec5acfb9031-Abstract.html) on MNIST digits.

# %%
# Load parameters
params = np.load("example_data/rbm_mnist.npz")
bv = params["bv"]
bh = params["bh"]
W = params["W"]

# %% [markdown]
# We can then initialize the factor graph for the RBM with

# %%
# Initialize factor graph
hidden_variables = vgroup.NDVarArray(num_states=2, shape=bh.shape)
visible_variables = vgroup.NDVarArray(num_states=2, shape=bv.shape)
fg = fgraph.FactorGraph(variable_groups=[hidden_variables, visible_variables])

# %% [markdown]
# [`NDVarArray`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.NDVarArray.html#pgmax.fg.groups.NDVarArray) is a convenient class for specifying a group of variables living on a multidimensional grid with the same number of states, and shares some similarities with [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html). The [`FactorGraph`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.FactorGraph.html#pgmax.fg.graph.FactorGraph) `fg` is initialized with a set of variables, which can be either a single [`VarGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.VarGroup.html#pgmax.fg.groups.VarGroup) (e.g. an [`NDVarArray`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.NDVarArray.html#pgmax.fg.groups.NDVarArray)), or a list of [`VarGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.VarGroup.html#pgmax.fg.groups.VarGroup)s. Once initialized, the set of variables in `fg` is fixed and cannot be changed.
#
# After initialization, `fg` does not have any factors. PGMax supports imperatively adding factors to a [`FactorGraph`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.FactorGraph.html#pgmax.fg.graph.FactorGraph). We can add the unary and pairwise factors by grouping them using [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup)

# %%
# Create unary factors
hidden_unaries = fgroup.EnumFactorGroup(
    variables_for_factors=[[hidden_variables[ii]] for ii in range(bh.shape[0])],
    factor_configs=np.arange(2)[:, None],
    log_potentials=np.stack([np.zeros_like(bh), bh], axis=1),
)
visible_unaries = fgroup.EnumFactorGroup(
    variables_for_factors=[[visible_variables[jj]] for jj in range(bv.shape[0])],
    factor_configs=np.arange(2)[:, None],
    log_potentials=np.stack([np.zeros_like(bv), bv], axis=1),
)

# Create pairwise factors
log_potential_matrix = np.zeros(W.shape + (2, 2)).reshape((-1, 2, 2))
log_potential_matrix[:, 1, 1] = W.ravel()

variables_for_factors = [
    [hidden_variables[ii], visible_variables[jj]]
    for ii in range(bh.shape[0])
    for jj in range(bv.shape[0])
]
pairwise_factors = fgroup.PairwiseFactorGroup(
    variables_for_factors=variables_for_factors,
    log_potential_matrix=log_potential_matrix,
)

# Add factors to the FactorGraph
fg.add_factors([hidden_unaries, visible_unaries, pairwise_factors])


# %% [markdown]
# PGMax implements convenient and computationally efficient [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup) for representing groups of similar factors. The code above makes use of [`EnumFactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.EnumFactorGroup.html#pgmax.fg.groups.EnumFactorGroup) and [`PairwiseFactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.PairwiseFactorGroup.html#pgmax.fg.groups.PairwiseFactorGroup).
#
# A [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup) takes as argument `variables_for_factors` which is a list of lists of the variables involved in the different factors, and additional arguments specific to each [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup) (e.g. `factor_configs` or `log_potential_matrix` here).
#
# In this example, since we construct `fg` with variables `hidden_variables` and `visible_variables`, which are both [`NDVarArray`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.NDVarArray.html#pgmax.fg.groups.NDVarArray)s, we can refer to the `ii`th hidden variable as `hidden_variables[ii]` and the `jj`th visible variable as `visible_variables[jj]`.
#
# An alternative way of creating the above factors is to add them iteratively without building the [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup)s as below. This approach is not recommended as it is not computationally efficient.
# ~~~python
# from pgmax import factor
# import itertools
# from tqdm import tqdm
#
# # Add unary factors
# for ii in range(bh.shape[0]):
#     unary_factor = factor.EnumFactor(
#         variables=[hidden_variables[ii]],
#         factor_configs=np.arange(2)[:, None],
#         log_potentials=np.array([0, bh[ii]]),
#     )
#     fg.add_factors(unary_factor)
#
# for jj in range(bv.shape[0]):
#     unary_factor = factor.EnumFactor(
#         variables=[visible_variables[jj]],
#         factor_configs=np.arange(2)[:, None],
#         log_potentials=np.array([0, bv[jj]]),
#     )
#     fg.add_factors(unary_factor)
#
# # Add pairwise factors
# factor_configs = np.array(list(itertools.product(np.arange(2), repeat=2)))
# for ii in tqdm(range(bh.shape[0])):
#     for jj in range(bv.shape[0]):
#         pairwise_factor = factor.EnumFactor(
#             variables=[hidden_variables[ii], visible_variables[jj]],
#             factor_configs=factor_configs,
#             log_potentials=np.array([0, 0, 0, W[ii, jj]]),
#         )
#         fg.add_factors(pairwise_factor)
# ~~~
#
# Once we have added the factors, we can run max-product LBP and get MAP decoding by
# ~~~python
# bp = infer.BP(fg.bp_state, temperature=0.0)
# bp_arrays = bp.run_bp(bp.init(), num_iters=100, damping=0.5)
# beliefs = bp.get_beliefs(bp_arrays)
# map_states = infer.decode_map_states(beliefs)
# ~~~
# and run sum-product LBP and get estimated marginals by
# ~~~python
# bp = infer.BP(fg.bp_state, temperature=1.0)
# bp_arrays = bp.run_bp(bp.init(), num_iters=100, damping=0.5)
# beliefs = bp.get_beliefs(bp_arrays)
# marginals = infer.get_marginals(beliefs)
# ~~~
# More generally, PGMax implements LBP with temperature, with `temperature=0.0` and `temperature=1.0` corresponding to the commonly used max/sum-product LBP respectively.
#
# Now we are ready to demonstrate PMP sampling from RBM. PMP perturbs the model with [Gumbel](https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html) unary potentials, and draws a sample from the RBM as the MAP decoding from running max-product LBP on the perturbed model

# %%
bp = infer.BP(fg.bp_state, temperature=0.0)

# %%
bp_arrays = bp.init(
    evidence_updates={
        hidden_variables: np.random.gumbel(size=(bh.shape[0], 2)),
        visible_variables: np.random.gumbel(size=(bv.shape[0], 2)),
    }
)
bp_arrays = bp.run_bp(bp_arrays, num_iters=100, damping=0.5)
beliefs = bp.get_beliefs(bp_arrays)

# %% [markdown]
# Here we use the `evidence_updates` argument of `bp.init` to perturb the model with Gumbel unary potentials. In general, `evidence_updates` can be used to incorporate evidence in the form of externally applied unary potentials in PGM inference.
#
# Visualizing the MAP decoding, we see that we have sampled an MNIST digit!

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(
    infer.decode_map_states(beliefs)[visible_variables].copy().reshape((28, 28)),
    cmap="gray",
)
ax.axis("off")

# %% [markdown]
# PGMax adopts a functional interface for implementing LBP: running LBP in PGMax starts with
# ~~~python
# bp = infer.BP(fg.bp_state, temperature=T)
# ~~~
# where the arguments of the `this_bp` are several useful functions to run LBP. In particular, `bp.init`, `bp.run_bp`, `bp.get_beliefs` are pure functions with no side-effects. This design choice means that we can easily apply JAX transformations like `jit`/`vmap`/`grad`, etc., to these functions, and additionally allows PGMax to seamlessly interact with other packages in the rapidly growing JAX ecosystem (see [here](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research) and [here](https://github.com/n2cholas/awesome-jax)).
#
# As an example of applying `jax.vmap` to `bp.init`/`bp.run_bp`/`bp.get_beliefs` to process a batch of samples/models in parallel, instead of drawing one sample at a time as above, we can draw a batch of samples in parallel as follows:

# %%
n_samples = 10
bp_arrays = jax.vmap(bp.init, in_axes=0, out_axes=0)(
    evidence_updates={
        hidden_variables: np.random.gumbel(size=(n_samples, bh.shape[0], 2)),
        visible_variables: np.random.gumbel(size=(n_samples, bv.shape[0], 2)),
    },
)
bp_arrays = jax.vmap(
    functools.partial(bp.run_bp, num_iters=100, damping=0.5),
    in_axes=0,
    out_axes=0,
)(bp_arrays)

beliefs = jax.vmap(bp.get_beliefs, in_axes=0, out_axes=0)(bp_arrays)
map_states = infer.decode_map_states(beliefs)

# %% [markdown]
# Visualizing the MAP decodings, we see that we have sampled 10 MNIST digits in parallel!

# %%
fig, ax = plt.subplots(2, 5, figsize=(20, 8))
for ii in range(10):
    ax[np.unravel_index(ii, (2, 5))].imshow(
        map_states[visible_variables][ii].copy().reshape((28, 28)), cmap="gray"
    )
    ax[np.unravel_index(ii, (2, 5))].axis("off")

fig.tight_layout()
