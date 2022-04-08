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

from pgmax.fg import graph
from pgmax.groups import enumeration
from pgmax.groups import variables as vgroup

# %% [markdown]
# The [`pgmax.fg.graph`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.html#module-pgmax.fg.graph) module contains core classes for specifying factor graphs and implementing LBP, while the [`pgmax.fg.groups`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.html#module-pgmax.fg.graph) module contains classes for specifying groups of variables/factors.
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
hidden_variables = vgroup.NDVariableArray(num_states=2, shape=bh.shape)
visible_variables = vgroup.NDVariableArray(num_states=2, shape=bv.shape)
fg = graph.FactorGraph(
    variables=dict(hidden=hidden_variables, visible=visible_variables),
)

# %% [markdown]
# [`NDVariableArray`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.NDVariableArray.html#pgmax.fg.groups.NDVariableArray) is a convenient class for specifying a group of variables living on a multidimensional grid with the same number of states, and shares some similarities with [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html). The [`FactorGraph`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.FactorGraph.html#pgmax.fg.graph.FactorGraph) `fg` is initialized with a set of variables, which can be either a single [`VariableGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.VariableGroup.html#pgmax.fg.groups.VariableGroup) (e.g. an [`NDVariableArray`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.NDVariableArray.html#pgmax.fg.groups.NDVariableArray)), or a list/dictionary of [`VariableGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.VariableGroup.html#pgmax.fg.groups.VariableGroup)s. Once initialized, the set of variables in `fg` is fixed and cannot be changed.
#
# After initialization, `fg` does not have any factors. PGMax supports imperatively adding factors to a [`FactorGraph`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.FactorGraph.html#pgmax.fg.graph.FactorGraph). We can add the unary and pairwise factors by grouping them using

# %%
# Add unary factors
fg.add_factor_group(
    factory=enumeration.EnumerationFactorGroup,
    variable_names_for_factors=[[("hidden", ii)] for ii in range(bh.shape[0])],
    factor_configs=np.arange(2)[:, None],
    log_potentials=np.stack([np.zeros_like(bh), bh], axis=1),
)

fg.add_factor_group(
    factory=enumeration.EnumerationFactorGroup,
    variable_names_for_factors=[[("visible", jj)] for jj in range(bv.shape[0])],
    factor_configs=np.arange(2)[:, None],
    log_potentials=np.stack([np.zeros_like(bv), bv], axis=1),
)

# Add pairwise factors
log_potential_matrix = np.zeros(W.shape + (2, 2)).reshape((-1, 2, 2))
log_potential_matrix[:, 1, 1] = W.ravel()

fg.add_factor_group(
    factory=enumeration.PairwiseFactorGroup,
    variable_names_for_factors=[
        [("hidden", ii), ("visible", jj)]
        for ii in range(bh.shape[0])
        for jj in range(bv.shape[0])
    ],
    log_potential_matrix=log_potential_matrix,
)


# %% [markdown]
# PGMax implements convenient and computationally efficient [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup) for representing Groups of similar factors. The code above makes use of [`EnumerationFactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.EnumerationFactorGroup.html#pgmax.fg.groups.EnumerationFactorGroup) and [`PairwiseFactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.PairwiseFactorGroup.html#pgmax.fg.groups.PairwiseFactorGroup), two [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup)s implemented in the [`pgmax.fg.groups`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.html#module-pgmax.fg.graph) module.
#
# A [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup) is created by calling [`fg.add_factor_group`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.FactorGraph.html#pgmax.fg.graph.FactorGraph.add_factor_group), which takes 2 arguments: `factory` which specifies the [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup) subclass, `variable_names_for_factors` which is a list of lists containing the name of the involved variables in the different factors, and additional arguments for the [`FactorGroup`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.FactorGroup.html#pgmax.fg.groups.FactorGroup) (e.g. `factor_configs` or `log_potential_matrix` here).
#
# In this example, since we construct `fg` with variables `dict(hidden=hidden_variables, visible=visible_variables)`, where `hidden_variables` and `visible_variables` are [`NDVariableArray`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.groups.NDVariableArray.html#pgmax.fg.groups.NDVariableArray)s, we can refer to the `ii`th hidden variable as `("hidden", ii)` and the `jj`th visible variable as `("visible", jj)`. In general, PGMax implements an intuitive scheme for automatically assigning names to the variables in a [`FactorGraph`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.FactorGraph.html#pgmax.fg.graph.FactorGraph).
#
# An alternative way of creating the above factors is to add them iteratively by calling [`fg.add_factor`](https://pgmax.readthedocs.io/en/latest/_autosummary/pgmax.fg.graph.FactorGraph.html#pgmax.fg.graph.FactorGraph.add_factor) as below. This approach is not recommended as it is not computationally efficient.
# ~~~python
# import itertools
# from tqdm import tqdm
#
# # Add unary factors
# for ii in range(bh.shape[0]):
#     fg.add_factor(
#         variable_names=[("hidden", ii)],
#         factor_configs=np.arange(2)[:, None],
#         log_potentials=np.array([0, bh[ii]]),
#     )
#
# for jj in range(bv.shape[0]):
#     fg.add_factor(
#         variable_names=[("visible", jj)],
#         factor_configs=np.arange(2)[:, None],
#         log_potentials=np.array([0, bv[jj]]),
#     )
#
# # Add pairwise factors
# factor_configs = np.array(list(itertools.product(np.arange(2), repeat=2)))
# for ii in tqdm(range(bh.shape[0])):
#     for jj in range(bv.shape[0]):
#         fg.add_factor(
#             variable_names=[("hidden", ii), ("visible", jj)],
#             factor_configs=factor_configs,
#             log_potentials=np.array([0, 0, 0, W[ii, jj]]),
#         )
# ~~~
#
# Once we have added the factors, we can run max-product LBP and get MAP decoding by
# ~~~python
# bp = graph.BP(fg.bp_state, temperature=0.0)
# bp_arrays = bp.run_bp(bp.init(), num_iters=100, damping=0.5)
# beliefs = bp.get_beliefs(bp_arrays)
# map_states = graph.decode_map_states(beliefs)
# ~~~
# and run sum-product LBP and get estimated marginals by
# ~~~python
# bp = graph.BP(fg.bp_state, temperature=1.0)
# bp_arrays = bp.run_bp(bp.init(), num_iters=100, damping=0.5)
# beliefs = bp.get_beliefs(bp_arrays)
# marginals = graph.get_marginals(beliefs)
# ~~~
# More generally, PGMax implements LBP with temperature, with `temperature=0.0` and `temperature=1.0` corresponding to the commonly used max/sum-product LBP respectively.
#
# Now we are ready to demonstrate PMP sampling from RBM. PMP perturbs the model with [Gumbel](https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html) unary potentials, and draws a sample from the RBM as the MAP decoding from running max-product LBP on the perturbed model

# %%
bp = graph.BP(fg.bp_state, temperature=0.0)

# %%
bp_arrays = bp.init(
    evidence_updates={
        "hidden": np.random.gumbel(size=(bh.shape[0], 2)),
        "visible": np.random.gumbel(size=(bv.shape[0], 2)),
    },
)
bp_arrays = bp.run_bp(bp_arrays, num_iters=100, damping=0.5)
beliefs = bp.get_beliefs(bp_arrays)
map_states = graph.decode_map_states(beliefs)

# %% [markdown]
# Here we use the `evidence_updates` argument of `run_bp` to perturb the model with Gumbel unary potentials. In general, `evidence_updates` can be used to incorporate evidence in the form of externally applied unary potentials in PGM inference.
#
# Visualizing the MAP decoding (Figure [fig:rbm_single_digit]), we see that we have sampled an MNIST digit!

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(map_states["visible"].copy().reshape((28, 28)), cmap="gray")
ax.axis("off")

# %% [markdown]
# PGMax adopts a functional interface for implementing LBP: running LBP in PGMax starts with
# ~~~python
# run_bp, get_bp_state, get_beliefs = graph.BP(fg.bp_state, num_iters=NUM_ITERS, temperature=T)
# ~~~
# where `run_bp` and `get_beliefs` are pure functions with no side-effects. This design choice means that we can easily apply JAX transformations like `jit`/`vmap`/`grad`, etc., to these functions, and additionally allows PGMax to seamlessly interact with other packages in the rapidly growing JAX ecosystem (see [here](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research) and [here](https://github.com/n2cholas/awesome-jax)). In what follows we demonstrate an example on applying `jax.vmap`, a convenient transformation for automatically vectorizing functions.
#
# Since we implement `run_bp`/`get_beliefs` as a pure function, we can apply `jax.vmap` to `run_bp`/`get_beliefs` to process a batch of samples/models in parallel. As an example, consider the PGMax implementation of PMP sampling from the RBM trained on MNIST images in Section [Tutorial: implementing LBP inference for RBMs with PGMax]. Instead of drawing one sample at a time
# ~~~python
# bp_arrays = run_bp(
#     evidence_updates={
#         "hidden": np.random.gumbel(size=(bh.shape[0], 2)),
#         "visible": np.random.gumbel(size=(bv.shape[0], 2)),
#     },
#     damping=0.5,
# )
# beliefs = get_beliefs(bp_arrays)
# map_states = graph.decode_map_states(beliefs)
# ~~~
# we can draw a batch of samples in parallel by transforming `run_bp`/`get_beliefs` with `jax.vmap`

# %%
n_samples = 10
bp_arrays = jax.vmap(bp.init, in_axes=0, out_axes=0)(
    evidence_updates={
        "hidden": np.random.gumbel(size=(n_samples, bh.shape[0], 2)),
        "visible": np.random.gumbel(size=(n_samples, bv.shape[0], 2)),
    },
)
bp_arrays = jax.vmap(
    functools.partial(bp.run_bp, num_iters=100, damping=0.5),
    in_axes=0,
    out_axes=0,
)(bp_arrays)
beliefs = jax.vmap(bp.get_beliefs, in_axes=0, out_axes=0)(bp_arrays)
map_states = graph.decode_map_states(beliefs)

# %% [markdown]
# Visualizing the MAP decodings (Figure [fig:rbm_multiple_digits]), we see that we have sampled 10 MNIST digits in parallel!

# %%
fig, ax = plt.subplots(2, 5, figsize=(20, 8))
for ii in range(10):
    ax[np.unravel_index(ii, (2, 5))].imshow(
        map_states["visible"][ii].copy().reshape((28, 28)), cmap="gray"
    )
    ax[np.unravel_index(ii, (2, 5))].axis("off")

fig.tight_layout()

# %%
