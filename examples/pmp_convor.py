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
# We use PGMax to reimplement the binary deconvolution experiment presented in the Section 5.6 of the [Perturb-and-max-product (PMP)](https://proceedings.neurips.cc/paper/2021/hash/07b1c04a30f798b5506c1ec5acfb9031-Abstract.html) Neurips 2021 paper.
#
# The original implementation is available on the [GitHub repository of the paper.](https://github.com/vicariousinc/perturb_and_max_product/blob/master/experiments/exp6_convor.py)

import functools

# %%
import jax
import numpy as np
from plot_images import plot_images
from tqdm.notebook import tqdm

from pgmax.factors import logical
from pgmax.fg import graph, groups

# %% [markdown]
# ### Load the data

# %% [markdown]
# We extract the data from the PMP paper, only keeping the first 20 images here for the sake of speed.

# %%
data = np.load("example_data/conv_problem.npz")["X"]
data = data[:20]

_ = plot_images(data[:, 0], nr=4)

# %% [markdown]
# ### Construct variable grid, initialize factor graph, and add factors

# %% [markdown]
# We build a factor graph equivalent to the one in the PMP paper.

# %%
n_images, n_chan, im_height, im_width = data.shape
n_feat, feat_height, feat_width = 5, 6, 6
s_height = im_height - feat_height + 1
s_width = im_width - feat_width + 1

S = groups.NDVariableArray(num_states=2, shape=(n_images, n_feat, s_height, s_width))
W = groups.NDVariableArray(
    num_states=2, shape=(n_chan, n_feat, feat_height, feat_width)
)
SW = groups.NDVariableArray(
    num_states=2,
    shape=(n_images, n_chan, im_height, im_width, n_feat, feat_height, feat_width),
)
X = groups.NDVariableArray(num_states=2, shape=data.shape)

fg = graph.FactorGraph(variables=dict(S=S, W=W, SW=SW, X=X))

# %%
variable_names_for_OR_factors = {}

# Add ANDFactors
for idx_img in tqdm(range(n_images)):
    for idx_chan in range(n_chan):
        for idx_s_height in range(s_height):
            for idx_s_width in range(s_width):
                for idx_feat in range(n_feat):
                    for idx_feat_height in range(feat_height):
                        for idx_feat_width in range(feat_width):
                            idx_img_height = idx_feat_height + idx_s_height
                            idx_img_width = idx_feat_width + idx_s_width
                            SW_var = (
                                "SW",
                                idx_img,
                                idx_chan,
                                idx_img_height,
                                idx_img_width,
                                idx_feat,
                                idx_feat_height,
                                idx_feat_width,
                            )

                            variable_names = [
                                ("S", idx_img, idx_feat, idx_s_height, idx_s_width),
                                (
                                    "W",
                                    idx_chan,
                                    idx_feat,
                                    idx_feat_height,
                                    idx_feat_width,
                                ),
                                SW_var,
                            ]

                            fg.add_factor_by_type(
                                variable_names=variable_names,
                                factor_type=logical.ANDFactor,
                            )

                            X_var = (idx_img, idx_chan, idx_img_height, idx_img_width)
                            if X_var not in variable_names_for_OR_factors:
                                variable_names_for_OR_factors[X_var] = [SW_var]
                            else:
                                variable_names_for_OR_factors[X_var].append(SW_var)


# Add ORFactors
for X_var, variable_names_for_OR_factor in variable_names_for_OR_factors.items():
    fg.add_factor_by_type(
        variable_names=variable_names_for_OR_factor + [("X",) + X_var],  # type: ignore
        factor_type=logical.ORFactor,
    )

for factor_type, factors in fg.factors.items():
    print(f"The factor graph contains {len(factors)} {factor_type}")

# %% [markdown]
# ### Run inference and visualize results

# %% [markdown]
# PMP perturbs the model by adding Gumbel noise to unary potentials, then samples from the joint posterior *p(S,W | X)*.

# %%
run_bp, get_bp_state, get_beliefs = graph.BP(fg.bp_state, 3000)

# %% [markdown]
# We first compute the evidence without perturbation, similarly to the PMP paper

# %%
from scipy.special import logit as invsigmoid

pe = 1e-100
pS = 1e-72
pW = 0.25

uS = np.zeros((S.shape) + (2,))
uS[..., 1] = invsigmoid(pS)

uW = np.zeros((W.shape) + (2,))
uW[..., 1] = invsigmoid(pW)

uX = np.zeros((data.shape) + (2,))
uX[..., 0] = (2 * data - 1) * invsigmoid(pe)

# %% [markdown]
# We draw a batch of samples from the posterior in parallel by transforming `run_bp`/`get_beliefs` with `jax.vmap`

# %%
np.random.seed(seed=42)
n_samples = 4

bp_arrays = jax.vmap(functools.partial(run_bp, damping=0.5), in_axes=0, out_axes=0)(
    evidence_updates={
        "S": uS[None] + np.random.gumbel(size=(n_samples,) + uS.shape),
        "W": uW[None] + np.random.gumbel(size=(n_samples,) + uW.shape),
        "SW": np.zeros(shape=(n_samples,) + SW.shape),
        "X": uX[None] + np.zeros(shape=(n_samples,) + uX.shape),
    },
)
beliefs = jax.vmap(get_beliefs, in_axes=0, out_axes=0)(bp_arrays)
map_states = graph.decode_map_states(beliefs)

# %% [markdown]
# Visualizing the MAP decoding, we see that we have 4 samples from the posterior!

# %%
_ = plot_images(map_states["W"].reshape(-1, feat_height, feat_width), nr=n_samples)

# %%
