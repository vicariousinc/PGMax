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

# %%
import functools

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit
from tqdm.notebook import tqdm

from pgmax.fg import graph
from pgmax.groups import logical, variables


# %%
def plot_images(images, display=True, nr=None):
    "Useful function for visualizing several images"
    n_images, H, W = images.shape
    images = images - images.min()
    images /= images.max() + 1e-10

    if nr is None:
        nr = nc = np.ceil(np.sqrt(n_images)).astype(int)
    else:
        nc = n_images // nr
        assert n_images == nr * nc
    big_image = np.ones(((H + 1) * nr + 1, (W + 1) * nc + 1, 3))
    big_image[..., :3] = 0
    big_image[:: H + 1] = [0.5, 0, 0.5]

    im = 0
    for r in range(nr):
        for c in range(nc):
            if im < n_images:
                big_image[
                    (H + 1) * r + 1 : (H + 1) * r + 1 + H,
                    (W + 1) * c + 1 : (W + 1) * c + 1 + W,
                    :,
                ] = images[im, :, :, None]
            im += 1

    if display:
        plt.figure(figsize=(10, 10))
        plt.imshow(big_image, interpolation="none")
    return big_image


# %% [markdown]
# ### Load the data

# %% [markdown]
# Our binary 2D convolution generative model uses two set of binary variables to form a set of binary images X:
#  - a set W of 2D binary features shared across images
#  - a set S of binary indicator variables representing whether each feature is present at each possible image location.
#
# Each binary entry of W and S is modeled with an independent Bernoulli prior. W and S are then combined by convolution, placing the features defined by W at the locations specified by S in order to form the image.
#
# We load the dataset of 100 images used in the PMP paper.
# We only keep the first 20 images here for the sake of speed.

# %%
data = np.load("example_data/conv_problem.npz")
W_gt = data["W"]
X_gt = data["X"]
X_gt = X_gt[:20]

_ = plot_images(X_gt[:, 0], nr=4)

# %% [markdown]
# We also visualize the four 2D binary features used to generate the images above.
#
# We aim at recovering these binary features using PGMax.

# %%
_ = plot_images(W_gt[0], nr=1)

# %% [markdown]
# ### Construct variable grid, initialize factor graph, and add factors

# %% [markdown]
# Our factor graph naturally includes the binary features W, the binary indicators of features locations S and the binary images obtained by convolution X.
#
# To generate X from W and S, we observe that a binary convolution can be represented by two set of logical factors:
#  - a first set of ANDFactors, which combine the joint activations in W and S. We store the children of these ANDFactors in an auxiliary variable SW
#  - a second set of ORFactors, which maps SW to X and model (binary) features overlapping.
#
# See Section 5.6 of the [PMP paper](https://proceedings.neurips.cc/paper/2021/hash/07b1c04a30f798b5506c1ec5acfb9031-Abstract.html) for more details.

# %%
# The dimensions of W used for the generation of X were (4, 5, 5) but we set them to (5, 6, 6)
# to simulate a more realistic scenario in which we do not know their ground truth values
n_feat, feat_height, feat_width = 5, 6, 6

n_images, n_chan, im_height, im_width = X_gt.shape
s_height = im_height - feat_height + 1
s_width = im_width - feat_width + 1

# Binary features
W = variables.NDVariableArray(
    num_states=2, shape=(n_chan, n_feat, feat_height, feat_width)
)

# Binary indicators of features locations
S = variables.NDVariableArray(num_states=2, shape=(n_images, n_feat, s_height, s_width))

# Auxiliary binary variables combining W and S
SW = variables.NDVariableArray(
    num_states=2,
    shape=(n_images, n_chan, im_height, im_width, n_feat, feat_height, feat_width),
)

# Binary images obtained by convolution
X = variables.NDVariableArray(num_states=2, shape=X_gt.shape)

# %%
# Factor graph
fg = graph.FactorGraph(variables=dict(S=S, W=W, SW=SW, X=X))

variable_names_for_ANDFactors = []
variable_names_for_ORFactors = []
variable_names_for_ORFactors_dict = {}

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

                            variable_names_for_ANDFactor = [
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
                            variable_names_for_ANDFactors.append(
                                variable_names_for_ANDFactor
                            )

                            X_var = (idx_img, idx_chan, idx_img_height, idx_img_width)
                            if X_var not in variable_names_for_ORFactors_dict:
                                variable_names_for_ORFactors_dict[X_var] = [SW_var]
                            else:
                                variable_names_for_ORFactors_dict[X_var].append(SW_var)


import time

start = time.time()
fg.add_factor_group(
    factory=logical.ANDFactorGroup,
    variable_names_for_factors=variable_names_for_ANDFactors,
)
print("Time", time.time() - start)

# Add ORFactors
for X_var, variable_names_for_ORFactor in variable_names_for_ORFactors_dict.items():
    variable_names_for_ORFactors.append(variable_names_for_ORFactor + [("X",) + X_var])

start = time.time()
fg.add_factor_group(
    factory=logical.ORFactorGroup,
    variable_names_for_factors=variable_names_for_ORFactors,
)
print("Time", time.time() - start)

for factor_type, factor_groups in fg.factor_groups.items():
    if len(factor_groups) > 0:
        assert len(factor_groups) == 1
        print(f"The factor graph contains {factor_groups[0].num_factors} {factor_type}")

# %%
start = time.time()
wiring = fg.wiring
print("Time", time.time() - start)

# %% [markdown]
# ### Run inference and visualize results

# %% [markdown]
# PMP perturbs the model by adding Gumbel noise to unary potentials, then samples from the joint posterior *p(W, S | X)*.
#
# Note that this posterior is highly multimodal: permuting the first dimension of W and the second dimension of S
# in the same manner does not change X, so this naturally results in multiple equivalent modes.

# %%
run_bp, get_bp_state, get_beliefs = graph.BP(fg.bp_state, 3000)

# %% [markdown]
# We first compute the evidence without perturbation, similar to the PMP paper.

# %%
pW = 0.25
pS = 1e-72
pX = 1e-100

# Sparsity inducing priors for W and S
uW = np.zeros((W.shape) + (2,))
uW[..., 1] = logit(pW)

uS = np.zeros((S.shape) + (2,))
uS[..., 1] = logit(pS)

# Likelihood the binary images given X
uX = np.zeros((X_gt.shape) + (2,))
uX[..., 0] = (2 * X_gt - 1) * logit(pX)

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
# Visualizing the MAP decoding, we see that we have 4 good random samples (one per row) from the posterior!
#
# Because we have used one extra feature for inference, each posterior sample recovers the 4 basic features used to generate the images, and includes an extra symbol.

# %%
_ = plot_images(map_states["W"].reshape(-1, feat_height, feat_width), nr=n_samples)
