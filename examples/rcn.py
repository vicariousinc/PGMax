# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import time
from typing import Dict

import jax
import matplotlib.pyplot as plt

# %%
# %matplotlib inline
import numpy as np
from jax import numpy as jnp
from jax import tree_util
from scipy.ndimage import maximum_filter
from scipy.signal import fftconvolve
from sklearn.datasets import fetch_openml

from pgmax.fg import graph, groups

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# %% [markdown]
# # 1. Load the data
# %%
hps, vps = 12, 12
train_size = 20
test_size = 20


def fetch_mnist_dataset(test_size: int, seed: int = 5):
    """Returns test images sampled randomly from the set of MNIST images.
    Args:
        test_size: Desired number of test images.
    Returns:
        test_set: A list of length test_size containing images from the MNIST test dataset.
        test_labels: Corresponding labels for the test images.
    """

    mnist_train_size = 60000
    num_per_class = test_size // 10

    print("Fetching the MNIST dataset")
    dataset = fetch_openml("mnist_784", as_frame=False, cache=True)
    print("Successfully downloaded the MNIST dataset")

    mnist_images = dataset["data"]
    mnist_labels = dataset["target"].astype("int")
    full_mnist_test_images = mnist_images[mnist_train_size:]
    full_mnist_test_labels = mnist_labels[mnist_train_size:]

    np.random.seed(seed)

    test_set = []
    test_labels = []
    for i in range(10):
        idxs = np.random.choice(
            np.argwhere(full_mnist_test_labels == i)[:, 0], num_per_class
        )
        for idx in idxs:
            img = full_mnist_test_images[idx].reshape(28, 28)
            img_arr = jax.image.resize(image=img, shape=(112, 112), method="bicubic")
            img = jnp.pad(
                img_arr,
                pad_width=tuple([(p, p) for p in (44, 44)]),
                mode="constant",
                constant_values=0,
            )

            test_set.append(img)
            test_labels.append(i)

    return test_set, np.array(test_labels)


# %%
test_set, test_labels = fetch_mnist_dataset(test_size)
train_labels = (
    np.array([[i] * (train_size // 10) for i in range(10)]).reshape(1, -1).squeeze()
)

# %% [markdown]
# # 2. Load the model

# %%
data = np.load("example_data/rcn.npz", allow_pickle=True, encoding="latin1")
frcs, edges, suppression_masks, filters = (
    data["frcs"],
    data["edges"],
    data["suppression_masks"],
    data["filters"],
)

M = (2 * hps + 1) * (2 * vps + 1)

# %% [markdown]
# # 3. Visualize loaded model

# %%
img = np.ones((200, 200))

frc, edge = frcs[4], edges[4]
plt.figure(figsize=(10, 10))
for e in edge:
    i1, i2, w = e
    f1, r1, c1 = frc[i1]
    f2, r2, c2 = frc[i2]

    img[r1, c1] = 0
    img[r2, c2] = 0
    plt.text((c1 + c2) // 2, (r1 + r2) // 2, str(w), color="green")
    plt.plot([c1, c2], [r1, r2], color="green", linewidth=0.5)

plt.imshow(img, cmap="gray")


# %% [markdown]
# ## 3.1 Visualize the filters

# %% [markdown]
# The filters are used to detect the oriented edges on a given image. They are pre-computed using Gabor filters.

# %%
plt.figure(figsize=(10, 10))
for i in range(filters.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(filters[i], cmap="gray")

# %% [markdown]
# # 4. Make pgmax graph

# %% [markdown]
# ## 4.1 Make variables

# %%
start = time.time()
assert frcs.shape[0] == edges.shape[0]

variables_all_models = {}
for idx in range(frcs.shape[0]):
    frc = frcs[idx]
    variables_all_models[idx] = groups.NDVariableArray(
        variable_size=M, shape=(frc.shape[0],)
    )

end = time.time()
print(f"Creating variables took {end-start:.3f} seconds.")


# %% [markdown]
# ## 4.2 Make factors

# %% [markdown]
# ### 4.2.1 Pre-compute the valid configs for different perturb radii.

# %%
def valid_configs(r: int) -> np.ndarray:
    """Returns the valid configurations for the potential matrix given the perturb radius.
    Args:
        r: Peturb radius
    Returns:
        config_matrix: A configuration matrix (shape n X 2) where n is the number of valid configurations.
    """

    rows = []
    cols = []
    index = 0
    for i in range(M):
        r1, c1 = -hps + i // (2 * vps + 1), -vps + i % (2 * vps + 1)

        r2_min = max(r1 - r, -hps)
        r2_max = min(r1 + r, hps)
        c2_min = max(c1 - r, -vps)
        c2_max = min(c1 + r, vps)

        for r2 in range(r2_min, r2_max + 1):
            for c2 in range(c2_min, c2_max + 1):
                j = c2 + vps + (2 * hps + 1) * (r2 + hps)
                rows.append(i)
                cols.append(j)
                index += 1

    return np.stack([rows, cols], axis=1)


max_perturb_radii = 25
phis = []
for r in range(max_perturb_radii):
    phi_r = valid_configs(r)
    phis.append(phi_r)

# %% [markdown]
# ### 4.2.2 Make the factor graph

# %%
start = end
fg = graph.FactorGraph(variables=variables_all_models)
for idx in range(edges.shape[0]):
    edge = edges[idx]

    for e in edge:
        i1, i2, r = e
        fg.add_factor(
            [(idx, i1), (idx, i2)],
            phis[r],
            np.zeros(phis[r].shape[0]),
        )

end = time.time()
print(f"Creating factors took {end-start:.3f} seconds.")


# %% [markdown]
# # 5. Run inference

# %% [markdown]
# ## 5.1 Helper functions to initialize the evidence for a given image

# %%
def get_bu_msg(img: np.ndarray) -> np.ndarray:
    """Computes the bottom-up messages given a test image.
    Args:
        img: The rgb image to compute bottom up messages on (H x W x 3).
    Returns:
        bu_msg: An array of shape [16 x H x W] denoting the presence or absence of an oriented line segment at a particular location.
                The elements of this array belong to the set {+1, -1}.
    """

    num_orients = 16
    brightness_diff_threshold = 40.0

    filtered = np.zeros((filters.shape[0],) + img.shape, dtype=np.float32)
    for i in range(filters.shape[0]):
        kern = filters[i, :, :]
        filtered[i] = fftconvolve(img, kern, mode="same")

    localized = np.zeros_like(filtered)
    cross_orient_max = filtered.max(0)
    filtered[filtered < 0] = 0
    for i, (layer, suppress_mask) in enumerate(zip(filtered, suppression_masks)):
        competitor_maxs = maximum_filter(layer, footprint=suppress_mask, mode="nearest")
        localized[i] = competitor_maxs <= layer
    localized[cross_orient_max > filtered] = 0

    # Threshold and binarize
    localized *= (filtered / brightness_diff_threshold).clip(0, 1)
    localized[localized < 1] = 0

    pooled_channel_weights = [(0, 1), (-1, 1), (1, 1)]
    pooled_channels = [-np.ones_like(sf) for sf in localized]
    for i, pc in enumerate(pooled_channels):
        for channel_offset, factor in pooled_channel_weights:
            ch = (i + channel_offset) % num_orients
            pos_chan = localized[ch]
            if factor != 1:
                pos_chan[pos_chan > 0] *= factor
            np.maximum(pc, pos_chan, pc)

    bu_msg = np.array(pooled_channels)
    bu_msg[bu_msg == 0] = -1
    return bu_msg


# %% [markdown]
# ## 5.1.1 Visualizing bu_msg for a sample image

# %% [markdown]
# bu_msg has shape (16, H, W) where each 1 <= f <= 16 denotes the present or absense of a oriented edge

# %%
r_test_img = test_set[4]
r_bu_msg = get_bu_msg(r_test_img)
img = np.ones((200, 200))

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.imshow(r_test_img, cmap="gray")
for i in range(r_bu_msg.shape[0]):
    img[r_bu_msg[i] > 0] = 0

plt.subplot(1, 2, 2)
plt.imshow(img, cmap="gray")

# %% [markdown]
# Showing the individual filter activations in r_bu_msg

# %%
plt.figure(figsize=(10, 10))

for i in range(r_bu_msg.shape[0]):
    plt.subplot(5, 4, i + 1)
    rbm = r_bu_msg[i]
    rbm[rbm == 1] = -2
    plt.imshow(rbm, cmap="gray")


# %%
def initialize_evidences(test_img: np.ndarray) -> Dict:
    """Computes the initial evidences to the PGMax factor graph given a test image.
    Args:
        test_img: The image to run inference on.
    Returns:
        evidence_updates: A dictionary containing the initial messages to all the variables in the factor graph.
    """

    bu_msg = get_bu_msg(test_img)
    # jnp_bu_msg = jnp.asarray(bu_msg)

    evidence_updates = {}
    for idx in range(frcs.shape[0]):
        frc = frcs[idx]

        unary_msg = -1 + np.zeros((frc.shape[0], M))
        # evidence_updates[idx] = get_evidence(jnp_bu_msg, frc)

        for v in range(frc.shape[0]):
            f, r, c = frc[v, :]
            evidence = bu_msg[f, r - hps : r + hps + 1, c - vps : c + vps + 1]
            unary_msg[v] = evidence.ravel()

        evidence_updates[idx] = unary_msg

    return evidence_updates


# from functools import partial


# @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
# def get_evidence(bu_msg, frc):
#     """
#     bu_msg: Array of shape (n_features, M, N)
#     frc: Array of shape (n_frcs, 3)
#     """

#     return jax.lax.dynamic_slice(
#         bu_msg[frc[0]],
#         jnp.array([frc[1] - hps, frc[2] - vps]),
#         jnp.array([2 * hps + 1, 2 * vps + 1])
#     ).ravel()

# %% [markdown]
# ## 5.2 Run map product inference on all test images

# %%
run_bp_fn, _, get_beliefs_fn = graph.BP(fg.bp_state, 30)
scores = np.zeros((len(test_set), frcs.shape[0]))
map_states_dict = {}

for test_idx in range(len(test_set)):
    img = test_set[test_idx]

    start = time.time()
    evidence_updates = initialize_evidences(img)

    end = time.time()
    print(f"Initializing evidences took {end-start:.3f} seconds for image {test_idx}.")

    start = end
    map_states = graph.decode_map_states(
        get_beliefs_fn(run_bp_fn(evidence_updates=evidence_updates))
    )
    end = time.time()
    print(f"Max product inference took {end-start:.3f} seconds for image {test_idx}.")

    map_states_dict[test_idx] = map_states
    start = end
    score = tree_util.tree_multimap(
        lambda evidence, map: jnp.sum(evidence[jnp.arange(map.shape[0]), map]),
        evidence_updates,
        map_states,
    )
    for ii in score:
        scores[test_idx, ii] = score[ii]
    end = time.time()
    print(f"Computing scores took {end-start:.3f} seconds for image {test_idx}.")


# %% [markdown]
# # 6. Compute metrics (accuracy)

# %%
test_preds = train_labels[scores.argmax(axis=1)]
accuracy = (test_preds == test_labels).sum() / test_labels.shape[0]

print(f"accuracy = {accuracy}")


# %% [markdown]
# # 7. Visualize predictions - backtrace for the top model

# %%
imgs = np.ones((20, 200, 200))
plt.figure(figsize=(15, 15))

pred_idxs = np.argmax(scores, axis=1)
n_plots = [0, 5, 10]
for ii, pred_idx in enumerate(n_plots):

    map_states = map_states_dict[pred_idx]
    map_state = map_states[pred_idxs[pred_idx]]
    frc = frcs[pred_idx]

    for v in range(frc.shape[0]):
        idx = map_state[v]
        f, r, c = frc[v]

        delta_r, delta_c = -hps + idx // (2 * vps + 1), -vps + idx % (2 * vps + 1)
        rd, cd = r + delta_r, c + delta_c
        imgs[ii, rd, cd] = 0

    plt.subplot(len(n_plots), 2, 1 + 2 * ii)
    plt.imshow(test_set[pred_idx], cmap="gray")

    plt.subplot(len(n_plots), 2, 2 + 2 * ii)
    plt.imshow(imgs[ii, :, :], cmap="gray")
