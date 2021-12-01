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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import os
import time

import jax
import matplotlib.pyplot as plt
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

# Use train_size = 100 if you have a gpu with atleast 8Gbs memory.
# Recommend that jax is installed with cuda enabled for this option.
train_size = 100
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
data = np.load("example_data/rcn_100.npz", allow_pickle=True, encoding="latin1")
frcs, edges, suppression_masks, filters = (
    data["frcs"][:train_size],
    data["edges"][:train_size],
    data["suppression_masks"],
    data["filters"],
)

M = (2 * hps + 1) * (2 * vps + 1)

# %% [markdown]
# # 3. Visualize loaded model

# %%
img = np.ones((200, 200))
pad = 55
frc, edge = frcs[4], edges[4]
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for e in edge:
    i1, i2, w = e
    f1, r1, c1 = frc[i1]
    f2, r2, c2 = frc[i2]

    img[r1, c1] = 0
    img[r2, c2] = 0
    ax.text(
        (c1 + c2) // 2 - pad, (r1 + r2) // 2 - pad, str(w), color="green", fontsize=15
    )
    ax.plot([c1 - pad, c2 - pad], [r1 - pad, r2 - pad], color="green", linewidth=0.5)

ax.axis("off")
ax.imshow(img[pad : 200 - pad, pad : 200 - pad], cmap="gray")


# %% [markdown]
# ## 3.1 Visualize the filters

# %% [markdown]
# The filters are used to detect the oriented edges on a given image. They are pre-computed using Gabor filters.

# %%
fig, ax = plt.subplots(4, 4, figsize=(10, 10))
for i in range(filters.shape[0]):
    idx = np.unravel_index(i, (4, 4))
    ax[idx].imshow(filters[i], cmap="gray")
    ax[idx].axis("off")

fig.tight_layout()

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
        A configuration matrix (shape n X 2) where n is the number of valid configurations.
    """

    configs = []
    for i, (r1, c1) in enumerate(np.array(np.unravel_index(np.arange(M), (25, 25))).T):
        r2_min = max(r1 - r, 0)
        r2_max = min(r1 + r, 2 * hps)
        c2_min = max(c1 - r, 0)
        c2_max = min(c1 + r, 2 * vps)
        j = np.ravel_multi_index(
            tuple(np.mgrid[r2_min : r2_max + 1, c2_min : c2_max + 1]), (25, 25)
        ).ravel()
        configs.append(np.stack([np.full(j.shape, fill_value=i), j], axis=1))

    return np.concatenate(configs)


max_perturb_radius = 25
valid_configs_list = [valid_configs(r) for r in range(max_perturb_radius)]

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
            valid_configs_list[r],
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
        An array of shape [16 x H x W] denoting the presence or absence of an oriented line segment at a particular location.
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
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(r_test_img, cmap="gray")
ax[0].axis("off")
ax[0].set_title("Input image", fontsize=30)
for i in range(r_bu_msg.shape[0]):
    img[r_bu_msg[i] > 0] = 0

ax[1].imshow(img, cmap="gray")
ax[1].axis("off")
ax[1].set_title("Max filter response across 16 channels", fontsize=30)
fig.tight_layout()


# %% [markdown]
# ## 5.2 Run map product inference on all test images

# %%
def get_evidence(bu_msg: np.ndarray, frc: np.ndarray):
    """Function to get evidence

    Args:
        bu_msg: Array of shape (n_features, 200, 200). Contains BU messages
        frc: Array of shape (n_frcs, 3).

    Returns:
        evidence; Array of shape (n_frcs, M). Contains evidence
    """
    evidence = np.zeros((frc.shape[0], M))
    for v, (f, r, c) in enumerate(frc):
        evidence[v] = bu_msg[f, r - hps : r + hps + 1, c - vps : c + vps + 1].ravel()

    return evidence


# %%
frcs_dict = {model_idx: frcs[model_idx] for model_idx in range(frcs.shape[0])}
run_bp, _, get_beliefs = graph.BP(fg.bp_state, 30)
scores = np.zeros((len(test_set), frcs.shape[0]))
map_states_dict = {}

for test_idx in range(len(test_set)):
    img = test_set[test_idx]

    start = time.time()
    bu_msg = get_bu_msg(img)
    evidence_updates = jax.tree_util.tree_map(
        lambda frc: get_evidence(bu_msg, frc), frcs_dict
    )
    end = time.time()
    print(f"Initializing evidences took {end-start:.3f} seconds for image {test_idx}.")

    start = end
    map_states = graph.decode_map_states(
        get_beliefs(run_bp(evidence_updates=evidence_updates))
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
best_model_idx = np.argmax(scores, axis=1)
test_preds = train_labels[best_model_idx]
accuracy = (test_preds == test_labels).sum() / test_labels.shape[0]

print(f"accuracy = {accuracy}")


# %% [markdown]
# # 7. Visualize predictions - backtrace for the top model

# %%
fig, ax = plt.subplots(5, 4, figsize=(16, 20))
for test_idx in range(20):
    idx = np.unravel_index(test_idx, (5, 4))
    map_state = map_states_dict[test_idx][best_model_idx[test_idx]]
    offsets = np.array(np.unravel_index(map_state, (25, 25))).T - np.array([hps, vps])
    activations = frcs[best_model_idx[test_idx]][:, 1:] + offsets
    for rd, cd in activations:
        ax[idx].plot(cd, rd, "r.")

    ax[idx].imshow(test_set[test_idx], cmap="gray")
    ax[idx].set_title(
        f"Ground Truth: {test_labels[test_idx]}, Pred: {test_preds[test_idx]}",
        fontsize=20,
    )
    ax[idx].axis("off")

fig.tight_layout()
