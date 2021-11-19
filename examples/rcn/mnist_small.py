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

import os
import time

import matplotlib.pyplot as plt

# %%
# %matplotlib inline
import numpy as np
from jax import numpy as jnp
from jax import tree_util
from load_data import get_mnist_data_iters
from preproc import Preproc

from pgmax.fg import graph, groups

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# %% [markdown]
# ## 1. Load the data
#
#
# %%
hps, vps = 12, 12
train_size = 20
test_size = 20
data_dir = "/storage/users/skushagra/MNIST/"

# +
train_set, test_set = get_mnist_data_iters(data_dir, train_size, test_size)

train_labels = -1 + np.zeros(len(train_set))
for i in range(len(train_set)):
    train_labels[i] = train_set[i][1]

test_labels = -1 + np.zeros(len(test_set))
for i in range(len(test_set)):
    test_labels[i] = test_set[i][1]
# -


# %% [markdown]
# ## 2. Load the model
#
#

# %%
directory = f"/storage/users/skushagra/pgmax_rcn_artifacts/model_science_{train_size}_{hps}_{vps}"
frcs = np.load(f"{directory}/frcs.npy", allow_pickle=True, encoding="latin1")
edges = np.load(f"{directory}/edges.npy", allow_pickle=True, encoding="latin1")
M = (2 * hps + 1) * (2 * vps + 1) + 1

# %% [markdown]
# ## 3. Visualize loaded model.
#
#

# %%
img = np.zeros((200, 200))

frc, edge = frcs[4], edges[4]
plt.figure(figsize=(10, 10))
for e in edge:
    i1, i2, w = e
    f1, r1, c1 = frc[i1]
    f2, r2, c2 = frc[i2]

    img[r1, c1] = 255
    img[r2, c2] = 255
    plt.text((c1 + c2) // 2, (r1 + r2) // 2, str(w), color="blue")
    plt.plot([c1, c2], [r1, r2], color="blue", linewidth=0.5)

plt.imshow(img, cmap="gray")


# %% [markdown]
# ## 4. Make pgmax graph
#
#

# %% [markdown]
# ## 3.1 Make variables
#
#

# %%
# +
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
# -

# %% [markdown]
# ## 3.2 Make factors
#
#

# %%

# %% [markdown]
# ## 3.2.1 Pre-compute the valid configs for different perturb radii.
#
#

# %%
# -
def valid_configs(r):
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
# ## 3.2.2 Make the factor graph
#
#

# %%
# -
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
# ## 4. Run inference
#
#

# %% [markdown]
# ## 4.1 Helper function to initialize the evidence for a given image
#
#

# %%
def initialize_evidences(test_img, frcs, hps, vps):
    preproc_layer = Preproc(cross_channel_pooling=True)
    bu_msg = preproc_layer.fwd_infer(test_img)

    evidence_updates = {}
    for idx in range(frcs.shape[0]):
        frc = frcs[idx]

        unary_msg = -1 + np.zeros((frc.shape[0], M))
        for v in range(frc.shape[0]):
            f, r, c = frc[v, :]
            evidence = bu_msg[f, r - hps : r + hps + 1, c - hps : c + hps + 1]
            indices = np.transpose(np.nonzero(evidence > 0))

            for index in indices:
                r1, c1 = index
                delta_r, delta_c = r1 - hps, c1 - vps

                index = delta_c + vps + (2 * hps + 1) * (delta_r + hps)
                unary_msg[v, index] = 1

        evidence_updates[idx] = unary_msg
    return bu_msg, evidence_updates


# %% [markdown]
# ## 4.2 Run map product inference on all test images
#
#

# %%
run_bp_fn, _, get_beliefs_fn = graph.BP(fg.bp_state, 30)
scores = np.zeros((len(test_set), frcs.shape[0]))
map_states_dict = {}

for test_idx in range(len(test_set)):
    img = test_set[test_idx][0]

    start = time.time()
    bu_msg, evidence_updates = initialize_evidences(img, frcs, hps, vps)
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

    # scores[test_idx, :] = score


# %% [markdown]
# ## 5. Compute metrics (accuracy)
#
#

# %%
test_preds = train_labels[scores.argmax(axis=1)]
accuracy = (test_preds == test_labels).sum() / test_labels.shape[0]

print(f"accuracy = {accuracy}")


# %% [markdown]
# ## 6. Visualize predictions (backtrace)
#
#

# %%
test_idx = 0
plt.imshow(test_set[test_idx][0], cmap="gray")


# %%
# ## 6.1 Backtrace of some models on this test image


# %%
# +
map_states = map_states_dict[test_idx]
imgs = np.ones((len(frcs), 200, 200))

for i in range(frcs.shape[0]):
    map_state = map_states[i]
    frc = frcs[i]

    for v in range(frc.shape[0]):
        idx = map_state[v]
        f, r, c = frc[v]

        delta_r, delta_c = -hps + idx // (2 * vps + 1), -vps + idx % (2 * vps + 1)
        rd, cd = r + delta_r, c + delta_c
        imgs[i, rd, cd] = 0
plt.figure(figsize=(15, 15))

for k, index in enumerate(range(0, len(train_set), 5)):
    plt.subplot(1, 4, 1 + k)
    plt.title(f" Model {int(train_labels[index])}")
    plt.imshow(imgs[index, :, :], cmap="gray")
# -


# %%
