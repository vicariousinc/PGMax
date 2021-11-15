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
import os
import time
from jax import jit

import matplotlib.pyplot as plt
import numpy as np
from helpers import (
    get_number_of_states,
    index_to_rc,
    initialize_evidences,
    visualize_graph,
)
from load_data import get_mnist_data_iters

from pgmax.fg import graph, groups

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# # 1. Load the data

hps, vps = 11, 11
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

# # 2. Load the model

directory = f"model_{train_size}_{hps}_{vps}"
frcs = np.load(f"{directory}/frcs.npy", allow_pickle=True, encoding='latin1')
edges = np.load(f"{directory}/edges.npy", allow_pickle=True, encoding='latin1')
phis = np.load(f"{directory}/phis.npy", allow_pickle=True, encoding='latin1')
M = get_number_of_states(hps, vps)

img = visualize_graph(frcs[4], edges[4])

# # 3. Make pgmax graph

# ## 3.1 Make variables

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

# ## 3.2 Make factors

# +
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

# -

# # 4. Run inference

# ## 4.1 Helper functins to compute score that will be useful later

@jit
def jax_get_pgmax_score(map_state, bu_msg, frc, hps, vps):
    score = 0
    for v in range(frc.shape[0]):

        idx = map_state[v]
        #if idx == 0: continue

        f, r, c = frc[v]
        
        rd = r - hps + (idx - 1) // (2 * vps + 1)
        cd = c - vps + (idx - 1) %  (2 * vps + 1)
        
        score += bu_msg[f, rd, cd]
    
    return score


def get_pgmax_scores(map_states, bu_msg, frcs, hps, vps, verbose=False):
    start = time.time()

    scores = [0] * frcs.shape[0]
    for i in range(frcs.shape[0]):
        scores[i] = jax_get_pgmax_score(map_states[i], bu_msg, frcs[i], hps, vps)

    end = time.time()
    print(f"Computing scores took {end-start:.3f} seconds.")
    return np.array(scores)


# ## 4.2 Run map product inference on all test images

# +
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
    map_states = graph.decode_map_states(get_beliefs_fn(run_bp_fn(evidence_updates=evidence_updates)))
    end = time.time()
    print(f"Max product inference took {end-start:.3f} seconds for image {test_idx}.")

    map_states_dict[test_idx] = map_states
    scores[test_idx, :] = get_pgmax_scores(map_states, bu_msg, frcs, hps, vps)
# -

# # 5. Compute metrics (accuracy)

test_preds = train_labels[scores.argmax(axis=1)]
accuracy = (test_preds == test_labels).sum() / test_labels.shape[0]

print(f"accuracy = {accuracy}")

# # 6. Visualize predictions (backtrace)

test_idx = 0
plt.imshow(test_set[test_idx][0], cmap="gray")

# ## 6.1 Backtrace of some models on this test image

# +
map_states = map_states_dict[test_idx]
imgs = np.ones((len(frcs), 200, 200))

for i in range(frcs.shape[0]):
    map_state = map_states[i]
    frc = frcs[i]

    for v in range(frc.shape[0]):
        idx = map_state[v]
        f, r, c = frc[v]

        delta_r, delta_c = index_to_rc(idx, hps, vps)
        rd, cd = r + delta_r, c + delta_c
        imgs[i, rd, cd] = 0
plt.figure(figsize=(15, 15))

for k, index in enumerate(range(0, len(train_set), 5)):
    plt.subplot(1, 4, 1+k)
    plt.title(f" Model {int(train_labels[index])}")
    plt.imshow(imgs[index, :, :], cmap="gray")
# -


