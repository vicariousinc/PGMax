# export
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from helpers import (
    get_number_of_states,
    index_to_rc,
    initialize_vertex_beliefs,
    rc_to_index,
    visualize_graph,
)
from load_data import get_mnist_data_iters

from pgmax.fg import graph, groups

os.environ["XLA_PYTHON_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

debug = False


# export
def make_pgmax_graph(frcs, edges, M):
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
    return fg


def do_inference_on_batch(fg, frcs, test_set):
    run_bp_fn, _, get_beliefs_fn = graph.BP(fg.bp_state, 30)

    scores = np.zeros((len(test_set), frcs.shape[0]))

    for test_idx in range(len(test_set)):
        img = test_set[test_idx][0]

        start = time.time()
        bu_msg, evidence_updates = initialize_vertex_beliefs(img, frcs, hps, vps)
        end = time.time()
        print(
            f"Initializing beliefs took {end-start:.3f} seconds for image {test_idx}."
        )
        start = end

        map_states = graph.decode_map_states(
            get_beliefs_fn(run_bp_fn(evidence_updates=evidence_updates))
        )
        end = time.time()
        print(
            f"Max product inference took {end-start:.3f} seconds for image {test_idx}."
        )

        scores[test_idx, :] = get_pgmax_scores(map_states, bu_msg, frcs, hps, vps)
    return scores


def get_pgmax_scores(map_states, bu_msg, frcs, hps, vps, verbose=False):
    start = time.time()

    scores = [0] * frcs.shape[0]
    for i in range(frcs.shape[0]):
        frc = frcs[i]

        # max_msg = -1 + np.zeros_like(bu_msg)
        score = 0
        for v in range(frc.shape[0]):

            idx = map_states[i][v]
            if idx == 0:
                continue

            f, r, c = frc[v]
            delta_r, delta_c = index_to_rc(idx, hps, vps)
            rd, cd = r + delta_r, c + delta_c

            # max_msg[f, rd, cd] = 1
            score += bu_msg[f, rd, cd]

        # scores[i] = (max_msg == bu_msg).sum() - np.prod(
        #     bu_msg.shape
        # )
        scores[i] = score

        # scores[i] -= model.sum_log_z

    end = time.time()
    if verbose:
        print(f"Computing scores took {end-start:.3f} seconds.")
    return np.array(scores)


# export
def get_next_data_batch(batch_size):
    i = 0
    while i < len(train_set):
        i_f = min(i + batch_size, len(train_set))
        fg = make_pgmax_graph(frcs[i:i_f], edges[i:i_f], M)
        j = 0
        while j < len(test_set):
            j_f = min(j + batch_size, len(test_set))
            yield fg, frcs[i:i_f], edges[i:i_f], test_set[j:j_f], (i, i_f, j, j_f)
            j += batch_size
        i += batch_size


# export
hps, vps = 11, 11
train_size = 100

directory = f"model_{train_size}_{hps}_{vps}"
frcs = np.load(f"{directory}/frcs.npy", allow_pickle=True)
edges = np.load(f"{directory}/edges.npy", allow_pickle=True)
phis = np.load(f"{directory}/phis.npy", allow_pickle=True)
M = get_number_of_states(hps, vps)

if debug:
    img = visualize_graph(frcs[4], edges[4])

# export
data_dir = "/storage/users/skushagra/MNIST/"
data_dir = "/home/skushagra/Documents/science_rcn/data/MNIST/"
train_set, test_set = get_mnist_data_iters(data_dir, train_size, 20)
print(f"Length of train set = {len(train_set)}")

train_labels = -1 + np.zeros(len(train_set))
test_labels = -1 + np.zeros(len(test_set))
for i in range(len(train_set)):
    train_labels[i] = train_set[i][1]
for i in range(len(test_set)):
    test_labels[i] = test_set[i][1]

# export
batch_size = 50
pgmax_scores = np.zeros((len(test_set), frcs.shape[0]))

# export
for batch_fg, batch_frcs, batch_edges, batch_test_set, indices in get_next_data_batch(
    batch_size
):
    start = time.time()

    i, i_f, j, j_f = indices
    scores = do_inference_on_batch(batch_fg, batch_frcs, batch_test_set)
    pgmax_scores[j:j_f, i:i_f] = scores

    end = time.time()
    print(f"Done with test images {j, j_f} on models {i, i_f} in {end-start} seconds")

# export
test_preds = train_labels[pgmax_scores.argmax(axis=1)]
accuracy = (test_preds == test_labels).sum() / test_labels.shape[0]

print(f"accuracy = {accuracy}")
