# export
import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from learning import Model, index_to_rc
from load_data import get_mnist_data_iters

from pgmax.fg import graph, groups

os.environ["XLA_PYTHON_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

debug = False


def make_all_models(train_set, factor_type="pgmax"):
    start = time.time()
    hps, vps = 11, 11
    models = []

    for idx in range(len(train_set)):
        img = train_set[idx][0]
        models.append(
            Model(img, hps, vps, max_cxn_length=100, factor_type=factor_type, alpha=1.0)
        )

        if idx % 10 == 0:
            print(f"Done making {idx+1}th model.")

    end = time.time()
    print(f"Making models took {end-start:.3f} seconds.")
    return models


def reset_device_memory():

    dvals = (x for x in gc.get_objects() if isinstance(x, jax.xla.DeviceArray))
    n_deleted = 0
    for dv in dvals:
        dv.delete()
        n_deleted += 1
        del dv
    del dvals
    gc.collect()
    return n_deleted


def make_pgmax_graph(models):
    start = time.time()

    variables_all_models = {}
    for model_idx, model in enumerate(models):
        M = model.factors[0].M
        variables_all_models[model_idx] = groups.NDVariableArray(
            variable_size=M, shape=(len(model.V),)
        )

    end = time.time()
    print(f"Creating variables took {end-start:.3f} seconds.")
    start = end

    fg = graph.FactorGraph(variables=variables_all_models)
    for model_idx, model in enumerate(models):
        for factor in model.factors:
            i1, i2 = factor.i1, factor.i2
            valid_states = factor.get_valid_states()

            fg.add_factor(
                [(model_idx, i1), (model_idx, i2)],
                valid_states,
                np.zeros(valid_states.shape[0]),
            )

    end = time.time()
    print(f"Creating factors took {end-start:.3f} seconds.")
    return fg


def do_inference_on_batch(fg, models, test_set):
    run_bp_fn, _, get_beliefs_fn = graph.BP(fg.bp_state, 30)

    scores = np.zeros((len(test_set), len(models)))

    for idx in range(len(test_set)):
        img = test_set[idx][0]

        start = time.time()
        bu_msgs = [0] * len(models)

        evidence_updates = {}
        for model_idx in range(len(models)):
            model = models[model_idx]
            bu_msgs[model_idx] = model.initialize_vertex_beliefs(img)

            unary_msgs = np.zeros((len(model.V), model.factors[0].M))
            for i, v in enumerate(model.V):
                unary_msgs[i, :] = v.get_initial_beliefs()

            evidence_updates[model_idx] = unary_msgs

        end = time.time()
        # print(f"Initializing beliefs took {end-start:.3f} seconds.")
        start = end

        map_states = graph.decode_map_states(
            get_beliefs_fn(run_bp_fn(evidence_updates=evidence_updates))
        )
        end = time.time()
        # print(f"Max product inference took {end-start:.3f} seconds.")

        scores[idx, :] = get_pgmax_scores(map_states, bu_msgs, models)
    return scores


def get_pgmax_scores(map_states, bu_msgs, models, verbose=False):
    start = time.time()

    scores = [0] * len(models)
    for model_idx, model in enumerate(models):
        hps, vps = model.hps, model.vps

        max_msg = -1 + np.zeros_like(bu_msgs[model_idx])

        for i, v in enumerate(model.V):
            idx = map_states[model_idx][i]

            f, r, c = v.get_frc()
            delta_r, delta_c = index_to_rc(idx, hps, vps)
            rd, cd = r + delta_r, c + delta_c
            max_msg[f, rd, cd] = 1

        scores[model_idx] = (max_msg == bu_msgs[model_idx]).sum() - np.prod(
            bu_msgs[model_idx].shape
        )

    end = time.time()
    if verbose:
        print(f"Computing scores took {end-start:.3f} seconds.")
    return np.array(scores)


def get_next_data_batch(batch_size):
    i = 0
    while i < len(train_set):
        i_f = min(i + batch_size, len(train_set))
        fg = make_pgmax_graph(models[i:i_f])
        j = 0
        while j < len(test_set):
            j_f = min(j + batch_size, len(test_set))
            yield fg, models[i:i_f], test_set[j:j_f], (i, i_f, j, j_f)
            j += batch_size
        i += batch_size


data_dir = "/storage/users/skushagra/MNIST/"
train_set, test_set = get_mnist_data_iters(data_dir, 20, 20)

train_labels = -1 + np.zeros(len(train_set))
test_labels = -1 + np.zeros(len(test_set))
for i in range(len(test_set)):
    train_labels[i] = test_set[i][1]
for i in range(len(test_set)):
    test_labels[i] = test_set[i][1]


batch_size = 50
models = make_all_models(train_set, factor_type="cpp")
pgmax_scores = np.zeros((len(test_set), len(models)))


for batch_fg, batch_models, batch_test_set, indices in get_next_data_batch(batch_size):
    i, i_f, j, j_f = indices
    scores = do_inference_on_batch(batch_fg, batch_models, batch_test_set)
    pgmax_scores[j:j_f, i:i_f] = scores
    print(f"Done with test images {j, j_f} on models {i, i_f}")


test_preds = train_labels[pgmax_scores.argmax(axis=1)]
accuracy = (test_preds == test_labels).sum() / test_labels.shape[0]

print(f"accuracy = {accuracy}")
