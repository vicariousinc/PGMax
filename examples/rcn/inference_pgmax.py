# export
import os
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
from learning import Model, index_to_rc
from load_data import get_mnist_data_iters

from pgmax.fg import graph, groups

os.environ["XLA_PYTHON_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

debug = False

# export


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


def do_pgmax_inference(
    test_img, fg, models, num_iters=10, damping_factor=0.5, verbose=False
):
    start = time.time()
    init_msgs = fg.get_init_msgs()
    bu_msgs = [0] * len(models)
    for model_idx, model in enumerate(models):
        bu_msgs[model_idx] = model.initialize_vertex_beliefs(test_img)

        unary_msgs = np.zeros((len(model.V), model.factors[0].M))
        for i, v in enumerate(model.V):
            unary_msgs[i, :] = v.get_initial_beliefs()

        init_msgs.evidence[model_idx] = unary_msgs

    end = time.time()
    if verbose:
        print(f"Initializing beliefs took {end-start:.3f} seconds.")
    start = end

    msgs = fg.run_bp(num_iters, damping_factor, init_msgs)
    map_states = fg.decode_map_states(msgs)

    end = time.time()
    if verbose:
        print(f"Max product inference took {end-start:.3f} seconds.")
    return map_states, bu_msgs


def get_pgmax_scores(map_states, bu_msgs, models, verbose=False):
    start = time.time()

    scores = [0] * len(models)
    for model_idx, model in enumerate(models):
        hps, vps = model.hps, model.vps

        max_msg = -1 + np.zeros_like(bu_msgs[model_idx])

        for i, v in enumerate(model.V):
            idx = map_states[(model_idx, i)]

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


# export
data_dir = "/storage/users/skushagra/MNIST/"
train_set, test_set = get_mnist_data_iters(data_dir, 20, 20)

# export
if debug:
    img = train_set[5][0]
    temp_model = Model(img, 11, 11, max_cxn_length=100, factor_type="pgmax", alpha=1.0)
    plt.figure(figsize=(10, 10))
    temp_model.visualize_graph()

# export
start = time.time()
hps, vps = 11, 11
models = []
train_labels = -1 + np.zeros(len(train_set))

for idx in range(len(train_set)):
    img = train_set[idx][0]

    train_labels[idx] = int(train_set[idx][1])
    models.append(
        Model(img, hps, vps, max_cxn_length=100, factor_type="pgmax", alpha=1.0)
    )

    if idx % 10 == 0:
        print(f"Done making {idx+1}th model.")


end = time.time()
print(f"Making models took {end-start:.3f} seconds.")

fg = make_pgmax_graph(models)

# export
test_labels = -1 + np.zeros(len(test_set))
pgmax_scores = np.zeros((len(test_set), len(models)))
init_msgs = fg.get_init_msgs()
update_msgs = fg.make_run_bp_function(50, 0.5)
for idx in range(len(test_set)):
    img = test_set[idx][0]
    test_labels[idx] = int(test_set[idx][1])
    start = time.time()
    bu_msgs = [0] * len(models)
    for model_idx, model in enumerate(models):
        bu_msgs[model_idx] = model.initialize_vertex_beliefs(img)
        unary_msgs = np.zeros((len(model.V), model.factors[0].M))
        for i, v in enumerate(model.V):
            unary_msgs[i, :] = v.get_initial_beliefs()

        init_msgs.evidence[model_idx] = unary_msgs

    end = time.time()
    print(f"Initializing beliefs took {end-start:.3f} seconds.")
    start = end
    msgs_after_bp = update_msgs(
        jax.device_put(init_msgs.ftov.value), jax.device_put(init_msgs.evidence.value)
    )
    msgs = graph.Messages(
        ftov=graph.FToVMessages(factor_graph=fg, init_value=msgs_after_bp),
        evidence=init_msgs.evidence,
    )
    map_states = fg.decode_map_states(msgs)
    end = time.time()
    print(f"Max product inference took {end-start:.3f} seconds.")
    pgmax_scores[idx, :] = get_pgmax_scores(map_states, bu_msgs, models)
    print(f"Done with test image {idx+1}")

# export
test_preds = train_labels[pgmax_scores.argmax(axis=1)]
accuracy = (test_preds == test_labels).sum() / test_labels.shape[0]
print(f"accuracy = {accuracy}")
