# export
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from learning import Model, index_to_rc, rc_to_index
from load_data import get_mnist_data_iters
from preproc import Preproc

from pgmax.fg import graph, groups

debug = False


def make_pgmax_graph(model):
    M = model.factors[0].M

    variables = groups.NDVariableArray(variable_size=M, shape=(len(model.V),))
    fg = graph.FactorGraph(variables=dict(h1_vertices=variables))

    for factor in model.factors:
        i1, i2 = factor.i1, factor.i2
        valid_states = factor.get_valid_states()

        fg.add_factor(
            [("h1_vertices", i1), ("h1_vertices", i2)],
            valid_states,
            np.zeros(valid_states.shape[0]),
        )

    return fg


def do_pgmax_inference(inf_img, fg, model, num_iters=10, damping_factor=0.5):
    model.initialize_vertex_beliefs(inf_img)
    init_beliefs = np.zeros((len(model.V), model.factors[0].M))

    for i, v in enumerate(model.V):
        init_beliefs[i, :] = v.get_initial_beliefs()
    init_msgs = fg.get_init_msgs()
    init_msgs.evidence["h1_vertices"] = init_beliefs

    msgs = fg.run_bp(num_iters, damping_factor, init_msgs)
    map_states = fg.decode_map_states(msgs)
    return map_states


def get_pgmax_score(map_states, model):
    neg_inf = -1000

    def _f(v, idx):
        belief = v.get_initial_beliefs()
        return belief[idx]

    score = 0
    for i, v in enumerate(model.V):
        idx = map_states[("h1_vertices", i)]
        score += _f(v, idx)

    return score


data_dir = "/home/skushagra/Documents/science_rcn/data/MNIST/"
train_set, test_set = get_mnist_data_iters(data_dir, 20, 20)

if debug:
    img = train_set[4][0]
    temp_model = Model(img, 11, 11, max_cxn_length=100, factor_type="pgmax", alpha=1.0)
    plt.figure(figsize=(10, 10))
    temp_model.visualize_graph()

hps, vps = 11, 11

models = []
fgs = []
train_labels = []

for idx in range(len(train_set)):
    img = train_set[idx][0]
    label = int(train_set[idx][1])

    model = Model(img, hps, vps, max_cxn_length=100, factor_type="pgmax", alpha=1.0)
    fg = make_pgmax_graph(model)

    models.append(model)
    fgs.append(fg)
    train_labels.append(label)

    print(f"Done making {idx+1}th factor graph.")


test_labels = []
test_preds = []
accuracy = 0

for idx in range(len(test_set)):
    img = test_set[idx][0]
    test_label = int(test_set[idx][1])
    test_labels.append(test_label)

    start = time.time()

    pg_max_scores = []
    for model, fg in zip(models, fgs):
        map_states = do_pgmax_inference(img, fg, model, num_iters=10)
        pg_max_scores.append(get_pgmax_score(map_states, model))
    pg_max_scores = np.array(pg_max_scores)

    test_pred = train_labels[pg_max_scores.argmax()]
    test_preds.append(test_pred)

    if test_pred == test_label:
        accuracy += 1

    print(f"Making predictions on {idx+1} image took {time.time() - start} seconds.")

accuracy = accuracy / len(test_set)

# export
print(f"accuracy = {accuracy}")
