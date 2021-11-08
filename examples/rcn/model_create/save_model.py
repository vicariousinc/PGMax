# +
# export
import os
import sys
from pathlib import Path

parent_dir = Path(os.path.abspath(os.getcwd())).parent.absolute()
sys.path.append(str(parent_dir))

import time

import matplotlib.pyplot as plt
import numpy as np
from learning import Model, get_number_of_states, index_to_rc, rc_to_index
from load_data import get_mnist_data_iters
from preproc import Preproc


# export
def save_models(train_set, hps, vps, save_dir, factor_type="cpp"):

    start = time.time()
    models = []
    frcs = []
    edges = []

    for idx in range(len(train_set)):
        img = train_set[idx][0]

        model = Model(
            img, hps, vps, max_cxn_length=100, factor_type=factor_type, alpha=1.0
        )
        frc = np.array([v.get_frc() for v in model.V])
        edge = np.array([(f.i1, f.i2, f.r) for f in model.factors])

        models.append(model)
        frcs.append(frc)
        edges.append(edge)

        if idx % 10 == 0:
            print(f"Done making {idx+1}th model.")

    np.save(f"{save_dir}/frcs.npy", frcs)
    np.save(f"{save_dir}/edges.npy", edges)

    end = time.time()
    print(f"Saving models took {end-start:.3f} seconds.")
    return models, frcs, edges


# export
def valid_configs(r, hps, vps):
    M = get_number_of_states(hps, vps)

    rows = []
    cols = []
    index = 0
    for i in range(1, M):
        r1, c1 = index_to_rc(i, hps, vps)

        r2_min = max(r1 - r, -hps)
        r2_max = min(r1 + r, hps)
        c2_min = max(c1 - r, -vps)
        c2_max = min(c1 + r, vps)

        for r2 in range(r2_min, r2_max + 1):
            for c2 in range(c2_min, c2_max + 1):
                j = rc_to_index(r2, c2, hps, vps)
                rows.append(i)
                cols.append(j)
                index += 1

    return np.stack([rows, cols], axis=1)


# export
def save_valid_configs(edges, save_dir):
    M = -1
    for i in range(len(train_set)):
        M = max(M, max(edges[i][:, 2]))

    phis = []
    for r in range(M + 1):
        phi_r = valid_configs(r, 11, 11)
        phis.append(phi_r)
        print(f"Done with  r = {r}")

    np.save(f"{save_dir}/phis.npy", phis)
    return phis


# export
data_dir = "/home/skushagra/Documents/science_rcn/data/MNIST/"
train_size = 1000
hps = 11
vps = 11
save_dir = f"{parent_dir}/model_{train_size}_{hps}_{vps}"
Path(save_dir).mkdir(parents=True, exist_ok=True)

# export
train_set, test_set = get_mnist_data_iters(data_dir, train_size, 20)
models, frcs, edges = save_models(train_set, hps, vps, save_dir, factor_type="cpp")
phis = save_valid_configs(edges, save_dir)
