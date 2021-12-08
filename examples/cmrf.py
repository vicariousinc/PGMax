# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from pgmax.fg import graph, groups


# %%
@nb.njit()
def contours_to_mess_bu(input_train, p_contour, n_clones):
    assert (input_train == 0).sum() + (
        input_train == 1
    ).sum() == input_train.size  # (binary inputs, where 0 is contour)
    assert len(p_contour) == 3
    n_states = n_clones.sum()
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    messages_bu = np.zeros(input_train.shape + (n_states,))
    for i in range(input_train.shape[0]):
        for r in range(input_train.shape[1]):
            for c in range(input_train.shape[2]):
                v = input_train[i, r, c]
                for d in range(3):
                    start, stop = state_loc[d : d + 2]
                    messages_bu[i, r, c, start:stop] = (
                        p_contour[d] if v == 0 else 1 - p_contour[d]
                    )

    return messages_bu


@nb.njit()
def img_to_mess_bu(images, n_clones):
    n_states = n_clones.sum()
    state_loc = np.hstack((np.array([0], dtype=n_clones.dtype), n_clones)).cumsum()
    messages_bu = np.zeros(images.shape + (n_states,))
    for i in range(images.shape[0]):
        for r in range(images.shape[1]):
            for c in range(images.shape[2]):
                v = images[i, r, c]
                start, stop = state_loc[v : v + 2]
                messages_bu[i, r, c, start:stop] = 1
    return messages_bu


# %%
data = np.load("/home/stannis/noisy_mnist_8_0.2.npz")
weights = np.load("/home/stannis/cmrf8_weights_15_mb50_lr1em2_nc64_emp.npz")
p_contour = data["p_contour"]
n_clones = weights["n_clones"]
evidence = contours_to_mess_bu(data["noisy_images_test"], p_contour, n_clones)
evidence[evidence == 0.0] = 1e-10
evidence = np.log(evidence)
targets = img_to_mess_bu(data["images_test"], n_clones)

# %%
M, N = data["images_train"].shape[-2:]
variables = groups.NDVariableArray(variable_size=np.sum(n_clones), shape=(M, N))
fg = graph.FactorGraph(variables)

# %%
# Add top-down factors
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii + 1, jj)] for ii in range(M - 1) for jj in range(N)
    ],
    log_potential_matrix=weights["logWtd"],
)
# Add left-right factors
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii, jj + 1)] for ii in range(M) for jj in range(N - 1)
    ],
    log_potential_matrix=weights["logWlr"],
)
# Add diagonal factors
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii + 1, jj + 1)] for ii in range(M - 1) for jj in range(N - 1)
    ],
    log_potential_matrix=weights["logWfd"],
)
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii - 1, jj + 1)] for ii in range(1, M) for jj in range(N - 1)
    ],
    log_potential_matrix=weights["logWsd"],
)

# %%
run_bp, _, get_beliefs = graph.BP(fg.bp_state, 15, 1.0)

# %%
idx = 0
marginals = graph.get_marginals(
    get_beliefs(run_bp(evidence_updates={None: evidence[idx]}, damping=0.0))
)
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].imshow(evidence[idx, ..., -2])
ax[1].imshow(marginals[..., -2] < 0.5)

# %%
logp = np.mean(np.log(np.sum(targets[idx] * marginals, axis=-1)))
