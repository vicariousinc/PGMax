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
import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
from jax.experimental import optimizers
from tqdm.notebook import tqdm

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
data = np.load("/storage/ltr/papers/icml2020_query_training/cmrf/noisy_mnist_8_0.2.npz")
weights = np.load(
    "/storage/ltr/papers/icml2020_query_training/cmrf/cmrf8_weights_15_mb50_lr1em2_nc64_emp.npz"
)
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
    name="td",
)
# Add left-right factors
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii, jj + 1)] for ii in range(M) for jj in range(N - 1)
    ],
    name="lr",
)
# Add diagonal factors
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii + 1, jj + 1)] for ii in range(M - 1) for jj in range(N - 1)
    ],
    name="fd",
)
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii - 1, jj + 1)] for ii in range(1, M) for jj in range(N - 1)
    ],
    name="sd",
)

# %%
run_bp, _, get_beliefs = graph.BP(fg.bp_state, 15, 1.0)


# %%
def loss(evidence, targets, log_potentials):
    marginals = graph.get_marginals(
        get_beliefs(
            run_bp(
                evidence_updates={None: evidence},
                log_potentials_updates=log_potentials,
                damping=0.0,
            )
        )
    )
    logp = jnp.mean(jnp.log(jnp.sum(targets * marginals, axis=-1)))
    return -logp


def batch_loss(evidence, targets, log_potentials):
    return jnp.mean(
        jax.vmap(loss, in_axes=(0, 0, None), out_axes=0)(
            evidence, targets, log_potentials
        )
    )


value_and_grad = jax.jit(jax.value_and_grad(batch_loss, argnums=2))

# %%
log_potentials = {
    "td": weights["logWtd"],
    "lr": weights["logWlr"],
    "fd": weights["logWfd"],
    "sd": weights["logWsd"],
}

init_fun, opt_update, get_params = optimizers.adam(2e-3)
opt_state = init_fun(log_potentials)


@jax.jit
def update(step, batch_evidence, batch_targets, opt_state):
    value, grad = value_and_grad(batch_evidence, batch_targets, get_params(opt_state))
    opt_state = opt_update(step, grad, opt_state)
    return value, opt_state


batch_size = 10
n_batches = evidence.shape[0] // batch_size
n_epochs = 10
with tqdm(total=n_epochs * n_batches) as pbar:
    for epoch in range(n_epochs):
        indices = np.random.permutation(evidence.shape[0])
        for idx in range(n_batches):
            batch_indices = indices[idx * batch_size : (idx + 1) * batch_size]
            batch_evidence, batch_targets = (
                evidence[batch_indices],
                targets[batch_indices],
            )
            value, opt_state = update(
                epoch * n_batches + idx, batch_evidence, batch_targets, opt_state
            )
            pbar.update()
            pbar.set_postfix(loss=value)
