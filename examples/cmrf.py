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
import joblib
import numpy as np
from jax.experimental import optimizers
from tqdm import tqdm

from pgmax.fg import graph, groups

# %%
data = np.load("/storage/ltr/papers/icml2020_query_training/cmrf/noisy_mnist_8_0.2.npz")
weights = np.load(
    "/storage/ltr/papers/icml2020_query_training/cmrf/cmrf8_weights_15_mb50_lr1em2_nc64_emp.npz"
)
n_clones = weights["n_clones"]
p_contour = np.repeat(data["p_contour"], n_clones)
p_contour[p_contour == 0.0] = 1e-10
p_contour = jax.device_put(p_contour)
prototype_targets = jax.device_put(
    np.array(
        [
            np.repeat(np.array([1, 0, 0]), n_clones),
            np.repeat(np.array([0, 1, 0]), n_clones),
            np.repeat(np.array([0, 0, 1]), n_clones),
        ]
    )
)
noisy_images = data["noisy_images_train"]
target_images = data["images_train"]

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
def loss(noisy_image, target_image, log_potentials):
    evidence = jnp.log(jnp.where(noisy_image[..., None] == 0, p_contour, 1 - p_contour))
    target = prototype_targets[target_image]
    marginals = graph.get_marginals(
        get_beliefs(
            run_bp(
                evidence_updates={None: evidence},
                log_potentials_updates=log_potentials,
                damping=0.0,
            )
        )
    )
    logp = jnp.mean(jnp.log(jnp.sum(target * marginals, axis=-1)))
    return -logp


def batch_loss(noisy_images, target_images, log_potentials):
    return jnp.mean(
        jax.vmap(loss, in_axes=(0, 0, None), out_axes=0)(
            noisy_images, target_images, log_potentials
        )
    )


value_and_grad = jax.jit(jax.value_and_grad(batch_loss, argnums=2))
init_fun, opt_update, get_params = optimizers.adam(2e-3)


@jax.jit
def update(step, batch_noisy_images, batch_target_images, opt_state):
    value, grad = value_and_grad(
        batch_noisy_images, batch_target_images, get_params(opt_state)
    )
    opt_state = opt_update(step, grad, opt_state)
    return value, opt_state


# %%
log_potentials = {
    "td": weights["logWtd"],
    "lr": weights["logWlr"],
    "fd": weights["logWfd"],
    "sd": weights["logWsd"],
}
opt_state = init_fun(log_potentials)

batch_size = 10
n_batches = noisy_images.shape[0] // batch_size
n_epochs = 10
with tqdm(total=n_epochs * n_batches) as pbar:
    for epoch in range(n_epochs):
        indices = np.random.permutation(noisy_images.shape[0])
        for idx in range(n_batches):
            batch_indices = indices[idx * batch_size : (idx + 1) * batch_size]
            batch_noisy_images, batch_target_images = (
                noisy_images[batch_indices],
                target_images[batch_indices],
            )
            step = epoch * n_batches + idx
            value, opt_state = update(
                step, batch_noisy_images, batch_target_images, opt_state
            )
            pbar.update()
            pbar.set_postfix(loss=value)
            if step % 100 == 0:
                joblib.dump(get_params(opt_state), "weights.joblib")
