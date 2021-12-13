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
import matplotlib.pyplot as plt
import numpy as np
from jax.example_libraries import optimizers
from tqdm.notebook import tqdm

from pgmax.fg import graph, groups

# %% [markdown]
# # Visualize a trained GMRF

# %%
# Load test data
data = np.load("example_data/noisy_mnist.npz")
noisy_images = data["noisy_images_test"]
target_images = data["images_test"]

# Load saved log potentials
log_potentials = dict(**np.load("example_data/gmrf_log_potentials.npz"))
n_clones = log_potentials.pop("n_clones")
p_contour = jax.device_put(np.repeat(data["p_contour"], n_clones))
prototype_targets = jax.device_put(
    np.array(
        [
            np.repeat(np.array([1, 0, 0]), n_clones),
            np.repeat(np.array([0, 1, 0]), n_clones),
            np.repeat(np.array([0, 0, 1]), n_clones),
        ]
    )
)

# %%
M, N = target_images.shape[-2:]
variable_size = np.sum(n_clones)
variables = groups.NDVariableArray(variable_size=variable_size, shape=(M, N))
fg = graph.FactorGraph(variables)

# %%
# Add top-down factors
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii + 1, jj)] for ii in range(M - 1) for jj in range(N)
    ],
    name="top_down",
)
# Add left-right factors
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii, jj + 1)] for ii in range(M) for jj in range(N - 1)
    ],
    name="left_right",
)
# Add diagonal factors
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii + 1, jj + 1)] for ii in range(M - 1) for jj in range(N - 1)
    ],
    name="diagonal0",
)
fg.add_factor_group(
    factory=groups.PairwiseFactorGroup,
    connected_variable_names=[
        [(ii, jj), (ii - 1, jj + 1)] for ii in range(1, M) for jj in range(N - 1)
    ],
    name="diagonal1",
)

# %%
run_bp, _, get_beliefs = graph.BP(fg.bp_state, 15, 1.0)

# %%
n_plots = 5
indices = np.random.permutation(noisy_images.shape[0])[:n_plots]
fig, ax = plt.subplots(n_plots, 3, figsize=(30, 10 * n_plots))
for plot_idx, idx in tqdm(enumerate(indices), total=n_plots):
    noisy_image = noisy_images[idx]
    target_image = target_images[idx]
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
    pred_image = np.argmax(
        np.stack(
            [
                np.sum(marginals[..., :-2], axis=-1),
                marginals[..., -2],
                marginals[..., -1],
            ],
            axis=-1,
        ),
        axis=-1,
    )
    ax[plot_idx, 0].imshow(noisy_image)
    ax[plot_idx, 0].axis("off")
    ax[plot_idx, 1].imshow(target_image)
    ax[plot_idx, 1].axis("off")
    ax[plot_idx, 2].imshow(pred_image)
    ax[plot_idx, 2].axis("off")
    if plot_idx == 0:
        ax[plot_idx, 0].set_title("Input noisy image", fontsize=40)
        ax[plot_idx, 1].set_title("Target image", fontsize=40)
        ax[plot_idx, 2].set_title("GMRF predicted image", fontsize=40)

fig.tight_layout()


# %% [markdown]
# # Train the model from scratch
#
# The following training loop requires a GPU with at least 11 GB of memory.

# %%
@jax.jit
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


@jax.jit
def batch_loss(noisy_images, target_images, log_potentials):
    return jnp.mean(
        jax.vmap(loss, in_axes=(0, 0, None), out_axes=0)(
            noisy_images, target_images, log_potentials
        )
    )


# %%
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
opt_state = init_fun(
    {
        "top_down": np.random.randn(variable_size, variable_size),
        "left_right": np.random.randn(variable_size, variable_size),
        "diagonal0": np.random.randn(variable_size, variable_size),
        "diagonal1": np.random.randn(variable_size, variable_size),
    }
)

noisy_images_train = data["noisy_images_train"]
target_images_train = data["images_train"]
batch_size = 10
n_epochs = 10
n_batches = noisy_images_train.shape[0] // batch_size
with tqdm(total=n_epochs * n_batches) as pbar:
    for epoch in range(n_epochs):
        indices = np.random.permutation(noisy_images_train.shape[0])
        for idx in range(n_batches):
            batch_indices = indices[idx * batch_size : (idx + 1) * batch_size]
            batch_noisy_images, batch_target_images = (
                noisy_images_train[batch_indices],
                target_images_train[batch_indices],
            )
            step = epoch * n_batches + idx
            value, opt_state = update(
                step, batch_noisy_images, batch_target_images, opt_state
            )
            pbar.update()
            pbar.set_postfix(loss=value)
