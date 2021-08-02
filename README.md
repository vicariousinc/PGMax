[![continuous-integration](https://github.com/vicariousinc/PGMax/actions/workflows/ci.yaml/badge.svg)](https://github.com/vicariousinc/PGMax/actions/workflows/ci.yaml)
[![PyPI version](https://badge.fury.io/py/pgmax.svg)](https://badge.fury.io/py/pgmax)

# PGMax
PGMax is a library for working with Factor Graphs in [JAX](https://jax.readthedocs.io/en/latest/). It currently provides an interface for specifying factor graphs of any type, as well as an efficient implementation of max-product belief propagation and inference on these graphs.

## Installation Instructions
### User
1. Install the library using pip via: `pip install pgmax`
1. By default this installs JAX for CPU. If you'd like to use JAX with a GPU and specific CUDA version (highly recommended), follow the official instructions [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Developer
1. Clone this project's [GitHub Repository](https://github.com/vicariousinc/PGMax)
1. Install Poetry by following [these instructions](https://python-poetry.org/docs/master/). Note: you may need to logout and log back in after running the install command for the `poetry --version` command to work in your shell environment.
1. Navigate to this project's directory and activate a poetry shell via the command `poetry shell`. This creates and activates a virtual environment for you to use with this project.
1. Install the project's dependencies into your virtual environment with the command `poetry install`. Your environment will now contain both developer and user dependencies!
    1. By default this installs JAX for CPU. If you'd like to use JAX with a GPU and specific CUDA version (highly recommended), follow the official instructions [here](https://github.com/google/jax#pip-installation-gpu-cuda).
1. Do `pre-commit install` to initialize pre-commit hooks