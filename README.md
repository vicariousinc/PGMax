[![continuous-integration](https://github.com/vicariousinc/PGMax/actions/workflows/ci.yaml/badge.svg)](https://github.com/vicariousinc/PGMax/actions/workflows/ci.yaml)

# PGMax
PGMax is a library for working with Factor Graphs in [JAX](https://jax.readthedocs.io/en/latest/). It currently provides an interface for specifying factor graphs of any type, as well as an efficient implementation of max-product belief propagation and inference on these graphs.

## Installation Instructions
### Developer
1. Install Poetry by following [these instructions](https://python-poetry.org/docs/master/). Note: you may need to logout and log back in after running the install command for the `poetry --version` command to work in your shell environment.
1. Navigate to this project's directory and activate a poetry shell via the command `poetry shell`. This creates and activates a virtual environment for you to use with this project.
1. Install the project's dependencies into your virtual environment with the command `poetry install`. Your environment will now contain both developer and user dependencies!
    1. By default this installs JAX for GPU with CUDA 11.1 or later. If you'd like to use JAX with a different CUDA version, do:
        ```
        CUDA_VERSION=cuda101 # Options are cuda101, cuda102, cuda110, cuda111 depending on your CUDA version (10.1, 10.2, ...)
        pip install --upgrade jax jaxlib==0.1.67+$CUDA_VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html
        ```
1. Do `pre-commit install` to initialize pre-commit hooks