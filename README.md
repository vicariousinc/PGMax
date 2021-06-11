# PGMax
Repository for the PGMax Library, which aims to implement Factor Graphs and efficient inference on these in JAX!

## Developer Installation Instructions
### Prerequisites
1. Install Poetry by following [these instructions](https://python-poetry.org/docs/). Note: you may need to logout and log back in after running the install command for the `poetry --version` command to work in your shell environment.
1. Navigate to this project's directory and activate a poetry shell via the command `poetry shell`. This creates and activates a virtual environment for you to use with this project.
1. Install the project's dependencies into your virtual environment with the command `poetry install`. Your environment will now contain both developer and user dependencies!
1. (Optional) If you'd like JAX to recognize and run on your GPU, you will need to run the below commands. Note that if your cuda version is 11.1 or greater, you should just use the CUDA_VERSION=cuda111 option:
    ```
    CUDA_VERSION=cuda101 # Options are cuda101, cuda102, cuda110, cuda111 depending on your CUDA version (10.1, 10.2, ...)
    pip install --upgrade jax jaxlib==0.1.67+$CUDA_VERSION -f https://storage.googleapis.com/jax-releases/jax_releases.html
    ```