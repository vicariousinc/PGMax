# PGMax
Temporary Placeholder Repository for the PGMax Library, which aims to implement Factor Graphs and efficient inference on these in JAX!

## Developer Installation Instructions
### Prerequisites
1. Install Poetry by following [these instructions](https://python-poetry.org/docs/). Note: you may need to logout and log back in after running the install command for the `poetry --version` command to work in your shell environment.
1. Navigate to this project's directory and activate a poetry shell via the command `poetry shell`. This creates and activates a virtual environment for you to use with this project.
1. Install the project's dependencies into your virtual environment with the command `poetry install`. Your environment will now contain both developer and user dependencies!
1. (Optional) If you'd like JAX to recognize and run on your GPU, you will need to run the below commands:
    ```
    PYTHON_VERSION=cp38  
    CUDA_VERSION=cuda101  # alternatives: cuda100, cuda101, cuda102, cuda110, check your cuda version
    PLATFORM=manylinux2010_x86_64  # alternatives: manylinux2010_x86_64
    BASE_URL='https://storage.googleapis.com/jax-releases'
    pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.55-$PYTHON_VERSION-none-$PLATFORM.whl
    pip install --upgrade jax
    ```
