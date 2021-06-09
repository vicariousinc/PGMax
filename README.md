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

## Experimental Framework and Workflow
### Experimental Framework
Our current overarching goal is to explore different implementation choices for belief propagation to settle on the best one for use in the PGMax library. To this end, we will benchmark the performance of various implementations on a graph-cut finding task on a 128 x 128 "sanity check" example image (the image, along with the Factor Graph derived from it can be seen in `test_notebooks/sanity_check_orig.ipynb` under the section "Setting up Image and Factor Graph"). Specifically, we will look at both the run-time (wall-clock) and memory needed to run 1000 iterations of belief propagation on the above-mentioned Factor Graph. Note that for the purposes of these experiments, we only care about benchmarking **belief propagation**, so we do not need to report the time taken or memory utilization of any other process (like MAP inference, creation of data-structures, etc.). In fact, discovering some implementation for which creating the necessary data structures given the current setup is inefficient, but belief propagation itself is extremely efficient, would be extremely useful (we can later modify the current interface for creating Factor Graphs to be conducive to creating such data structures!).

### Experiment Workflow
