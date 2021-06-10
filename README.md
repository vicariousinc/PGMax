# PGMax
Temporary Placeholder Repository for the PGMax Library, which aims to implement Factor Graphs and efficient inference on these in JAX!

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

## Experimental Framework and Workflow
### Experimental Framework
Our current overarching goal is to explore different implementation choices for belief propagation to settle on the best one for use in the PGMax library. To this end, we will benchmark the performance of various implementations on a graph-cut finding task on a 128 x 128 "sanity check" example image (the image, along with the Factor Graph derived from it can be seen in `test_notebooks/sanity_check_orig.ipynb` under the section "Setting up Image and Factor Graph"). Specifically, we will look at both the run-time (wall-clock) and memory needed to run 1000 iterations of belief propagation on the above-mentioned Factor Graph. Note that for the purposes of these experiments, we only care about benchmarking **belief propagation**, so we do not need to report the time taken or memory utilization of any other process (like MAP inference, creation of data-structures, etc.). In fact, discovering some implementation for which creating the necessary data structures given the current setup is inefficient, but belief propagation itself is extremely efficient, would be extremely useful (we can later modify the current interface for creating Factor Graphs to be conducive to creating such data structures!).

### Experiment Workflow
1. Write a new version of Max-Product Belief Propagation (MPBP) that makes some different design decisions than previously-written versions (which can all be found under the `pgmax` folder)
    1. The I/O for your implementation is specified under the "Belief Propagation" section of `test_notebooks/sanity_check_orig.ipynb`. Essentially, you must write a method that accepts the following arguments:
        1. a FactorGraph (`pgmax/node_classes/FactorGraph`)
        1. a dict mapping VariableNodes (`pgmax/node_classes/VariableNode`) to np.arrays of evidence for each variable
        1. some number of iterations (int kept at 1000 for these experiments)
        1. some damping factor (float kept at 0.5 for these experiments)

        Your method must then return a dictionary mapping each VariableNode in the input FactorGraph to an integer representing the MAP state of that variable.
    1. You will most-likely want to implement BP in the following series of high-level steps
        1. Use the input data structures (FactorGraph and dictionary for evidence) to compile some JAX-compatible data-structures like arrays, etc.
        1. Use these data structures to write functions for message-passing and updating with damping that can all be wrapped in some loop.
        1. Use these data structures to write some function to perform MAP inference on the updated messages
        1. Convert the final MAP inference results to the output dictionary required by the I/O spec
1. Analyze the timing and memory required for 1000 iterations of BP in your implementation