# Experimental Framework and Workflow
## Experimental Framework
Our current overarching goal is to explore different implementation choices for belief propagation to settle on the best one for use in the PGMax library. To this end, we will benchmark the performance of various implementations on a graph-cut finding task on a 150 x 150 "sanity check" example image (the image, along with the Factor Graph derived from it can be seen in `test_notebooks/sanity_check_orig.ipynb` under the section "Setting up Image and Factor Graph"). Specifically, we will look at both the run-time (wall-clock) and memory needed to run 1000 iterations of belief propagation on the above-mentioned Factor Graph. Note that for the purposes of these experiments, we only care about benchmarking **belief propagation**, so we do not need to report the time taken or memory utilization of any other process (like MAP inference, creation of data-structures, etc.). In fact, discovering some implementation for which creating the necessary data structures given the current setup is inefficient, but belief propagation itself is extremely efficient, would be extremely useful (we can later modify the current interface for creating Factor Graphs to be conducive to creating such data structures!).

## Experiment Workflow
1. Write a new version of Max-Product Belief Propagation (MPBP) that makes some different design decisions than previously-written versions (which can all be found under the `pgmax/contrib/mpbp` folder). Write your implementation in a **separate file** under the `pgmax/contrib/mpbp` folder.
    1. The I/O for your implementation is specified under the "Belief Propagation" section of `test_notebooks/sanity_check_orig.ipynb`. Essentially, you must write a method that accepts the following arguments:
        1. a FactorGraph (`pgmax/contrib/interface/node_classes/FactorGraph`)
        1. a dict mapping VariableNodes (`pgmax/contrib/interface/node_classes/VariableNode`) to np.arrays of evidence for each variable
        1. some number of iterations (int kept at 1000 for these experiments)
        1. some damping factor (float kept at 0.5 for these experiments)

        Your method must then return a dictionary mapping each VariableNode in the input FactorGraph to an integer representing the MAP state of that variable.

    1. You will most-likely want to implement BP in the following series of high-level steps (to do this, you can copy the `pgmax/contrib/mpbp/mp_belief_prop_jax_orig.py` file and then build upon/edit it).
        1. Use the input data structures (FactorGraph and dictionary for evidence) to compile some JAX-compatible data-structures like arrays, etc. You can do this somewhat easily by modifying the `compile_jax_data_structures()` function under `pgmax/contrib/mpbp/mp_belief_prop_jax_orig.py`. Feel free to use this function as-is if you just want the current data-structures (documented in the function's docstring).
        1. Use these data structures to write functions for message-passing and updating with damping that can all be wrapped in some loop.
        1. Use these same data structures to write some function to perform MAP inference on the updated messages. You can use or modify the `compute_map_estimate_jax()` function from `pgmax/contrib/mpbp/mp_belief_prop_jax_orig.py`.
        1. Convert the final MAP inference results to the output dictionary required by the I/O spec. You can use or modify the `convert_map_to_dict()` function from `pgmax/contrib/mpbp/mp_belief_prop_jax_orig.py`.

1. Verify that your implementation is correct by running the `test_notebooks/sanity_check_orig.ipynb` (or making a copy) and using your new implementation for BP and MAP inference. The output of the notebook should look exactly as in the image on the following Confluence page (random seed is fixed, so there should be no differences): [https://vicarious.atlassian.net/l/c/8MegBH2a](https://vicarious.atlassian.net/l/c/8MegBH2a)


1. Analyze the timing and memory required for 1000 iterations of BP (and only for this!) in your implementation
    1. To perform timing analysis, simply use the `timeit` library. Be careful to use JAX's `block_until_ready()` function to ensure the time taken for BP is accurately captured.
        1. For example, the `pgmax/contrib/mpbp/mp_belief_prop_jax_orig.py` file performs timing analysis of the belief propagation process as below:
            ```
            msg_update_start_time = timer()
            msgs_arr = run_mpbp_update_loop(
                msgs_arr,
                evidence_arr,
                neighbors_vtof_arr,
                neighbors_ftov_arr,
                neighbor_vars_valid_configs_arr,
                num_iters,
            ).block_until_ready()
            msg_update_end_time = timer()
            print(
                f"Message Passing completed in: {msg_update_end_time - msg_update_start_time}s"
            )
            ```
        1. Be sure to run the entire BP loop once before timing to allow JAX to compile all the functions. This way, you can measure the execution time independent of the compilation time!
    1. To perform memory usage analysis, simply run `nvidia-smi` in a separate terminal on your workstation in polling mode (eg `watch -n 0.5 nvidia-smi`, which polls every 0.5s). Record the maximum `GPU Memory Usage` you see.

1. Record the timing and memory usage, along with relevant implementation changes made, at the following Confluence page: [https://vicarious.atlassian.net/l/c/8MegBH2a](https://vicarious.atlassian.net/l/c/8MegBH2a)