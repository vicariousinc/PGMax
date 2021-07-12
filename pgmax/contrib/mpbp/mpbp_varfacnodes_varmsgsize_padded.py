from timeit import default_timer as timer
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import pgmax.contrib.interface.node_classes_with_factortypes as node_classes

from pgmax.contrib.mpbp.mpbp_varnode_fac_lowmem import (  # isort:skip
    convert_map_to_dict,
)

NEG_INF = (
    -100000.0
)  # A large negative value to use as -inf for numerical stability reasons

# NOTE: This file contains an implementation of max-product belief propagation that uses padding to handle
# messages of different sizes.


def run_mp_belief_prop_and_compute_map(
    fg: node_classes.FactorGraph,
    evidence: Dict[node_classes.VariableNode, np.ndarray],
    num_iters: int,
    damping_factor: float,
) -> Dict[node_classes.VariableNode, int]:
    """Performs max-product belief propagation on a FactorGraph fg for num_iters iterations and returns the MAP
    estimate.

    Args
        fg: A FactorGraph object upon which to do belief propagation
        evidence: Each entry represents the constant, evidence message that's passed to the corresponding
            VariableNode that acts as the key
        num_iters: The number of iterations for which to perform message passing
        damping_factor: The damping factor to use for message updates between one timestep and the next

    Returns:
        A dictionary mapping each variable to its MAP estimate value
    """
    # NOTE: This currently assumes all variable nodes have the same size. Thus, all messages have the same size

    start_time = timer()
    (
        msgs_arr,
        evidence_arr,
        edges_to_var_arr,
        factor_configs,
        edge_vals_to_config_summary_indices,
        var_to_indices_dict,
        num_val_configs,
    ) = compile_jax_data_structures(fg, evidence)
    end_time = timer()
    print(f"Data structures compiled in: {end_time - start_time}s")

    # Convert all arrays to jnp.ndarrays for use in BP
    # (Comments on the right show cumulative memory usage as each of these lines execute)
    msgs_arr = jax.device_put(msgs_arr)  # 69 MiB
    evidence_arr = jax.device_put(evidence_arr)  # 85 MiB
    edges_to_var_arr = jax.device_put(edges_to_var_arr)
    factor_configs = jax.device_put(factor_configs)
    edge_vals_to_config_summary_indices = jax.device_put(
        edge_vals_to_config_summary_indices
    )

    @jax.partial(jax.jit, static_argnames=("num_val_configs", "num_iters"))
    def run_mpbp_update_loop(
        msgs_arr,
        evidence_arr,
        edges_to_var_arr,
        factor_configs,
        edge_vals_to_config_summary_indices,
        num_val_configs,
        num_iters,
    ):
        "Function wrapper that leverages jax.lax.scan to efficiently perform BP"

        def mpbp_update_step(msgs_arr, x):
            # Variable to Factor messages update
            updated_vtof_msgs = pass_var_to_fac_messages_jnp(
                msgs_arr,
                evidence_arr,
                edges_to_var_arr,
            )
            # Factor to Variable messages update
            updated_ftov_msgs = pass_fac_to_var_messages_jnp(
                msgs_arr,
                factor_configs,
                edge_vals_to_config_summary_indices,
                num_val_configs,
            )
            # Damping before final message update
            msgs_arr = damp_and_update_messages(
                updated_vtof_msgs, updated_ftov_msgs, msgs_arr, damping_factor
            )
            return msgs_arr, None

        msgs_arr, _ = jax.lax.scan(mpbp_update_step, msgs_arr, None, num_iters)
        return msgs_arr

    # Run the entire BP loop once to allow JAX to compile
    msg_comp_start_time = timer()
    msgs_arr = run_mpbp_update_loop(
        msgs_arr,
        evidence_arr,
        edges_to_var_arr,
        factor_configs,
        edge_vals_to_config_summary_indices,
        num_val_configs,
        num_iters,
    ).block_until_ready()
    msg_comp_end_time = timer()
    print(
        f"First Time Message Passing completed in: {msg_comp_end_time - msg_comp_start_time}s"
    )

    msg_update_start_time = timer()
    msgs_arr = run_mpbp_update_loop(
        msgs_arr,
        evidence_arr,
        edges_to_var_arr,
        factor_configs,
        edge_vals_to_config_summary_indices,
        num_val_configs,
        num_iters,
    ).block_until_ready()
    msg_update_end_time = timer()
    print(
        f"Second Time Message Passing completed in: {msg_update_end_time - msg_update_start_time}s"
    )

    map_start_time = timer()
    map_arr = compute_map_estimate_jax(msgs_arr, evidence_arr, edges_to_var_arr)
    map_end_time = timer()
    print(f"MAP inference took {map_end_time - map_start_time}s")

    map_conversion_start = timer()
    var_map_estimate = convert_map_to_dict(map_arr, var_to_indices_dict)
    map_conversion_end = timer()
    print(f"MAP conversion to dict took {map_conversion_end - map_conversion_start}")

    return var_map_estimate  # type: ignore


def compile_jax_data_structures(
    fg: node_classes.FactorGraph, evidence: Dict[node_classes.VariableNode, np.ndarray]
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[node_classes.VariableNode, int],
    int,
]:
    """Creates data-structures that can be efficiently used with JAX for MPBP.

    Args:
        fg: A FactorGraph object upon which to do belief propagation
        evidence: Each entry represents the constant, evidence message that's passed to the corresponding
            VariableNode that acts as the key

    Returns:
        tuple containing data structures useful for message passing updates in JAX:
            msgs_arr: Array shape is (2, num_edges, max_msg_size). This holds all the messages. the 0th index of the
                0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to v-> f msgs. To make this
                a regularly-shaped array, messages are padded with a large negative value
            evidence_arr: Array shape is shape (num_var_nodes, max_msg_size). evidence_arr[x,:] corresponds to the evidence
                for the variable node at var_neighbors_arr[x,:,:]
            edges_to_var_arr: Array shape is (num_edges,). The ith entry is an integer representing which variable this edge is
                connected to.
            factor_configs: Maximum array shape is bounded by (3, num_factors * max_num_configs * max_config_size). The 0th axis is
                essentially a flattened mapping from factors to edges, with some repetitions so that it has the same shape as
                the other axes; the entries provide a flattened set of indices that index into neighboring edges for each factor.
                The 1st axis contains a flat list of valid configurations, and the 2nd axis contains segmentation masks (i.e, all
                entries corresponding to factor 0 config 0 are labelled 0, all entries corresponding to factor 0 config 1 are labelled
                1, and so on).
            edge_vals_to_config_summary_indices: Maximum array shape is bounded by (2, num_edges * msg_size * max_config_size). The
                0th axis contains indices corresponding to the configurations that involve a particular edge taking a particular
                value, and the 1st axis contains a segmentation mask of these values (i.e, all configuration indices corresponding to
                edge 0 value 0 will be labelled 0, all indices corresponding to edge 0 value 1 will be labelled 1 and so on.) Note that
                the length of the 1st axis will be the same as factor_configs
            var_to_indices_dict: for a particular var_node key, var_to_indices_dict[var_node] will yield the number in edges_to_var_arr
                corresponding to var_node
            num_val_configs: the total number of valid configurations for factors in the factor graph.
    """
    num_edges = fg.count_num_edges()
    max_msg_size = fg.find_max_msg_size()

    # This below loop constructs the following data structures that are returned:
    # - evidence_arr
    # - edges_to_var_arr
    # It also pads msgs_arr with NEG_INF
    fac_to_var_msg_to_index_dict = {}
    # Initialize all entries in the evidence array to be negative infinity
    evidence_arr = np.ones((len(fg.variable_nodes), max_msg_size)) * NEG_INF
    edges_to_var_arr = np.zeros(num_edges, dtype=int)
    msgs_arr = np.zeros((2, num_edges, max_msg_size))
    var_to_indices_dict = {}
    tmp_fac_to_index_dict: Dict[node_classes.FactorNode, int] = {}
    edge_counter = 0
    fac_index = 0
    for var_index, var_node in enumerate(fg.variable_nodes):
        for fac_node_neighbor in var_node.neighbors:
            fac_to_var_msg_to_index_dict[(fac_node_neighbor, var_node)] = edge_counter
            edges_to_var_arr[edge_counter] = var_index
            if tmp_fac_to_index_dict.get(fac_node_neighbor) is None:
                tmp_fac_to_index_dict[fac_node_neighbor] = fac_index
                fac_index += 1
            # Pad msgs_arr with NEG_INF
            msgs_arr[0, edge_counter, var_node.num_states :] = (
                np.ones(max_msg_size - var_node.num_states) * NEG_INF
            )
            msgs_arr[1, edge_counter, var_node.num_states :] = (
                np.ones(max_msg_size - var_node.num_states) * NEG_INF
            )
            edge_counter += 1
        evidence_arr[var_index, : var_node.num_states] = evidence[var_node]
        var_to_indices_dict[var_node] = var_index

    # Create and populate remaining data structures
    # Start by defining some useful constants that will be necessary to construct the arrays
    # we're interested in
    num_factors = fg.count_num_factor_nodes()
    max_num_configs = fg.find_max_num_valid_configs()
    max_config_size = fg.find_max_valid_config_size()

    # Initialize these arrays to the maximum possible size they could be. Make each of the values -100 so
    # unallocated rows can be recognized and deleted later
    factor_configs = (
        np.ones((3, num_factors * max_num_configs * max_config_size), dtype=int) * -100
    )
    edge_vals_to_config_summary_indices = (
        np.ones((2, num_edges * max_msg_size * max_num_configs), dtype=int) * -100
    )

    configs_insertion_index = 0
    configs_counter = 0
    flat_edge_configs_counter = 0
    for fac_node, fac_index in sorted(
        tmp_fac_to_index_dict.items(), key=lambda item: item[1]
    ):
        # Update factor_configs[1,2]
        var_configs = fac_node.factor_type.neighbor_configs_arr
        num_configs, len_of_each_config = var_configs.shape
        config_indices_arr = np.arange(configs_counter, configs_counter + num_configs)
        segs_to_insert = np.repeat(config_indices_arr, len_of_each_config)
        configs_counter += num_configs
        configs_to_insert = var_configs.flatten()
        factor_configs[
            1,
            configs_insertion_index : configs_insertion_index
            + configs_to_insert.shape[0],
        ] = configs_to_insert
        factor_configs[
            2,
            configs_insertion_index : configs_insertion_index
            + configs_to_insert.shape[0],
        ] = segs_to_insert

        # Loop thru all neighboring variables to update edge_vals_to_config_summary_indices
        surr_edge_indices = []
        for var_node in fac_node.neighbors:
            edge_idx = fac_to_var_msg_to_index_dict[(fac_node, var_node)]
            surr_edge_indices.append(edge_idx)
            config_column_index = fac_node.neighbor_to_index_mapping[var_node]
            for msg_idx in range(var_node.num_states):
                flattened_msg_idx_val = (edge_idx * max_msg_size) + msg_idx
                config_indices_for_msg_idx = config_indices_arr[
                    var_configs[:, config_column_index] == msg_idx
                ]
                num_configs_for_msg_idx = config_indices_for_msg_idx.shape[0]
                edge_vals_to_config_summary_indices[
                    0,
                    flat_edge_configs_counter : flat_edge_configs_counter
                    + num_configs_for_msg_idx,
                ] = config_indices_for_msg_idx
                edge_vals_to_config_summary_indices[
                    1,
                    flat_edge_configs_counter : flat_edge_configs_counter
                    + num_configs_for_msg_idx,
                ] = np.repeat(flattened_msg_idx_val, num_configs_for_msg_idx)
                flat_edge_configs_counter += num_configs_for_msg_idx

        # Update factor_configs[0]
        surr_edges_arr = np.tile(np.array(surr_edge_indices, dtype=int), num_configs)
        factor_configs[
            0,
            configs_insertion_index : configs_insertion_index
            + configs_to_insert.shape[0],
        ] = surr_edges_arr

        configs_insertion_index += configs_to_insert.shape[0]

    # Finally, delete all misc rows that have not been changed from their initial value!
    factor_configs = np.delete(
        factor_configs, np.where(factor_configs[2, :] == -100)[0], axis=1
    )
    edge_vals_to_config_summary_indices = np.delete(
        edge_vals_to_config_summary_indices,
        np.where(edge_vals_to_config_summary_indices[1, :] == -100)[0],
        axis=1,
    )

    # Grab the final necessary parameter from the factor graph
    num_val_configs = fg.count_total_num_valid_configs()

    return (
        msgs_arr,
        evidence_arr,
        edges_to_var_arr,
        factor_configs,
        edge_vals_to_config_summary_indices,
        var_to_indices_dict,
        num_val_configs,
    )


@jax.jit
def pass_var_to_fac_messages_jnp(
    msgs_arr: jnp.array,
    evidence_arr: jnp.array,
    edges_to_var_arr: jnp.array,
) -> jnp.array:
    """
    passes messages from VariableNodes to FactorNodes and computes a new updated set of messages using JAX

    Args:
        msgs_arr: Array shape is (2, num_edges, msg_size). This holds all the messages. the 0th index of the
            0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to v-> f msgs.
       evidence_arr: Array shape is shape (num_var_nodes, msg_size). evidence_arr[x,:] corresponds to the evidence
            for a particular VariableNode x.
        edges_to_var_arr: Array len is num_edges. The ith entry is an integer representing which variable this edge is connected
            to.
    Returns:
        Array of shape (num_edges, msg_size) corresponding to the updated v->f messages after normalization and clipping
    """
    # For each variable, sum the neighboring factor to variable messages and the evidence.
    var_sums_arr = (
        jax.ops.segment_sum(
            msgs_arr[0], edges_to_var_arr, num_segments=evidence_arr.shape[0]
        )
        + evidence_arr
    )
    updated_vtof_msgs = var_sums_arr[edges_to_var_arr] - msgs_arr[0]

    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = updated_vtof_msgs - updated_vtof_msgs.max(
        axis=1, keepdims=True
    )
    clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, 1000)

    return clipped_updated_msgs


@jax.partial(jax.jit, static_argnames=("num_val_configs"))
def pass_fac_to_var_messages_jnp(
    msgs_arr: jnp.ndarray,
    factor_configs: jnp.ndarray,
    edge_vals_to_config_summary_indices: jnp.ndarray,
    num_val_configs: int,
) -> jnp.ndarray:

    """
    passes messages from FactorNodes to VariableNodes and computes a new, updated set of messages using JAX

    Args:
        msgs_arr: Array shape is (2, num_edges, msg_size). This holds all the messages. the 0th index of the
            0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to v-> f msgs.
        factor_configs: Maximum array shape is bounded by (3, num_factors * max_num_configs * max_config_size). The 0th axis is
            essentially a flattened mapping from factors to edges, with some repetitions so that it has the same shape as
            the other axes; the entries provide a flattened set of indices that index into neighboring edges for each factor.
            The 1st axis contains a flat list of valid configurations, and the 2nd axis contains segmentation masks (i.e, all
            entries corresponding to factor 0 config 0 are labelled 0, all entries corresponding to factor 0 config 1 are labelled
            1, and so on).
        edge_vals_to_config_summary_indices: Maximum array shape is bounded by (2, num_edges * msg_size * max_config_size). The
            0th axis contains indices corresponding to the configurations that involve a particular edge taking a particular
            value, and the 1st axis contains a segmentation mask of these values (i.e, all configuration indices corresponding to
            edge 0 value 0 will be labelled 0, all indices corresponding to edge 0 value 1 will be labelled 1 and so on.) Note that
            the length of the 1st axis will be the same as factor_configs
        num_val_configs: the total number of valid configurations for factors in the factor graph.

    Returns:
        Array of shape (num_edges, msg_size) corresponding to the updated f->v messages after normalization and clipping
    """

    # Update Strategy
    # Stage 1: Generate a "summary" array that has an entry for every valid configuration. The elements of
    #   this array are simply the sums of messages across each valid config.
    # Stage 2: Utilize the info from edge_vals_to_config_summary_indices to apply the scattering operation and generate
    #   a flat set of output messages.

    # Generate an array of shape bounded by (num_factors * num_configs_per_factor * size_per_config, msg_size) such that
    # the messages are ordered according to factor (so factor_to_surr_edges_indices[1,:] will contain the index of the
    # factor that each message neighbors)
    fac_neighboring_msgs = msgs_arr[1, factor_configs[0]]
    # Generate summary array for each factor and each config
    fac_config_indices = jnp.arange(fac_neighboring_msgs.shape[0])
    fac_config_summary_sum = jax.ops.segment_sum(
        fac_neighboring_msgs[fac_config_indices, factor_configs[1]],
        factor_configs[2],
        num_segments=num_val_configs,
    )

    # Update Step 2
    flat_msgs = msgs_arr[1].flatten()
    # Perform scattering in a flattened format
    updated_ftov_msgs_flat = (
        jnp.full(shape=(flat_msgs.shape[0],), fill_value=NEG_INF)
        .at[edge_vals_to_config_summary_indices[1]]
        .max(
            fac_config_summary_sum[edge_vals_to_config_summary_indices[0]]
            - flat_msgs[edge_vals_to_config_summary_indices[1]]
        )
    )
    # Reshape the messages back into their expected shape
    updated_ftov_msgs = jnp.reshape(
        updated_ftov_msgs_flat, (msgs_arr[1].shape[0], msgs_arr[1].shape[1])
    )

    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = updated_ftov_msgs - updated_ftov_msgs.max(
        axis=1, keepdims=True
    )
    clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, 1000)

    return clipped_updated_msgs


@jax.partial(jax.jit, static_argnames=("damping_factor"))
def damp_and_update_messages(
    updated_vtof_msgs: jnp.ndarray,
    updated_ftov_msgs: jnp.ndarray,
    original_msgs_arr: jnp.ndarray,
    damping_factor: float,
) -> jnp.ndarray:
    """
    updates messages using previous messages, new messages and damping factor

    Args:
        updated_vtof_msgs: Array shape is (num_edges, msg_size). This corresponds to the updated
            v->f messages after normalization and clipping
        updated_ftov_msgs: Array shape is (num_edges, msg_size). This corresponds to the updated
            f->v messages after normalization and clipping
        original_msgs_arr: Array shape is (2, num_edges, msg_size). This holds all the messages prior to updating.
            the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds
            to v-> f msgs.
        damping_factor (float): The damping factor to use when updating messages.

    Returns:
        updated_msgs_arr: Array shape is (2, num_edges, msg_size). This holds all the updated messages.
            The 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds
            to v-> f msgs.
    """
    updated_msgs_arr = jnp.zeros_like(original_msgs_arr)
    damped_vtof_msgs = (damping_factor * original_msgs_arr[1]) + (
        1 - damping_factor
    ) * updated_vtof_msgs
    damped_ftov_msgs = (damping_factor * original_msgs_arr[0]) + (
        1 - damping_factor
    ) * updated_ftov_msgs
    updated_msgs_arr = updated_msgs_arr.at[1].set(damped_vtof_msgs)
    updated_msgs_arr = updated_msgs_arr.at[0].set(damped_ftov_msgs)
    return updated_msgs_arr


@jax.jit
def compute_map_estimate_jax(
    msgs_arr: jnp.ndarray,
    evidence_arr: jnp.ndarray,
    edges_to_var_arr: jnp.ndarray,
) -> jnp.ndarray:
    """
    uses messages computed by message passing to derive the MAP estimate for every variable node

    Args:
        msgs_arr: Array shape is (2, num_edges, msg_size). This holds all the messages. the 0th index of the
            0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to v-> f msgs.
        evidence_arr: Array shape is shape (num_var_nodes, msg_size). evidence_arr[x,:] corresponds to the evidence
            for a particular VariableNode x.
        edges_to_var_arr: Array len is num_edges. The ith entry is an integer representing which variable this edge is connected
            to.

    Returns:
        an array of size num_var_nodes where each index corresponds to the MAP state of a particular variable node
    """

    neighbor_and_evidence_sum = (
        jax.ops.segment_sum(
            msgs_arr[0], edges_to_var_arr, num_segments=evidence_arr.shape[0]
        )
        + evidence_arr
    )
    return neighbor_and_evidence_sum.argmax(1)
