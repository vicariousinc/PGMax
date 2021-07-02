from timeit import default_timer as timer
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import pgmax.contrib.interface.node_classes_with_factortypes as node_classes
import pgmax.contrib.mpbp.utils as utils

NEG_INF = (
    -100000.0
)  # A large negative value to use as -inf for numerical stability reasons

NEG_INF_INT = -10000


def debug(fg: node_classes.FactorGraph,
    evidence: Dict[node_classes.VariableNode, np.ndarray],
    num_iters: int,
    damping_factor: float,
) -> Dict[node_classes.VariableNode, int]:

    (
        msgs_arr,
        evidence_arr,
        msg_vals_to_var_arr,
        factor_configs,
        factor_configs_segs_lens,
        edge_vals_to_config_summary_indices,
        normalization_indices_arr,
        var_to_indices_dict,
        num_val_configs,
    ) = compile_jax_data_structures(fg, evidence)

    max_config_segs_len = int(np.max(factor_configs_segs_lens))
    # Convert all arrays to jnp.ndarrays for use in BP
    # (Comments on the right show cumulative memory usage as each of these lines execute)
    msgs_arr = jax.device_put(msgs_arr)  # 69 MiB
    evidence_arr = jax.device_put(evidence_arr)  # 85 MiB
    msg_vals_to_var_arr = jax.device_put(msg_vals_to_var_arr)
    factor_configs = jax.device_put(factor_configs)
    factor_configs_segs_lens = jax.device_put(factor_configs_segs_lens)
    edge_vals_to_config_summary_indices = jax.device_put(
        edge_vals_to_config_summary_indices
    )
    normalization_indices_arr = jax.device_put(normalization_indices_arr)

    with jax.disable_jit():
        for i in range(num_iters):
            # Variable to Factor messages update
            updated_vtof_msgs = pass_var_to_fac_messages_jnp(
                msgs_arr, evidence_arr, msg_vals_to_var_arr, normalization_indices_arr
            )
            # Factor to Variable messages update
            updated_ftov_msgs = pass_fac_to_var_messages_jnp(
                msgs_arr,
                factor_configs,
                factor_configs_segs_lens,
                edge_vals_to_config_summary_indices,
                normalization_indices_arr,
                max_config_segs_len,
            )
            # Damping before final message update
            msgs_arr = damp_and_update_messages(
                updated_vtof_msgs, updated_ftov_msgs, msgs_arr, damping_factor
            )
    
    map_arr = compute_map_estimate_jax(msgs_arr, evidence_arr, msg_vals_to_var_arr)
    var_map_estimate = convert_map_to_dict(map_arr, var_to_indices_dict)

    return var_map_estimate


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
        msg_vals_to_var_arr,
        factor_configs,
        factor_configs_segs_lens,
        edge_vals_to_config_summary_indices,
        normalization_indices_arr,
        var_to_indices_dict,
        num_val_configs,
    ) = compile_jax_data_structures(fg, evidence)
    end_time = timer()
    print(f"Data structures compiled in: {end_time - start_time}s")

    max_config_segs_len = int(np.max(factor_configs_segs_lens))
    # Convert all arrays to jnp.ndarrays for use in BP
    # (Comments on the right show cumulative memory usage as each of these lines execute)
    msgs_arr = jax.device_put(msgs_arr)  # 69 MiB
    evidence_arr = jax.device_put(evidence_arr)  # 85 MiB
    msg_vals_to_var_arr = jax.device_put(msg_vals_to_var_arr)
    factor_configs = jax.device_put(factor_configs)
    factor_configs_segs_lens = jax.device_put(factor_configs_segs_lens)
    edge_vals_to_config_summary_indices = jax.device_put(
        edge_vals_to_config_summary_indices
    )
    normalization_indices_arr = jax.device_put(normalization_indices_arr)
    

    @jax.partial(jax.jit, static_argnames=("max_config_segs_len", "num_iters"))
    def run_mpbp_update_loop(
        msgs_arr,
        evidence_arr,
        msg_vals_to_var_arr,
        factor_configs,
        factor_configs_segs_lens,
        edge_vals_to_config_summary_indices,
        normalization_indices_arr,
        max_config_segs_len,
        num_iters,
    ):
        "Function wrapper that leverages jax.lax.scan to efficiently perform BP"

        def mpbp_update_step(msgs_arr, x):
            # Variable to Factor messages update
            updated_vtof_msgs = pass_var_to_fac_messages_jnp(
                msgs_arr, evidence_arr, msg_vals_to_var_arr, normalization_indices_arr
            )
            # Factor to Variable messages update
            updated_ftov_msgs = pass_fac_to_var_messages_jnp(
                msgs_arr,
                factor_configs,
                factor_configs_segs_lens,
                edge_vals_to_config_summary_indices,
                normalization_indices_arr,
                max_config_segs_len,
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
        msg_vals_to_var_arr,
        factor_configs,
        factor_configs_segs_lens,
        edge_vals_to_config_summary_indices,
        normalization_indices_arr,
        max_config_segs_len,
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
        msg_vals_to_var_arr,
        factor_configs,
        factor_configs_segs_lens,
        edge_vals_to_config_summary_indices,
        normalization_indices_arr,
        max_config_segs_len,
        num_iters,
    ).block_until_ready()
    msg_update_end_time = timer()
    print(
        f"Second Time Message Passing completed in: {msg_update_end_time - msg_update_start_time}s"
    )

    map_start_time = timer()
    map_arr = compute_map_estimate_jax(msgs_arr, evidence_arr, msg_vals_to_var_arr)
    map_end_time = timer()
    print(f"MAP inference took {map_end_time - map_start_time}s")

    map_conversion_start = timer()
    var_map_estimate = convert_map_to_dict(map_arr, var_to_indices_dict)
    map_conversion_end = timer()
    print(f"MAP conversion to dict took {map_conversion_end - map_conversion_start}")

    return var_map_estimate


def compile_jax_data_structures(
    fg: node_classes.FactorGraph, evidence: Dict[node_classes.VariableNode, np.ndarray]
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[node_classes.VariableNode, List[int]],
    int,
]:
    """Creates data-structures that can be efficiently used with JAX for MPBP.

    Args:
        fg: A FactorGraph object upon which to do belief propagation
        evidence: Each entry represents the constant, evidence message that's passed to the corresponding
            VariableNode that acts as the key

    Returns:
        tuple containing data structures useful for message passing updates in JAX:
            msgs_arr: Maximum array shape is bounded (2, num_edges * max_msg_size). This holds all the flattened messages.
                the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to
                v->f msgs. To make this a regularly-shaped array, messages are padded with a large negative value
            evidence_arr: Maximum array shape is bounded by (num_var_nodes * max_msg_size). This array contains the fully-flattened
                set of evidence messages for each variable node
            msg_vals_to_var_arr: Maximum array shape is bounded by (num_edges * max_msg_size,). This array maps messages that are
                contained in msgs_arr into a shape that is compatible with evidence_arr. So, for a particular entry in msgs_arr
                (i.e msgs_arr[0,i]), msg_vals_to_var_arr[i] provides an index into evidence_arr such that
                evidence_arr[msg_vals_to_var_arr[i]] is the evidence value that needs to be added to msgs_arr[0,i] to perform the
                variable to factor update
            factor_configs: Maximum array shape is bounded by (num_factors * max_num_configs * max_config_size,). This array just
                contains a flat list of valid configuration indices such that msgs_arr[1, factor_configs[0]] gives a flattened array of
                all the message values from the valid configurations.
            factor_configs_segs_lens: Maximum array shape is bounded by (num_factors * max_num_configs * max_config_size,). This array
                simply provides one number for each segment in factor_configs that represents how long the segment is. Thus, each index simply
                provides the length of each configuration (i.e, index 0 specifies the length of the 0th config, which will be represented in the
                first few elements of factor_configs)
            edge_vals_to_config_summary_indices: Maximum array shape is bounded by (2, num_edges * msg_size * max_config_size). The
                0th axis contains indices corresponding to the configurations that involve a particular edge taking a particular
                value, and the 1st axis contains a segmentation mask of these values (i.e, all configuration indices corresponding to
                edge 0 value 0 will be labelled 0, all indices corresponding to edge 0 value 1 will be labelled 1 and so on.) Note that
                the length of the 1st axis will be the same as factor_configs
            normalization_indices_arr: Maximum array shape is bounded by (num_edges * max_msg_size,). For every entry along the 1st axis of
                msgs_arr, this array contains the index of the entry this must be subtracted by to produce message normalization.
            var_to_indices_dict: for a particular var_node key, var_to_indices_dict[var_node] will yield the indices in evidence_arr
                that correspond to messages surrounding var_node
            num_val_configs: the total number of valid configurations for factors in the factor graph.
    """
    num_edges = fg.count_num_edges()
    max_msg_size = fg.find_max_msg_size()

    # This below loop constructs the following data structures that are returned:
    # - msgs_arr
    # - evidence_arr
    # - msg_vals_to_var_arr
    # - normalization_indices_arr

    # maps a (fac_node, var_node) tuple to a list of ints representing the indices at which
    # the message vals for this tuple can be found
    fac_to_var_msg_to_indices_dict = {}

    # Initialize all entries in the evidence array to be NEG_INF. Later, we will
    # delete all these values that have not been changed from NEG_INF
    evidence_arr = np.ones((len(fg.variable_nodes) * max_msg_size)) * NEG_INF_INT
    msg_vals_to_var_arr = np.ones(num_edges * max_msg_size, dtype=int) * NEG_INF_INT
    msgs_arr = np.zeros((2, num_edges * max_msg_size))
    normalization_indices_arr = (
        np.ones(num_edges * max_msg_size, dtype=int) * NEG_INF_INT
    )
    var_to_indices_dict = {}
    tmp_fac_to_index_dict: Dict[node_classes.FactorNode, int] = {}
    edge_msg_counter = 0
    fac_index = 0
    var_evidence_msg_index = 0
    for var_node in fg.variable_nodes:
        # var_evidence_msg_index keeps track of until what point evidence_arr has already been filled.
        evidence_arr[
            var_evidence_msg_index : var_evidence_msg_index + var_node.num_states
        ] = evidence[var_node]
        evidence_indices = np.arange(
            var_evidence_msg_index, var_evidence_msg_index + var_node.num_states
        )
        # Populate msg_vals_to_var_arr so as to be sorted
        msg_vals_to_var_arr[
            edge_msg_counter : edge_msg_counter + (var_node.num_states * var_node.num_neighbors())
        ] = np.repeat(evidence_indices, var_node.num_neighbors())
        
        # Populate the normalization_indices_arr to point to the index of the first entry for every message
        normalization_indices_arr[
            edge_msg_counter : edge_msg_counter + (var_node.num_states * var_node.num_neighbors())
        ] = np.tile(np.arange(edge_msg_counter, edge_msg_counter + var_node.num_neighbors()), var_node.num_states)

        for neighbor_num, fac_node_neighbor in enumerate(var_node.neighbors):
            fac_to_var_msg_to_indices_dict[(fac_node_neighbor, var_node)] = list(
                range(edge_msg_counter + neighbor_num, edge_msg_counter + (var_node.num_states * var_node.num_neighbors()), var_node.num_neighbors())
            )
            if tmp_fac_to_index_dict.get(fac_node_neighbor) is None:
                tmp_fac_to_index_dict[fac_node_neighbor] = fac_index
                fac_index += 1
            
        # Populate var_to_indices_dict with the fact that all the messages of evidence corresponding to var_node will be found
        # at the evidence_indices
        var_to_indices_dict[var_node] = evidence_indices
        var_evidence_msg_index += var_node.num_states
        edge_msg_counter += (var_node.num_states * var_node.num_neighbors()) # This keeps track of how much msg_vals_to_var_arr has been filled

    # Finally, delete all the unnecessary rows
    msgs_arr = np.delete(msgs_arr, np.s_[edge_msg_counter::], axis=1)
    evidence_arr = np.delete(evidence_arr, np.s_[var_evidence_msg_index::])
    msg_vals_to_var_arr = np.delete(msg_vals_to_var_arr, np.s_[edge_msg_counter::])

    # Create and populate remaining data structures
    # Start by defining some useful constants that will be necessary to construct the arrays
    # we're interested in
    num_factors = fg.count_num_factor_nodes()
    max_num_configs = fg.find_max_num_valid_configs()
    max_config_size = fg.find_max_valid_config_size()

    # Initialize these arrays to the maximum possible size they could be. Make each of the values NEG_INF so
    # unallocated rows can be recognized and deleted later
    factor_configs = (
        np.ones(num_factors * max_num_configs * max_config_size, dtype=int)
        * NEG_INF_INT
    )
    factor_configs_segs_lens = (
        np.ones(num_factors * max_num_configs * max_config_size, dtype=int)
        * NEG_INF_INT
    )
    edge_vals_to_config_summary_indices = (
        np.ones((2, num_edges * max_msg_size * max_num_configs), dtype=int)
        * NEG_INF_INT
    )

    configs_insertion_index = 0
    configs_counter = 0
    flat_edge_configs_counter = 0
    segs_insertion_index = 0
    for fac_node, fac_index in sorted(
        tmp_fac_to_index_dict.items(), key=lambda item: item[1]
    ):
        var_configs = fac_node.factor_type.neighbor_configs_arr
        num_configs, len_of_each_config = var_configs.shape
        config_indices_arr = np.arange(configs_counter, configs_counter + num_configs)
        segs_to_insert = np.repeat(len_of_each_config, num_configs)
        configs_counter += num_configs

        neighboring_msgs_idxs = []
        # Loop thru all neighboring variables to update edge_vals_to_config_summary_indices
        for var_node in fac_node.neighbors:
            neighboring_msgs_idxs.append(
                fac_to_var_msg_to_indices_dict[(fac_node, var_node)]
            )
            config_column_index = fac_node.neighbor_to_index_mapping[var_node]
            for msg_idx in range(var_node.num_states):
                flattened_msg_idx_val = fac_to_var_msg_to_indices_dict[
                    (fac_node, var_node)
                ][msg_idx]
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

        # Update factor_configs
        # First, get all the indices of the neighboring msgs and pad them with -1s
        padded_neighbor_indices_arr = utils.pad(neighboring_msgs_idxs, -1)
        # Next, use the configs to index into this above array to get all the message indices
        # for the valid configs
        configs_to_insert = np.take_along_axis(
            padded_neighbor_indices_arr, var_configs.T, axis=1
        ).T.flatten()
        # Finally, insert the flattened array of these valid config message indices into factor_configs
        factor_configs[
            configs_insertion_index : configs_insertion_index
            + configs_to_insert.shape[0],
        ] = configs_to_insert
        factor_configs_segs_lens[segs_insertion_index: segs_insertion_index + segs_to_insert.shape[0]] = segs_to_insert

        configs_insertion_index += configs_to_insert.shape[0]
        segs_insertion_index += segs_to_insert.shape[0]

    # Finally, delete all misc rows that have not been changed from their initial value!
    factor_configs = np.delete(
        factor_configs, np.where(factor_configs == NEG_INF_INT)[0])
    factor_configs_segs_lens = np.delete(factor_configs_segs_lens, np.where(factor_configs_segs_lens == NEG_INF_INT)[0])
    edge_vals_to_config_summary_indices = np.delete(
        edge_vals_to_config_summary_indices,
        np.where(edge_vals_to_config_summary_indices[1, :] == NEG_INF_INT)[0],
        axis=1,
    )

    # Grab the final necessary parameter from the factor graph
    num_val_configs = fg.count_total_num_valid_configs()

    return (
        msgs_arr,
        evidence_arr,
        msg_vals_to_var_arr,
        factor_configs,
        factor_configs_segs_lens,
        edge_vals_to_config_summary_indices,
        normalization_indices_arr,
        var_to_indices_dict,
        num_val_configs,
    )


@jax.jit
def pass_var_to_fac_messages_jnp(
    msgs_arr: jnp.array,
    evidence_arr: jnp.array,
    msg_vals_to_var_arr: jnp.array,
    normalization_indices_arr: jnp.array,
) -> jnp.array:
    """
    passes messages from VariableNodes to FactorNodes and computes a new updated set of messages using JAX

    Args:
        msgs_arr: Maximum array shape is bounded (2, num_edges * max_msg_size). This holds all the flattened messages.
            the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to
            v->f msgs. To make this a regularly-shaped array, messages are padded with a large negative value
        evidence_arr: Maximum array shape is bounded by (num_var_nodes * max_msg_size). This array contains the fully-flattened
            set of evidence messages for each variable node
        msg_vals_to_var_arr: Maximum array shape is bounded by (num_edges * max_msg_size,). This array maps messages that are
            contained in msgs_arr into a shape that is compatible with evidence_arr. So, for a particular entry in msgs_arr
            (i.e msgs_arr[0,i]), msg_vals_to_var_arr[i] provides an index into evidence_arr such that
            evidence_arr[msg_vals_to_var_arr[i]] is the evidence value that needs to be added to msgs_arr[0,i] to perform the
            variable to factor update
        normalization_indices_arr: Maximum array shape is bounded by (num_edges * max_msg_size,). For every entry along the 1st axis of
            msgs_arr, this array contains the index of the entry this must be subtracted by to produce message normalization.
    Returns:
        Array of shape (num_edges, msg_size) corresponding to the updated v->f messages after normalization and clipping
    """
    # For each variable, sum the neighboring factor to variable messages and the evidence.
    var_sums_arr = (
        jax.ops.segment_sum(
            msgs_arr[0], msg_vals_to_var_arr, num_segments=evidence_arr.shape[0]
        )
        + evidence_arr
    )
    updated_vtof_msgs = var_sums_arr[msg_vals_to_var_arr] - msgs_arr[0]

    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = (
        updated_vtof_msgs - updated_vtof_msgs[normalization_indices_arr]
    )
    clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, 1000)

    return clipped_updated_msgs

# @jax.partial(jax.jit, static_argnames=("max_msg_count"))
# def pass_var_to_fac_messages_jnp_segsum6(
#     msgs_arr: jnp.array,
#     evidence_arr: jnp.array,
#     msg_seg_counts_arr: jnp.array,
#     normalization_indices_arr: jnp.array,
#     max_msg_count: int
# ) -> jnp.array:
#     """
#     passes messages from VariableNodes to FactorNodes and computes a new updated set of messages using JAX

#     Args:
#         msgs_arr: Maximum array shape is bounded (2, num_edges * max_msg_size). This holds all the flattened messages.
#             the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to
#             v->f msgs. To make this a regularly-shaped array, messages are padded with a large negative value
#         evidence_arr: Maximum array shape is bounded by (num_var_nodes * max_msg_size). This array contains the fully-flattened
#             set of evidence messages for each variable node
#         msg_seg_counts_arr: Maximum array shape is bounded by (num_edges * max_msg_size,). This array contains integers representing the 
#             length of each segment of messages in msgs_arr for use in a segment sum operation
#         normalization_indices_arr: Maximum array shape is bounded by (num_edges * max_msg_size,). For every entry along the 1st axis of
#             msgs_arr, this array contains the index of the entry this must be subtracted by to produce message normalization.
#         max_msg_count: the max of msg_seg_counts_arr
#     Returns:
#         Array of shape (num_edges, msg_size) corresponding to the updated v->f messages after normalization and clipping
#     """
#     # For each variable, sum the neighboring factor to variable messages and the evidence.
#     var_sums_arr = (
#         utils.segment_sum_opt(
#             msgs_arr[0], msg_seg_counts_arr, max_msg_count
#         )
#         + evidence_arr
#     )
#     updated_vtof_msgs = var_sums_arr[msg_vals_to_var_arr] - msgs_arr[0]

#     # Normalize and clip messages (between -1000 and 1000) before returning
#     normalized_updated_msgs = (
#         updated_vtof_msgs - updated_vtof_msgs[normalization_indices_arr]
#     )
#     clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, 1000)

#     return clipped_updated_msgs

@jax.partial(jax.jit, static_argnames=("max_config_segs_len"))
def pass_fac_to_var_messages_jnp(
    msgs_arr: jnp.ndarray,
    factor_configs: jnp.ndarray,
    factor_configs_segs_lens: jnp.ndarray,
    edge_vals_to_config_summary_indices: jnp.ndarray,
    normalization_indices_arr: jnp.ndarray,
    max_config_segs_len: int,
) -> jnp.ndarray:

    """
    passes messages from FactorNodes to VariableNodes and computes a new, updated set of messages using JAX

    Args:
        msgs_arr: Maximum array shape is bounded (2, num_edges * max_msg_size). This holds all the flattened messages.
            the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to
            v->f msgs. To make this a regularly-shaped array, messages are padded with a large negative value
        factor_configs: Maximum array shape is bounded by (num_factors * max_num_configs * max_config_size,). This array just
            contains a flat list of valid configuration indices such that msgs_arr[1, factor_configs[0]] gives a flattened array of
            all the message values from the valid configurations.
        factor_configs_segs_lens: Maximum array shape is bounded by (num_factors * max_num_configs * max_config_size,). This array
            simply provides one number for each segment in factor_configs that represents how long the segment is. Thus, each index simply
            provides the length of each configuration (i.e, index 0 specifies the length of the 0th config, which will be represented in the
            first few elements of factor_configs)
        edge_vals_to_config_summary_indices: Maximum array shape is bounded by (2, num_edges * msg_size * max_config_size). The
            0th axis contains indices corresponding to the configurations that involve a particular edge taking a particular
            value, and the 1st axis contains a segmentation mask of these values (i.e, all configuration indices corresponding to
            edge 0 value 0 will be labelled 0, all indices corresponding to edge 0 value 1 will be labelled 1 and so on.) Note that
            the length of the 1st axis will be the same as factor_configs
        normalization_indices_arr: Maximum array shape is bounded by (num_edges * max_msg_size,). For every entry along the 1st axis of
            msgs_arr, this array contains the index of the entry this must be subtracted by to produce message normalization.
        max_config_segs_len: the max of factor_configs_segs_len

    Returns:
        Array of shape (num_edges, msg_size) corresponding to the updated f->v messages after normalization and clipping
    """

    # Update Strategy
    # Stage 1: Generate a "summary" array that has an entry for every valid configuration. The elements of
    #   this array are simply the sums of messages across each valid config.
    # Stage 2: Utilize the info from edge_vals_to_config_summary_indices to apply the scattering operation and generate
    #   a flat set of output messages.

    # Generate summary array for each factor and each config
    fac_config_summary_sum = utils.segment_sum_opt(
        msgs_arr[1, factor_configs],
        factor_configs_segs_lens,
        max_config_segs_len,
    )

    # Update Step 2
    # Perform scattering in a flattened format
    updated_ftov_msgs = (
        jnp.full(shape=(msgs_arr[1].shape[0],), fill_value=NEG_INF)
        .at[edge_vals_to_config_summary_indices[1]]
        .max(fac_config_summary_sum[edge_vals_to_config_summary_indices[0]])
    ) - msgs_arr[1]

    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = (
        updated_ftov_msgs - updated_ftov_msgs[normalization_indices_arr]
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
        updated_vtof_msgs: Maximum array shape is bounded by (num_edges * max_msg_size). This corresponds to the updated
            v->f messages after normalization and clipping
        updated_ftov_msgs: Maximum array shape is bounded by (num_edges * max_msg_size). This corresponds to the updated
            f->v messages after normalization and clipping
        original_msgs_arr: Maximum array shape is bounded (2, num_edges * max_msg_size). This holds all the flattened messages.
            the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to
            v->f msgs. To make this a regularly-shaped array, messages are padded with a large negative value
        damping_factor (float): The damping factor to use when updating messages.

    Returns:
        updated_msgs_arr: Maximum array shape is bounded (2, num_edges * max_msg_size). This holds all the updated messages.
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
    msg_vals_to_var_arr: jnp.ndarray,
) -> jnp.ndarray:
    """
    uses messages computed by message passing to derive the MAP estimate for every variable node

    Args:
        msgs_arr: Maximum array shape is bounded (2, num_edges * max_msg_size). This holds all the flattened messages.
            the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to
            v->f msgs. To make this a regularly-shaped array, messages are padded with a large negative value
        evidence_arr: Maximum array shape is bounded by (num_var_nodes * max_msg_size). This array contains the fully-flattened
            set of evidence messages for each variable node
        msg_vals_to_var_arr: Maximum array shape is bounded by (num_edges * max_msg_size,). This array maps messages that are
            contained in msgs_arr into a shape that is compatible with evidence_arr. So, for a particular entry in msgs_arr
            (i.e msgs_arr[0,i]), msg_vals_to_var_arr[i] provides an index into evidence_arr such that
            evidence_arr[msg_vals_to_var_arr[i]] is the evidence value that needs to be added to msgs_arr[0,i] to perform the
            variable to factor update

    Returns:
        an array of the same shape as evidence_arr corresponding to the final message values of each VariableNode
    """

    var_sums_arr = (
        jax.ops.segment_sum(
            msgs_arr[0], msg_vals_to_var_arr, num_segments=evidence_arr.shape[0]
        )
        + evidence_arr
    )
    return var_sums_arr


def convert_map_to_dict(
    map_arr: jnp.ndarray,
    var_to_indices_dict: Dict[node_classes.VariableNode, List[int]],
) -> Dict[node_classes.VariableNode, int]:
    """converts the array after MAP inference to a dict format expected by the viz code

    Args:
        map_arr: an array of the same shape as evidence_arr corresponding to the final
            message values of each VariableNode
        var_to_indices_dict: for a particular var_node key, var_to_indices_dict[var_node] will
            yield the indices in evidence_arr that correspond to messages surrounding var_node
    Returns:
        a dict mapping each VariableNode and its MAP state
    """
    var_map_dict = {}

    map_np_arr = np.array(map_arr)

    for var_node in var_to_indices_dict.keys():
        var_map_dict[var_node] = np.argmax(map_np_arr[var_to_indices_dict[var_node]])

    return var_map_dict
