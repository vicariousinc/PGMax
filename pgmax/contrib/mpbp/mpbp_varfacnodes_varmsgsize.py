from timeit import default_timer as timer
from typing import Dict, Tuple

import jax
import numpy as np

import pgmax.contrib.interface.node_classes_with_factortypes as node_classes

from pgmax.contrib.mpbp.mpbp_varfacnodes_unpadded import (  # isort:skip
    run_mp_belief_prop_and_compute_map,
    pass_var_to_fac_messages_jnp,
    pass_fac_to_var_messages_jnp,
    damp_and_update_messages,
    compute_map_estimate_jax,
    convert_map_to_dict,
)

NEG_INF = (
    -100000.0
)  # A large negative value to use as -inf for numerical stability reasons


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
            evidence_arr: Array shape is shape (num_var_nodes, msg_size). evidence_arr[x,:] corresponds to the evidence
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
    edges_to_fac_arr = np.zeros((num_edges, 3), dtype=int)
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
            edges_to_fac_arr[edge_counter, 0] = tmp_fac_to_index_dict[fac_node_neighbor]
            edges_to_fac_arr[
                edge_counter, 1
            ] = fac_node_neighbor.neighbor_to_index_mapping[var_node]
            edges_to_fac_arr[edge_counter, 2] = fg.factor_type_to_index_dict[
                fac_node_neighbor.factor_type
            ]
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
