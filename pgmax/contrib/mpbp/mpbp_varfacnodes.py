import itertools
from timeit import default_timer as timer
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import pgmax.contrib.interface.node_classes_with_factortypes as node_classes

from pgmax.contrib.mpbp.mpbp_varnode_fac_lowmem import (  # isort:skip
    compute_map_estimate_jax,
    convert_map_to_dict,
    damp_and_update_messages,
    pass_var_to_fac_messages_jnp,
)

NEG_INF = (
    -100000.0
)  # A large negative value to use as -inf for numerical stability reasons


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
        var_neighbors_arr,
        edges_to_var_arr,
        edge_to_fac_indices_arr,
        factor_to_type_confs_inds,
        factor_to_surr_edges_indices,
        factor_type_valid_confs_arr,
        var_to_indices_dict,
    ) = compile_jax_data_structures(fg, evidence)
    end_time = timer()
    print(f"Data structures compiled in: {end_time - start_time}s")

    # Convert all arrays to jnp.ndarrays for use in BP
    # (Comments on the right show cumulative memory usage as each of these lines execute)
    msgs_arr = jax.device_put(msgs_arr)  # 69 MiB
    evidence_arr = jax.device_put(evidence_arr)  # 85 MiB
    var_neighbors_arr = jax.device_put(var_neighbors_arr)  # 85 MiB
    edges_to_var_arr = jax.device_put(edges_to_var_arr)  # 85 MiB
    edge_to_fac_indices_arr = jax.device_put(edge_to_fac_indices_arr)  # 149 MiB
    factor_to_type_confs_inds = jax.device_put(factor_to_type_confs_inds)  # 149 MiB
    factor_to_surr_edges_indices = jax.device_put(
        factor_to_surr_edges_indices
    )  # 149 MiB
    factor_type_valid_confs_arr = jax.device_put(factor_type_valid_confs_arr)  # 149 MiB

    @jax.partial(jax.jit, static_argnames=("num_iters"))
    def run_mpbp_update_loop(
        msgs_arr,
        evidence_arr,
        var_neighbors_arr,
        edges_to_var_arr,
        edge_to_fac_indices_arr,
        factor_to_type_confs_inds,
        factor_to_surr_edges_indices,
        factor_type_valid_confs_arr,
        num_iters,
    ):
        "Function wrapper that leverages jax.lax.scan to efficiently perform BP"

        def mpbp_update_step(msgs_arr, x):
            # Variable to Factor messages update
            updated_vtof_msgs = pass_var_to_fac_messages_jnp(
                msgs_arr,
                evidence_arr,
                var_neighbors_arr,
                edges_to_var_arr,
            )
            # Factor to Variable messages update
            updated_ftov_msgs = pass_fac_to_var_messages_jnp(
                msgs_arr,
                edge_to_fac_indices_arr,
                factor_to_type_confs_inds,
                factor_to_surr_edges_indices,
                factor_type_valid_confs_arr,
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
        var_neighbors_arr,
        edges_to_var_arr,
        edge_to_fac_indices_arr,
        factor_to_type_confs_inds,
        factor_to_surr_edges_indices,
        factor_type_valid_confs_arr,
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
        var_neighbors_arr,
        edges_to_var_arr,
        edge_to_fac_indices_arr,
        factor_to_type_confs_inds,
        factor_to_surr_edges_indices,
        factor_type_valid_confs_arr,
        num_iters,
    ).block_until_ready()
    msg_update_end_time = timer()
    print(
        f"Second Time Message Passing completed in: {msg_update_end_time - msg_update_start_time}s"
    )

    map_start_time = timer()
    map_arr = compute_map_estimate_jax(msgs_arr, evidence_arr, var_neighbors_arr)
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
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[node_classes.VariableNode, int],
]:
    """Creates data-structures that can be efficiently used with JAX for MPBP.

    Args:
        fg: A FactorGraph object upon which to do belief propagation
        evidence: Each entry represents the constant, evidence message that's passed to the corresponding
            VariableNode that acts as the key

    Returns:
        tuple containing data structures useful for message passing updates in JAX:
            msgs_arr: Array shape is (2, num_edges + 1, msg_size). This holds all the messages. the 0th index
                of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to v-> f
                msgs. The last row is just an extra row of 0's that represents a "null message" which will never
                be updated.
            evidence_arr: Array shape is shape (num_var_nodes, msg_size). evidence_arr[x,:] corresponds to the evidence
                for the variable node at var_neighbors_arr[x,:,:]
            var_neighbors_arr: Array shape is (num_variables x max_num_var_neighbors). var_neighbors_arr[i,:] represents
                all the indices into msgs_arr[0,:,:] that correspond to neighboring f->v messages
            edges_to_var_arr: Array len is num_edges. The ith entry is an integer corresponding to the index into
                var_node_neighboring_indices that represents the variable connected to this edge
            edge_to_fac_indices_arr: Array shape is (num_edges x 2). edge_to_fac_indices_arr[x,0] is an index into
                the 0th axis of factor_to_type_conf corresponding to the factor that edge x is connected to.
                edge_to_fac_indices_arr[x,1] is an index into the 2nd axis of factor_type_valid_confs_arr to
                get the column within the configurations that corresponds to e.
            factor_to_type_confs_inds: Array len is num_factors. factor_to_type_confs_inds[f] is an index into the 0th axis of
                factor_type_valid_confs_arr that yields the valid configurations for this factor
            factor_to_surr_edges_indices: Array shape is (num_factors x max_num_fac_neighbors). factor_to_surr_edges_indices[f,:]
                provides indices into the 0th axis of msgs_arr[0,:,:] corresponding to all messages surrounding factor f
            factor_type_valid_confs_arr: Array shape is (num_fac_types x max_num_valid_configs x max_num_fac_neighbors)
                factor_type_valid_confs_arr[f,:,:] contains all the valid configurations for any factor of factor_type f
                In order to make this a regularly-sized array, we pad the max_num_fac_neighbors dimension with
                -1s, and the max_num_valid_configs dimension with a repetition of the last row of valid configs (so that
                there will be multiple copies of the same configuration, which won't affect the max operation in max-product
                belief propagation).
            var_to_indices_dict: for a particular var_node key, var_to_indices_dict[var_node]
                contains the row index into var_neighbors_arr that corresponds to var_node
    """
    # NOTE: This currently assumes all variable nodes have the same size. Thus, all messages have the same size

    num_edges = fg.count_num_edges()
    msg_size = fg.variable_nodes[0].num_states
    # Initialize np arrays to hold messages and evidence. We will convert these to
    # jnp arrays later
    msgs_arr = np.zeros((2, num_edges + 1, msg_size))

    # The below loop does the following:
    # - Makes a mapping from (fac_node,var_node) and (var_node, fac_node) as keys to
    #   indices in the msgs_arr by looping thru all edges (fac_to_var_msg_to_index_dict)
    # - Populates the evidence
    # - Makes a list that contains an entry for every variable node corresponding to all
    #   indices in msgs_arr that will correspond to this node's neighbors (var_node_neighboring_indices)
    # - Makes an array of len num_edges, where the ith entry is an integer corresponding to the index into
    #   var_node_neighboring_indices that represents the variable connected to this edge
    # - Populates edges_to_fac_arr
    # - Makes a dict that goes from a VariableNode object to the index in the list at which its
    #   neighbors are contained.
    fac_to_var_msg_to_index_dict = {}
    evidence_arr = np.zeros((len(fg.variable_nodes), msg_size))
    var_neighbors_list = [None for _ in range(len(fg.variable_nodes))]
    edges_to_var_arr = np.zeros(num_edges, dtype=int)
    edges_to_fac_arr = np.zeros((num_edges, 3), dtype=int)
    var_to_indices_dict = {}
    tmp_fac_to_index_dict: Dict[node_classes.FactorNode, int] = {}
    edge_counter = 0
    fac_index = 0
    for var_index, var_node in enumerate(fg.variable_nodes):
        var_node_neighboring_indices = []
        for fac_node_neighbor in var_node.neighbors:
            fac_to_var_msg_to_index_dict[(fac_node_neighbor, var_node)] = edge_counter
            var_node_neighboring_indices.append(edge_counter)
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
            edge_counter += 1
        var_neighbors_list[var_index] = var_node_neighboring_indices  # type: ignore
        evidence_arr[var_index, :] = evidence[var_node]
        var_to_indices_dict[var_node] = var_index

    # Convert the neighbors lists and neighbor vars valid configs into regularly-shaped arrays
    var_neighbors_arr = np.array(
        list(itertools.zip_longest(*var_neighbors_list, fillvalue=-1))  # type: ignore
    ).T

    # Define some useful constants
    num_fac_types = fg.count_num_factor_types()
    max_num_valid_configs = fg.find_max_num_valid_configs()
    max_num_fac_neighbors = fg.find_max_num_factor_neighbors()
    num_factors = fg.count_num_factor_nodes()

    # Construct and populate factor_type_valid_confs_arr
    factor_type_valid_confs_arr = (
        np.ones(
            (num_fac_types, max_num_valid_configs, max_num_fac_neighbors), dtype=int
        )
        * -1
    )
    tmp_factype_to_index_dict = {}
    for fti in range(num_fac_types):
        fac_type = fg.factor_types[fti]
        tmp_factype_to_index_dict[fac_type] = fti
        (
            num_config_rows,
            num_config_cols,
        ) = fac_type.neighbor_configs_arr.shape
        pad_arr = np.tile(
            fac_type.neighbor_configs_arr[-1],
            max_num_valid_configs - num_config_rows,
        ).reshape(-1, num_config_cols)
        padded_valid_configs = np.vstack([fac_type.neighbor_configs_arr, pad_arr])

        factor_type_valid_confs_arr[fti, :, :num_config_cols] = padded_valid_configs

    factor_to_type_confs_inds = np.zeros(num_factors, dtype=int)
    factor_to_surr_edges_indices = (
        np.ones((num_factors, max_num_fac_neighbors), dtype=int) * -1
    )
    edge_to_fac_indices_arr = np.zeros((num_edges, 2), dtype=int)
    for fac_node, fac_index in tmp_fac_to_index_dict.items():
        factor_to_type_confs_inds[fac_index] = tmp_factype_to_index_dict[
            fac_node.factor_type
        ]
        surr_edge_indices = []
        for var_node in fac_node.neighbors:
            edge_idx = fac_to_var_msg_to_index_dict[(fac_node, var_node)]
            surr_edge_indices.append(edge_idx)
            edge_to_fac_indices_arr[edge_idx, 0] = fac_index
            edge_to_fac_indices_arr[edge_idx, 1] = fac_node.neighbor_to_index_mapping[
                var_node
            ]
        surr_edges_arr = np.array(surr_edge_indices, dtype=int)
        factor_to_surr_edges_indices[
            fac_index, : surr_edges_arr.shape[0]
        ] = surr_edges_arr

    return (
        msgs_arr,
        evidence_arr,
        var_neighbors_arr,
        edges_to_var_arr,
        edge_to_fac_indices_arr,
        factor_to_type_confs_inds,
        factor_to_surr_edges_indices,
        factor_type_valid_confs_arr,
        var_to_indices_dict,
    )


@jax.jit
def pass_fac_to_var_messages_jnp(
    msgs_arr: jnp.ndarray,
    edge_to_fac_indices_arr: jnp.ndarray,
    factor_to_type_confs_inds: jnp.ndarray,
    factor_to_surr_edges_indices: jnp.ndarray,
    factor_type_valid_confs_arr: jnp.ndarray,
) -> jnp.ndarray:

    """
    passes messages from FactorNodes to VariableNodes and computes a new, updated set of messages using JAX

    Args:
        msgs_arr: Array shape is (2, num_edges + 1, msg_size). This holds all the messages. the 0th index
            of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to v-> f
            msgs. The last row is just an extra row of 0's that represents a "null message" which will never
            be updated.
        edge_to_fac_indices_arr: Array shape is (num_edges x 2). edge_to_fac_indices_arr[x,0] is an index into
            the 0th axis of factor_to_type_conf corresponding to the factor that edge x is connected to.
            edge_to_fac_indices_arr[x,1] is an index into the 2nd axis of factor_type_valid_confs_arr to
            get the column within the configurations that corresponds to e.
        factor_to_type_confs_inds: Array shape is (num_factors,). factor_to_type_confs_inds[f] is an index into the 0th axis of
            factor_type_valid_confs_arr that yields the valid configurations for this factor
        factor_to_surr_edges_indices: Array shape is (num_factors x max_num_fac_neighbors). factor_to_surr_edges_indices[f,:]
            provides indices into the 0th axis of msgs_arr[0,:,:] corresponding to all messages surrounding factor f
        factor_type_valid_confs_arr: Array shape is (num_fac_types x max_num_valid_configs x max_num_fac_neighbors)
            factor_type_valid_confs_arr[f,:,:] contains all the valid configurations for any factor of factor_type f
            In order to make this a regularly-sized array, we pad the max_num_fac_neighbors dimension with
            -1s, and the max_num_valid_configs dimension with a repetition of the last row of valid configs (so that
            there will be multiple copies of the same configuration, which won't affect the max operation in max-product
            belief propagation).

    Returns:
        Array of shape (num_edges, msg_size) corresponding to the updated f->v messages after normalization and clipping
    """

    # Update Strategy:
    # 1. For each factor, we use factor_to_type_confs_inds, factor_to_surr_edges_indices, and factor_type_valid_confs_arr
    # to generate a new array of shape (num_factors x max_num_configs) that simply contains one number for each valid config
    # that is just the sum across each valid config.
    # 2. We generate an array of shape (num_edges x max_num_configs) that contains one number for each valid config that is
    # the message value of the variable node involved in the edge for that particular config

    _, num_edges, msg_size = msgs_arr.shape
    num_edges -= 1  # account for the extra null message row

    # Create convenience arrays to index all edges and all factors
    edge_indices = jnp.arange(num_edges)

    # Update Step 1
    factor_config_sums = jnp.sum(
        msgs_arr[
            1,
            factor_to_surr_edges_indices[:, None],
            factor_type_valid_confs_arr[factor_to_type_confs_inds],
        ],
        axis=2,
    )

    # Update Step 2
    valid_confs_curr_edge = factor_type_valid_confs_arr[factor_to_type_confs_inds][
        edge_to_fac_indices_arr[:, 0], :, edge_to_fac_indices_arr[:, 1]
    ]
    curr_edge_conf_value_arr = msgs_arr[1, edge_indices[:, None], valid_confs_curr_edge]

    updated_ftov_msgs = (
        jnp.full(shape=(num_edges, msg_size), fill_value=NEG_INF)
        .at[edge_indices[:, None], valid_confs_curr_edge]
        .max(
            factor_config_sums[edge_to_fac_indices_arr[:, 0]] - curr_edge_conf_value_arr
        )
    )

    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = updated_ftov_msgs - updated_ftov_msgs[:, [0]]
    clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, 1000)

    return clipped_updated_msgs
