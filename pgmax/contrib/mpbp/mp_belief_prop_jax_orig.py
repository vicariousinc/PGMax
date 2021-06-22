import itertools
from timeit import default_timer as timer
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import pgmax.contrib.interface.node_classes as node_classes


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
        neighbors_vtof_arr,
        neighbors_ftov_arr,
        neighbor_vars_valid_configs_arr,
        var_neighbors_arr,
        var_to_indices_dict,
    ) = compile_jax_data_structures(fg, evidence)
    end_time = timer()
    print(f"Data structures compiled in: {end_time - start_time}s")

    # Convert all arrays to jnp.ndarrays for use in BP
    msgs_arr = jax.device_put(msgs_arr)
    evidence_arr = jax.device_put(evidence_arr)
    neighbors_vtof_arr = jax.device_put(neighbors_vtof_arr)
    neighbors_ftov_arr = jax.device_put(neighbors_ftov_arr)
    neighbor_vars_valid_configs_arr = jax.device_put(neighbor_vars_valid_configs_arr)
    var_neighbors_arr = jax.device_put(var_neighbors_arr)

    @jax.partial(jax.jit, static_argnames=("num_iters"))
    def run_mpbp_update_loop(
        msgs_arr,
        evidence_arr,
        neighbors_vtof_arr,
        neighbors_ftov_arr,
        neighbor_vars_valid_configs_arr,
        num_iters,
    ):
        "Function wrapper that leverages jax.lax.scan to efficiently perform BP"

        def mpbp_update_step(msgs_arr, x):
            # Variable to Factor messages update
            updated_vtof_msgs = pass_var_to_fac_messages_jnp(
                msgs_arr, evidence_arr, neighbors_ftov_arr
            )
            # Factor to Variable messages update
            updated_ftov_msgs = pass_fac_to_var_messages_jnp(
                msgs_arr, neighbor_vars_valid_configs_arr, neighbors_vtof_arr
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
    _ = run_mpbp_update_loop(
        msgs_arr,
        evidence_arr,
        neighbors_vtof_arr,
        neighbors_ftov_arr,
        neighbor_vars_valid_configs_arr,
        num_iters,
    ).block_until_ready()
    msg_comp_end_time = timer()
    print(
        f"First time Message Passing completed in: {msg_comp_end_time - msg_comp_start_time}s"
    )

    # Rerun the loop and time it!
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
        f"Second time Message Passing completed in: {msg_update_end_time - msg_update_start_time}s"
    )

    map_start_time = timer()
    map_arr = compute_map_estimate_jax(msgs_arr, evidence_arr, var_neighbors_arr)
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
            evidence_arr: Array shape is shape (num_edges, msg_size). evidence_arr[x,:] corresponds to the evidence
                needed to compute the message contained in msgs_arr[1,x,:]
            neighbors_vtof_arr: Array shape is (num_edges x max_num_fac_neighbors). neighbors_vtof_list[x,:] is an
                array of integers that represent the indices into the 1st axis of msgs_arr[1,:,:] that correspond to
                the messages needed to update the message for msgs_arr[0,x,:]. In order to make this a regularly-sized
                array, we pad each row with -1's to refer to the "null message".
            neighbors_ftov_arr: Array shape is (num_edges x max_num_var_neighbors). neighbors_ftov_list[x,:] is an
                array of integers that represent the indices into the 1st axis of msgs_arr[0,:,:] that correspond to the
                messages needed to update the message for msgs_arr[1,x,:]. In order to make this a regularly-sized array,
                we pad each row with -1's to refer to the "null message".
            neighbor_vars_valid_configs_arr: Array shape is (num_edges x msg_size x max_num_valid_configs x max_num_fac_neighbors))
                neighboring_vars_valid_configs[x,:,:] contains an array of arrays, such that the 0th array
                contains an array of valid states such that whatever variable corresponds to msgs_arr[0,x,:] is
                in state 0. In order to make this a regularly-sized array, we pad the max_num_fac_neighbors dimension with
                -1s, and the max_num_valid_configs dimension with a repetition of the last row of valid configs (so that
                there will be multiple copies of the same configuration)
            var_neighbors_arr: Array shape is (num_variables x max_num_var_neighbors). var_neighbors_arr[i,:] represent
                all the indices into msgs_arr[0,:,:] that correspond to neighboring f-> messages
            var_to_indices_dict: for a particular var_node key, var_to_indices_dict[var_node]
                contains the row index into var_neighbors_arr that corresponds to var_node
    """
    # NOTE: This currently assumes all variable nodes have the same size. Thus, all messages have the same size

    num_edges = fg.count_num_edges()
    msg_size = fg.variable_nodes[0].num_states
    # Initialize np arrays to hold messages and evidence. We will convert these to
    # jnp arrays later
    msgs_arr = np.zeros((2, num_edges + 1, msg_size))
    evidence_arr = np.zeros((num_edges, msg_size))

    # Make a mapping from (fac_node,var_node) and (var_node, fac_node) as keys to
    # indices in the msgs_arr by looping thru all edges. Simultaneously populate the evidence
    # Also make a list that contains an entry for every variable node corresponding to all indices in msgs_arr
    # that will correspond to this node's neighbors. Finally, make a dict that goes from a VariableNode object
    # to the index in the list at which its neighbors are contained.
    fac_to_var_msg_to_index_dict = {}
    var_neighbors_list = [None for _ in range(len(fg.variable_nodes))]
    var_to_indices_dict = {}
    edge_counter = 0
    for var_index, var_node in enumerate(fg.variable_nodes):
        var_node_neighboring_indices = []
        for fac_node_neighbor in var_node.neighbors:
            fac_to_var_msg_to_index_dict[(fac_node_neighbor, var_node)] = edge_counter
            evidence_arr[edge_counter, :] = evidence[var_node]
            var_node_neighboring_indices.append(edge_counter)
            edge_counter += 1
        var_neighbors_list[var_index] = var_node_neighboring_indices  # type: ignore
        var_to_indices_dict[var_node] = var_index

    # Convert the neighbors lists and neighbor vars valid configs into regularly-shaped arrays
    var_neighbors_arr = np.array(
        list(itertools.zip_longest(*var_neighbors_list, fillvalue=-1))  # type: ignore
    ).T

    # NOTE: The below loop to make the neighbors lists is EXTREMELY naive. It can likely be easily
    # optimized...

    # Loop thru all keys of the above-created dict. For each (fac_node, var_node) key, consider the
    # fac_node and add all indices along the 1st axis (since we already know the 0th axis value)
    # of msgs_arr of neighboring vars (except var_node) to a list to keep track of them. Do the same
    # for var_node.
    neighbors_vtof_list = [None for _ in range(num_edges)]
    neighbors_ftov_list = [None for _ in range(num_edges)]
    for k in fac_to_var_msg_to_index_dict.keys():
        index_to_insert_at = fac_to_var_msg_to_index_dict[k]
        curr_fac_node = k[0]
        curr_var_node = k[1]
        fac_neighbor_indices = []
        for vn in curr_fac_node.neighbors:
            if vn != curr_var_node:
                neighboring_index = fac_to_var_msg_to_index_dict[(curr_fac_node, vn)]
                fac_neighbor_indices.append(neighboring_index)
        neighbors_vtof_list[index_to_insert_at] = fac_neighbor_indices  # type: ignore
        var_neighbor_indices = []
        for fn in curr_var_node.neighbors:
            if fn != curr_fac_node:
                neighboring_index = fac_to_var_msg_to_index_dict[(fn, curr_var_node)]
                var_neighbor_indices.append(neighboring_index)
        neighbors_ftov_list[index_to_insert_at] = var_neighbor_indices  # type: ignore

    # Convert the neighbors lists and neighbor vars valid configs into regularly-shaped arrays
    neighbors_vtof_arr = np.array(
        list(itertools.zip_longest(*neighbors_vtof_list, fillvalue=-1))  # type: ignore
    ).T
    neighbors_ftov_arr = np.array(
        list(itertools.zip_longest(*neighbors_ftov_list, fillvalue=-1))  # type: ignore
    ).T

    # Get the maximum number of neighbors for any factor
    max_num_valid_configs = fg.find_max_num_valid_configs()
    max_num_fac_neighbors = neighbors_vtof_arr.shape[1]
    neighbor_vars_valid_configs_arr = (
        np.ones((num_edges, msg_size, max_num_valid_configs, max_num_fac_neighbors))
        * -1
    )

    for k in fac_to_var_msg_to_index_dict.keys():
        index_to_insert_at = fac_to_var_msg_to_index_dict[k]
        curr_fac_node = k[0]
        curr_var_node = k[1]
        # Populate the list of valid configurations for the edge curr_fac_node - curr_var_node
        # by looping thru all possible states curr_var_node might take
        curr_var_node_index = curr_fac_node.neighbor_to_index_mapping[curr_var_node]
        for var_state in range(msg_size):
            valid_configs = curr_fac_node.neighbor_config_list[
                curr_fac_node.neighbor_config_list[:, curr_var_node_index] == var_state
            ]
            # Now, get the valid configs and neighboring msgs such that the curr_var's index is excluded
            valid_configs_without_curr_var = np.delete(
                valid_configs, curr_var_node_index, axis=1
            )

            # We now need to pad this array to have max_num_valid_configs rows. To do this, we'll simply
            # copy the last row and append it to the array
            num_config_rows, num_config_cols = valid_configs_without_curr_var.shape
            pad_arr = np.tile(
                valid_configs_without_curr_var[-1],
                max_num_valid_configs - num_config_rows,
            ).reshape(-1, num_config_cols)
            padded_valid_configs_without_curr_var = np.vstack(
                [valid_configs_without_curr_var, pad_arr]
            )

            # Insert valid_configs_without_curr_var into the left of the correct location of
            # neighbor_vars_valid_configs_arr. All other vals are already -1 because of init.
            neighbor_vars_valid_configs_arr[
                index_to_insert_at, var_state, :, :num_config_cols
            ] = padded_valid_configs_without_curr_var

    # Make sure all the neighbor arrays are int types
    neighbors_vtof_arr = neighbors_vtof_arr.astype(int)
    neighbors_ftov_arr = neighbors_ftov_arr.astype(int)
    neighbor_vars_valid_configs_arr = neighbor_vars_valid_configs_arr.astype(int)

    return (
        msgs_arr,
        evidence_arr,
        neighbors_vtof_arr,
        neighbors_ftov_arr,
        neighbor_vars_valid_configs_arr,
        var_neighbors_arr,
        var_to_indices_dict,
    )


@jax.jit
def pass_var_to_fac_messages_jnp(
    msgs_arr: jnp.ndarray,
    evidence_arr: jnp.ndarray,
    neighbors_ftov_arr: jnp.ndarray,
) -> jnp.ndarray:
    """
    passes messages from VariableNodes to FactorNodes and computes a new updated set of messages using JAX

    Args:
        msgs_arr: Array shape is (2, num_edges + 1, msg_size). This holds all the messages. the 0th index
            of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to v-> f
            msgs. The last row is just an extra row of 0's that represents a "null message" which will never
            be updated.
        evidence_arr: Array shape is shape (num_edges, msg_size). evidence_arr[x,:] corresponds to the evidence
            needed to compute the message contained in msgs_arr[1,x,:]
        neighbors_ftov_arr: Array shape is (num_edges x max_num_var_neighbors). neighbors_ftov_list[x,:] is an
                array of integers that represent the indices into the 1st axis of msgs_arr[0,:,:] that correspond to the
                messages needed to update the message for msgs_arr[1,x,:]. In order to make this a regularly-sized array,
                we pad each row with -1's to refer to the "null message".
    Returns:
        Array of shape (num_edges, msg_size) corresponding to the updated v->f messages after normalization and clipping
    """
    _, num_edges, _ = msgs_arr.shape
    num_edges -= 1  # account for the extra null message row
    indices_arr = jnp.arange(num_edges)  # make an array to index into the correct rows

    # For every row, neighbors_ftov_arr contains the indices of the neighboring fac to var messages
    # Use indices_arr to obtain these indices, then index into msgs_arr[0] using these.
    # Finally, sum these neighboring messages and the evidence to get the final answer!
    expanded_evidence = evidence_arr[indices_arr, None]
    neighboring_ftov_msgs = msgs_arr[0, neighbors_ftov_arr[indices_arr, :], :]
    updated_vtof_msgs = jnp.sum(
        jnp.concatenate((expanded_evidence, neighboring_ftov_msgs), 1), axis=1
    )

    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = updated_vtof_msgs - updated_vtof_msgs[:, [0]]
    clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, 1000)

    return clipped_updated_msgs


@jax.jit
def pass_fac_to_var_messages_jnp(
    msgs_arr: jnp.ndarray,
    neighbor_vars_valid_configs_arr: jnp.ndarray,
    neighbors_vtof_arr: jnp.ndarray,
) -> jnp.ndarray:
    """
    passes messages from VariableNodes to FactorNodes and computes a new, updated set of messages using JAX

    Args:
        msgs_arr: Array shape is (2, num_edges + 1, msg_size). This holds all the messages. the 0th index
            of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to v-> f
            msgs. The last row is just an extra row of 0's that represents a "null message" which will never
            be updated.
        neighbor_vars_valid_configs_arr: Array shape is (num_edges x msg_size x max_num_valid_configs x max_num_fac_neighbors))
                neighboring_vars_valid_configs[x,:,:] contains an array of arrays, such that the 0th array
                contains an array of valid states such that whatever variable corresponds to msgs_arr[0,x,:] is
                in state 0. In order to make this a regularly-sized array, we pad the max_num_fac_neighbors dimension with
                -1s, and the max_num_valid_configs dimension with a repetition of the last row of valid configs (so that
                there will be multiple copies of the same configuration)
        neighbors_vtof_arr: Array shape is (num_edges x max_num_fac_neighbors). neighbors_vtof_list[x,:] is an
                array of integers that represent the indices into the 1st axis of msgs_arr[1,:,:] that correspond to
                the messages needed to update the message for msgs_arr[0,x,:]. In order to make this a regularly-sized
                array, we pad each row with -1's to refer to the "null message".

    Returns:
        Array of shape (num_edges, msg_size) corresponding to the updated f->v messages after normalization and clipping
    """
    _, num_edges, _ = msgs_arr.shape
    num_edges -= 1  # account for the extra null message row
    indices_arr = jnp.arange(num_edges)  # make an array to index into the correct rows

    neighboring_vtof_msgs = msgs_arr[1, neighbors_vtof_arr[indices_arr, :], :]

    # Expand dims of the vtof msgs and rearrange axes of neighbor_valid_configs
    # so the below take_along_axis operation goes thru
    neighboring_vtof_msgs_padded = neighboring_vtof_msgs[:, :, :, None]
    configs_for_var_state = jnp.swapaxes(
        neighbor_vars_valid_configs_arr[indices_arr, :, :, :], 1, 3
    )
    # Take the values corresponding to the right configs for each possible state, sum these and take
    # the max to get the value of each message at each possible state
    updated_ftov_msgs = (
        jnp.take_along_axis(
            neighboring_vtof_msgs_padded[indices_arr, :], configs_for_var_state, axis=2
        )
        .sum(1)
        .max(1)
    )

    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = updated_ftov_msgs - updated_ftov_msgs[:, [0]]
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
        original_msgs_arr: Array shape is (2, num_edges + 1, msg_size). This holds all the messages prior to updating.
            the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds
            to v-> f msgs. The last row is just an extra row of 0's that represents a "null message" which will never
            be updated.
        damping_factor (float): The damping factor to use when updating messages.

    Returns:
        updated_msgs_arr: Array shape is (2, num_edges + 1, msg_size). This holds all the updated messages.
            The 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds
            to v-> f msgs. The last row is just an extra row of 0's that represents a "null message" which will never
            be updated.
    """
    updated_msgs_arr = jnp.zeros_like(original_msgs_arr)
    damped_vtof_msgs = (damping_factor * original_msgs_arr[1, :-1, :]) + (
        1 - damping_factor
    ) * updated_vtof_msgs
    damped_ftov_msgs = (damping_factor * original_msgs_arr[0, :-1, :]) + (
        1 - damping_factor
    ) * updated_ftov_msgs
    updated_msgs_arr = updated_msgs_arr.at[1, :-1, :].set(damped_vtof_msgs)
    updated_msgs_arr = updated_msgs_arr.at[0, :-1, :].set(damped_ftov_msgs)
    return updated_msgs_arr


@jax.jit
def compute_map_estimate_jax(
    msgs_arr: jnp.ndarray,
    evidence_arr: jnp.ndarray,
    var_neighbors_arr: jnp.ndarray,
) -> jnp.ndarray:
    """
    uses messages computed by message passing to derive the MAP estimate for every variable node

    Args:
        msgs_arr: Array shape is (2, num_edges + 1, msg_size). This holds all the messages. the 0th index
            of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to v-> f
            msgs. The last row is just an extra row of 0's that represents a "null message" which will never
            be updated.
        evidence_arr: Array shape is shape (num_edges, msg_size). evidence_arr[x,:] corresponds to the evidence
            needed to compute the message contained in msgs_arr[1,x,:]
        var_neighbors_arr: Array shape is (num_variables x max_num_var_neighbors). var_neighbors_arr[i,:] represent
                all the indices into msgs_arr[0,:,:] that correspond to neighboring f-> messages

    Returns:
        an array of size num_var_nodes where each index corresponds to the MAP state of a particular variable node
    """

    var_indices = jnp.arange(var_neighbors_arr.shape[0])
    neighboring_msgs_sum = msgs_arr[0, var_neighbors_arr[var_indices, :], :].sum(1)
    # For any var_index, we know that evidence_arr[var_neighbors_arr[var_index,:]] will be an array of the same
    # evidence messages (since all messages connected to the same var_node must have the same evidence).
    # Thus, to get only one of these messages, we can do evidence_arr[var_neighbors_arr[var_index,0]]
    neighbor_and_evidence_sum = (
        neighboring_msgs_sum + evidence_arr[var_neighbors_arr[var_indices, 0]]
    )
    return neighbor_and_evidence_sum.argmax(1)


def convert_map_to_dict(
    map_arr: jnp.ndarray, var_to_indices_dict: Dict[node_classes.VariableNode, int]
) -> Dict[node_classes.VariableNode, int]:
    """converts the array after MAP inference to a dict format expected by the viz code

    Args:
        map_arr: an array of size num_var_nodes where each index corresponds to the MAP state of
            a particular variable node
        var_to_indices_dict: for a particular var_node key, var_to_indices_dict[var_node]
                contains the row index into var_neighbors_arr that corresponds to var_node
    Returns:
        a dict mapping each VariableNode and its MAP state
    """
    var_map_dict = {}

    map_np_arr = np.array(map_arr)

    for var_node in var_to_indices_dict.keys():
        var_map_dict[var_node] = map_np_arr[var_to_indices_dict[var_node]]

    return var_map_dict
