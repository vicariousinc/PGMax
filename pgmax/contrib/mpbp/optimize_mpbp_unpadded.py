from timeit import default_timer as timer
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from pgmax.contrib.mpbp.mpbp_varfacnodes_varmsgsize_unpadded import compile_jax_data_structures
import pgmax.contrib.interface.node_classes_with_factortypes as node_classes

NEG_INF = (
    -100000.0
)  # A large negative value to use as -inf for numerical stability reasons
damping_factor = 0.5

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
        _,
        normalization_indices_arr,
        var_to_indices_dict,
        num_val_configs,
    ) = compile_jax_data_structures(fg, evidence)
    end_time = timer()
    print(f"Data structures compiled in: {end_time - start_time}s")

    edge_msg_sizes = jax.device_put(np.bincount(normalization_indices_arr // 3))
    max_edge_msg_size = int(jnp.max(edge_msg_sizes))

    # Convert all arrays to jnp.ndarrays for use in BP
    # (Comments on the right show cumulative memory usage as each of these lines execute)
    msgs_arr = jax.device_put(msgs_arr)
    evidence_arr = jax.device_put(evidence_arr)
    msg_vals_to_var_arr = jax.device_put(msg_vals_to_var_arr)
    factor_configs = jax.device_put(factor_configs)
    edge_msg_sizes = jax.device_put(edge_msg_sizes)
    normalization_indices_arr = jax.device_put(normalization_indices_arr)

    @jax.partial(jax.jit, static_argnames=("num_val_configs", "num_iters", "max_edge_msg_size"))
    def run_mpbp_update_loop(
        msgs_arr,
        evidence_arr,
        msg_vals_to_var_arr,
        factor_configs,
        edge_msg_sizes,
        max_edge_msg_size,
        num_val_configs,
        num_iters,
    ):
        "Function wrapper that leverages jax.lax.scan to efficiently perform BP"
        def mpbp_update_step(msgs_arr, x):
            # Variable to Factor messages update
            updated_vtof_msgs = pass_var_to_fac_messages_jnp(
                msgs_arr, evidence_arr, msg_vals_to_var_arr, edge_msg_sizes, max_edge_msg_size
            )
            # Factor to Variable messages update
            updated_ftov_msgs = pass_fac_to_var_messages_jnp(
                msgs_arr,
                factor_configs,
                edge_msg_sizes,
                max_edge_msg_size,
                num_val_configs
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
        edge_msg_sizes,
        max_edge_msg_size,
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
        msg_vals_to_var_arr,
        factor_configs,
        edge_msg_sizes,
        max_edge_msg_size,
        num_val_configs,
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


@jax.partial(jax.jit, static_argnames="max_segment_length")
def segment_max_opt(data, segments_lengths, max_segment_length):
    @jax.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
    def get_max(data, start_index, segment_length):
        return jnp.max(
            jnp.where(
                jnp.arange(max_segment_length) < segment_length,
                jax.lax.dynamic_slice(
                    data, jnp.array([start_index]), [max_segment_length]
                ),
                NEG_INF,
            )
        )

    start_indices = jnp.concatenate(
        [
            jnp.full(shape=(1,), fill_value=int(NEG_INF), dtype=int),
            jnp.cumsum(segments_lengths),
        ]
    )[:-1]
    expanded_data = jnp.concatenate([data, jnp.zeros(max_segment_length)])
    return get_max(expanded_data, start_indices, segments_lengths)


@jax.partial(jax.jit, static_argnames="max_edge_msg_size")
def pass_var_to_fac_messages_jnp(
    msgs_arr: jnp.array,
    evidence_arr: jnp.array,
    msg_vals_to_var_arr: jnp.array,
    edge_msg_sizes: jnp.array,
    max_edge_msg_size: int,
) -> jnp.array:
    """
    passes messages from VariableNodes to FactorNodes and computes a new updated set of messages using JAX

    Args:
        msgs_arr: Maximum array shape is bounded (2, num_edges * max_msg_size). This holds all the flattened messages.
            the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to
            v->f msgs.
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
    var_sums_arr = evidence_arr.at[msg_vals_to_var_arr].add(msgs_arr[0])
    updated_vtof_msgs = var_sums_arr[msg_vals_to_var_arr] - msgs_arr[0]
    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = updated_vtof_msgs - jnp.repeat(
        segment_max_opt(updated_vtof_msgs, edge_msg_sizes, max_edge_msg_size),
        edge_msg_sizes,
        total_repeat_length=msgs_arr.shape[1],
    )
    clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, 1000)

    return clipped_updated_msgs


@jax.partial(jax.jit, static_argnames=("num_val_configs", "max_edge_msg_size"))
def pass_fac_to_var_messages_jnp(
    msgs_arr: jnp.ndarray,
    factor_configs: jnp.ndarray,
    edge_msg_sizes: jnp.ndarray,
    max_edge_msg_size: int,
    num_val_configs: int,
) -> jnp.ndarray:

    """
    passes messages from FactorNodes to VariableNodes and computes a new, updated set of messages using JAX

    Args:
        msgs_arr: Maximum array shape is bounded (2, num_edges * max_msg_size). This holds all the flattened messages.
            the 0th index of the 0th axis corresponds to f->v msgs while the 1st index of the 0th axis corresponds to
            v->f msgs.
        factor_configs: Maximum array shape is bounded by (2, num_factors * max_num_configs * max_config_size). The 0th axis
            contains a flat list of valid configuration indices such that msgs_arr[1, factor_configs[0]] gives a flattened array of
            all the message values from the valid configurations. The 1st axis contains segmentation masks corresponding to the 0th
            axis (i.e, all entries corresponding to factor 0 config 0 are labelled 0, all entries corresponding to factor 0 config 1
            are labelled 1, and so on).
        normalization_indices_arr: Maximum array shape is bounded by (num_edges * max_msg_size,). For every entry along the 1st axis of
            msgs_arr, this array contains the index of the entry this must be subtracted by to produce message normalization.
        num_val_configs: the total number of valid configurations for factors in the factor graph.

    Returns:
        Array of shape (num_edges, msg_size) corresponding to the updated f->v messages after normalization and clipping
    """

    # Update Strategy
    # Stage 1: Generate a "summary" array that has an entry for every valid configuration. The elements of
    #   this array are simply the sums of messages across each valid config.
    # Stage 2: Utilize the info from edge_vals_to_config_summary_indices to apply the scattering operation and generate
    #   a flat set of output messages.

    # Generate summary array for each factor and each config
    # fac_config_summary_sum = jax.ops.segment_sum(
    #     msgs_arr[1, factor_configs[0]],
    #     factor_configs[1],
    #     num_segments=num_val_configs,
    # )
    fac_config_summary_sum = (
        jnp.zeros(shape=(num_val_configs,))
        .at[factor_configs[1]]
        .add(msgs_arr[1, factor_configs[0]])
    )

    # Update Step 2
    # Perform scattering in a flattened format
    updated_ftov_msgs = (
        jnp.full(shape=(msgs_arr[1].shape[0],), fill_value=NEG_INF)
        .at[factor_configs[0]]
        .max(fac_config_summary_sum[factor_configs[1]])
    ) - msgs_arr[1]

    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = updated_ftov_msgs - jnp.repeat(
        segment_max_opt(updated_ftov_msgs, edge_msg_sizes, max_edge_msg_size),
        edge_msg_sizes,
        total_repeat_length=msgs_arr.shape[1],
    )
    clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, 1000)

    return clipped_updated_msgs


@jax.jit
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


@jax.partial(
    jax.jit, static_argnames=("num_val_configs", "num_iters", "max_edge_msg_size")
)
def run_mpbp_update_loop(
    msgs_arr,
    evidence_arr,
    msg_vals_to_var_arr,
    factor_configs,
    num_val_configs,
    edge_msg_sizes,
    num_iters,
    max_edge_msg_size: int,
):
    "Function wrapper that leverages jax.lax.scan to efficiently perform BP"

    def mpbp_update_step(msgs_arr, x):
        # Variable to Factor messages update
        updated_vtof_msgs = pass_var_to_fac_messages_jnp(
            msgs_arr,
            evidence_arr,
            msg_vals_to_var_arr,
            edge_msg_sizes,
            max_edge_msg_size,
        )
        # Factor to Variable messages update
        updated_ftov_msgs = pass_fac_to_var_messages_jnp(
            msgs_arr,
            factor_configs,
            edge_msg_sizes,
            max_edge_msg_size,
            num_val_configs,
        )
        # Damping before final message update
        msgs_arr = damp_and_update_messages(
            updated_vtof_msgs, updated_ftov_msgs, msgs_arr, damping_factor
        )
        return msgs_arr, None

    msgs_arr, _ = jax.lax.scan(mpbp_update_step, msgs_arr, None, num_iters)
    return msgs_arr

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