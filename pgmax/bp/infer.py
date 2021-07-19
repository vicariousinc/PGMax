import jax
import jax.numpy as jnp

import pgmax.bp.bp_utils as bp_utils

NEG_INF = (
    -100000.0
)  # A large negative value to use as -inf for numerical stability reasons


@jax.jit
def pass_var_to_fac_messages(
    msgs: jnp.array,
    evidence: jnp.array,
    var_states_for_edges: jnp.array,
) -> jnp.array:
    """
    passes messages from VariableNodes to FactorNodes and computes a new updated set of messages using JAX

    Args:
        msgs: Array of shape (num_edge_state,). This holds all the flattened factor to variable messages.
        evidence: Maximum array shape is bounded by (num_var_nodes * max_msg_size). This array contains the fully-flattened
            set of evidence messages for each variable node
    Returns:
        Array of shape (num_edge_state,). This holds all the flattened variable to factor messages.
    """
    # For each variable, sum the neighboring factor to variable messages and the evidence.
    var_sums_arr = evidence.at[var_states_for_edges].add(msgs)
    vtof_msgs = var_sums_arr[var_states_for_edges] - msgs
    return vtof_msgs


@jax.partial(jax.jit, static_argnames=("num_val_configs"))
def pass_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    factor_configs_edge_states: jnp.ndarray,
    num_val_configs: int,
) -> jnp.ndarray:

    """
    passes messages from FactorNodes to VariableNodes and computes a new, updated set of messages using JAX

    Args:
        vtof_msgs: Array of shape (num_edge_state,). This holds all the flattened variable to factor messages.
        factor_configs_edge_states: Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global factor_config and edge_state indices
            factor_configs_edge_states[ii, 0] contains the global factor config index
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index
        num_val_configs: the total number of valid configurations for factors in the factor graph.

    Returns:
        Array of shape (num_edge_state,). This holds all the flattened factor to variable messages.
    """

    # Update Strategy
    # Stage 1: Generate a "summary" array that has an entry for every valid configuration. The elements of
    #   this array are simply the sums of messages across each valid config.
    # Stage 2: Utilize the info from edge_vals_to_config_summary_indices to apply the scattering operation and generate
    #   a flat set of output messages.

    # Generate summary array for each factor and each config
    fac_config_summary_sum = (
        jnp.zeros(shape=(num_val_configs,))
        .at[factor_configs_edge_states[..., 0]]
        .add(vtof_msgs[factor_configs_edge_states[..., 1]])
    )

    # Update Step 2
    # Perform scattering in a flattened format
    updated_ftov_msgs = (
        jnp.full(shape=(vtof_msgs.shape[0],), fill_value=NEG_INF)
        .at[factor_configs_edge_states[..., 1]]
        .max(fac_config_summary_sum[factor_configs_edge_states[..., 0]])
    ) - vtof_msgs

    return updated_ftov_msgs


@jax.partial(jax.jit, static_argnames=("max_msg_size"))
def normalize_and_clip_msgs(
    msgs: jnp.ndarray,
    edges_num_states: jnp.ndarray,
    max_msg_size: int,
) -> jnp.ndarray:
    """
    updates messages using previous messages, new messages and damping factor

    Args:
        msgs: Array of shape (num_edge_state,). This holds all the flattened factor to variable messages.
        edges_num_states: Array of shape (num_edges,). Number of states for the variables connected to each edge
        max_msg_size: the max of edges_num_states

    Returns:
        Array of shape (num_edge_state,). This holds all the flattened factor to variable messages
            after normalization and clipping to the range [-1000,0]
    """
    normalized_updated_msgs = msgs - jnp.repeat(
        bp_utils.segment_max_opt(msgs, edges_num_states, max_msg_size),
        edges_num_states,
        total_repeat_length=msgs.shape[0],
    )
    # Clip message values to be always greater than -1000
    clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, None)
    return clipped_updated_msgs
