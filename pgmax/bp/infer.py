"""A module containing the core message-passing functions for belief propagation"""

import jax
import jax.numpy as jnp

import pgmax.bp.bp_utils as bp_utils

NEG_INF = -100000.0  # A large negative value to use as -inf to avoid NaN's


@jax.jit
def pass_var_to_fac_messages(
    ftov_msgs: jnp.array,
    evidence: jnp.array,
    var_states_for_edges: jnp.array,
) -> jnp.array:
    """Passes messages from Variables to Factors.

    The update works by first summing the evidence and neighboring factor to variable messages for
    each variable. Next, it subtracts messages from the correct elements of this sum to yield the
    correct updated messages.

    Args:
        ftov_msgs: Array of shape (num_edge_state,). This holds all the flattened factor to variable
            messages.
        Array of shape (num_var_states,) representing the flattened evidence for each variable
    Returns:
        Array of shape (num_edge_state,). This holds all the flattened variable to factor messages.
    """
    var_sums_arr = evidence.at[var_states_for_edges].add(ftov_msgs)
    vtof_msgs = var_sums_arr[var_states_for_edges] - ftov_msgs
    return vtof_msgs


@jax.partial(jax.jit, static_argnames=("num_val_configs"))
def pass_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    factor_configs_edge_states: jnp.ndarray,
    factor_configs_log_potentials: jnp.ndarray,
    num_val_configs: int,
) -> jnp.ndarray:

    """Passes messages from Factors to Variables.

    The update is performed in two steps. First, a "summary" array is generated that has an entry for every valid
    configuration for every factor. The elements of this array are simply the sums of messages across each valid
    config. Then, the info from edge_vals_to_config_summary_indices is used to apply the scattering operation and
    generate a flat set of output messages.

    Args:
        vtof_msgs: Array of shape (num_edge_state,). This holds all the flattened variable to factor messages.
        factor_configs_edge_states: Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global factor_config and edge_state indices
            factor_configs_edge_states[ii, 0] contains the global factor config index
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index
        factor_configs_log_potentials: Array of shape (num_val_configs, ). An entry at index i is the log potential
            function value for the configuration with global factor config index i.
        num_val_configs: the total number of valid configurations for factors in the factor graph.

    Returns:
        Array of shape (num_edge_state,). This holds all the flattened factor to variable messages.
    """
    fac_config_summary_sum = (
        jnp.zeros(shape=(num_val_configs,))
        .at[factor_configs_edge_states[..., 0]]
        .add(vtof_msgs[factor_configs_edge_states[..., 1]])
    ) + factor_configs_log_potentials
    ftov_msgs = (
        jnp.full(shape=(vtof_msgs.shape[0],), fill_value=NEG_INF)
        .at[factor_configs_edge_states[..., 1]]
        .max(fac_config_summary_sum[factor_configs_edge_states[..., 0]])
    ) - vtof_msgs
    return ftov_msgs


@jax.partial(jax.jit, static_argnames=("max_msg_size"))
def normalize_and_clip_msgs(
    msgs: jnp.ndarray,
    edges_num_states: jnp.ndarray,
    max_msg_size: int,
) -> jnp.ndarray:
    """Performs normalization and clipping of flattened messages

    Normalization is done by subtracting the maximum value of every message from every element of every message,
    clipping is done to keep every message value in the range [-1000, 0].

    Args:
        msgs: Array of shape (num_edge_state,). This holds all the flattened factor to variable messages.
        edges_num_states: Array of shape (num_edges,). Number of states for the variables connected to each edge
        max_msg_size: the max of edges_num_states

    Returns:
        Array of shape (num_edge_state,). This holds all the flattened factor to variable messages
            after normalization and clipping
    """
    msgs = msgs - jnp.repeat(
        bp_utils.segment_max_opt(msgs, edges_num_states, max_msg_size),
        edges_num_states,
        total_repeat_length=msgs.shape[0],
    )
    # Clip message values to be always greater than -1000
    msgs = jnp.clip(msgs, -1000, None)
    return msgs
