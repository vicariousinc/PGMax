"""A module containing the core message-passing functions for belief propagation"""

import functools

import jax
import jax.numpy as jnp

from pgmax.bp import bp_utils


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
        evidence: Array of shape (num_var_states,) representing the flattened evidence for each variable
        var_states_for_edges: Array of shape (num_edge_states,)
            Global variable state indices for each edge state
    Returns:
        Array of shape (num_edge_state,). This holds all the flattened variable to factor messages.
    """
    var_sums_arr = evidence.at[var_states_for_edges].add(ftov_msgs)
    vtof_msgs = var_sums_arr[var_states_for_edges] - ftov_msgs
    return vtof_msgs


@functools.partial(jax.jit, static_argnames=("max_msg_size"))
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
