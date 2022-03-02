"""A module containing the core message-passing functions for belief propagation"""

import functools

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
        evidence: Array of shape (num_var_states,) representing the flattened evidence for each variable
        var_states_for_edges: Array of shape (num_edge_states,)
            Global variable state indices for each edge state
    Returns:
        Array of shape (num_edge_state,). This holds all the flattened variable to factor messages.
    """
    var_sums_arr = evidence.at[var_states_for_edges].add(ftov_msgs)
    vtof_msgs = var_sums_arr[var_states_for_edges] - ftov_msgs
    return vtof_msgs


@functools.partial(jax.jit, static_argnames=("num_val_configs", "temperature"))
def pass_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    factor_configs_edge_states: jnp.ndarray,
    log_potentials: jnp.ndarray,
    num_val_configs: int,
    temperature: float,
) -> jnp.ndarray:

    """Passes messages from Factors to Variables.

    The update is performed in two steps. First, a "summary" array is generated that has an entry for every valid
    configuration for every factor. The elements of this array are simply the sums of messages across each valid
    config. Then, the info from factor_configs_edge_states is used to apply the scattering operation and
    generate a flat set of output messages.

    Args:
        vtof_msgs: Array of shape (num_edge_state,). This holds all the flattened variable to factor messages.
        factor_configs_edge_states: Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global factor_config and edge_state indices
            factor_configs_edge_states[ii, 0] contains the global factor config index
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index
        log_potentials: Array of shape (num_val_configs, ). An entry at index i is the log potential
            function value for the configuration with global factor config index i.
        num_val_configs: the total number of valid configurations for factors in the factor graph.
        temperature: Temperature for loopy belief propagation.
            1.0 corresponds to sum-product, 0.0 corresponds to max-product.

    Returns:
        Array of shape (num_edge_state,). This holds all the flattened factor to variable messages.
    """
    fac_config_summary_sum = (
        jnp.zeros(shape=(num_val_configs,))
        .at[factor_configs_edge_states[..., 0]]
        .add(vtof_msgs[factor_configs_edge_states[..., 1]])
    ) + log_potentials
    max_factor_config_summary_for_edge_states = (
        jnp.full(shape=(vtof_msgs.shape[0],), fill_value=NEG_INF)
        .at[factor_configs_edge_states[..., 1]]
        .max(fac_config_summary_sum[factor_configs_edge_states[..., 0]])
    )
    ftov_msgs = max_factor_config_summary_for_edge_states - vtof_msgs
    if temperature != 0.0:
        ftov_msgs = ftov_msgs + (
            temperature
            * jnp.log(
                jnp.full(shape=(vtof_msgs.shape[0],), fill_value=jnp.exp(NEG_INF))
                .at[factor_configs_edge_states[..., 1]]
                .add(
                    jnp.exp(
                        (
                            fac_config_summary_sum[factor_configs_edge_states[..., 0]]
                            - max_factor_config_summary_for_edge_states[
                                factor_configs_edge_states[..., 1]
                            ]
                        )
                        / temperature
                    )
                )
            )
        )
    return ftov_msgs


@jax.jit
def pass_OR_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    parents_states: jnp.ndarray,
    children_states: jnp.ndarray,
) -> jnp.ndarray:

    """Passes messages from OR Factors to Variables.

    Args:
        vtof_msgs: Array of shape (num_edge_states,).
            This holds all the flattened (binary) variables to factor messages.
        parents_states: Array of shape (num_parents, 2)
            parents_states[ii, 0] contains the global factor index
            parents_states[ii, 1] contains the message index of the parent variable's state 0.
            The parent variable's state 1 is parents_states[ii, 2] + 1
        children_states: Array of shape (num_factors, 2)
            children_states[ii, 0] contains the global factor index
            children_states[ii, 1] contains the message index of the child variable's state 0
            The child variable's state 1 is children_states[ii, 1] + 1

    Returns:
        Array of shape (num_edge_states,). This holds all the flattened factor to variable messages.
    """
    num_top_vars = parents_states.shape[0]
    num_factors = children_states.shape[0]

    factor_indices = parents_states[..., 0]
    top_tof_msgs = (
        vtof_msgs[parents_states[..., 1] + 1] - vtof_msgs[parents_states[..., 1]]
    )
    bottom_tof_msgs = (
        vtof_msgs[children_states[..., 1] + 1] - vtof_msgs[children_states[..., 1]]
    )

    def _get_argmaxes(top_tof_msgs, factor_indices):
        maxes = (
            jnp.full(shape=num_factors, fill_value=-jnp.inf)
            .at[factor_indices]
            .max(top_tof_msgs)
        )
        only_maxes_pos = jnp.arange(num_top_vars) - num_top_vars * (
            top_tof_msgs != maxes[factor_indices]
        )
        argmaxes = (
            jnp.full(shape=num_factors, fill_value=-jnp.inf)
            .at[factor_indices]
            .max(only_maxes_pos)
            .astype(jnp.int32)
        )
        return argmaxes

    # Get top incoming argmaxes for each factor
    first_argmaxes = _get_argmaxes(top_tof_msgs, factor_indices)
    second_argmaxes = _get_argmaxes(
        top_tof_msgs.at[first_argmaxes].set(-jnp.inf), factor_indices
    )

    top_tof_msgs_pos = jnp.maximum(0.0, top_tof_msgs)
    sum_top_tof_msgs_pos_inc = (
        jnp.full(shape=num_factors, fill_value=0.0)
        .at[factor_indices]
        .add(top_tof_msgs_pos)
    )

    # Outgoing messages to top variables
    top_msgs = jnp.minimum(
        bottom_tof_msgs[factor_indices]
        + sum_top_tof_msgs_pos_inc[factor_indices]
        - top_tof_msgs_pos,
        jnp.maximum(0.0, -top_tof_msgs[first_argmaxes][factor_indices]),
    )
    top_msgs = top_msgs.at[first_argmaxes].set(
        jnp.minimum(
            bottom_tof_msgs
            + sum_top_tof_msgs_pos_inc
            - top_tof_msgs_pos[first_argmaxes],
            jnp.maximum(0.0, -top_tof_msgs[second_argmaxes]),
        )
    )

    # Special case for factor with single parents
    has_single_parents = (first_argmaxes == second_argmaxes).astype(jnp.float32)
    top_msgs = top_msgs.at[first_argmaxes].set(
        (1 - has_single_parents) * top_msgs[first_argmaxes]
        + has_single_parents * bottom_tof_msgs
    )

    # Outgoing messages to bottom variables
    bottom_msgs = sum_top_tof_msgs_pos_inc + jnp.minimum(
        0.0, top_tof_msgs[first_argmaxes]
    )

    bottom_tov0_msgs = jnp.minimum(0.0, -bottom_msgs)
    bottom_tov1_msgs = jnp.minimum(0.0, bottom_msgs)
    top_tov0_msgs = jnp.minimum(0.0, -top_msgs)
    top_tov1_msgs = jnp.minimum(0.0, top_msgs)

    ftov_msgs = jnp.full(shape=(vtof_msgs.shape[0],), fill_value=-jnp.inf)
    ftov_msgs = ftov_msgs.at[children_states[..., 1]].set(bottom_tov0_msgs)
    ftov_msgs = ftov_msgs.at[children_states[..., 1] + 1].set(bottom_tov1_msgs)
    ftov_msgs = ftov_msgs.at[parents_states[..., 1]].set(top_tov0_msgs)
    ftov_msgs = ftov_msgs.at[parents_states[..., 1] + 1].set(top_tov1_msgs)
    return ftov_msgs


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
