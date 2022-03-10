"""A module containing the core message-passing functions for belief propagation"""

import functools

import jax
import jax.numpy as jnp
from jax.nn import log_sigmoid, sigmoid

import pgmax.bp.bp_utils as bp_utils


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
            factor_configs_edge_states[ii, 0] contains the global factor config index,
            which takes into account all the enumeration factors
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index,
            which takes into account all the enumeration and OR factors
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
        jnp.full(shape=(vtof_msgs.shape[0],), fill_value=bp_utils.NEG_INF)
        .at[factor_configs_edge_states[..., 1]]
        .max(fac_config_summary_sum[factor_configs_edge_states[..., 0]])
    )
    ftov_msgs = max_factor_config_summary_for_edge_states - vtof_msgs
    if temperature != 0.0:
        ftov_msgs = ftov_msgs + (
            temperature
            * jnp.log(
                jnp.full(
                    shape=(vtof_msgs.shape[0],), fill_value=jnp.exp(bp_utils.NEG_INF)
                )
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


@functools.partial(jax.jit, static_argnames=("temperature"))
def pass_OR_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    parents_edge_states: jnp.ndarray,
    children_edge_states: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:

    """Passes messages from OR Factors to Variables.

    Args:
        vtof_msgs: Array of shape (num_edge_state,).
            This holds all the flattened (binary) variables to factor messages.
        parents_edge_states: Array of shape (num_parents, 2)
            parents_edge_states[ii, 0] contains the global factor index,
            which takes into account all the OR factors
            parents_edge_states[ii, 1] contains the message index of the parent variable's state 0,
            which takes into account all the enumeration and OR factors
            The parent variable's state 1 is parents_edge_states[ii, 2] + 1
        children_edge_states: Array of shape (num_factors,)
            children_edge_states[ii] contains the message index of the child variable's state 0,
            which takes into account all the enumeration and OR factors
            The child variable's state 1 is children_edge_states[ii, 1] + 1
        temperature: Temperature for loopy belief propagation.
            1.0 corresponds to sum-product, 0.0 corresponds to max-product.

    Returns:
        Array of shape (num_edge_state,). This holds all the flattened factor to variable messages.
    """
    num_factors = children_edge_states.shape[0]

    factor_indices = parents_edge_states[..., 0]

    parents_tof_msgs = (
        vtof_msgs[parents_edge_states[..., 1] + 1]
        - vtof_msgs[parents_edge_states[..., 1]]
    )
    children_tof_msgs = (
        vtof_msgs[children_edge_states + 1] - vtof_msgs[children_edge_states]
    )

    # We treat the max-product case separately.
    if temperature == 0.0:
        # Get the first and second argmaxes for the incoming parents messages of each factor
        _, first_parents_argmaxes = bp_utils.get_maxes_and_argmaxes(
            parents_tof_msgs, factor_indices, num_factors
        )
        _, second_parents_argmaxes = bp_utils.get_maxes_and_argmaxes(
            parents_tof_msgs.at[first_parents_argmaxes].set(bp_utils.NEG_INF),
            factor_indices,
            num_factors,
        )

        parents_tof_msgs_pos = jnp.maximum(0.0, parents_tof_msgs)
        sum_parents_tof_msgs_pos = (
            jnp.full(shape=(num_factors,), fill_value=0.0)
            .at[factor_indices]
            .add(parents_tof_msgs_pos)
        )

        # Outgoing messages to parents variables
        # See https://arxiv.org/pdf/2111.02458.pdf, Appendix C.3
        parents_msgs = jnp.minimum(
            children_tof_msgs[factor_indices]
            + sum_parents_tof_msgs_pos[factor_indices]
            - parents_tof_msgs_pos,
            jnp.maximum(0.0, -parents_tof_msgs[first_parents_argmaxes][factor_indices]),
        )
        parents_msgs = parents_msgs.at[first_parents_argmaxes].set(
            jnp.minimum(
                children_tof_msgs
                + sum_parents_tof_msgs_pos
                - parents_tof_msgs_pos[first_parents_argmaxes],
                jnp.maximum(0.0, -parents_tof_msgs[second_parents_argmaxes]),
            )
        )

        # Outgoing messages to children variables
        children_msgs = sum_parents_tof_msgs_pos + jnp.minimum(
            0.0, parents_tof_msgs[first_parents_argmaxes]
        )
    else:

        def g(x):
            # assert jnp.all(x >= 0)
            return jnp.where(
                x == 0.0,
                0.0,
                x + temperature * jnp.log(1.0 - jnp.exp(-x / temperature)),
            )

        log_sig_parents_tof_msgs = -temperature * log_sigmoid(
            -parents_tof_msgs / temperature
        )
        sum_log_sig_parents_tof_msgs = (
            jnp.full(shape=(num_factors,), fill_value=0.0)
            .at[factor_indices]
            .add(log_sig_parents_tof_msgs)
        )
        g_sum_log_sig_parents_minus_id = g(
            sum_log_sig_parents_tof_msgs[factor_indices] - log_sig_parents_tof_msgs
        )

        # Outgoing messages to parents variables
        parents_msgs = -temperature * jnp.log(
            sigmoid(g_sum_log_sig_parents_minus_id / temperature)
            + sigmoid(-g_sum_log_sig_parents_minus_id / temperature)
            * jnp.exp(-children_tof_msgs[factor_indices] / temperature)
        )

        # Outgoing messages to children variables
        children_msgs = g(sum_log_sig_parents_tof_msgs)

    # Special case: factors with a single parent
    num_parents = jnp.bincount(factor_indices, length=num_factors)
    first_elements = jnp.concatenate(
        [jnp.zeros(1, dtype=int), jnp.cumsum(num_parents)]
    )[:-1]
    parents_msgs = parents_msgs.at[first_elements].set(
        jnp.where(num_parents == 1, children_tof_msgs, parents_msgs[first_elements]),
    )

    ftov_msgs = jnp.zeros_like(vtof_msgs)
    ftov_msgs = ftov_msgs.at[parents_edge_states[..., 1] + 1].set(parents_msgs)
    ftov_msgs = ftov_msgs.at[children_edge_states + 1].set(children_msgs)
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
