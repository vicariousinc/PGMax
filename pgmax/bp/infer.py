import jax
import jax.numpy as jnp

import pgmax.bp.utils as utils

NEG_INF = (
    -100000.0
)  # A large negative value to use as -inf for numerical stability reasons
NEG_INF_INT = -10000


def run_bp_and_infer(
    msgs: jnp.ndarray,
    evidence: jnp.ndarray,
    edges_num_states: jnp.ndarray,
    var_states_for_edges: jnp.ndarray,
    factor_configs_edge_states: jnp.ndarray,
    num_iters: int,
    damping_factor: float,
):
    # NOTE: We can use a boolean variable here to indicate max-product vs sum-product!
    """
    performs belief propagation given the specified data-structures for num_iters iterations and returns the
    output of inference

    Args:
        msgs: Array of shape (num_edge_states). This holds all the flattened factor to variable messages.
        evidence: Maximum array shape is bounded by (num_var_nodes * max_msg_size). This array contains the fully-flattened
            set of evidence messages for each variable node
        edges_num_states: Array of shape (num_edges,). Number of states for the variables connected to each edge
        var_states_for_edges: Array of shape (num_edge_states,). Global variable state indices for each edge state
        factor_configs_edge_states: Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global factor_config and edge_state indices
            factor_configs_edge_states[ii, 0] contains the global factor config index
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index
        num_iters: The number of iterations for which to perform message passing
        damping_factor: The damping factor to use for message updates between one timestep and the next

    Returns:
        ???: Really not sure yet. Will figure out after implementing!
    """
    max_num_edges = int(jnp.max(edges_num_states))
    normalized_msgs = msgs - jnp.repeat(
        utils.segment_max_opt(msgs, edges_num_states, max_num_edges),
        edges_num_states,
        total_repeat_length=msgs.shape[0],
    )  # Normalize the messages to ensure the maximum value is 0.
    num_val_configs = int(factor_configs_edge_states[-1, 0])

    def message_passing_loop(
        normalized_msgs,
        evidence,
        var_states_for_edges,
        edges_num_states,
        max_num_edges,
        damping_factor,
        num_iters,
    ):
        "Function wrapper that leverages jax.lax.scan to efficiently perform belief propagation"

        def message_passing_step(normalized_msgs, _):
            vtof_msgs = _pass_var_to_fac_messages(
                normalized_msgs,
                evidence,
                var_states_for_edges,
                edges_num_states,
                max_num_edges,
            )
            ftov_msgs = _pass_fac_to_var_messages(
                vtof_msgs,
                factor_configs_edge_states,
                edges_num_states,
                max_num_edges,
                num_val_configs,
            )
            normalized_msgs = _damp_and_update_messages(
                normalized_msgs,
                ftov_msgs,
                edges_num_states,
                max_num_edges,
                damping_factor,
            )
            return normalized_msgs, None

        normalized_msgs, _ = jax.lax.scan(
            message_passing_step, normalized_msgs, None, num_iters
        )
        return normalized_msgs

    msgs_after_bp = message_passing_loop(
        normalized_msgs,
        evidence,
        var_states_for_edges,
        edges_num_states,
        max_num_edges,
        damping_factor,
        num_iters,
    )

    final_var_states = _compute_final_var_states(
        msgs_after_bp, evidence, var_states_for_edges
    )

    return final_var_states


@jax.partial(jax.jit, static_argnames="max_num_edges")
def _pass_var_to_fac_messages(
    msgs: jnp.array,
    evidence: jnp.array,
    var_states_for_edges: jnp.array,
    edges_num_states: jnp.array,
    max_num_edges: int,
) -> jnp.array:
    """
    passes messages from VariableNodes to FactorNodes and computes a new updated set of messages using JAX

    Args:
        msgs: Array of shape (num_edge_states). This holds all the flattened factor to variable messages.
        evidence: Maximum array shape is bounded by (num_var_nodes * max_msg_size). This array contains the fully-flattened
            set of evidence messages for each variable node
        var_states_for_edges: Array of shape (num_edge_states,). Global variable state indices for each edge state
        edges_num_states: Array of shape (num_edges,). Number of states for the variables connected to each edge
        max_num_edges: the max of edges_num_states
    Returns:
        Array of shape (num_edge_states). This holds all the flattened variable to factor messages.
    """
    # For each variable, sum the neighboring factor to variable messages and the evidence.
    var_sums_arr = evidence.at[var_states_for_edges].add(msgs)
    vtof_msgs = var_sums_arr[var_states_for_edges] - msgs
    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_vtof_msgs = vtof_msgs - jnp.repeat(
        utils.segment_max_opt(vtof_msgs, edges_num_states, max_num_edges),
        edges_num_states,
        total_repeat_length=msgs.shape[0],
    )
    clipped_vtof_msgs = jnp.clip(
        normalized_vtof_msgs, -1000, None
    )  # TODO: Take out clipping here and just clip the final updates msgs

    return clipped_vtof_msgs


@jax.partial(jax.jit, static_argnames=("num_val_configs", "max_num_edges"))
def _pass_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    factor_configs_edge_states: jnp.ndarray,
    edges_num_states: jnp.ndarray,
    max_num_edges: int,
    num_val_configs: int,
) -> jnp.ndarray:

    """
    passes messages from FactorNodes to VariableNodes and computes a new, updated set of messages using JAX

    Args:
        vtof_msgs: Array of shape (num_edge_states). This holds all the flattened variable to factor messages.
        factor_configs_edge_states: Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global factor_config and edge_state indices
            factor_configs_edge_states[ii, 0] contains the global factor config index
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index
        edges_num_states: Array of shape (num_edges,). Number of states for the variables connected to each edge
        max_num_edges: the max of edges_num_states
        num_val_configs: the total number of valid configurations for factors in the factor graph.

    Returns:
        Array of shape (num_edge_states). This holds all the flattened factor to variable messages.
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

    # Normalize and clip messages (between -1000 and 1000) before returning
    normalized_updated_msgs = updated_ftov_msgs - jnp.repeat(
        utils.segment_max_opt(updated_ftov_msgs, edges_num_states, max_num_edges),
        edges_num_states,
        total_repeat_length=vtof_msgs.shape[0],
    )
    clipped_updated_msgs = jnp.clip(normalized_updated_msgs, -1000, None)

    return clipped_updated_msgs  # TODO: Take out clipping here and just clip the final updates msgs


@jax.partial(jax.jit, static_argnames=("damping_factor", "max_num_edges"))
def _damp_and_update_messages(
    original_msgs: jnp.ndarray,
    new_msgs: jnp.ndarray,
    edges_num_states: jnp.ndarray,
    max_num_edges: int,
    damping_factor: float,
) -> jnp.ndarray:
    """
    updates messages using previous messages, new messages and damping factor

    Args:
        original_msgs: Array of shape (num_edge_states). This holds all the flattened factor to variable messages
            prior to bp updating.
        new_msgs: msgs: Array of shape (num_edge_states). This holds all the flattened factor to variable messages
            after bp updating but before damping.
        edges_num_states: Array of shape (num_edges,). Number of states for the variables connected to each edge
        max_num_edges: the max of edges_num_states
        damping_factor (float): The damping factor to use when updating messages.

    Returns:
        Array of shape (num_edge_states). This holds all the updated flattened factor to variable messages
            after damping.
    """
    delta_msgs = new_msgs - original_msgs
    damped_updated_msgs = original_msgs + damping_factor * delta_msgs
    normalized_updated_msgs = damped_updated_msgs - jnp.repeat(
        utils.segment_max_opt(damped_updated_msgs, edges_num_states, max_num_edges),
        edges_num_states,
        total_repeat_length=damped_updated_msgs.shape[0],
    )

    return normalized_updated_msgs


@jax.jit
def _compute_final_var_states(
    msgs: jnp.array,
    evidence: jnp.array,
    var_states_for_edges: jnp.array,
) -> jnp.array:
    """
    passes messages from VariableNodes to FactorNodes and computes a new updated set of messages using JAX

    Args:
        msgs: Array of shape (num_edge_states). This holds all the flattened factor to variable messages.
        evidence: Maximum array shape is bounded by (num_var_nodes * max_msg_size). This array contains the fully-flattened
            set of evidence messages for each variable node
        var_states_for_edges: Array of shape (num_edge_states,). Global variable state indices for each edge state
        edges_num_states: Array of shape (num_edges,). Number of states for the variables connected to each edge
        max_num_edges: the max of edges_num_states
    Returns:
        Array of shape (num_edge_states). This holds all the flattened variable to factor messages.
    """
    # For each variable, sum the neighboring factor to variable messages and the evidence.
    var_sums_arr = evidence.at[var_states_for_edges].add(msgs)
    return var_sums_arr
