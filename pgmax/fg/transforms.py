from dataclasses import replace
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from pgmax.bp import infer
from pgmax.fg import graph


def BP(bp_state: graph.BPState, num_iters: int):
    max_msg_size = int(jnp.max(bp_state.fg_state.wiring.edges_num_states))
    num_val_configs = (
        int(bp_state.fg_state.wiring.factor_configs_edge_states[-1, 0]) + 1
    )

    @jax.jit
    def run_bp(
        log_potentials_updates: Optional[Dict[Any, jnp.ndarray]] = None,
        ftov_msgs_updates: Optional[Dict[Any, jnp.ndarray]] = None,
        evidence_updates: Optional[Dict[Any, jnp.ndarray]] = None,
        damping: float = 0.5,
    ):
        """Function to perform belief propagation.

        Specifically, belief propagation is run for num_iters iterations and
        returns the resulting messages.

        Args:
            num_iters: The number of iterations for which to perform message passing
            damping: The damping factor to use for message updates between one timestep and the next
            bp_state: Initial messages to start the belief propagation.

        Returns:
            ftov messages after running BP for num_iters iterations
        """
        # Retrieve the necessary data structures from the compiled self.wiring and
        # convert these to jax arrays.
        log_potentials = jax.device_put(bp_state.log_potentials.value)
        if log_potentials_updates is not None:
            log_potentials = graph.update_log_potentials(
                log_potentials, log_potentials_updates, bp_state.fg_state
            )

        ftov_msgs = jax.device_put(bp_state.ftov_msgs.value)
        if ftov_msgs_updates is not None:
            ftov_msgs = graph.update_ftov_msgs(
                ftov_msgs, ftov_msgs_updates, bp_state.fg_state
            )

        evidence = jax.device_put(bp_state.evidence.value)
        if evidence_updates is not None:
            evidence = graph.update_evidence(
                evidence, evidence_updates, bp_state.fg_state
            )

        wiring = jax.device_put(bp_state.fg_state.wiring)
        # Normalize the messages to ensure the maximum value is 0.
        ftov_msgs = infer.normalize_and_clip_msgs(
            ftov_msgs, wiring.edges_num_states, max_msg_size
        )

        def update(msgs, _):
            # Compute new variable to factor messages by message passing
            vtof_msgs = infer.pass_var_to_fac_messages(
                msgs,
                evidence,
                wiring.var_states_for_edges,
            )
            # Compute new factor to variable messages by message passing
            ftov_msgs = infer.pass_fac_to_var_messages(
                vtof_msgs,
                wiring.factor_configs_edge_states,
                log_potentials,
                num_val_configs,
            )
            # Use the results of message passing to perform damping and
            # update the factor to variable messages
            delta_msgs = ftov_msgs - msgs
            msgs = msgs + (1 - damping) * delta_msgs
            # Normalize and clip these damped, updated messages before returning
            # them.
            msgs = infer.normalize_and_clip_msgs(
                msgs,
                wiring.edges_num_states,
                max_msg_size,
            )
            return msgs, None

        ftov_msgs, _ = jax.lax.scan(update, ftov_msgs, None, num_iters)
        return ftov_msgs

    def get_bp_state(ftov_msgs):
        return replace(
            bp_state,
            ftov_msgs=graph.FToVMessages(
                fg_state=bp_state.ftov_msgs.fg_state, value=ftov_msgs
            ),
        )

    return run_bp, get_bp_state


def DecodeMAPStates(bp_state: graph.BPState):
    def decode_map_states(
        variable_name: Any = None,
    ) -> Union[int, Dict[Tuple[Any, ...], int]]:
        var_states_for_edges = jax.device_put(
            bp_state.fg_state.wiring.var_states_for_edges
        )
        evidence = jax.device_put(bp_state.evidence.value)
        beliefs = evidence.at[var_states_for_edges].add(bp_state.ftov_msgs.value)
        if variable_name is None:
            variables_to_map_states: Dict[Tuple[Any, ...], int] = {}
            for variable_name in bp_state.ftov_msgs.fg_state.variable_group.keys:
                variable = bp_state.ftov_msgs.fg_state.variable_group[variable_name]
                start_index = bp_state.ftov_msgs.fg_state.vars_to_starts[variable]
                variables_to_map_states[variable_name] = int(
                    jnp.argmax(beliefs[start_index : start_index + variable.num_states])
                )

            return variables_to_map_states
        else:
            variable = bp_state.ftov_msgs.fg_state.variable_group[variable_name]
            start_index = bp_state.ftov_msgs.fg_state.vars_to_starts[variable]
            return int(
                jnp.argmax(beliefs[start_index : start_index + variable.num_states])
            )

    return decode_map_states
