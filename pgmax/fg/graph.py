from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import pgmax.bp.bp_utils as bp_utils
import pgmax.bp.infer as infer
from pgmax.fg import fg_utils, nodes


@dataclass
class FactorGraph:
    """Base class for factor graph
    Concrete factor graphs inherits from this class, and specifies get_evidence to generate
    the evidence array, and optionally init_msgs (default to initializing all messages to 0)

    Args:
        variables: List of involved variables
        factors: List of involved factors
    """

    variables: Sequence[nodes.Variable]
    factors: Sequence[nodes.EnumerationFactor]

    def __post_init__(self):
        vars_num_states_cumsum = np.insert(
            np.array(
                [variable.num_states for variable in self.variables], dtype=int
            ).cumsum(),
            0,
            0,
        )
        self._vars_to_starts = MappingProxyType(
            {
                variable: vars_num_states_cumsum[vv]
                for vv, variable in enumerate(self.variables)
            }
        )
        self.num_var_states = vars_num_states_cumsum[-1]

    def compile_wiring(self) -> None:
        """Compile wiring for belief propagation inference using JAX"""
        wirings = [
            factor.compile_wiring(self._vars_to_starts) for factor in self.factors
        ]
        self._wiring = fg_utils.concatenate_enumeration_wirings(wirings)

    def get_wiring(self) -> nodes.EnumerationWiring:
        """Compile (if not already done) and return wiring for belief propagation inference using JAX"""
        if not hasattr(self, "_wiring"):
            self.compile_wiring()
        return self._wiring

    def init_evidence(self, data: Any, context: Any = None) -> jnp.ndarray:
        """Function to generate evidence array. Need to be overwritten for concrete factor graphs

        Args:
            data: Data for generating evidence
            context: Optional context for generating evidence

        Returns:
            None, but must set the self._evidence attribute to a jnp.array of shape (num_var_states,)
        """
        self._evidence = None
        raise NotImplementedError("get_evidence function needs to be implemented")

    def output_inference(
        self, final_var_states: jnp.ndarray, context: Any = None
    ) -> Any:
        """Function to take the result of message passing and output the inference result for
            each variable

        Args:
            final_var_states: an array of shape (num_var_states,) that is the result of belief
                propagation
            context: Optional context for using this array

        Returns:
            Any data-structure of the user's choice that contains the MAP result derived from
                final_var_states
        """
        raise NotImplementedError("output_inference function needs to be implemented")

    def init_msgs(self, context: Any = None):
        """Initialize messages. By default it initializes all messages to 0.
        Can be overwritten to support customized initialization schemes

        Args:
            context: Optional context for initializing messages

        Returns:
            None, but must set the self._init_msgs attribute to a jnp.array of shape (num_edge_state,)
        """
        if not hasattr(self, "_wiring"):
            self.compile_wiring()

        self._init_msgs = jnp.zeros(self._wiring.var_states_for_edges.shape[0])

    def run_bp_and_infer(
        self,
        num_iters: int,
        damping_factor: float,
    ) -> jnp.ndarray:
        """
        performs belief propagation given the specified data-structures for num_iters iterations and returns the
        output of inference

        Args:
            evidence: Array of shape (num_var_states,). This array contains the fully-flattened
                set of evidence messages for each variable node
            num_iters: The number of iterations for which to perform message passing
            damping_factor: The damping factor to use for message updates between one timestep and the next

        Returns:
            an array of the same shape as evidence that contains the values for each state for each variables after BP
        """
        if not hasattr(self, "_init_msgs"):
            raise RuntimeError(
                "Please call the init_msgs method on your factor graph before attempting to call"
                + "the run_bp_and_infer method. Also, ensure the init_msgs_method sets the self._init_msgs attribute"
            )

        if not hasattr(self, "_evidence") or self._evidence is None:
            raise RuntimeError(
                "Please call the get_evidence method on your factor graph before attempting to call"
                + "the run_bp_and_infer method. Also, ensure the get_evidence sets the self._evidence attribute"
            )

        if not hasattr(self, "_wiring"):
            self.compile_wiring()

        # Retrieve the necessary data structures from the compiled self._wiring and
        # convert these to jax arrays.
        msgs = self._init_msgs
        evidence = self._evidence
        edges_num_states = jax.device_put(self._wiring.edges_num_states)
        var_states_for_edges = jax.device_put(self._wiring.var_states_for_edges)
        factor_configs_edge_states = jax.device_put(
            self._wiring.factor_configs_edge_states
        )
        max_msg_size = int(jnp.max(edges_num_states))

        # Normalize the messages to ensure the maximum value is 0.
        normalized_msgs = msgs - jnp.repeat(
            bp_utils.segment_max_opt(msgs, edges_num_states, max_msg_size),
            edges_num_states,
            total_repeat_length=msgs.shape[0],
        )
        num_val_configs = int(factor_configs_edge_states[-1, 0])

        def message_passing_loop(
            normalized_msgs,
            evidence,
            var_states_for_edges,
            edges_num_states,
            max_msg_size,
            damping_factor,
            num_iters,
        ):
            "Function wrapper that leverages jax.lax.scan to efficiently perform belief propagation"

            def message_passing_step(original_msgs, _):
                # Compute new variable to factor messages by message passing
                vtof_msgs = infer.pass_var_to_fac_messages(
                    original_msgs,
                    evidence,
                    var_states_for_edges,
                )
                # Compute new factor to variable messages by message passing
                ftov_msgs = infer.pass_fac_to_var_messages(
                    vtof_msgs,
                    factor_configs_edge_states,
                    num_val_configs,
                )
                # Use the results of message passing to perform damping and
                # update the factor to variable messages
                delta_msgs = ftov_msgs - original_msgs
                damped_updated_msgs = original_msgs + (
                    (1 - damping_factor) * delta_msgs
                )
                # Normalize and clip these damped, updated messages before returning
                # them.
                normalized_and_clipped_msgs = infer.normalize_and_clip_msgs(
                    damped_updated_msgs,
                    edges_num_states,
                    max_msg_size,
                )
                return normalized_and_clipped_msgs, None

            final_msgs, _ = jax.lax.scan(
                message_passing_step, normalized_msgs, None, num_iters
            )
            return final_msgs

        msgs_after_bp = message_passing_loop(
            normalized_msgs,
            evidence,
            var_states_for_edges,
            edges_num_states,
            max_msg_size,
            damping_factor,
            num_iters,
        )

        # Compute the final states for every variable and return these
        final_var_states = evidence.at[var_states_for_edges].add(msgs_after_bp)

        return final_var_states
