"""A module containing the core class to specify a Factor Graph."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from pgmax import utils
from pgmax.bp import infer
from pgmax.fg import fg_utils, groups, nodes


@dataclass(frozen=True, eq=False)
class FactorGraph:
    """Base class to represent a factor graph.

    Concrete factor graphs inherits from this class, and specifies get_evidence to generate
    the evidence array, and optionally init_msgs (default to initializing all messages to 0)

    Args:
        factor_groups: a tuple containing all the FactorGroups that are part of this FactorGraph

    Attributes:
        variables: tuple. contains involved variables
        factors: tuple. contains involved factors
        num_var_states: int. represents the sum of all variable states of all variables in the
            FactorGraph
        _vars_to_starts: MappingProxyType[nodes.Variable, int]. maps every variable to an int
            representing an index in the evidence array at which the first entry of the evidence
            for that particular variable should be placed.
    """

    factor_groups: Tuple[groups.FactorGroup, ...]

    def __post_init__(self):
        self.factors = sum(
            [factor_group.factors for factor_group in self.factor_groups], ()
        )
        self.variables = sum(
            [
                factor_group.variable_group.variables
                for factor_group in self.factor_groups
            ],
            (),
        )

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

    @utils.cached_property
    def wiring(self) -> nodes.EnumerationWiring:
        """Function to compile wiring for belief propagation.

        If wiring has already beeen compiled, do nothing.

        Returns:
            compiled wiring from each individual factor
        """
        wirings = [
            factor.compile_wiring(self._vars_to_starts) for factor in self.factors
        ]
        wiring = fg_utils.concatenate_enumeration_wirings(wirings)
        return wiring

    @utils.cached_property
    def factor_configs_log_potentials(self) -> np.ndarray:
        """Function to compile potential array for belief propagation..

        If potential array has already beeen compiled, do nothing.

        Returns:
            a jnp array representing the log of the potential function for each
                valid configuration
        """
        return np.concatenate(
            [factor.factor_configs_log_potentials for factor in self.factors]
        )

    def get_evidence(self, data: Any, context: Any = None) -> jnp.ndarray:
        """Function to generate evidence array. Need to be overwritten for concrete factor graphs

        Args:
            data: Data for generating evidence
            context: Optional context for generating evidence

        Returns:
            Array of shape (num_var_states,) representing the flattened evidence for each variable
        """
        raise NotImplementedError("get_evidence function needs to be implemented")

    def get_init_msgs(self, context: Any = None):
        """Function to initialize messages.

        By default it initializes all messages to 0. Can be overwritten to support
        customized initialization schemes

        Args:
            context: Optional context for initializing messages

        Returns:
            array of shape (num_edge_state,) representing initialized factor to variable
                messages
        """
        return jnp.zeros(self.wiring.var_states_for_edges.shape[0])

    def run_bp(
        self,
        num_iters: int,
        damping_factor: float,
        init_msgs: jnp.ndarray = None,
        msgs_context: Any = None,
        evidence_data: Any = None,
        evidence_context: Any = None,
    ) -> jnp.ndarray:
        """Function to perform belief propagation.

        Specifically, belief propagation is run on messages obtained from the self.get_init_msgs
        method for num_iters iterations and returns the resulting messages.

        Args:
            evidence: Array of shape (num_var_states,). This array contains the fully-flattened
                set of evidence messages for each variable node
            num_iters: The number of iterations for which to perform message passing
            damping_factor: The damping factor to use for message updates between one timestep and the next
            init_msgs: array of shape (num_edge_state,) representing the initial messaged on which to perform
                belief propagation. If this argument is none, messages are generated by calling self.get_init_msgs()
            msgs_context: Optional context for initializing messages
            evidence_data: Data for generating evidence
            evidence_context: Optional context for generating evidence

        Returns:
            an array of shape (num_edge_state,) that contains the message values after running BP for num_iters iterations
        """
        # Retrieve the necessary data structures from the compiled self.wiring and
        # convert these to jax arrays.
        if init_msgs is not None:
            msgs = init_msgs
        else:
            msgs = self.get_init_msgs(msgs_context)

        wiring = jax.device_put(self.wiring)
        factor_configs_log_potentials = jax.device_put(
            self.factor_configs_log_potentials
        )
        evidence = self.get_evidence(evidence_data, evidence_context)
        max_msg_size = int(jnp.max(wiring.edges_num_states))

        # Normalize the messages to ensure the maximum value is 0.
        msgs = infer.normalize_and_clip_msgs(
            msgs, wiring.edges_num_states, max_msg_size
        )
        num_val_configs = int(wiring.factor_configs_edge_states[-1, 0]) + 1

        @jax.jit
        def message_passing_step(msgs, _):
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
                factor_configs_log_potentials,
                num_val_configs,
            )
            # Use the results of message passing to perform damping and
            # update the factor to variable messages
            delta_msgs = ftov_msgs - msgs
            msgs = msgs + (1 - damping_factor) * delta_msgs
            # Normalize and clip these damped, updated messages before returning
            # them.
            msgs = infer.normalize_and_clip_msgs(
                msgs,
                wiring.edges_num_states,
                max_msg_size,
            )
            return msgs, None

        msgs_after_bp, _ = jax.lax.scan(message_passing_step, msgs, None, num_iters)

        return msgs_after_bp

    def decode_map_states(
        self, msgs: jnp.ndarray, evidence_data: Any = None, evidence_context: Any = None
    ) -> Dict[nodes.Variable, int]:
        """Function to compute the output of MAP inference on input messages.

        The final states are computed based on evidence obtained from the self.get_evidence
        method as well as the internal wiring.

        Args:
            msgs: an array of shape (num_edge_state,) that correspond to messages to perform inference
                upon
            evidence_data: Data for generating evidence
            evidence_context: Optional context for generating evidence

        Returns:
            a dictionary mapping variables to their MAP state
        """
        # NOTE: Having to regenerate the evidence here is annoying - there must be a better way to handle evidence and
        # message initialization.
        evidence = self.get_evidence(evidence_data, evidence_context)
        var_states_for_edges = jax.device_put(self.wiring.var_states_for_edges)
        final_var_states = evidence.at[var_states_for_edges].add(msgs)
        var_to_map_dict = {}
        final_var_states_np = np.array(final_var_states)
        for var in self.variables:
            start_index = self._vars_to_starts[var]
            var_to_map_dict[var] = np.argmax(
                final_var_states_np[start_index : start_index + var.num_states]
            )
        return var_to_map_dict
