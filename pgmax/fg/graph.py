"""A module containing the core class to specify a Factor Graph."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax.bp import infer
from pgmax.fg import fg_utils, groups, nodes


@dataclass
class FactorGraph:
    """Base class to represent a factor graph.

    Concrete factor graphs inherits from this class, and specifies get_evidence to generate
    the evidence array, and optionally init_msgs (default to initializing all messages to 0)

    Args:
        variable_groups: A container containing multiple variable groups.
            Supported containers include mapping, sequence and single VariableGroup.
            For a mapping, the keys of the mapping are used to index the variable groups.
            For a sequence, the indices of the sequence are used to index the variable groups.

    Attributes:
        _comp_var_group: CompositeVariableGroup. contains all involved VariableGroups
        _factors: list. contains all involved factors
        num_var_states: int. represents the sum of all variable states of all variables in the
            FactorGraph
        _vars_to_starts: MappingProxyType[nodes.Variable, int]. maps every variable to an int
            representing an index in the evidence array at which the first entry of the evidence
            for that particular variable should be placed.
        _vars_to_evidence: Dict[nodes.Variable, np.ndarray]. maps every variable to an np.ndarray
            representing the evidence for that variable
        _vargroups_set: Set[groups.VariableGroup]. keeps track of all the VariableGroup's that have
            been added to this FactorGraph
    """

    variable_groups: Union[
        Mapping[Any, groups.VariableGroup],
        Sequence[groups.VariableGroup],
        groups.VariableGroup,
    ]

    def __post_init__(self):
        if isinstance(self.variable_groups, groups.CompositeVariableGroup):
            self._comp_var_group = self.variable_groups
        elif isinstance(self.variable_groups, groups.VariableGroup):
            self._comp_var_group = groups.CompositeVariableGroup([self.variable_groups])
        else:
            self._comp_var_group = groups.CompositeVariableGroup(self.variable_groups)

        vars_num_states_cumsum = np.insert(
            np.array(
                [variable.num_states for variable in self._comp_var_group.variables],
                dtype=int,
            ).cumsum(),
            0,
            0,
        )
        self._vars_to_starts = MappingProxyType(
            {
                variable: vars_num_states_cumsum[vv]
                for vv, variable in enumerate(self._comp_var_group.variables)
            }
        )
        self.num_var_states = vars_num_states_cumsum[-1]

        self._vars_to_evidence: Dict[nodes.Variable, np.ndarray] = {}

        self._factors: List[nodes.EnumerationFactor] = []

    def add_factors(self, new_factors: Sequence[nodes.EnumerationFactor]) -> None:
        """Function to add factors to this FactorGraph.

        Args:
            new_factors: a sequence of factors to be added to the graph

        Raises:
            ValueError: if new_factors is empty
        """

        if len(new_factors) == 0:
            raise ValueError("No Factors or FactorGroups to add!")
        self._factors.extend(new_factors)

    def add_factor_groups(
        self, new_factor_groups: Sequence[groups.FactorGroup]
    ) -> None:
        """Function to add FactorGroups to this FactorGraph.

        Args:
            new_factor_groups: a sequence of FactorGroups to be added to the graph

        Raises:
            ValueError: if new_factor_groups is empty
        """
        if len(new_factor_groups) == 0:
            raise ValueError("No Factors or FactorGroups to add!")
        for factor_group in new_factor_groups:
            self._factors.extend(factor_group.factors)

    @property
    def curr_wiring(self) -> nodes.EnumerationWiring:
        """Function to compile wiring for belief propagation.

        If wiring has already beeen compiled, do nothing.

        Returns:
            compiled wiring from each individual factor
        """
        wirings = [
            factor.compile_wiring(self._vars_to_starts) for factor in self._factors
        ]
        wiring = fg_utils.concatenate_enumeration_wirings(wirings)
        return wiring

    @property
    def curr_factor_configs_log_potentials(self) -> np.ndarray:
        """Function to compile potential array for belief propagation..

        If potential array has already beeen compiled, do nothing.

        Returns:
            a jnp array representing the log of the potential function for each
                valid configuration
        """
        return np.concatenate(
            [factor.factor_configs_log_potentials for factor in self._factors]
        )

    def get_curr_evidence(self, evidence_default_mode) -> np.ndarray:
        """Function to generate evidence array. Need to be overwritten for concrete factor graphs

        Args:
            evidence_default: a string representing a setting that specifies the default evidence value for any variable
                whose evidence was not explicitly specified using 'update_evidence'. Currently, the following modes are
                implemented
                - 'zeros': set unspecified nodes to 0

        Returns:
            Array of shape (num_var_states,) representing the flattened evidence for each variable

        Raises:
            NotImplementedError: if evidence_default is a string that is not listed
        """
        evidence = np.zeros(self.num_var_states)

        for var in self._comp_var_group.variables:
            start_index = self._vars_to_starts[var]
            if self._vars_to_evidence.get(var) is not None:
                evidence[
                    start_index : start_index + var.num_states
                ] = self._vars_to_evidence[var]
            else:
                if evidence_default_mode == "zeros":
                    evidence[start_index : start_index + var.num_states] = np.zeros(
                        var.num_states
                    )
                else:
                    raise NotImplementedError(
                        f"evidence_default_mode {evidence_default_mode} is not yet implemented"
                    )

        return evidence

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
        return jnp.zeros(self.curr_wiring.var_states_for_edges.shape[0])

    def update_evidence(
        self,
        variable_group: groups.VariableGroup,
        evidence_values: Union[Dict[Any, np.ndarray], np.ndarray],
    ):
        """Function to update the evidence for variables in the FactorGraph.

        Args:
            variable_group: a VariableGroup that contains all the variables to be updated.
            evidence_values: a container for np.ndarrays representing the evidence
                Currently supported containers are:
                - an np.ndarray: if variable_group is of type NDVariableArray, then evidence_values
                can simply be an np.ndarray with num_var_array_dims + 1 dimensions where
                num_var_array_dims is the number of dimensions of the NDVariableArray, and the
                +1 represents a dimension (that should be the final dimension) for the evidence.
                Note that the size of the final dimension should be the same as
                variable_group.variable_size.
                - a dictionary: if variable_group is of type GenericVariableGroup, then evidence_values
                must be a dictionary mapping keys of variable_group to np.ndarrays of evidence values.
                Note that each np.ndarray in the dictionary values must have the same size as
                variable_group.variable_size.

        Raises:
            ValueError:
                - if variable_group was not passed into this FactorGraph as part of self.variable_groups
                - or if variable_group is an NDVariableArray and:
                    - evidence_values is anything other than an np.ndarray or
                    - the dimensions of evidence_values aren't num_var_array_dims + 1 or
                    - the last dimension of evidence_values doesn't have the same size as
                    variable_group.variable_size,
                - or if variable_group is a GenericVariableGroup and:
                    - evidence_values is not a dictionary or
                    - there is a value in the evidence_values dictionary with array size that's
                    not the same as variable_group.variable_size
            NotImplementedError: if variable_group is neither a NDVariableArray or GenericVariableGroup
        """
        if isinstance(variable_group, groups.NDVariableArray):
            if not isinstance(evidence_values, np.ndarray):
                raise ValueError(
                    "Expected np.ndarray of evidence_values for variable_group of type NDVariableArray"
                )
            if evidence_values.shape != variable_group.shape + tuple(
                [variable_group.variable_size]
            ):
                raise ValueError(
                    f"Expected evidence values to have shape {variable_group.shape + tuple([variable_group.variable_size])}"
                    + f" instead, got {evidence_values.shape}"
                )
            for idx, _ in np.ndenumerate(evidence_values[..., 0]):
                self._vars_to_evidence[variable_group[idx]] = evidence_values[idx]

        elif isinstance(variable_group, groups.GenericVariableGroup):
            if not isinstance(evidence_values, Dict):
                raise ValueError(
                    "Expected dict of evidence_values for variable_group of type NDVariableArray"
                )
            for idx, evidence_val in evidence_values.items():
                if evidence_val.shape != (variable_group.variable_size,):
                    raise ValueError(
                        f"evidence_values contains a value of shape {evidence_val.shape}, but expected shape {(variable_group.variable_size,)}"
                    )
                self._vars_to_evidence[variable_group[idx]] = evidence_val

        else:
            raise NotImplementedError

    def run_bp(
        self,
        num_iters: int,
        damping_factor: float,
        init_msgs: jnp.ndarray = None,
        msgs_context: Any = None,
        evidence_default_mode: str = "zero",
    ) -> jnp.ndarray:
        """Function to perform belief propagation.

        Specifically, belief propagation is run on messages obtained from the self.get_init_msgs
        method for num_iters iterations and returns the resulting messages.

        Args:
            num_iters: The number of iterations for which to perform message passing
            damping_factor: The damping factor to use for message updates between one timestep and the next
            init_msgs: array of shape (num_edge_state,) representing the initial messaged on which to perform
                belief propagation. If this argument is none, messages are generated by calling self.get_init_msgs()
            msgs_context: Optional context for initializing messages
            evidence_default: a string representing a setting that specifies the default evidence value for any variable
                whose evidence was not explicitly specified using 'update_evidence'

        Returns:
            an array of shape (num_edge_state,) that contains the message values after running BP for num_iters iterations
        """
        # Retrieve the necessary data structures from the compiled self.wiring and
        # convert these to jax arrays.
        if init_msgs is not None:
            msgs = init_msgs
        else:
            msgs = self.get_init_msgs(msgs_context)

        wiring = jax.device_put(self.curr_wiring)
        evidence = jax.device_put(self.get_curr_evidence(evidence_default_mode))
        factor_configs_log_potentials = jax.device_put(
            self.curr_factor_configs_log_potentials
        )
        # evidence = self.get_evidence(evidence_data, evidence_context)
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
        self, msgs: jnp.ndarray, evidence_default_mode: str = "zero"
    ) -> Dict[nodes.Variable, int]:
        """Function to computes the output of MAP inference on input messages.

        The final states are computed based on evidence obtained from the self.get_evidence
        method as well as the internal wiring.

        Args:
            msgs: an array of shape (num_edge_state,) that correspond to messages to perform inference
                upon
            evidence_default: a string representing a setting that specifies the default evidence value for any variable
                whose evidence was not explicitly specified using 'update_evidence'

        Returns:
            a dictionary mapping variables to their MAP state
        """
        var_states_for_edges = jax.device_put(self.curr_wiring.var_states_for_edges)
        evidence = jax.device_put(self.get_curr_evidence(evidence_default_mode))
        final_var_states = evidence.at[var_states_for_edges].add(msgs)
        var_to_map_dict = {}
        final_var_states_np = np.array(final_var_states)
        for var in self._comp_var_group.variables:
            start_index = self._vars_to_starts[var]
            var_to_map_dict[var] = np.argmax(
                final_var_states_np[start_index : start_index + var.num_states]
            )
        return var_to_map_dict
