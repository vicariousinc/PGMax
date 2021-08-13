"""A module containing the core class to specify a Factor Graph."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax.bp import infer
from pgmax.fg import fg_utils, groups, nodes

from typing import Optional  # isort:skip


@dataclass
class FactorGraph:
    """Base class to represent a factor graph.

    Concrete factor graphs inherits from this class, and specifies get_evidence to generate
    the evidence array, and optionally init_msgs (default to initializing all messages to 0)

    Args:
        variable_groups: A container containing multiple VariableGroups, or a CompositeVariableGroup.
            If not a CompositeVariableGroup, supported containers include mapping, sequence and single
            VariableGroup.
            For a mapping, the keys of the mapping are used to index the variable groups.
            For a sequence, the indices of the sequence are used to index the variable groups.
            Note that a CompositeVariableGroup will be created from this input, and the individual
            VariableGroups will need to be accessed by indexing this.

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

    def add_factors(
        self,
        variable_group_key: Optional[Tuple[Any, ...]],
        FactorFactory: Callable,
        kwargs: Dict[str, Any],
    ) -> None:
        """Function to add factors to this FactorGraph.

        Args:
            variable_group_key: optional tuple that represents the index into the CompositeVariableGroup
                (self._comp_var_group) that is created when the FactorGraph is instantiated. If this is
                set to None, then it is assumed that the entire CompositeVariableGroup is being indexed.
            FactorFactory: a callable class definition. This class must be a sub-class of FactorGroup
                and contain a string key for every keyword argument (EXCEPT the variable_group keyword)
                that might be necessary to instantiate this class.
            kwargs: a dictionary of keyword args

        Raises:
            ValueError: if new_factors is empty
        """
        if variable_group_key is None:
            variable_group = self._comp_var_group
        else:
            variable_group = self._comp_var_group[variable_group_key]
        kwargs["variable_group"] = variable_group
        new_factor = FactorFactory(**kwargs)
        if not isinstance(new_factor, groups.FactorGroup):
            raise ValueError(
                f"The FactorFactory did not instantiate a subclass of FactorGroup as expected. Instead, the class was {type(new_factor)}"
            )
        self._factors.extend(new_factor.factors)

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

    def get_curr_evidence(self, evidence_default_mode: str) -> np.ndarray:
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
        key: Union[Tuple[Any, ...], Any],
        evidence: Union[Dict[Any, np.ndarray], np.ndarray],
    ):
        """Function to update the evidence for variables in the FactorGraph.

        Args:
            key: tuple that represents the index into the CompositeVariableGroup
                (self._comp_var_group) that is created when the FactorGraph is instantiated. Note that
                this can be an index referring to an entire VariableGroup (in which case, the evidence
                is set for the entire VariableGroup at once), or to an individual Variable within the
                CompositeVariableGroup.
            evidence: a container for np.ndarrays representing the evidence
                Currently supported containers are:
                - an np.ndarray: if key indexes an NDVariableArray, then evidence_values
                can simply be an np.ndarray with num_var_array_dims + 1 dimensions where
                num_var_array_dims is the number of dimensions of the NDVariableArray, and the
                +1 represents a dimension (that should be the final dimension) for the evidence.
                Note that the size of the final dimension should be the same as
                variable_group.variable_size. if key indexes a particular variable, then this array
                must be of the same size as variable.num_states
                - a dictionary: if key indexes a GenericVariableGroup, then evidence_values
                must be a dictionary mapping keys of variable_group to np.ndarrays of evidence values.
                Note that each np.ndarray in the dictionary values must have the same size as
                variable_group.variable_size.
        """
        if key in self._comp_var_group.container_keys:
            self._vars_to_evidence.update(
                self._comp_var_group.variable_group_container[key].get_vars_to_evidence(
                    evidence
                )
            )
        else:
            self._vars_to_evidence[self._comp_var_group[key]] = evidence

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
