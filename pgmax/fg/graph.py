"""A module containing the core class to specify a Factor Graph."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax.bp import infer
from pgmax.fg import fg_utils, groups, nodes


@dataclass
class FactorGraph:
    """Class for representing a factor graph

    Args:
        variable_groups: A container containing multiple VariableGroups, or a CompositeVariableGroup.
            If not a CompositeVariableGroup, supported containers include mapping, sequence and single
            VariableGroup.
            For a mapping, the keys of the mapping are used to index the variable groups.
            For a sequence, the indices of the sequence are used to index the variable groups.
            Note that a CompositeVariableGroup will be created from this input, and the individual
            VariableGroups will need to be accessed by indexing this.
        evidence_default: string representing a setting that specifies the default evidence value for
            any variable whose evidence was not explicitly specified using 'set_evidence'

    Attributes:
        _composite_variable_group: CompositeVariableGroup. contains all involved VariableGroups
        _factor_groups: List of added factor groups
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
        Mapping[Hashable, groups.VariableGroup],
        Sequence[groups.VariableGroup],
        groups.VariableGroup,
    ]
    evidence_default_mode: str = "zeros"

    def __post_init__(self):
        if isinstance(self.variable_groups, groups.CompositeVariableGroup):
            self._composite_variable_group = self.variable_groups
        elif isinstance(self.variable_groups, groups.VariableGroup):
            self._composite_variable_group = groups.CompositeVariableGroup(
                [self.variable_groups]
            )
        else:
            self._composite_variable_group = groups.CompositeVariableGroup(
                self.variable_groups
            )

        vars_num_states_cumsum = np.insert(
            np.array(
                [
                    variable.num_states
                    for variable in self._composite_variable_group.variables
                ],
                dtype=int,
            ).cumsum(),
            0,
            0,
        )
        self._vars_to_starts = MappingProxyType(
            {
                variable: vars_num_states_cumsum[vv]
                for vv, variable in enumerate(self._composite_variable_group.variables)
            }
        )
        self.num_var_states = vars_num_states_cumsum[-1]
        self._vars_to_evidence: Dict[nodes.Variable, np.ndarray] = {}
        self._factor_groups: List[groups.FactorGroup] = []
        self._named_factor_groups: Dict[Hashable, groups.FactorGroup] = {}
        self._total_factor_num_states: int = 0
        self._factor_group_to_starts: Dict[groups.FactorGroup, int] = {}

    def add_factors(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Function to add factors to this FactorGraph.

        Args:
            *args: optional sequence of arguments. If specified, and if there is no
                "factor_factory" key specified as part of the **kwargs, then these args
                are taken to specify the arguments to be used to instantiate an
                EnumerationFactor. If there is a "factor_factory" key, then these args
                are taken to specify the arguments to be used to construct the class
                specified by the "factor_factory" argument. Note that either *args or
                **kwargs must be specified.
            **kwargs: optional mapping of keyword arguments. If specified, and if there
                is no "factor_factory" key specified as part of this mapping, then these
                args are taken to specify the arguments to be used to instantiate an
                EnumerationFactor (specify a kwarg with the key 'keys' to indicate the
                indices of variables ot be indexed to create the EnumerationFactor).
                If there is a "factor_factory" key, then these args are taken to specify
                the arguments to be used to construct the class specified by the
                "factor_factory" argument. Note that either *args or **kwargs must be
                specified.
        """
        name = kwargs.pop("name", None)
        factor_factory = kwargs.pop("factor_factory", None)
        if factor_factory is not None:
            factor_group = factor_factory(
                self._composite_variable_group, *args, **kwargs
            )
        else:
            if len(args) > 0:
                new_args = list(args)
                new_args[0] = [args[0]]
                factor_group = groups.EnumerationFactorGroup(
                    self._composite_variable_group, *new_args, **kwargs
                )
            else:
                keys = kwargs.pop("keys")
                kwargs["connected_var_keys"] = [keys]
                factor_group = groups.EnumerationFactorGroup(
                    self._composite_variable_group, **kwargs
                )

        self._factor_groups.append(factor_group)
        self._factor_group_to_starts[factor_group] = self._total_factor_num_states
        self._total_factor_num_states += np.sum(factor_group.factor_num_states)
        if name is not None:
            self._named_factor_groups[name] = factor_group

    def get_factor(self, key: Any) -> Tuple[nodes.EnumerationFactor, int]:
        if key in self._named_factor_groups:
            if len(self._named_factor_groups[key].factors) != 1:
                raise ValueError(
                    f"Invalid factor key {key}. "
                    "Please provide a key for an individual factor, "
                    "not a factor group"
                )

            factor_group = self._named_factor_groups[key]
            factor = factor_group.factors[0]
            start = self._factor_group_to_starts[factor_group]
        else:
            if not (isinstance(key, tuple) and key[0] in self._named_factor_groups):
                raise ValueError(
                    f"Invalid factor key {key}. "
                    "Please provide a key either for an individual named factor, "
                    "or a tuple specifying name of the factor group and index "
                    "of individual factors"
                )

            factor_group = self._named_factor_groups[key[0]]
            factor = factor_group[key[1:]]
            start = self._factor_group_to_starts[factor_group] + np.sum(
                factor_group.factor_num_states[: factor_group.factors.index(factor)]
            )

        return factor, start

    @property
    def wiring(self) -> nodes.EnumerationWiring:
        """Function to compile wiring for belief propagation.

        If wiring has already beeen compiled, do nothing.

        Returns:
            compiled wiring from each individual factor
        """
        wirings = [
            factor_group.compile_wiring(self._vars_to_starts)
            for factor_group in self._factor_groups
        ]
        wiring = fg_utils.concatenate_enumeration_wirings(wirings)
        return wiring

    @property
    def factor_configs_log_potentials(self) -> np.ndarray:
        """Function to compile potential array for belief propagation..

        If potential array has already beeen compiled, do nothing.

        Returns:
            a jnp array representing the log of the potential function for each
                valid configuration
        """
        return np.concatenate(
            [
                factor_group.factor_group_log_potentials
                for factor_group in self._factor_groups
            ]
        )

    @property
    def evidence(self) -> np.ndarray:
        """Function to generate evidence array. Need to be overwritten for concrete factor graphs

        Returns:
            Array of shape (num_var_states,) representing the flattened evidence for each variable

        Raises:
            NotImplementedError: if self.evidence_default is a string that is not listed
        """
        if self.evidence_default_mode == "zeros":
            evidence = np.zeros(self.num_var_states)
        elif self.evidence_default_mode == "random":
            evidence = np.random.gumbel(size=self.num_var_states)
        else:
            raise NotImplementedError(
                f"evidence_default_mode {self.evidence_default_mode} is not yet implemented"
            )

        for var, evidence_val in self._vars_to_evidence.items():
            start_index = self._vars_to_starts[var]
            evidence[start_index : start_index + var.num_states] = evidence_val

        return evidence

    @property
    def factors(self) -> Tuple[nodes.EnumerationFactor, ...]:
        """List of individual factors in the factor graph"""
        return sum([factor_group.factors for factor_group in self._factor_groups], ())

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

    def set_evidence(
        self,
        key: Union[Tuple[Any, ...], Any],
        evidence: Union[Dict[Hashable, np.ndarray], np.ndarray],
    ) -> None:
        """Function to update the evidence for variables in the FactorGraph.

        Args:
            key: tuple that represents the index into the CompositeVariableGroup
                (self._composite_variable_group) that is created when the FactorGraph is instantiated. Note that
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
        if key in self._composite_variable_group.container_keys:
            self._vars_to_evidence.update(
                self._composite_variable_group.variable_group_container[
                    key
                ].get_vars_to_evidence(evidence)
            )
        else:
            self._vars_to_evidence[self._composite_variable_group[key]] = evidence

    def run_bp(
        self,
        num_iters: int,
        damping_factor: float,
        init_msgs: jnp.ndarray = None,
        msgs_context: Any = None,
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
        evidence = jax.device_put(self.evidence)
        factor_configs_log_potentials = jax.device_put(
            self.factor_configs_log_potentials
        )
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

    def decode_map_states(self, msgs: jnp.ndarray) -> Dict[Tuple[Any, ...], int]:
        """Function to computes the output of MAP inference on input messages.

        The final states are computed based on evidence obtained from the self.get_evidence
        method as well as the internal wiring.

        Args:
            msgs: an array of shape (num_edge_state,) that correspond to messages to perform inference
                upon

        Returns:
            a dictionary mapping each variable key to the MAP states of the corresponding variable
        """
        var_states_for_edges = jax.device_put(self.wiring.var_states_for_edges)
        evidence = jax.device_put(self.evidence)
        final_var_states = evidence.at[var_states_for_edges].add(msgs)
        var_key_to_map_dict: Dict[Tuple[Any, ...], int] = {}
        final_var_states_np = np.array(final_var_states)
        for var_key in self._composite_variable_group.keys:
            var = self._composite_variable_group[var_key]
            start_index = self._vars_to_starts[var]
            var_key_to_map_dict[var_key] = np.argmax(
                final_var_states_np[start_index : start_index + var.num_states]
            )
        return var_key_to_map_dict


@dataclass
class Messages:
    factor_graph: FactorGraph
    default_mode: str = "zeros"

    def __post_init__(self):
        self._message_updates: Dict[int, jnp.ndarray] = {}

    def __getitem__(self, keys: Tuple[Any, Any]) -> jnp.ndarray:
        if not (
            isinstance(keys, tuple)
            and len(keys) == 2
            and keys[1] in self.factor_graph._composite_variable_group.keys
        ):
            raise ValueError("")

    @typing.overload
    def __setitem__(
        self,
        keys: Tuple[Any, Any],
        data: Union[np.ndarray, jnp.ndarray],
    ) -> None:
        """Setting messages from a factor to a variable

        Args:
            keys: A tuple of length 2
                keys[0] is the key of the factor
                keys[1] is the key of the variable
            data: An array containing messages from factor keys[0]
                to variable keys[1]
        """

    @typing.overload
    def __setitem__(
        self,
        keys: Any,
        data: Union[np.ndarray, jnp.ndarray],
    ) -> None:
        """Spreading beliefs at a variable to all connected factors

        Args:
            keys: The key of the variable
            data: An array containing the beliefs to be spread at variable keys
        """

    def __setitem__(self, keys, data) -> None:
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and keys[1] in self.factor_graph._composite_variable_group.keys
        ):
            factor, start = self.factor_graph.get_factor(keys[0])
            variable = self.factor_graph._composite_variable_group[keys[1]]
            self._message_updates[
                start
                + np.sum(factor.edges_num_states[: factor.variables.index(variable)])
            ] = data
        elif keys in self.factor_graph._composite_variable_group.keys:
            pass
        else:
            raise ValueError("")

    @property
    def data(self) -> jnp.ndarray:
        pass
