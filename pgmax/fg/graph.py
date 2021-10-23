"""A module containing the core class to specify a Factor Graph."""

from __future__ import annotations

import collections
import typing
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    Hashable,
    Mapping,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np

from pgmax.bp import infer
from pgmax.fg import fg_utils, groups, nodes
from pgmax.utils import cached_property


@dataclass
class FactorGraph:
    """Class for representing a factor graph

    Args:
        variables: A single VariableGroup or a container containing variable groups.
            If not a single VariableGroup, supported containers include mapping and sequence.
            For a mapping, the keys of the mapping are used to index the variable groups.
            For a sequence, the indices of the sequence are used to index the variable groups.
            Note that if not a single VariableGroup, a CompositeVariableGroup will be created from
            this input, and the individual VariableGroups will need to be accessed by indexing.
        messages_default_mode: default mode for initializing messages.
            Allowed values are "zeros" and "random".
        evidence_default_mode: default mode for initializing evidence.
            Allowed values are "zeros" and "random".
            Any variable whose evidence was not explicitly specified using 'set_evidence'

    Attributes:
        _variable_group: VariableGroup. contains all involved VariableGroups
        num_var_states: int. represents the sum of all variable states of all variables in the
            FactorGraph
        _vars_to_starts: MappingProxyType[nodes.Variable, int]. maps every variable to an int
            representing an index in the evidence array at which the first entry of the evidence
            for that particular variable should be placed.
        _vargroups_set: Set[groups.VariableGroup]. keeps track of all the VariableGroup's that have
            been added to this FactorGraph
        _named_factor_groups: Dict[Hashable, groups.FactorGroup]. A dictionary mapping the names of
            named factor groups to the corresponding factor groups.
            We only support setting messages from factors within explicitly named factor groups
            to connected variables.
        _total_factor_num_states: int. Current total number of edge states for the added factors.
        _factor_group_to_starts: Dict[groups.FactorGroup, int]. Maps a factor group to its
            corresponding starting index in the flat message array.
    """

    variables: Union[
        Mapping[Hashable, groups.VariableGroup],
        Sequence[groups.VariableGroup],
        groups.VariableGroup,
    ]
    messages_default_mode: str = "zeros"
    evidence_default_mode: str = "zeros"

    def __post_init__(self):
        if isinstance(self.variables, groups.VariableGroup):
            self._variable_group = self.variables
        else:
            self._variable_group = groups.CompositeVariableGroup(self.variables)

        vars_num_states_cumsum = np.insert(
            np.array(
                [variable.num_states for variable in self._variable_group.variables],
                dtype=int,
            ).cumsum(),
            0,
            0,
        )
        self._vars_to_starts = MappingProxyType(
            {
                variable: vars_num_states_cumsum[vv]
                for vv, variable in enumerate(self._variable_group.variables)
            }
        )
        self.num_var_states = vars_num_states_cumsum[-1]
        self._named_factor_groups: Dict[Hashable, groups.FactorGroup] = {}
        self._total_factor_num_states: int = 0
        self._factor_group_to_starts: OrderedDict[
            groups.FactorGroup, int
        ] = collections.OrderedDict()
        self._variables_to_factors: OrderedDict[
            FrozenSet, nodes.EnumerationFactor
        ] = collections.OrderedDict()
        self._factor_to_starts: OrderedDict[
            nodes.EnumerationFactor, int
        ] = collections.OrderedDict()

    def __hash__(self) -> int:
        return hash(self.factor_groups)

    def add_factor(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Function to add factor/factor group to this FactorGraph.

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
                "factor_factory" argument.
                If there is a "name" key, we add the added factor/factor group to the list
                of named factors within the factor graph.
                Note that either *args or **kwargs must be specified.
        """
        name = kwargs.pop("name", None)
        if name in self._named_factor_groups:
            raise ValueError(
                f"A factor group with the name {name} already exists. Please choose a different name!"
            )

        factor_factory = kwargs.pop("factor_factory", None)
        if factor_factory is not None:
            factor_group = factor_factory(self._variable_group, *args, **kwargs)
        else:
            if len(args) > 0:
                new_args = list(args)
                new_args[0] = [args[0]]
                factor_group = groups.EnumerationFactorGroup(
                    self._variable_group, *new_args, **kwargs
                )
            else:
                keys = kwargs.pop("keys")
                kwargs["connected_var_keys"] = [keys]
                factor_group = groups.EnumerationFactorGroup(
                    self._variable_group, **kwargs
                )

        self._factor_group_to_starts[factor_group] = self._total_factor_num_states
        factor_num_states_cumsum = np.insert(
            factor_group.factor_num_states.cumsum(), 0, 0
        )
        for vv, variables in enumerate(factor_group._variables_to_factors):
            if variables in self._variables_to_factors:
                raise ValueError(
                    f"A factor involving variables {variables} already exists. Please merge the corresponding factors."
                )

            factor = factor_group._variables_to_factors[variables]
            self._variables_to_factors[variables] = factor
            self._factor_to_starts[factor] = (
                self._factor_group_to_starts[factor_group]
                + factor_num_states_cumsum[vv]
            )

        self._total_factor_num_states += factor_num_states_cumsum[-1]
        if name is not None:
            self._named_factor_groups[name] = factor_group

    @cached_property
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

    @cached_property
    def log_potentials(self) -> np.ndarray:
        """Function to compile potential array for belief propagation..

        If potential array has already beeen compiled, do nothing.

        Returns:
            a jnp array representing the log of the potential function for each
                valid configuration
        """
        return np.concatenate(
            [
                factor_group.factor_group_log_potentials
                for factor_group in self.factor_groups
            ]
        )

    @cached_property
    def factors(self) -> Tuple[nodes.EnumerationFactor, ...]:
        """Tuple of individual factors in the factor graph"""
        return tuple(self._variables_to_factors.values())

    @property
    def factor_groups(self) -> Tuple[groups.FactorGroup, ...]:
        """Tuple of factor groups in the factor graph"""
        return tuple(self._factor_group_to_starts.keys())

    def get_init_msgs(self) -> Messages:
        """Function to initialize messages.

        Returns:
            Initialized messages
        """
        return Messages(
            ftov=FToVMessages(
                factor_graph=self, default_mode=self.messages_default_mode
            ),
            evidence=Evidence(
                factor_graph=self, default_mode=self.evidence_default_mode
            ),
        )

    def run_bp(
        self,
        num_iters: int,
        damping_factor: float,
        init_msgs: Optional[Messages] = None,
    ) -> Messages:
        """Function to perform belief propagation.

        Specifically, belief propagation is run for num_iters iterations and
        returns the resulting messages.

        Args:
            num_iters: The number of iterations for which to perform message passing
            damping_factor: The damping factor to use for message updates between one timestep and the next
            init_msgs: Initial messages to start the belief propagation.
                If None, construct init_msgs by calling self.get_init_msgs()

        Returns:
            ftov messages after running BP for num_iters iterations
        """
        # Retrieve the necessary data structures from the compiled self.wiring and
        # convert these to jax arrays.
        if init_msgs is None:
            init_msgs = self.get_init_msgs()

        msgs = jax.device_put(init_msgs.ftov.value)
        evidence = jax.device_put(init_msgs.evidence.value)
        wiring = jax.device_put(self.wiring)
        log_potentials = jax.device_put(self.log_potentials)
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
                log_potentials,
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
        return Messages(
            ftov=FToVMessages(factor_graph=self, value=msgs_after_bp),
            evidence=init_msgs.evidence,
        )

    def decode_map_states(self, msgs: Messages) -> Dict[Tuple[Any, ...], int]:
        """Function to computes the output of MAP inference on input messages.

        The final states are computed based on evidence obtained from the self.get_evidence
        method as well as the internal wiring.

        Args:
            msgs: ftov messages for deciding MAP states

        Returns:
            a dictionary mapping each variable key to the MAP states of the corresponding variable
        """
        var_states_for_edges = jax.device_put(self.wiring.var_states_for_edges)
        evidence = jax.device_put(msgs.evidence.value)
        final_var_states = evidence.at[var_states_for_edges].add(msgs.ftov.value)
        var_key_to_map_dict: Dict[Tuple[Any, ...], int] = {}
        for var_key in self._variable_group.keys:
            var = self._variable_group[var_key]
            start_index = self._vars_to_starts[var]
            var_key_to_map_dict[var_key] = int(
                jnp.argmax(final_var_states[start_index : start_index + var.num_states])
            )
        return var_key_to_map_dict


@dataclass
class FToVMessages:
    """Class for storing and manipulating factor to variable messages.

    Args:
        factor_graph: associated factor graph
        default_mode: default mode for initializing ftov messages.
            Allowed values include "zeros" and "random"
            If value is None, defaults to "zeros"
        value: Optionally specify initial value for ftov messages

    Attributes:
        _message_updates: Dict[int, jnp.ndarray]. A dictionary containing
            the message updates to make on top of initial message values.
            Maps starting indices to the message values to update with.
    """

    factor_graph: FactorGraph
    default_mode: Optional[str] = None
    value: Optional[Union[np.ndarray, jnp.ndarray]] = None

    def __post_init__(self):
        if self.default_mode is not None and self.value is not None:
            raise ValueError("Should specify only one of default_mode and value.")

        if self.default_mode is None and self.value is None:
            self.default_mode = "zeros"

        if self.value is None:
            if self.default_mode == "zeros":
                self.value = jnp.zeros(self.factor_graph._total_factor_num_states)
            elif self.default_mode == "random":
                self.value = jax.device_put(
                    np.random.gumbel(size=(self.factor_graph._total_factor_num_states,))
                )
            else:
                raise ValueError(
                    f"Unsupported default message mode {self.default_mode}. "
                    "Supported default modes are zeros or random"
                )
        else:
            value = jax.device_put(self.value)
            if not value.shape == (self.factor_graph._total_factor_num_states,):
                raise ValueError(
                    f"Expected messages shape {(self.factor_graph._total_factor_num_states,)}. "
                    f"Got {value.shape}."
                )

    def __getitem__(self, keys: Tuple[Any, Any]) -> jnp.ndarray:
        """Function to query messages from a factor to a variable

        Args:
            keys: a tuple of length 2, with keys[0] being the key for
                factor, and keys[1] being the key for variable

        Returns:
            An array containing the current ftov messages from factor
            keys[0] to variable keys[1]
        """
        if not (
            isinstance(keys, tuple)
            and len(keys) == 2
            and keys[1] in self.factor_graph._variable_group.keys
        ):
            raise ValueError(
                f"Invalid keys {keys}. Please specify a tuple of factor, variable "
                "keys to get the messages from a named factor to a variable"
            )

        factor = self.factor_graph._variables_to_factors[frozenset(keys[0])]
        variable = self.factor_graph._variable_group[keys[1]]
        start = self.factor_graph._factor_to_starts[factor] + np.sum(
            factor.edges_num_states[: factor.variables.index(variable)]
        )
        msgs = jax.device_put(self.value)[start : start + variable.num_states]
        return jax.device_put(msgs)

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
            data: An array containing the beliefs to be spread uniformly
                across all factor to variable messages involving this
                variable.
        """

    def __setitem__(self, keys, data) -> None:
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and keys[1] in self.factor_graph._variable_group.keys
        ):
            factor = self.factor_graph._variables_to_factors[frozenset(keys[0])]
            variable = self.factor_graph._variable_group[keys[1]]
            start = self.factor_graph._factor_to_starts[factor] + np.sum(
                factor.edges_num_states[: factor.variables.index(variable)]
            )
            if data.shape != (variable.num_states,):
                raise ValueError(
                    f"Given message shape {data.shape} does not match expected "
                    f"shape f{(variable.num_states,)} from factor {keys[0]} "
                    f"to variable {keys[1]}."
                )

            self.value = (
                jax.device_put(self.value)
                .at[start : start + variable.num_states]
                .set(data)
            )
        elif keys in self.factor_graph._variable_group.keys:
            variable = self.factor_graph._variable_group[keys]
            if data.shape != (variable.num_states,):
                raise ValueError(
                    f"Given belief shape {data.shape} does not match expected "
                    f"shape f{(variable.num_states,)} for variable {keys}."
                )

            starts = np.nonzero(
                self.factor_graph.wiring.var_states_for_edges
                == self.factor_graph._vars_to_starts[variable]
            )[0]
            for start in starts:
                self.value = (
                    jax.device_put(self.value)
                    .at[start : start + variable.num_states]
                    .st(data / starts.shape[0])
                )
        else:
            raise ValueError(
                "Invalid keys for setting messages. "
                "Supported keys include a tuple of length 2 with factor "
                "and variable keys for directly setting factor to variable "
                "messages, or a valid variable key for spreading expected "
                "beliefs at a variable"
            )


@dataclass
class Evidence:
    """Class for storing and manipulating evidence

    Args:
        factor_graph: associated factor graph
        default_mode: default mode for initializing evidence.
            Allowed values include "zeros" and "random"
            If value is None, defaults to "zeros"
        value: Optionally specify initial value for evidence

    Attributes:
        _evidence_updates: Dict[nodes.Variable, np.ndarray]. maps every variable to an np.ndarray
            representing the evidence for that variable
    """

    factor_graph: FactorGraph
    default_mode: Optional[str] = None
    value: Optional[Union[np.ndarray, jnp.ndarray]] = None

    def __post_init__(self):
        if self.default_mode is not None and self.value is not None:
            raise ValueError("Should specify only one of default_mode and value.")

        if self.default_mode is None and self.value is None:
            self.default_mode = "zeros"

        if self.value is None and self.default_mode not in ("zeros", "random"):
            raise ValueError(
                f"Unsupported default evidence mode {self.default_mode}. "
                "Supported default modes are zeros or random"
            )

        if self.value is None:
            if self.default_mode == "zeros":
                self.value = jnp.zeros(self.factor_graph.num_var_states)
            else:
                self.value = jax.device_put(
                    np.random.gumbel(size=(self.factor_graph.num_var_states,))
                )
        else:
            self.value = jax.device_put(self.value)

    def __getitem__(self, key: Any) -> jnp.ndarray:
        """Function to query evidence for a variable

        Args:
            key: key for the variable

        Returns:
            evidence for the queried variable
        """
        variable = self.factor_graph._variable_group[key]
        start = self.factor_graph._vars_to_starts[variable]
        evidence = jax.device_put(self.value)[start : start + variable.num_states]
        return evidence

    def __setitem__(
        self,
        key: Any,
        evidence: Union[Dict[Hashable, np.ndarray], np.ndarray],
    ) -> None:
        """Function to update the evidence for variables

        Args:
            key: tuple that represents the index into the VariableGroup
                (self.factor_graph._variable_group) that is created when the FactorGraph is instantiated. Note that
                this can be an index referring to an entire VariableGroup (in which case, the evidence
                is set for the entire VariableGroup at once), or to an individual Variable within the
                VariableGroup.
            evidence: a container for np.ndarrays representing the evidence
                Currently supported containers are:
                - an np.ndarray: if key indexes an NDVariableArray, then evidence_values
                can simply be an np.ndarray with num_var_array_dims + 1 dimensions where
                num_var_array_dims is the number of dimensions of the NDVariableArray, and the
                +1 represents a dimension (that should be the final dimension) for the evidence.
                Note that the size of the final dimension should be the same as
                variable_group.variable_size. if key indexes a particular variable, then this array
                must be of the same size as variable.num_states
                - a dictionary: if key indexes a VariableDict, then evidence_values
                must be a dictionary mapping keys of variable_group to np.ndarrays of evidence values.
                Note that each np.ndarray in the dictionary values must have the same size as
                variable_group.variable_size.
        """
        if key in self.factor_graph._variable_group.container_keys:
            if key == slice(None):
                variable_group = self.factor_graph._variable_group
            else:
                variable_group = (
                    self.factor_graph._variable_group.variable_group_container[key]
                )

            for var, evidence_val in variable_group.get_vars_to_evidence(
                evidence
            ).items():
                start_index = self.factor_graph._vars_to_starts[var]
                self.value = (
                    jax.device_put(self.value)
                    .at[start_index : start_index + evidence_val.shape[0]]
                    .set(evidence_val)
                )
        else:
            var = self.factor_graph._variable_group[key]
            start_index = self.factor_graph._vars_to_starts[var]
            self.value = (
                jax.device_put(self.value)
                .at[start_index : start_index + var.num_states]
                .set(evidence)
            )


@dataclass
class Messages:
    """Container class for factor to variable messages and evidence.

    Args:
        ftov: factor to variable messages
        evidence: evidence
    """

    ftov: FToVMessages
    evidence: Evidence
