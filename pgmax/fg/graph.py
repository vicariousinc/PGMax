"""A module containing the core class to specify a Factor Graph."""

from __future__ import annotations

import collections
import copy
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
        _factor_group_to_msgs_starts: Dict[groups.FactorGroup, int]. Maps a factor group to its
            corresponding starting index in the flat message array.
    """

    variables: Union[
        Mapping[Hashable, groups.VariableGroup],
        Sequence[groups.VariableGroup],
        groups.VariableGroup,
    ]

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
        self.num_var_states = vars_num_states_cumsum[-1]
        self._vars_to_starts = MappingProxyType(
            {
                variable: vars_num_states_cumsum[vv]
                for vv, variable in enumerate(self._variable_group.variables)
            }
        )
        self._named_factor_groups: Dict[Hashable, groups.FactorGroup] = {}
        self._variables_to_factors: OrderedDict[
            FrozenSet, nodes.EnumerationFactor
        ] = collections.OrderedDict()
        # For ftov messages
        self._total_factor_num_states: int = 0
        self._factor_group_to_msgs_starts: OrderedDict[
            groups.FactorGroup, int
        ] = collections.OrderedDict()
        self._factor_to_msgs_starts: OrderedDict[
            nodes.EnumerationFactor, int
        ] = collections.OrderedDict()
        # For log potentials
        self._total_factor_num_configs: int = 0
        self._factor_group_to_potentials_starts: OrderedDict[
            groups.FactorGroup, int
        ] = collections.OrderedDict()
        self._factor_to_potentials_starts: OrderedDict[
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

        self._factor_group_to_msgs_starts[factor_group] = self._total_factor_num_states
        self._factor_group_to_potentials_starts[
            factor_group
        ] = self._total_factor_num_configs
        factor_num_states_cumsum = np.insert(
            factor_group.factor_num_states.cumsum(), 0, 0
        )
        factor_group_num_configs = 0
        for vv, variables in enumerate(factor_group._variables_to_factors):
            if variables in self._variables_to_factors:
                raise ValueError(
                    f"A factor involving variables {variables} already exists. Please merge the corresponding factors."
                )

            factor = factor_group._variables_to_factors[variables]
            self._variables_to_factors[variables] = factor
            self._factor_to_msgs_starts[factor] = (
                self._factor_group_to_msgs_starts[factor_group]
                + factor_num_states_cumsum[vv]
            )
            self._factor_group_to_potentials_starts[factor] = (
                self._factor_group_to_potentials_starts[factor_group]
                + vv * factor.log_potentials.shape[0]
            )
            factor_group_num_configs += factor.log_potentials.shape[0]

        if (
            factor_group_num_configs
            != factor_group.factor_group_log_potentials.shape[0]
        ):
            raise ValueError(
                "Factors in a factor group should have the same number of valid configurations."
            )

        self._total_factor_num_states += factor_num_states_cumsum[-1]
        self._total_factor_num_configs += factor_group_num_configs
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
        return tuple(self._factor_group_to_msgs_starts.keys())

    @cached_property
    def fg_state(self) -> FactorGraphState:
        return FactorGraphState(
            variable_group=self._variable_group,
            vars_to_starts=self._vars_to_starts,
            num_var_states=self.num_var_states,
            total_factor_num_states=self._total_factor_num_states,
            variables_to_factors=copy.copy(self._variables_to_factors),
            named_factor_groups=copy.copy(self._named_factor_groups),
            factor_group_to_potentials_starts=copy.copy(
                self._factor_group_to_potentials_starts
            ),
            factor_to_potentials_starts=copy.copy(self._factor_to_potentials_starts),
            factor_to_msgs_starts=copy.copy(self._factor_to_msgs_starts),
            log_potentials=self.log_potentials,
            wiring=self.wiring,
        )

    @property
    def bp_state(self) -> BPState:
        return BPState(
            log_potentials=LogPotentials(fg_state=self.fg_state),
            ftov_msgs=FToVMessages(fg_state=self.fg_state),
            evidence=Evidence(fg_state=self.fg_state),
        )


@dataclass(frozen=True, eq=False)
class FactorGraphState:
    variable_group: groups.VariableGroup
    vars_to_starts: Mapping[nodes.Variable, int]
    num_var_states: int
    total_factor_num_states: int
    variables_to_factors: Mapping[FrozenSet, nodes.EnumerationFactor]
    named_factor_groups: Mapping[Hashable, groups.FactorGroup]
    factor_group_to_potentials_starts: Mapping[groups.FactorGroup, int]
    factor_to_potentials_starts: Mapping[nodes.EnumerationFactor, int]
    factor_to_msgs_starts: Mapping[nodes.EnumerationFactor, int]
    log_potentials: np.ndarray
    wiring: nodes.EnumerationWiring

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            if isinstance(getattr(self, field), np.ndarray):
                getattr(self, field).flags.writeable = False

            if isinstance(getattr(self, field), Mapping):
                object.__setattr__(self, field, MappingProxyType(getattr(self, field)))


@dataclass(frozen=True, eq=False)
class BPState:
    """Container class for belief propagation states, including log potentials,
    ftov messages and evidence (unary log potentials).

    Args:
        log_potentials: log potentials of the model
        ftov_msgs: factor to variable messages
        evidence: evidence
    """

    log_potentials: LogPotentials
    ftov_msgs: FToVMessages
    evidence: Evidence

    def __post_init__(self):
        if (self.log_potentials.fg_state != self.ftov_msgs.fg_state) or (
            self.ftov_msgs.fg_state != self.evidence.fg_state
        ):
            raise ValueError(
                "log_potentials, ftov_msgs and evidence should be derived from the same fg_state."
            )

    @property
    def fg_state(self) -> FactorGraphState:
        return self.log_potentials.fg_state


@jax.partial(jax.jit, static_argnames="fg_state")
def update_log_potentials(
    log_potentials: jnp.ndarray,
    updates: Dict[Any, jnp.ndarray],
    fg_state: FactorGraphState,
) -> jnp.ndarray:
    for key in updates:
        data = updates[key]
        if key in fg_state.named_factor_groups:
            factor_group = fg_state.named_factor_groups[key]
            if data.shape != factor_group.factor_group_log_potentials.shape:
                raise ValueError(
                    f"Expected log potentials shape {factor_group.factor_group_log_potentials.shape} "
                    f"for factor group {key}. Got {data.shape}."
                )

            start = fg_state.factor_group_to_potentials_starts[factor_group]
            log_potentials = log_potentials.at[
                start : start + factor_group.factor_group_log_potentials.shape[0]
            ].set(data)
        elif frozenset(key) in fg_state.variables_to_factors:
            factor = fg_state.variables_to_factors[frozenset(key)]
            if data.shape != factor.log_potentials.shape:
                raise ValueError(
                    f"Expected log potentials shape {factor.log_potentials.shape} "
                    f"for factor {key}. Got {data.shape}."
                )

            start = fg_state.factor_to_potentials_starts[factor]
            log_potentials = log_potentials.at[
                start : start + factor.log_potentials.shape[0]
            ].set(data)
        else:
            raise ValueError(f"Invalid key {key} for log potentials updates.")

    return log_potentials


@dataclass(frozen=True, eq=False)
class LogPotentials:

    fg_state: FactorGraphState
    value: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.value is None:
            object.__setattr__(
                self, "value", jax.device_put(self.fg_state.log_potentials)
            )
        else:
            if not self.value.shape == self.fg_state.log_potentials.shape:
                raise ValueError(
                    f"Expected log potentials shape {self.fg_state.log_potentials.shape}. "
                    f"Got {self.value.shape}."
                )

            object.__setattr__(self, "value", jax.device_put(self.value))

    def __getitem__(self, key: Any):
        if key in self.fg_state.named_factor_groups:
            factor_group = self.fg_state.named_factor_groups[key]
            start = self.fg_state.factor_group_to_potentials_starts[factor_group]
            log_potentials = jax.device_put(self.value)[
                start : start + factor_group.factor_group_log_potentials.shape[0]
            ]
        elif frozenset(key) in self.fg_state.variables_to_factors:
            factor = self.fg_state.variables_to_factors[frozenset(key)]
            start = self.fg_state.factor_to_potentials_starts[factor]
            log_potentials = jax.device_put(self.value)[
                start : start + factor.log_potentials.shape[0]
            ]
        else:
            raise ValueError("")

        return log_potentials

    def __setitem__(
        self,
        key: Any,
        data: Union[np.ndarray, jnp.ndarray],
    ):
        object.__setattr__(
            self,
            "value",
            update_log_potentials(
                jax.device_put(self.value), {key: jax.device_put(data)}, self.fg_state
            ),
        )


@jax.partial(jax.jit, static_argnames="fg_state")
def update_ftov_msgs(
    ftov_msgs: jnp.ndarray, updates: Dict[Any, jnp.ndarray], fg_state: FactorGraphState
) -> jnp.ndarray:
    for keys in updates:
        data = updates[keys]
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and keys[1] in fg_state.variable_group.keys
        ):
            factor = fg_state.variables_to_factors[frozenset(keys[0])]
            variable = fg_state.variable_group[keys[1]]
            start = fg_state.factor_to_msgs_starts[factor] + np.sum(
                factor.edges_num_states[: factor.variables.index(variable)]
            )
            if data.shape != (variable.num_states,):
                raise ValueError(
                    f"Given message shape {data.shape} does not match expected "
                    f"shape f{(variable.num_states,)} from factor {keys[0]} "
                    f"to variable {keys[1]}."
                )

            ftov_msgs = ftov_msgs.at[start : start + variable.num_states].set(data)
        elif keys in fg_state.variable_group.keys:
            variable = fg_state.variable_group[keys]
            if data.shape != (variable.num_states,):
                raise ValueError(
                    f"Given belief shape {data.shape} does not match expected "
                    f"shape f{(variable.num_states,)} for variable {keys}."
                )

            starts = np.nonzero(
                fg_state.wiring.var_states_for_edges
                == fg_state.vars_to_starts[variable]
            )[0]
            for start in starts:
                ftov_msgs = ftov_msgs.at[start : start + variable.num_states].st(
                    data / starts.shape[0]
                )
        else:
            raise ValueError(
                "Invalid keys for setting messages. "
                "Supported keys include a tuple of length 2 with factor "
                "and variable keys for directly setting factor to variable "
                "messages, or a valid variable key for spreading expected "
                "beliefs at a variable"
            )

    return ftov_msgs


@dataclass(frozen=True, eq=False)
class FToVMessages:
    """Class for storing and manipulating factor to variable messages.

    Args:
        factor_graph: associated factor graph
        value: Optionally specify initial value for ftov messages

    Attributes:
        _message_updates: Dict[int, jnp.ndarray]. A dictionary containing
            the message updates to make on top of initial message values.
            Maps starting indices to the message values to update with.
    """

    fg_state: FactorGraphState
    value: Optional[Union[np.ndarray, jnp.ndarray]] = None

    def __post_init__(self):
        if self.value is None:
            object.__setattr__(
                self, "value", jnp.zeros(self.fg_state.total_factor_num_states)
            )
        else:
            if not self.value.shape == (self.fg_state.total_factor_num_states,):
                raise ValueError(
                    f"Expected messages shape {(self.fg_state.total_factor_num_states,)}. "
                    f"Got {self.value.shape}."
                )

            object.__setattr__(self, "value", jax.device_put(self.value))

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
            and keys[1] in self.fg_state.variable_group.keys
        ):
            raise ValueError(
                f"Invalid keys {keys}. Please specify a tuple of factor, variable "
                "keys to get the messages from a named factor to a variable"
            )

        factor = self.fg_state.variables_to_factors[frozenset(keys[0])]
        variable = self.fg_state.variable_group[keys[1]]
        start = self.fg_state.factor_to_msgs_starts[factor] + np.sum(
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
        object.__setattr__(
            self,
            "value",
            update_ftov_msgs(
                jax.device_put(self.value), {keys: jax.device_put(data)}, self.fg_state
            ),
        )


@jax.partial(jax.jit, static_argnames="fg_state")
def update_evidence(
    evidence: jnp.ndarray, updates: Dict[Any, jnp.ndarray], fg_state: FactorGraphState
) -> jnp.ndarray:
    for key in updates:
        data = updates[key]
        if key in fg_state.variable_group.container_keys:
            if key is None:
                variable_group = fg_state.variable_group
            else:
                assert isinstance(
                    fg_state.variable_group, groups.CompositeVariableGroup
                )
                variable_group = fg_state.variable_group[key]

            for var, evidence_val in variable_group.get_vars_to_evidence(data).items():
                start_index = fg_state.vars_to_starts[var]
                evidence = evidence.at[
                    start_index : start_index + evidence_val.shape[0]
                ].set(evidence_val)
        else:
            var = fg_state.variable_group[key]
            start_index = fg_state.vars_to_starts[var]
            evidence = evidence.at[start_index : start_index + var.num_states].set(data)

    return evidence


@dataclass(frozen=True, eq=False)
class Evidence:
    """Class for storing and manipulating evidence

    Args:
        factor_graph: associated factor graph
        value: Optionally specify initial value for evidence

    Attributes:
        _evidence_updates: Dict[nodes.Variable, np.ndarray]. maps every variable to an np.ndarray
            representing the evidence for that variable
    """

    fg_state: FactorGraphState
    value: Optional[Union[np.ndarray, jnp.ndarray]] = None

    def __post_init__(self):
        if self.value is None:
            object.__setattr__(self, "value", jnp.zeros(self.fg_state.num_var_states))
        else:
            object.__setattr__(self, "value", jax.device_put(self.value))

    def __getitem__(self, key: Any) -> jnp.ndarray:
        """Function to query evidence for a variable

        Args:
            key: key for the variable

        Returns:
            evidence for the queried variable
        """
        variable = self.fg_state.variable_group[key]
        start = self.fg_state.vars_to_starts[variable]
        evidence = jax.device_put(self.value)[start : start + variable.num_states]
        return evidence

    def __setitem__(
        self,
        key: Any,
        data: Union[Dict[Hashable, np.ndarray], np.ndarray],
    ) -> None:
        """Function to update the evidence for variables

        Args:
            key: tuple that represents the index into the VariableGroup
                (self.fg_state.variable_group) that is created when the FactorGraph is instantiated. Note that
                this can be an index referring to an entire VariableGroup (in which case, the evidence
                is set for the entire VariableGroup at once), or to an individual Variable within the
                VariableGroup.
            data: a container for np.ndarrays representing the evidence
                Currently supported containers are:
                - an np.ndarray: if key indexes an NDVariableArray, then data
                can simply be an np.ndarray with num_var_array_dims + 1 dimensions where
                num_var_array_dims is the number of dimensions of the NDVariableArray, and the
                +1 represents a dimension (that should be the final dimension) for the evidence.
                Note that the size of the final dimension should be the same as
                variable_group.variable_size. if key indexes a particular variable, then this array
                must be of the same size as variable.num_states
                - a dictionary: if key indexes a VariableDict, then data
                must be a dictionary mapping keys of variable_group to np.ndarrays of evidence values.
                Note that each np.ndarray in the dictionary values must have the same size as
                variable_group.variable_size.
        """
        object.__setattr__(
            self,
            "value",
            update_evidence(
                jax.device_put(self.value), {key: jax.device_put(data)}, self.fg_state
            ),
        )
