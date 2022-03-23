from __future__ import annotations

"""A module containing the core class to specify a Factor Graph."""

import collections
import copy
import functools
import typing
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from pgmax.bp import infer
from pgmax.factors import FAC_TO_VAR_UPDATES
from pgmax.fg import groups, nodes
from pgmax.utils import cached_property


@dataclass
class FactorGraph:
    """Class for representing a factor graph.
    Factors in a graph are clustered in factor groups, which are grouped according to their factor types.

    Args:
        variables: A single VariableGroup or a container containing variable groups.
            If not a single VariableGroup, supported containers include mapping and sequence.
            For a mapping, the keys of the mapping are used to index the variable groups.
            For a sequence, the indices of the sequence are used to index the variable groups.
            Note that if not a single VariableGroup, a CompositeVariableGroup will be created from
            this input, and the individual VariableGroups will need to be accessed by indexing.
    """

    variables: Union[
        Mapping[Any, groups.VariableGroup],
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
        # See FactorGraphState docstrings for documentation on the following fields
        self._num_var_states = vars_num_states_cumsum[-1]
        self._vars_to_starts = MappingProxyType(
            {
                variable: vars_num_states_cumsum[vv]
                for vv, variable in enumerate(self._variable_group.variables)
            }
        )
        self._named_factor_groups: Dict[Hashable, groups.FactorGroup] = {}
        self._variables_to_factors: OrderedDict[
            FrozenSet, nodes.Factor
        ] = collections.OrderedDict()
        self._factor_types_to_groups: OrderedDict[
            Type, List[groups.FactorGroup]
        ] = collections.OrderedDict(
            [(factor_type, []) for factor_type in FAC_TO_VAR_UPDATES]
        )

    def __hash__(self) -> int:
        all_factor_groups = tuple(
            [
                factor_group
                for factor_groups_per_type in self.factor_groups.values()
                for factor_group in factor_groups_per_type
            ]
        )
        return hash(all_factor_groups)

    def add_factor(
        self,
        variable_names: List,
        factor_configs: np.ndarray,
        log_potentials: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        """Function to add a single factor to the FactorGraph.

        Args:
            variable_names: A list containing the connected variable names.
                Variable names are tuples of the type (variable_group_name, variable_name_within_variable_group)
            factor_configs: Array of shape (num_val_configs, num_variables)
                An array containing explicit enumeration of all valid configurations.
                If the connected variables have n1, n2, ... states, 1 <= num_val_configs <= n1 * n2 * ...
                factor_configs[config_idx, variable_idx] represents the state of variable_names[variable_idx]
                in the configuration factor_configs[config_idx].
            log_potentials: Optional array of shape (num_val_configs,).
                If specified, log_potentials[config_idx] contains the log of the potential value for
                the valid configuration factor_configs[config_idx].
                If None, it is assumed the log potential is uniform 0 and such an array is automatically
                initialized.
        """
        factor_group = groups.EnumerationFactorGroup(
            self._variable_group,
            variable_names_for_factors=[variable_names],
            factor_configs=factor_configs,
            log_potentials=log_potentials,
        )
        self._register_factor_group(factor_group, name)

    def add_factor_by_type(
        self, variable_names: List, factor_type: type, *args, **kwargs
    ) -> None:
        """Function to add a single factor to the FactorGraph.

        Args:
            variable_names: A list containing the connected variable names.
                Variable names are tuples of the type (variable_group_name, variable_name_within_variable_group)
            factor_type: Type of factor to be added
            args: Args to be passed to the factor_type.
            kwargs: kwargs to be passed to the factor_type, and an optional "name" argument
                for specifying the name of a named factor group.

        Example:
            To add an ORFactor to a FactorGraph fg, run::

                fg.add_factor_by_type(
                    variable_names=variables_names_for_OR_factor,
                    factor_type=logical.ORFactor
                )
        """
        if factor_type not in FAC_TO_VAR_UPDATES:
            raise ValueError(
                f"Type {factor_type} is not one of the supported factor types {FAC_TO_VAR_UPDATES.keys()}"
            )

        name = kwargs.pop("name", None)
        variables = tuple(self._variable_group[variable_names])
        factor = factor_type(variables, *args, **kwargs)
        factor_group = groups.SingleFactorGroup(
            variable_group=self._variable_group,
            variable_names=variable_names,
            factor=factor,
        )
        self._register_factor_group(factor_group, name)

    def add_factor_group(self, factory: Callable, *args, **kwargs) -> None:
        """Add a factor group to the factor graph

        Args:
            factory: Factory function that takes args and kwargs as input and outputs a factor group.
            args: Args to be passed to the factory function.
            kwargs: kwargs to be passed to the factory function, and an optional "name" argument
                for specifying the name of a named factor group.
        """
        name = kwargs.pop("name", None)
        factor_group = factory(self._variable_group, *args, **kwargs)
        self._register_factor_group(factor_group, name)

    def _register_factor_group(
        self, factor_group: groups.FactorGroup, name: Optional[str] = None
    ) -> None:
        """Register a factor group to the factor graph, by updating the factor graph state.

        Args:
            factor_group: The factor group to be registered to the factor graph.
            name: Optional name of the factor group.

        Raises:
            ValueError: If the factor group with the same name or a factor involving the same variables
                already exists in the factor graph.
        """
        if name in self._named_factor_groups:
            raise ValueError(
                f"A factor group with the name {name} already exists. Please choose a different name!"
            )

        for variables, factor in factor_group._variables_to_factors.items():
            if variables in self._variables_to_factors:
                raise ValueError(
                    f"A factor involving variables {variables} already exists. Please merge the corresponding factors."
                )
            self._variables_to_factors[variables] = factor

        factor_type = type(factor)
        self._factor_types_to_groups[factor_type].append(factor_group)
        if name is not None:
            self._named_factor_groups[name] = factor_group

    @functools.lru_cache(None)
    def compute_offsets(self) -> None:
        """Compute factor messages offsets for the factor types, factor groups and factors
        in the flattened array of message.
        Also compute log potentials offsets for factor groups and factors.

        See FactorGraphState for documentation on the following fields

        If offsets have already beeen compiled, do nothing.
        """
        # Message offsets for ftov messages
        self._factor_type_to_msgs_range = collections.OrderedDict()
        self._factor_group_to_msgs_starts = collections.OrderedDict()
        self._factor_to_msgs_starts = collections.OrderedDict()
        factor_num_states_cumsum = 0

        # Log potentials offsets
        self._factor_type_to_potentials_range = collections.OrderedDict()
        self._factor_group_to_potentials_starts = collections.OrderedDict()
        self._factor_to_potentials_starts = collections.OrderedDict()
        factor_num_configs_cumsum = 0

        for factor_type, factors_groups_by_type in self.factor_groups.items():
            factor_type_num_states_start = factor_num_states_cumsum
            factor_type_num_configs_start = factor_num_configs_cumsum
            for factor_group in factors_groups_by_type:
                self._factor_group_to_msgs_starts[
                    factor_group
                ] = factor_num_states_cumsum
                self._factor_group_to_potentials_starts[
                    factor_group
                ] = factor_num_configs_cumsum

                for factor in factor_group.factors:
                    self._factor_to_msgs_starts[factor] = factor_num_states_cumsum
                    self._factor_to_potentials_starts[
                        factor
                    ] = factor_num_configs_cumsum

                    factor_num_states_cumsum += np.sum(factor.edges_num_states)
                    factor_num_configs_cumsum += factor.log_potentials.shape[0]

            self._factor_type_to_msgs_range[factor_type] = (
                factor_type_num_states_start,
                factor_num_states_cumsum,
            )
            self._factor_type_to_potentials_range[factor_type] = (
                factor_type_num_configs_start,
                factor_num_configs_cumsum,
            )

        self._total_factor_num_states = factor_num_states_cumsum
        self._total_factor_num_configs = factor_num_configs_cumsum

    @cached_property
    def wiring(self) -> OrderedDict[Type, nodes.Wiring]:
        """Function to compile wiring for belief propagation.

        If wiring has already beeen compiled, do nothing.

        Returns:
            A dictionnary mapping each factor type to its wiring.
        """
        wiring = collections.OrderedDict(
            [
                (
                    factor_type,
                    [
                        factor.compile_wiring(self._vars_to_starts)
                        for factor in self.factors[factor_type]
                    ],
                )
                for factor_type in self.factors
            ]
        )
        wiring = collections.OrderedDict(
            [
                (factor_type, factor_type.concatenate_wirings(wiring[factor_type]))
                for factor_type in wiring
            ]
        )
        return wiring

    @cached_property
    def log_potentials(self) -> OrderedDict[Type, np.ndarray]:
        """Function to compile potential array for belief propagation.

        If potential array has already beeen compiled, do nothing.

        Returns:
            A dictionnary mapping each factor type to the array of the log of the potential
                function for each valid configuration
        """
        log_potentials = collections.OrderedDict()
        for factor_type, factors_groups_by_type in self.factor_groups.items():
            if len(factors_groups_by_type) == 0:
                log_potentials[factor_type] = np.empty((0,))
            else:
                log_potentials[factor_type] = np.concatenate(
                    [
                        factor_group.factor_group_log_potentials
                        for factor_group in factors_groups_by_type
                    ]
                )

        return log_potentials

    @cached_property
    def factors(self) -> OrderedDict[Type, Tuple[nodes.Factor, ...]]:
        """Mapping factor type to individual factors in the factor graph"""
        factors: OrderedDict[Type, Tuple[nodes.Factor, ...]] = collections.OrderedDict(
            [
                (
                    factor_type,
                    tuple(
                        [
                            factor
                            for factor_group in self.factor_groups[factor_type]
                            for factor in factor_group.factors
                        ]
                    ),
                )
                for factor_type in self.factor_groups
            ]
        )
        return factors

    @property
    def factor_groups(self) -> OrderedDict[Type, List[groups.FactorGroup]]:
        """Tuple of factor groups in the factor graph"""
        return self._factor_types_to_groups

    @cached_property
    def fg_state(self) -> FactorGraphState:
        """Current factor graph state given the added factors."""
        # Preliminary computations
        self.compute_offsets()
        log_potentials = np.concatenate(
            [self.log_potentials[factor_type] for factor_type in self.log_potentials]
        )

        return FactorGraphState(
            variable_group=self._variable_group,
            vars_to_starts=self._vars_to_starts,
            num_var_states=self._num_var_states,
            total_factor_num_states=self._total_factor_num_states,
            variables_to_factors=copy.copy(self._variables_to_factors),
            named_factor_groups=copy.copy(self._named_factor_groups),
            factor_type_to_msgs_range=copy.copy(self._factor_type_to_msgs_range),
            factor_type_to_potentials_range=copy.copy(
                self._factor_type_to_potentials_range
            ),
            factor_group_to_potentials_starts=copy.copy(
                self._factor_group_to_potentials_starts
            ),
            factor_to_potentials_starts=copy.copy(self._factor_to_potentials_starts),
            factor_to_msgs_starts=copy.copy(self._factor_to_msgs_starts),
            log_potentials=log_potentials,
            wiring=self.wiring,
        )

    @property
    def bp_state(self) -> BPState:
        """Relevant information for doing belief propagation."""
        # Preliminary computations
        self.compute_offsets()

        return BPState(
            log_potentials=LogPotentials(fg_state=self.fg_state),
            ftov_msgs=FToVMessages(fg_state=self.fg_state),
            evidence=Evidence(fg_state=self.fg_state),
        )


@dataclass(frozen=True, eq=False)
class FactorGraphState:
    """FactorGraphState.

    Args:
        variable_group: A variable group containing all the variables in the FactorGraph.
        vars_to_starts: Maps variables to their starting indices in the flat evidence array.
            flat_evidence[vars_to_starts[variable]: vars_to_starts[variable] + variable.num_var_states]
            contains evidence to the variable.
        num_var_states: Total number of variable states.
        total_factor_num_states: Size of the flat ftov messages array.
        variables_to_factors: Maps sets of connected variables (in the form of frozensets of
            variable names) to corresponding factors.
        named_factor_groups: Maps the names of named factor groups to the corresponding factor groups.
        factor_type_to_msgs_range: Maps factors types to their start and end indices in the flat ftov messages.
        factor_type_to_potentials_range: Maps factor types to their start and end indices in the flat log potentials.
        factor_group_to_potentials_starts: Maps factor groups to their starting indices in the flat log potentials.
        factor_to_msgs_starts: Maps factors to their starting indices in the flat ftov messages.
        factor_to_potentials_starts: Maps factors to their starting indices in the flat log potentials.
        log_potentials: Flat log potentials array concatenated for each factor type.
        wiring: Wiring derived for each factor type.
    """

    variable_group: groups.VariableGroup
    vars_to_starts: Mapping[nodes.Variable, int]
    num_var_states: int
    total_factor_num_states: int
    variables_to_factors: Mapping[FrozenSet, nodes.Factor]
    named_factor_groups: Mapping[Hashable, groups.FactorGroup]
    factor_group_to_potentials_starts: OrderedDict[groups.FactorGroup, int]
    factor_to_potentials_starts: OrderedDict[nodes.Factor, int]
    factor_to_msgs_starts: OrderedDict[nodes.Factor, int]
    factor_type_to_msgs_range: OrderedDict[type, Tuple[int, int]]
    factor_type_to_potentials_range: OrderedDict[type, Tuple[int, int]]
    log_potentials: OrderedDict[type, None | np.ndarray]
    wiring: OrderedDict[type, nodes.Wiring]

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
        evidence: evidence (unary log potentials) for variables.

    Raises:
        ValueError: If log_potentials, ftov_msgs or evidence are not derived from the same
            FactorGraphState.
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


@functools.partial(jax.jit, static_argnames="fg_state")
def update_log_potentials(
    log_potentials: jnp.ndarray,
    updates: Dict[Any, jnp.ndarray],
    fg_state: FactorGraphState,
) -> jnp.ndarray:
    """Function to update log_potentials.

    Args:
        log_potentials: A flat jnp array containing log_potentials.
        updates: A dictionary containing updates for log_potentials
        fg_state: Factor graph state

    Returns:
        A flat jnp array containing updated log_potentials.

    Raises: ValueError if
        (1) Provided log_potentials shape does not match the expected log_potentials shape.
        (2) Provided name is not valid for log_potentials updates.
    """
    for name in updates:
        data = updates[name]
        if name in fg_state.named_factor_groups:
            factor_group = fg_state.named_factor_groups[name]

            flat_data = factor_group.flatten(data)
            if flat_data.shape != factor_group.factor_group_log_potentials.shape:
                raise ValueError(
                    f"Expected log potentials shape {factor_group.factor_group_log_potentials.shape} "
                    f"for factor group {name}. Got incompatible data shape {data.shape}."
                )

            start = fg_state.factor_group_to_potentials_starts[factor_group]
            log_potentials = log_potentials.at[start : start + flat_data.shape[0]].set(
                flat_data
            )
        elif frozenset(name) in fg_state.variables_to_factors:
            factor = fg_state.variables_to_factors[frozenset(name)]

            if data.shape != factor.log_potentials.shape:
                raise ValueError(
                    f"Expected log potentials shape {factor.log_potentials.shape} "
                    f"for factor {name}. Got {data.shape}."
                )

            start = fg_state.factor_to_potentials_starts[factor]
            log_potentials = log_potentials.at[
                start : start + factor.log_potentials.shape[0]
            ].set(data)
        else:
            raise ValueError(f"Invalid name {name} for log potentials updates.")

    return log_potentials


@dataclass(frozen=True, eq=False)
class LogPotentials:
    """Class for storing and manipulating log potentials.

    Args:
        fg_state: Factor graph state
        value: Optionally specify an initial value

    Raises:
        ValueError: If provided value shape does not match the expected log_potentials shape.
    """

    fg_state: FactorGraphState
    value: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.value is None:
            object.__setattr__(self, "value", self.fg_state.log_potentials)
        else:
            if not self.value.shape == self.fg_state.log_potentials.shape:
                raise ValueError(
                    f"Expected log potentials shape {self.fg_state.log_potentials.shape}. "
                    f"Got {self.value.shape}."
                )

            object.__setattr__(self, "value", self.value)

    def __getitem__(self, name: Any) -> np.ndarray:
        """Function to query log potentials for a named factor group or a factor.

        Args:
            name: Name of a named factor group, or a frozenset containing the set
                of connected variables for the queried factor.

        Returns:
            The queried log potentials.
        """
        value = cast(np.ndarray, self.value)
        if not isinstance(name, Hashable):
            name = frozenset(name)

        if name in self.fg_state.named_factor_groups:
            factor_group = self.fg_state.named_factor_groups[name]
            start = self.fg_state.factor_group_to_potentials_starts[factor_group]
            log_potentials = value[
                start : start + factor_group.factor_group_log_potentials.shape[0]
            ]
        elif frozenset(name) in self.fg_state.variables_to_factors:
            factor = self.fg_state.variables_to_factors[frozenset(name)]
            start = self.fg_state.factor_to_potentials_starts[factor]
            log_potentials = value[start : start + factor.log_potentials.shape[0]]
        else:
            raise ValueError(f"Invalid name {name} for log potentials updates.")

        return log_potentials

    def __setitem__(
        self,
        name: Any,
        data: Union[np.ndarray, jnp.ndarray],
    ):
        """Set the log potentials for a named factor group or a factor.

        Args:
            name: Name of a named factor group, or a frozenset containing the set
                of connected variables for the queried factor.
            data: Array containing the log potentials for the named factor group
                or the factor.
        """
        if not isinstance(name, Hashable):
            name = frozenset(name)

        object.__setattr__(
            self,
            "value",
            np.asarray(
                update_log_potentials(
                    jax.device_put(self.value),
                    {name: jax.device_put(data)},
                    self.fg_state,
                )
            ),
        )


@functools.partial(jax.jit, static_argnames="fg_state")
def update_ftov_msgs(
    ftov_msgs: jnp.ndarray, updates: Dict[Any, jnp.ndarray], fg_state: FactorGraphState
) -> jnp.ndarray:
    """Function to update ftov_msgs.

    Args:
        ftov_msgs: A flat jnp array containing ftov_msgs.
        updates: A dictionary containing updates for ftov_msgs
        fg_state: Factor graph state

    Returns:
        A flat jnp array containing updated ftov_msgs.

    Raises: ValueError if:
        (1) provided ftov_msgs shape does not match the expected ftov_msgs shape.
        (2) provided name is not valid for ftov_msgs updates.
    """
    for names in updates:
        data = updates[names]
        if (
            isinstance(names, tuple)
            and len(names) == 2
            and names[1] in fg_state.variable_group.names
        ):
            factor = fg_state.variables_to_factors[frozenset(names[0])]
            variable = fg_state.variable_group[names[1]]
            start = fg_state.factor_to_msgs_starts[factor] + np.sum(
                factor.edges_num_states[: factor.variables.index(variable)]
            )
            if data.shape != (variable.num_states,):
                raise ValueError(
                    f"Given message shape {data.shape} does not match expected "
                    f"shape {(variable.num_states,)} from factor {names[0]} "
                    f"to variable {names[1]}."
                )

            ftov_msgs = ftov_msgs.at[start : start + variable.num_states].set(data)
        elif names in fg_state.variable_group.names:
            variable = fg_state.variable_group[names]
            if data.shape != (variable.num_states,):
                raise ValueError(
                    f"Given belief shape {data.shape} does not match expected "
                    f"shape {(variable.num_states,)} for variable {names}."
                )

            var_states_for_edges = np.concatenate(
                [
                    wiring_by_type.var_states_for_edges
                    for wiring_by_type in fg_state.wiring.values()
                ]
            )

            starts = np.nonzero(
                var_states_for_edges == fg_state.vars_to_starts[variable]
            )[0]
            for start in starts:
                ftov_msgs = ftov_msgs.at[start : start + variable.num_states].set(
                    data / starts.shape[0]
                )
        else:
            raise ValueError(
                "Invalid names for setting messages. "
                "Supported names include a tuple of length 2 with factor "
                "and variable names for directly setting factor to variable "
                "messages, or a valid variable name for spreading expected "
                "beliefs at a variable"
            )

    return ftov_msgs


@dataclass(frozen=True, eq=False)
class FToVMessages:
    """Class for storing and manipulating factor to variable messages.

    Args:
        fg_state: Factor graph state
        value: Optionally specify initial value for ftov messages

    Raises: ValueError if provided value does not match expected ftov messages shape.
    """

    fg_state: FactorGraphState
    value: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.value is None:
            object.__setattr__(
                self, "value", np.zeros(self.fg_state.total_factor_num_states)
            )
        else:
            if not self.value.shape == (self.fg_state.total_factor_num_states,):
                raise ValueError(
                    f"Expected messages shape {(self.fg_state.total_factor_num_states,)}. "
                    f"Got {self.value.shape}."
                )

            object.__setattr__(self, "value", self.value)

    def __getitem__(self, names: Tuple[Any, Any]) -> np.ndarray:
        """Function to query messages from a factor to a variable

        Args:
            names: a tuple of length 2, with names[0] being the name for
                factor, and names[1] being the name for variable

        Returns:
            An array containing the current ftov messages from factor
            names[0] to variable names[1]

        Raises: ValueError if provided names are not valid for querying ftov messages.
        """
        value = cast(np.ndarray, self.value)
        if not (
            isinstance(names, tuple)
            and len(names) == 2
            and names[1] in self.fg_state.variable_group.names
        ):
            raise ValueError(
                f"Invalid names {names}. Please specify a tuple of factor, variable "
                "names to get the messages from a named factor to a variable"
            )

        factor = self.fg_state.variables_to_factors[frozenset(names[0])]
        variable = self.fg_state.variable_group[names[1]]
        start = self.fg_state.factor_to_msgs_starts[factor] + np.sum(
            factor.edges_num_states[: factor.variables.index(variable)]
        )
        msgs = value[start : start + variable.num_states]
        return msgs

    @typing.overload
    def __setitem__(
        self,
        names: Tuple[Any, Any],
        data: Union[np.ndarray, jnp.ndarray],
    ) -> None:
        """Setting messages from a factor to a variable

        Args:
            names: A tuple of length 2
                names[0] is the name of the factor
                names[1] is the name of the variable
            data: An array containing messages from factor names[0]
                to variable names[1]
        """

    @typing.overload
    def __setitem__(
        self,
        names: Any,
        data: Union[np.ndarray, jnp.ndarray],
    ) -> None:
        """Spreading beliefs at a variable to all connected factors

        Args:
            names: The name of the variable
            data: An array containing the beliefs to be spread uniformly
                across all factor to variable messages involving this
                variable.
        """

    def __setitem__(self, names, data) -> None:
        if (
            isinstance(names, tuple)
            and len(names) == 2
            and names[1] in self.fg_state.variable_group.names
        ):
            names = (frozenset(names[0]), names[1])

        object.__setattr__(
            self,
            "value",
            np.asarray(
                update_ftov_msgs(
                    jax.device_put(self.value),
                    {names: jax.device_put(data)},
                    self.fg_state,
                )
            ),
        )


@functools.partial(jax.jit, static_argnames="fg_state")
def update_evidence(
    evidence: jnp.ndarray, updates: Dict[Any, jnp.ndarray], fg_state: FactorGraphState
) -> jnp.ndarray:
    """Function to update evidence.

    Args:
        evidence: A flat jnp array containing evidence.
        updates: A dictionary containing updates for evidence
        fg_state: Factor graph state

    Returns:
        A flat jnp array containing updated evidence.
    """
    for name in updates:
        data = updates[name]
        if name in fg_state.variable_group.container_names:
            if name is None:
                variable_group = fg_state.variable_group
            else:
                assert isinstance(
                    fg_state.variable_group, groups.CompositeVariableGroup
                )
                variable_group = fg_state.variable_group.variable_group_container[name]

            start_index = fg_state.vars_to_starts[variable_group.variables[0]]
            flat_data = variable_group.flatten(data)
            evidence = evidence.at[start_index : start_index + flat_data.shape[0]].set(
                flat_data
            )
        else:
            var = fg_state.variable_group[name]
            start_index = fg_state.vars_to_starts[var]
            evidence = evidence.at[start_index : start_index + var.num_states].set(data)
    return evidence


@dataclass(frozen=True, eq=False)
class Evidence:
    """Class for storing and manipulating evidence

    Args:
        fg_state: Factor graph state
        value: Optionally specify initial value for evidence

    Raises: ValueError if provided value does not match expected evidence shape.
    """

    fg_state: FactorGraphState
    value: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.value is None:
            object.__setattr__(self, "value", np.zeros(self.fg_state.num_var_states))
        else:
            if self.value.shape != (self.fg_state.num_var_states,):
                raise ValueError(
                    f"Expected evidence shape {(self.fg_state.num_var_states,)}. "
                    f"Got {self.value.shape}."
                )

            object.__setattr__(self, "value", self.value)

    def __getitem__(self, name: Any) -> np.ndarray:
        """Function to query evidence for a variable

        Args:
            name: name for the variable

        Returns:
            evidence for the queried variable
        """
        value = cast(np.ndarray, self.value)
        variable = self.fg_state.variable_group[name]
        start = self.fg_state.vars_to_starts[variable]
        evidence = value[start : start + variable.num_states]
        return evidence

    def __setitem__(
        self,
        name: Any,
        data: np.ndarray,
    ) -> None:
        """Function to update the evidence for variables

        Args:
            name: The name of a variable group or a single variable.
                If name is the name of a variable group, updates are derived by using the variable group to
                flatten the data.
                If name is the name of a variable, data should be of an array shape (num_states,)
                If name is None, updates are derived by using self.fg_state.variable_group to flatten the data.
            data: Array containing the evidence updates.
        """
        object.__setattr__(
            self,
            "value",
            np.asarray(
                update_evidence(
                    jax.device_put(self.value),
                    {name: jax.device_put(data)},
                    self.fg_state,
                ),
            ),
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class BPArrays:
    """Container for the relevant flat arrays used in belief propagation.

    Args:
        log_potentials: Flat log potentials array.
        ftov_msgs: Flat factor to variable messages array.
        evidence: Flat evidence array.
    """

    log_potentials: Union[np.ndarray, jnp.ndarray]
    ftov_msgs: Union[np.ndarray, jnp.ndarray]
    evidence: Union[np.ndarray, jnp.ndarray]

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            if isinstance(getattr(self, field), np.ndarray):
                getattr(self, field).flags.writeable = False

    def tree_flatten(self):
        return jax.tree_util.tree_flatten(asdict(self))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data.unflatten(children))


def BP(
    bp_state: BPState, num_iters: int, temperature: float = 0.0
) -> Tuple[Callable, Callable, Callable]:
    """Function for generating belief propagation functions.

    Args:
        bp_state: Belief propagation state.
        num_iters: Number of belief propagation iterations.
        temperature: Temperature for loopy belief propagation.
            1.0 corresponds to sum-product, 0.0 corresponds to max-product.

    Returns:
        Tuple containing\n
        \trun_bp: Function for running belief propagation for num_iters.\n
        \t\tOptionally takes as input log_potentials updates, ftov_msgs updates,
        \t\tevidence updates, and damping factor, and outputs a BPArrays.\n
        \tget_bp_state: Function to reconstruct the BPState from BPArrays.\n
        \tget_beliefs: Function to calculate beliefs from BPArrays.\n
    """
    wiring = bp_state.fg_state.wiring
    edges_num_states = np.concatenate(
        [wiring[factor_type].edges_num_states for factor_type in FAC_TO_VAR_UPDATES]
    )
    max_msg_size = int(np.max(edges_num_states))
    var_states_for_edges = np.concatenate(
        [wiring[factor_type].var_states_for_edges for factor_type in FAC_TO_VAR_UPDATES]
    )
    inference_arguments: Dict[type, Mapping] = {
        factor_type: wiring[factor_type].inference_arguments
        for factor_type in FAC_TO_VAR_UPDATES
    }
    factor_type_to_msgs_range = bp_state.fg_state.factor_type_to_msgs_range
    factor_type_to_potentials_range = bp_state.fg_state.factor_type_to_potentials_range

    def run_bp(
        log_potentials_updates: Optional[Dict[Any, jnp.ndarray]] = None,
        ftov_msgs_updates: Optional[Dict[Any, jnp.ndarray]] = None,
        evidence_updates: Optional[Dict[Any, jnp.ndarray]] = None,
        damping: float = 0.5,
    ) -> BPArrays:
        """Function to perform belief propagation.

        Specifically, belief propagation is run for num_iters iterations and
        returns a BPArrays containing the updated log_potentials, ftov_msgs and evidence.

        Args:
            log_potentials_updates: Dictionary containing optional log_potentials updates.
            ftov_msgs_updates: Dictionary containing optional ftov_msgs updates.
            evidence_updates: Dictionary containing optional evidence updates.
            damping: The damping factor to use for message updates between one timestep and the next

        Returns:
            A BPArrays containing the updated log_potentials, ftov_msgs and evidence.
        """
        log_potentials = jax.device_put(bp_state.log_potentials.value)
        if log_potentials_updates is not None:
            log_potentials = update_log_potentials(
                log_potentials, log_potentials_updates, bp_state.fg_state
            )

        ftov_msgs = bp_state.ftov_msgs.value
        if ftov_msgs_updates is not None:
            ftov_msgs = update_ftov_msgs(
                ftov_msgs, ftov_msgs_updates, bp_state.fg_state
            )

        evidence = bp_state.evidence.value
        if evidence_updates is not None:
            evidence = update_evidence(evidence, evidence_updates, bp_state.fg_state)

        # Normalize the messages to ensure the maximum value is 0.
        ftov_msgs = infer.normalize_and_clip_msgs(
            ftov_msgs, edges_num_states, max_msg_size
        )

        @jax.checkpoint
        def update(msgs: jnp.ndarray, _) -> Tuple[jnp.ndarray, None]:
            # Compute new variable to factor messages by message passing
            vtof_msgs = infer.pass_var_to_fac_messages(
                msgs,
                evidence,
                var_states_for_edges,
            )
            ftov_msgs = jnp.zeros_like(vtof_msgs)
            for factor_type in FAC_TO_VAR_UPDATES:
                msgs_start, msgs_end = factor_type_to_msgs_range[factor_type]
                potentials_start, potentials_end = factor_type_to_potentials_range[
                    factor_type
                ]
                ftov_msgs_type = FAC_TO_VAR_UPDATES[factor_type](
                    vtof_msgs=vtof_msgs[msgs_start:msgs_end],
                    log_potentials=log_potentials[potentials_start:potentials_end],
                    temperature=temperature,
                    **inference_arguments[factor_type],
                )
                ftov_msgs = ftov_msgs.at[msgs_start:msgs_end].set(ftov_msgs_type)

            # Use the results of message passing to perform damping and
            # update the factor to variable messages
            delta_msgs = ftov_msgs - msgs
            msgs = msgs + (1 - damping) * delta_msgs
            # Normalize and clip these damped, updated messages before
            # returning them.
            msgs = infer.normalize_and_clip_msgs(msgs, edges_num_states, max_msg_size)
            return msgs, None

        ftov_msgs, _ = jax.lax.scan(update, ftov_msgs, None, num_iters)
        return BPArrays(
            log_potentials=log_potentials, ftov_msgs=ftov_msgs, evidence=evidence
        )

    def get_bp_state(bp_arrays: BPArrays) -> BPState:
        """Reconstruct the BPState from a BPArrays

        Args:
            bp_arrays: A BPArrays containing arrays for belief propagation.

        Returns:
            The corresponding BPState
        """
        return BPState(
            log_potentials=LogPotentials(
                fg_state=bp_state.fg_state, value=bp_arrays.log_potentials
            ),
            ftov_msgs=FToVMessages(
                fg_state=bp_state.fg_state,
                value=bp_arrays.ftov_msgs,
            ),
            evidence=Evidence(fg_state=bp_state.fg_state, value=bp_arrays.evidence),
        )

    @jax.jit
    def get_beliefs(bp_arrays: BPArrays) -> Any:
        """Calculate beliefs from given BPArrays

        Args:
            bp_arrays: A BPArrays containing arrays for belief propagation.

        Returns:
            beliefs: An array or a PyTree container containing the beliefs for the variables.
        """
        beliefs = bp_state.fg_state.variable_group.unflatten(
            jax.device_put(bp_arrays.evidence)
            .at[jax.device_put(var_states_for_edges)]
            .add(bp_arrays.ftov_msgs)
        )
        return beliefs

    return run_bp, get_bp_state, get_beliefs


@jax.jit
def decode_map_states(beliefs: Any) -> Any:
    """Function to decode MAP states given the calculated beliefs.

    Args:
        beliefs: An array or a PyTree container containing beliefs for different variables.

    Returns:
        An array or a PyTree container containing the MAP states for different variables.
    """
    map_states = jax.tree_util.tree_map(
        lambda x: jnp.argmax(x, axis=-1),
        beliefs,
    )
    return map_states


@jax.jit
def get_marginals(beliefs: Any) -> Any:
    """Function to get marginal probabilities given the calculated beliefs.

    Args:
        beliefs: An array or a PyTree container containing beliefs for different variables.

    Returns:
        An array or a PyTree container containing the marginal probabilities different variables.
    """
    marginals = jax.tree_util.tree_map(
        lambda x: jnp.exp(x - logsumexp(x, axis=-1, keepdims=True)),
        beliefs,
    )
    return marginals
