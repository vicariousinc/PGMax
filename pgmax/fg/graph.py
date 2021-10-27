"""A module containing the core class to specify a Factor Graph."""
from __future__ import annotations

import collections
import copy
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
        self._num_var_states = vars_num_states_cumsum[-1]
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
        variable_names: List,
        factor_configs: np.ndarray,
        log_potentials: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ) -> None:
        """Function to add a single factor to the FactorGraph.

        Args:
            variable_names: A list containing the involved variable names.
            factor_configs: Array of shape (num_val_configs, num_variables)
                An array containing explicit enumeration of all valid configurations
            log_potentials: Optional array of shape (num_val_configs,) or (num_factors, num_val_configs).
                If specified, it contains the log of the potential value for every possible configuration.
                If none, it is assumed the log potential is uniform 0 and such an array is automatically
                initialized.
        """
        if name in self._named_factor_groups:
            raise ValueError(
                f"A factor group with the name {name} already exists. Please choose a different name!"
            )

        factor_group = groups.EnumerationFactorGroup(
            self._variable_group,
            connected_var_keys=[variable_names],
            factor_configs=factor_configs,
            log_potentials=log_potentials,
        )
        self._register_factor_group(factor_group)

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
        if name in self._named_factor_groups:
            raise ValueError(
                f"A factor group with the name {name} already exists. Please choose a different name!"
            )

        """Register a factor group to the factor graph, by updating the factor graph state.

        Args:
            factor_group: The factor group to be registered to the factor graph.
            name: Optional name of the factor group.
        """
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
            self._factor_to_potentials_starts[factor] = (
                self._factor_group_to_potentials_starts[factor_group]
                + vv * factor.log_potentials.shape[0]
            )
            factor_group_num_configs += factor.log_potentials.shape[0]

        self._total_factor_num_states += factor_num_states_cumsum[-1]
        self._total_factor_num_configs += factor_group_num_configs
        if name is not None:
            self._named_factor_groups[name] = factor_group

    @cached_property
    def wiring(self) -> nodes.EnumerationWiring:
        """Function to compile wiring for belief propagation.

        If wiring has already beeen compiled, do nothing.

        Returns:
            Compiled wiring from individual factors.
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
            A jnp array representing the log of the potential function for each
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
        """Current factor graph state given the added factors."""
        return FactorGraphState(
            variable_group=self._variable_group,
            vars_to_starts=self._vars_to_starts,
            num_var_states=self._num_var_states,
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
        """Relevant information for doing belief propagation."""
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
        variables_to_factors: Maps sets of involved variables (in the form of frozensets of
            variable names) to corresponding factors.
        named_factor_groups: Maps the names of named factor groups to the corresponding factor groups.
        factor_group_to_potentials_starts: Maps factor groups to their starting indices in the flat log potentials.
        factor_to_potentials_starts: Maps factors to their starting indices in the flat log potentials.
        factor_to_msgs_starts: Maps factors to their starting indices in the flat ftov messages.
        log_potentials: Flat log potentials array.
        wiring: Wiring derived from the current set of factors.
    """

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
        evidence: evidence (unary log potentials) for variables.
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
    """Function to update log_potentials.

    Args:
        log_potentials: A flat jnp array containing log_potentials.
        updates: A dictionary containing updates for log_potentials
        fg_state: Factor graph state

    Returns:
        A flat jnp array containing updated log_potentials.
    """
    for key in updates:
        data = updates[key]
        if key in fg_state.named_factor_groups:
            factor_group = fg_state.named_factor_groups[key]
            flat_data = factor_group.flatten(data)
            if flat_data.shape != factor_group.factor_group_log_potentials.shape:
                raise ValueError(
                    f"Expected log potentials shape {factor_group.factor_group_log_potentials.shape} "
                    f"for factor group {key}. Got incompatible data shape {data.shape}."
                )

            start = fg_state.factor_group_to_potentials_starts[factor_group]
            log_potentials = log_potentials.at[start : start + flat_data.shape[0]].set(
                flat_data
            )
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
    """Class for storing and manipulating log potentials.

    Args:
        fg_state: Factor graph state
        value: Optionally specify an initial value
    """

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
        """Function to query log potentials for a named factor group or a factor.

        Args:
            key: Name of a named factor group, or a frozenset containing the set
                of involved variables for the queried factor.

        Returned:
            The quried log potentials.
        """
        if not isinstance(key, Hashable):
            key = frozenset(key)

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
            raise ValueError(f"Invalid key {key} for log potentials updates.")

        return log_potentials

    def __setitem__(
        self,
        key: Any,
        data: Union[np.ndarray, jnp.ndarray],
    ):
        """Set the log potentials for a named factor group or a factor.

        Args:
            key: Name of a named factor group, or a frozenset containing the set
                of involved variables for the queried factor.
            data: Array containing the log potentials for the named factor group
                or the factor.
        """
        if not isinstance(key, Hashable):
            key = frozenset(key)

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
    """Function to update ftov_msgs.

    Args:
        ftov_msgs: A flat jnp array containing ftov_msgs.
        updates: A dictionary containing updates for ftov_msgs
        fg_state: Factor graph state

    Returns:
        A flat jnp array containing updated ftov_msgs.
    """
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
                    f"shape {(variable.num_states,)} from factor {keys[0]} "
                    f"to variable {keys[1]}."
                )

            ftov_msgs = ftov_msgs.at[start : start + variable.num_states].set(data)
        elif keys in fg_state.variable_group.keys:
            variable = fg_state.variable_group[keys]
            if data.shape != (variable.num_states,):
                raise ValueError(
                    f"Given belief shape {data.shape} does not match expected "
                    f"shape {(variable.num_states,)} for variable {keys}."
                )

            starts = np.nonzero(
                fg_state.wiring.var_states_for_edges
                == fg_state.vars_to_starts[variable]
            )[0]
            for start in starts:
                ftov_msgs = ftov_msgs.at[start : start + variable.num_states].set(
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
        fg_state: Factor graph state
        value: Optionally specify initial value for ftov messages
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
        if (
            isinstance(keys, tuple)
            and len(keys) == 2
            and keys[1] in self.fg_state.variable_group.keys
        ):
            keys = (frozenset(keys[0]), keys[1])

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
    """Function to update evidence.

    Args:
        evidence: A flat jnp array containing evidence.
        updates: A dictionary containing updates for evidence
        fg_state: Factor graph state

    Returns:
        A flat jnp array containing updated evidence.
    """
    for key in updates:
        data = updates[key]
        if key in fg_state.variable_group.container_keys:
            if key is None:
                variable_group = fg_state.variable_group
            else:
                assert isinstance(
                    fg_state.variable_group, groups.CompositeVariableGroup
                )
                variable_group = fg_state.variable_group.variable_group_container[key]

            start_index = fg_state.vars_to_starts[variable_group.variables[0]]
            flat_data = variable_group.flatten(data)
            evidence = evidence.at[start_index : start_index + flat_data.shape[0]].set(
                flat_data
            )
        else:
            var = fg_state.variable_group[key]
            start_index = fg_state.vars_to_starts[var]
            evidence = evidence.at[start_index : start_index + var.num_states].set(data)

    return evidence


@dataclass(frozen=True, eq=False)
class Evidence:
    """Class for storing and manipulating evidence

    Args:
        fg_state: Factor graph state
        value: Optionally specify initial value for evidence
    """

    fg_state: FactorGraphState
    value: Optional[Union[np.ndarray, jnp.ndarray]] = None

    def __post_init__(self):
        if self.value is None:
            object.__setattr__(self, "value", jnp.zeros(self.fg_state.num_var_states))
        else:
            if self.value.shape != (self.fg_state.num_var_states,):
                raise ValueError(
                    f"Expected evidence shape {(self.fg_state.num_var_states,)}. "
                    f"Got {self.value.shape}."
                )

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
        data: np.ndarray,
    ) -> None:
        """Function to update the evidence for variables

        Args:
            key: The name of a variable group or a single variable.
                If key is the name of a variable group, updates are derived by using the variable group to
                flatten the data.
                If key is the name of a variable, data should be of an array shape (variable_size,)
                If key is None, updates are derived by using self.fg_state.variable_group to flatten the data.
            data: Array containing the evidence updates.
        """
        object.__setattr__(
            self,
            "value",
            update_evidence(
                jax.device_put(self.value), {key: jax.device_put(data)}, self.fg_state
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


def BP(bp_state: BPState, num_iters: int):
    """Function for generating belief propagation functions.

    Args:
        bp_state: Belief propagation state.
        num_iters: Number of belief propagation iterations.

    Returns:
        run_bp: Function for running belief propagation for num_iters.
            Optionally takes as input log_potentials updates, ftov_msgs updates,
            evidence updates, and damping factor, and outputs a BPArrays.
        get_bp_state: Function to reconstruct the BPState from BPArrays.
        get_beliefs: Function to calculate beliefs from BPArrays.
    """
    wiring = jax.device_put(bp_state.fg_state.wiring)
    max_msg_size = int(jnp.max(wiring.edges_num_states))
    num_val_configs = int(wiring.factor_configs_edge_states[-1, 0]) + 1

    @jax.jit
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

        ftov_msgs = jax.device_put(bp_state.ftov_msgs.value)
        if ftov_msgs_updates is not None:
            ftov_msgs = update_ftov_msgs(
                ftov_msgs, ftov_msgs_updates, bp_state.fg_state
            )

        evidence = jax.device_put(bp_state.evidence.value)
        if evidence_updates is not None:
            evidence = update_evidence(evidence, evidence_updates, bp_state.fg_state)

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
            # Normalize and clip these damped, updated messages before
            # returning them.
            msgs = infer.normalize_and_clip_msgs(
                msgs,
                wiring.edges_num_states,
                max_msg_size,
            )
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
    def get_beliefs(bp_arrays: BPArrays):
        """Calculate beliefs from a given BPArrays

        Args:
            bp_arrays: A BPArrays containing arrays for belief propagation.

        Returns:
            beliefs: An array or a PyTree container containing the beliefs for the variables.
        """
        evidence = jax.device_put(bp_arrays.evidence)
        beliefs = bp_state.fg_state.variable_group.unflatten(
            evidence.at[wiring.var_states_for_edges].add(bp_arrays.ftov_msgs)
        )
        return beliefs

    return run_bp, get_bp_state, get_beliefs


@jax.jit
def decode_map_states(beliefs: Any):
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
