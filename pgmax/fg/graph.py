from __future__ import annotations

"""A module containing the core class to specify a Factor Graph."""

import collections
import copy
import functools
import inspect
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
    Set,
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
        variable_groups: A single VariableGroup or a list of VariableGroups.
    """

    variable_groups: Union[groups.VariableGroup, Sequence[groups.VariableGroup]]

    def __post_init__(self):
        import time

        start = time.time()
        if isinstance(self.variable_groups, groups.VariableGroup):
            self.variable_groups = [self.variable_groups]

        # Useful objects to build the FactorGraph
        self._factor_types_to_groups: OrderedDict[
            Type, List[groups.FactorGroup]
        ] = collections.OrderedDict(
            [(factor_type, []) for factor_type in FAC_TO_VAR_UPDATES]
        )
        self._factor_types_to_variables_for_factors: OrderedDict[
            Type, Set[FrozenSet]
        ] = collections.OrderedDict(
            [(factor_type, set()) for factor_type in FAC_TO_VAR_UPDATES]
        )

        # See FactorGraphState docstrings for documentation on the following fields
        self._vars_to_starts: OrderedDict[
            Tuple[int, int], int
        ] = collections.OrderedDict()
        vars_num_states_cumsum = 0
        for variable_group in self.variable_groups:
            vg_num_states = variable_group.num_states.flatten()
            vg_num_states_cumsum = np.insert(np.cumsum(vg_num_states), 0, 0)
            self._vars_to_starts.update(
                zip(
                    variable_group.variables,
                    vars_num_states_cumsum + vg_num_states_cumsum[:-1],
                )
            )
            vars_num_states_cumsum += vg_num_states_cumsum[-1]
        self._num_var_states = vars_num_states_cumsum
        print("Init", time.time() - start)

    def __hash__(self) -> int:
        return hash(self.factor_groups)

    def add_factors(
        self,
        factor_group: Optional[groups.FactorGroup] = None,
        factor: Optional[nodes.Factor] = None,
    ) -> None:
        """Add a FactorGroup or a single Factor to the FactorGraph, by updating the FactorGraphState.

        Args:
            factor_group: The FactorGroup to be added to the FactorGraph.
            factor: The Factor to be added to the FactorGraph.

        Raises:
            ValueError: If
                (1) Both a Factor and a FactorGroup are added
                (2) The FactorGroup involving the same variables already exists in the FactorGraph.
        """
        if factor is None and factor_group is None:
            raise ValueError("A Factor or a FactorGroup is required")

        if factor is not None and factor_group is not None:
            raise ValueError("Cannot simultaneously add a Factor and a FactorGroup")

        if factor is not None:
            factor_group = groups.SingleFactorGroup(
                variables_for_factors=[factor.variables],
                factor=factor,
            )
        assert factor_group is not None

        factor_type = factor_group.factor_type
        for var_names_for_factor in factor_group.variables_for_factors:
            var_names = frozenset(var_names_for_factor)
            if var_names in self._factor_types_to_variables_for_factors[factor_type]:
                raise ValueError(
                    f"A Factor of type {factor_type} involving variables {var_names} already exists. Please merge the corresponding factors."
                )
            self._factor_types_to_variables_for_factors[factor_type].add(var_names)

        self._factor_types_to_groups[factor_type].append(factor_group)

    @functools.lru_cache(None)
    def compute_offsets(self) -> None:
        """Compute factor messages offsets for the factor types and factor groups
        in the flattened array of message.
        Also compute log potentials offsets for factor groups.

        See FactorGraphState for documentation on the following fields

        If offsets have already beeen compiled, do nothing.
        """
        # Message offsets for ftov messages
        self._factor_type_to_msgs_range = collections.OrderedDict()
        self._factor_group_to_msgs_starts = collections.OrderedDict()
        factor_num_states_cumsum = 0

        # Log potentials offsets
        self._factor_type_to_potentials_range = collections.OrderedDict()
        self._factor_group_to_potentials_starts = collections.OrderedDict()
        factor_num_configs_cumsum = 0

        for factor_type, factors_groups_by_type in self._factor_types_to_groups.items():
            factor_type_num_states_start = factor_num_states_cumsum
            factor_type_num_configs_start = factor_num_configs_cumsum
            for factor_group in factors_groups_by_type:
                self._factor_group_to_msgs_starts[
                    factor_group
                ] = factor_num_states_cumsum
                self._factor_group_to_potentials_starts[
                    factor_group
                ] = factor_num_configs_cumsum

                factor_num_states_cumsum += factor_group.total_num_states
                factor_num_configs_cumsum += (
                    factor_group.factor_group_log_potentials.shape[0]
                )

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
                        factor_group.compile_wiring(self._vars_to_starts)
                        for factor_group in self._factor_types_to_groups[factor_type]
                    ],
                )
                for factor_type in self._factor_types_to_groups
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

        If potential array has already been compiled, do nothing.

        Returns:
            A dictionnary mapping each factor type to the array of the log of the potential
                function for each valid configuration
        """
        log_potentials = collections.OrderedDict()
        for factor_type, factors_groups_by_type in self._factor_types_to_groups.items():
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
        """Mapping factor type to individual factors in the factor graph.
        This function is only called on demand when the user requires it."""
        print(
            "Factors have not been added to the factor graph yet, this may take a while..."
        )

        factors: OrderedDict[Type, Tuple[nodes.Factor, ...]] = collections.OrderedDict(
            [
                (
                    factor_type,
                    tuple(
                        [
                            factor
                            for factor_group in self._factor_types_to_groups[
                                factor_type
                            ]
                            for factor in factor_group.factors
                        ]
                    ),
                )
                for factor_type in self._factor_types_to_groups
            ]
        )
        return factors

    @property
    def factor_groups(self) -> Tuple[groups.FactorGroup, ...]:
        """Tuple of factor groups in the factor graph"""
        return tuple(
            [
                factor_group
                for factor_groups_per_type in self._factor_types_to_groups.values()
                for factor_group in factor_groups_per_type
            ]
        )

    @cached_property
    def fg_state(self) -> FactorGraphState:
        """Current factor graph state given the added factors."""
        # Preliminary computations
        self.compute_offsets()
        log_potentials = np.concatenate(
            [self.log_potentials[factor_type] for factor_type in self.log_potentials]
        )
        assert isinstance(self.variable_groups, list)

        return FactorGraphState(
            variable_groups=self.variable_groups,
            vars_to_starts=self._vars_to_starts,
            num_var_states=self._num_var_states,
            total_factor_num_states=self._total_factor_num_states,
            factor_groups=self.factor_groups,
            factor_type_to_msgs_range=copy.copy(self._factor_type_to_msgs_range),
            factor_type_to_potentials_range=copy.copy(
                self._factor_type_to_potentials_range
            ),
            factor_group_to_potentials_starts=copy.copy(
                self._factor_group_to_potentials_starts
            ),
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
        variable_groups: VariableGroups in the FactorGraph.
        vars_to_starts: Maps variables to their starting indices in the flat evidence array.
            flat_evidence[vars_to_starts[variable]: vars_to_starts[variable] + variable.num_var_states]
            contains evidence to the variable.
        num_var_states: Total number of variable states.
        total_factor_num_states: Size of the flat ftov messages array.
        factor_groups: FactorGroups in the FactorGraph
        factor_type_to_msgs_range: Maps factors types to their start and end indices in the flat ftov messages.
        factor_type_to_potentials_range: Maps factor types to their start and end indices in the flat log potentials.
        factor_group_to_potentials_starts: Maps factor groups to their starting indices in the flat log potentials.
        log_potentials: Flat log potentials array concatenated for each factor type.
        wiring: Wiring derived for each factor type.
    """

    variable_groups: Sequence[groups.VariableGroup]
    vars_to_starts: Mapping[Tuple[Any, int], int]
    num_var_states: int
    total_factor_num_states: int
    factor_groups: Tuple[groups.FactorGroup, ...]
    factor_type_to_msgs_range: OrderedDict[type, Tuple[int, int]]
    factor_type_to_potentials_range: OrderedDict[type, Tuple[int, int]]
    factor_group_to_potentials_starts: OrderedDict[groups.FactorGroup, int]
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
    for factor_group, data in updates.items():
        if factor_group in fg_state.factor_groups:
            flat_data = factor_group.flatten(data)
            if flat_data.shape != factor_group.factor_group_log_potentials.shape:
                raise ValueError(
                    f"Expected log potentials shape {factor_group.factor_group_log_potentials.shape} "
                    f"for factor group. Got incompatible data shape {data.shape}."
                )

            start = fg_state.factor_group_to_potentials_starts[factor_group]
            log_potentials = log_potentials.at[start : start + flat_data.shape[0]].set(
                flat_data
            )
        else:
            raise ValueError("Invalid FactorGroup for log potentials updates.")

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

    def __getitem__(self, factor_group: groups.FactorGroup) -> np.ndarray:
        """Function to query log potentials for a FactorGroup.

        Args:
            factor_group: Queried FactorGroup

        Returns:
            The queried log potentials.
        """
        value = cast(np.ndarray, self.value)
        if factor_group in self.fg_state.factor_groups:
            start = self.fg_state.factor_group_to_potentials_starts[factor_group]
            log_potentials = value[
                start : start + factor_group.factor_group_log_potentials.shape[0]
            ]
        else:
            raise ValueError("Invalid FactorGroup for log potentials updates.")
        return log_potentials

    def __setitem__(
        self,
        factor_group: Any,
        data: Union[np.ndarray, jnp.ndarray],
    ):
        """Set the log potentials for a FactorGroup

        Args:
            factor_group: FactorGroup
            data: Array containing the log potentials for the FactorGroup
        """
        object.__setattr__(
            self,
            "value",
            np.asarray(
                update_log_potentials(
                    jax.device_put(self.value),
                    {factor_group: jax.device_put(data)},
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
        (2) provided variable is not in the FactorGraph.
    """
    for variable, data in updates.items():
        if variable in fg_state.vars_to_starts:
            if data.shape != (variable[1],):
                raise ValueError(
                    f"Given belief shape {data.shape} does not match expected "
                    f"shape {(variable[1],)} for variable {variable}."
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
                ftov_msgs = ftov_msgs.at[start : start + variable[1]].set(
                    data / starts.shape[0]
                )
        else:
            raise ValueError("Provided variable is not in the FactorGraph")
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

    def __setitem__(
        self,
        variable: Tuple[Any, int],
        data: Union[np.ndarray, jnp.ndarray],
    ) -> None:
        """Spreading beliefs at a variable to all connected Factors

        Args:
            variable: A tuple representing a variable
            data: An array containing the beliefs to be spread uniformly
                across all factors to variable messages involving this variable.
        """

        object.__setattr__(
            self,
            "value",
            np.asarray(
                update_ftov_msgs(
                    jax.device_put(self.value),
                    {variable: jax.device_put(data)},
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
    for name, data in updates.items():
        # Name is a variable_group or a variable
        if name in fg_state.variable_groups:
            first_variable = name.variables[0]
            start_index = fg_state.vars_to_starts[first_variable]
            flat_data = name.flatten(data)
            evidence = evidence.at[start_index : start_index + flat_data.shape[0]].set(
                flat_data
            )
        elif name in fg_state.vars_to_starts:
            start_index = fg_state.vars_to_starts[name]
            evidence = evidence.at[start_index : start_index + name[1]].set(data)
        else:
            raise ValueError(
                "Got evidence for a variable or a VariableGroup not in the FactorGraph!"
            )
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

    def __getitem__(self, variable: Tuple[Any, int]) -> np.ndarray:
        """Function to query evidence for a variable

        Args:
            variable: Variable queried

        Returns:
            evidence for the queried variable
        """
        value = cast(np.ndarray, self.value)
        start = self.fg_state.vars_to_starts[variable]
        evidence = value[start : start + variable[1]]
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
                If name is None, updates are derived by using self.fg_state.variable_groups to flatten the data.
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


@dataclass(frozen=True, eq=False)
class BeliefPropagation:
    """Belief propagation functions.

    Arguments:
        init: Function to create log_potentials, ftov_msgs and evidence.
            Args:
                log_potentials_updates: Optional dictionary containing log_potentials updates.
                ftov_msgs_updates: Optional dictionary containing ftov_msgs updates.
                evidence_updates: Optional dictionary containing evidence updates.
            Returns:
                A BPArrays with the log_potentials, ftov_msgs and evidence.

        update: Function to update log_potentials, ftov_msgs and evidence.
            Args:
                bp_arrays: Optional arrays of log_potentials, ftov_msgs, evidence.
                log_potentials_updates: Optional dictionary containing log_potentials updates.
                ftov_msgs_updates: Optional dictionary containing ftov_msgs updates.
                evidence_updates: Optional dictionary containing evidence updates.
            Returns:
                A BPArrays with the updated log_potentials, ftov_msgs and evidence.

        run_bp: Function to run belief propagation for num_iters with a damping_factor.
            Args:
                bp_arrays: Initial arrays of log_potentials, ftov_msgs, evidence.
                num_iters: Number of belief propagation iterations.
                damping: The damping factor to use for message updates between one timestep and the next.
            Returns:
                A BPArrays containing the updated ftov_msgs.

        get_bp_state: Function to reconstruct the BPState from a BPArrays.
            Args:
                bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.
            Returns:
                The reconstructed BPState

        get_beliefs: Function to calculate beliefs from a BPArrays.
            Args:
                bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.
            Returns:
                beliefs: Beliefs returned by belief propagation.
    """

    init: Callable
    update: Callable
    run_bp: Callable
    to_bp_state: Callable
    get_beliefs: Callable


def BP(bp_state: BPState, temperature: float = 0.0) -> BeliefPropagation:
    """Function for generating belief propagation functions.

    Args:
        bp_state: Belief propagation state.
        temperature: Temperature for loopy belief propagation.
            1.0 corresponds to sum-product, 0.0 corresponds to max-product.

    Returns:
        Belief propagation functions.
    """
    wiring = bp_state.fg_state.wiring
    edges_num_states = np.concatenate(
        [wiring[factor_type].edges_num_states for factor_type in FAC_TO_VAR_UPDATES]
    )
    max_msg_size = int(np.max(edges_num_states))

    var_states_for_edges = np.concatenate(
        [wiring[factor_type].var_states_for_edges for factor_type in FAC_TO_VAR_UPDATES]
    )

    # Inference argumnets per factor type
    inference_arguments: Dict[type, Mapping] = {}
    for factor_type in FAC_TO_VAR_UPDATES:
        this_inference_arguments = inspect.getfullargspec(
            FAC_TO_VAR_UPDATES[factor_type]
        ).args
        this_inference_arguments.remove("vtof_msgs")
        this_inference_arguments.remove("log_potentials")
        this_inference_arguments.remove("temperature")
        this_inference_arguments = {
            key: getattr(wiring[factor_type], key) for key in this_inference_arguments
        }
        inference_arguments[factor_type] = this_inference_arguments

    factor_type_to_msgs_range = bp_state.fg_state.factor_type_to_msgs_range
    factor_type_to_potentials_range = bp_state.fg_state.factor_type_to_potentials_range

    def update(
        bp_arrays: Optional[BPArrays] = None,
        log_potentials_updates: Optional[Dict[Any, jnp.ndarray]] = None,
        ftov_msgs_updates: Optional[Dict[Any, jnp.ndarray]] = None,
        evidence_updates: Optional[Dict[Any, jnp.ndarray]] = None,
    ) -> BPArrays:
        """Function to update belief propagation log_potentials, ftov_msgs, evidence.

        Args:
            bp_arrays: Optional arrays of log_potentials, ftov_msgs, evidence.
            log_potentials_updates: Optional dictionary containing log_potentials updates.
            ftov_msgs_updates: Optional dictionary containing ftov_msgs updates.
            evidence_updates: Optional dictionary containing evidence updates.

        Returns:
            A BPArrays with the updated log_potentials, ftov_msgs and evidence.
        """
        if bp_arrays is not None:
            log_potentials = bp_arrays.log_potentials
            evidence = bp_arrays.evidence
            ftov_msgs = bp_arrays.ftov_msgs
        else:
            log_potentials = jax.device_put(bp_state.log_potentials.value)
            ftov_msgs = bp_state.ftov_msgs.value
            evidence = bp_state.evidence.value

        if log_potentials_updates is not None:
            log_potentials = update_log_potentials(
                log_potentials, log_potentials_updates, bp_state.fg_state
            )

        if ftov_msgs_updates is not None:
            ftov_msgs = update_ftov_msgs(
                ftov_msgs, ftov_msgs_updates, bp_state.fg_state
            )

        if evidence_updates is not None:
            evidence = update_evidence(evidence, evidence_updates, bp_state.fg_state)

        return BPArrays(
            log_potentials=log_potentials, ftov_msgs=ftov_msgs, evidence=evidence
        )

    def run_bp(
        bp_arrays: BPArrays,
        num_iters: int,
        damping: float = 0.5,
    ) -> BPArrays:
        """Function to run belief propagation for num_iters with a damping_factor.

        Args:
            bp_arrays: Initial arrays of log_potentials, ftov_msgs, evidence.
            num_iters: Number of belief propagation iterations.
            damping: The damping factor to use for message updates between one timestep and the next.

        Returns:
            A BPArrays containing the updated ftov_msgs.
        """
        log_potentials = bp_arrays.log_potentials
        evidence = bp_arrays.evidence
        ftov_msgs = bp_arrays.ftov_msgs

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
            # Normalize and clip these damped, updated messages before returning them.
            msgs = infer.normalize_and_clip_msgs(msgs, edges_num_states, max_msg_size)
            return msgs, None

        ftov_msgs, _ = jax.lax.scan(update, ftov_msgs, None, num_iters)

        return BPArrays(
            log_potentials=log_potentials, ftov_msgs=ftov_msgs, evidence=evidence
        )

    def to_bp_state(bp_arrays: BPArrays) -> BPState:
        """Function to reconstruct the BPState from a BPArrays

        Args:
            bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.

        Returns:
            The reconstructed BPState
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

    def unflatten_beliefs(flat_beliefs, variable_groups) -> Dict[Hashable, Any]:
        """Function that returns unflattened beliefs from the flat beliefs

        Args:
            flat_beliefs: Flattened array of beliefs
            variable_groups: All the variable groups in the FactorGraph.
        """
        beliefs = {}
        start = 0
        for variable_group in variable_groups:
            num_states = variable_group.num_states
            assert isinstance(num_states, np.ndarray)
            length = num_states.sum()

            beliefs[variable_group] = variable_group.unflatten(
                flat_beliefs[start : start + length]
            )
            start += length
        return beliefs

    @jax.jit
    def get_beliefs(bp_arrays: BPArrays) -> Dict[Hashable, Any]:
        """Function to calculate beliefs from a BPArrays

        Args:
            bp_arrays: A BPArrays containing log_potentials, ftov_msgs, evidence.

        Returns:
            beliefs: Beliefs returned by belief propagation.
        """

        flat_beliefs = (
            jax.device_put(bp_arrays.evidence)
            .at[jax.device_put(var_states_for_edges)]
            .add(bp_arrays.ftov_msgs)
        )
        return unflatten_beliefs(flat_beliefs, bp_state.fg_state.variable_groups)

    bp = BeliefPropagation(
        init=functools.partial(update, None),
        update=update,
        run_bp=run_bp,
        to_bp_state=to_bp_state,
        get_beliefs=get_beliefs,
    )
    return bp


@jax.jit
def decode_map_states(beliefs: Dict[Hashable, Any]) -> Any:
    """Function to decode MAP states given the calculated beliefs.

    Args:
        beliefs: An array or a PyTree container containing beliefs for different variables.

    Returns:
        An array or a PyTree container containing the MAP states for different variables.
    """
    return jax.tree_util.tree_map(lambda x: jnp.argmax(x, axis=-1), beliefs)


@jax.jit
def get_marginals(beliefs: Dict[Hashable, Any]) -> Any:
    """Function to get marginal probabilities given the calculated beliefs.

    Args:
        beliefs: An array or a PyTree container containing beliefs for different variables.

    Returns:
        An array or a PyTree container containing the marginal probabilities different variables.
    """
    return jax.tree_util.tree_map(
        lambda x: jnp.exp(x - logsumexp(x, axis=-1, keepdims=True)), beliefs
    )
