"""A module containing the core class to build a factor graph."""

import collections
import copy
import functools
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    OrderedDict,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np

from pgmax import factor, fgroup, vgroup
from pgmax.factor import FAC_TO_VAR_UPDATES
from pgmax.utils import cached_property


@dataclass(frozen=True, eq=False)
class FactorGraphState:
    """FactorGraphState.

    Args:
        variable_groups: VarGroups in the FactorGraph.
        vars_to_starts: Maps variables to their starting indices in the flat evidence array.
            flat_evidence[vars_to_starts[variable]: vars_to_starts[variable] + variable.num_var_states]
            contains evidence to the variable.
        num_var_states: Total number of variable states.
        total_factor_num_states: Size of the flat ftov messages array.
        factor_type_to_msgs_range: Maps factors types to their start and end indices in the flat ftov messages.
        factor_type_to_potentials_range: Maps factor types to their start and end indices in the flat log potentials.
        factor_group_to_potentials_starts: Maps factor groups to their starting indices in the flat log potentials.
        log_potentials: Flat log potentials array concatenated for each factor type.
        wiring: Wiring derived for each factor type.
    """

    variable_groups: Sequence[vgroup.VarGroup]
    vars_to_starts: Mapping[Tuple[int, int], int]
    num_var_states: int
    total_factor_num_states: int
    factor_type_to_msgs_range: OrderedDict[type, Tuple[int, int]]
    factor_type_to_potentials_range: OrderedDict[type, Tuple[int, int]]
    factor_group_to_potentials_starts: OrderedDict[fgroup.FactorGroup, int]
    log_potentials: OrderedDict[type, Union[None, np.ndarray]]
    wiring: OrderedDict[type, factor.Wiring]

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            if isinstance(getattr(self, field), np.ndarray):
                getattr(self, field).flags.writeable = False

            if isinstance(getattr(self, field), Mapping):
                object.__setattr__(self, field, MappingProxyType(getattr(self, field)))


@dataclass
class FactorGraph:
    """Class for representing a factor graph.
    Factors in a graph are clustered in factor groups, which are grouped according to their factor types.

    Args:
        variable_groups: A single VarGroup or a list of VarGroups.
    """

    variable_groups: Union[vgroup.VarGroup, Sequence[vgroup.VarGroup]]

    def __post_init__(self):
        if isinstance(self.variable_groups, vgroup.VarGroup):
            self.variable_groups = [self.variable_groups]

        # Useful objects to build the FactorGraph
        self._factor_types_to_groups: OrderedDict[
            Type, List[fgroup.FactorGroup]
        ] = collections.OrderedDict(
            [(factor_type, []) for factor_type in FAC_TO_VAR_UPDATES]
        )
        self._factor_types_to_variables_for_factors: OrderedDict[
            Type, Set[FrozenSet]
        ] = collections.OrderedDict(
            [(factor_type, set()) for factor_type in FAC_TO_VAR_UPDATES]
        )

        # See FactorGraphState docstrings for documentation on the following fields
        self._vars_to_starts: Dict[Tuple[int, int], int] = {}

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

    def __hash__(self) -> int:
        all_factor_groups = tuple(
            [
                factor_group
                for factor_groups_per_type in self._factor_types_to_groups.values()
                for factor_group in factor_groups_per_type
            ]
        )
        return hash(all_factor_groups)

    def add_factors(
        self,
        factors: Union[
            factor.Factor,
            fgroup.FactorGroup,
            Sequence[Union[factor.Factor, fgroup.FactorGroup]],
        ],
    ) -> None:
        """Add a single Factor, a FactorGroup or a list of single Factors and FactorGroups to the FactorGraph,
        by updating the FactorGraphState.

        Args:
            factors: The Factor, FactorGroup or list of Factors and FactorGroups to be added to the FactorGraph.

        Raises:
            ValueError: A FactorGroup involving the same variables already exists in the FactorGraph.
        """
        if isinstance(factors, list):
            for this_factor in factors:
                self.add_factors(this_factor)
            return None

        if isinstance(factors, fgroup.FactorGroup):
            factor_group = factors
        elif isinstance(factors, factor.Factor):
            factor_group = fgroup.SingleFactorGroup(
                variables_for_factors=[factors.variables],
                factor=factors,
            )

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

                factor_num_states_cumsum += factor_group.factor_edges_num_states.sum()
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
    def wiring(self) -> OrderedDict[Type, factor.Wiring]:
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
    def factors(self) -> OrderedDict[Type, Tuple[factor.Factor, ...]]:
        """Mapping factor type to individual factors in the factor graph.
        This function is only called on demand when the user requires it."""
        print(
            "Factors have not been added to the factor graph yet, this may take a while..."
        )

        factors: OrderedDict[Type, Tuple[factor.Factor, ...]] = collections.OrderedDict(
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
    def factor_groups(self) -> OrderedDict[Type, List[fgroup.FactorGroup]]:
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
        assert isinstance(self.variable_groups, list)

        return FactorGraphState(
            variable_groups=self.variable_groups,
            vars_to_starts=self._vars_to_starts,
            num_var_states=self._num_var_states,
            total_factor_num_states=self._total_factor_num_states,
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
    def bp_state(self) -> Any:
        """Relevant information for doing belief propagation."""
        # Preliminary computations
        self.compute_offsets()

        from pgmax.infer import bp_state

        return bp_state.BPState(
            log_potentials=bp_state.LogPotentials(fg_state=self.fg_state),
            ftov_msgs=bp_state.FToVMessages(fg_state=self.fg_state),
            evidence=bp_state.Evidence(fg_state=self.fg_state),
        )
