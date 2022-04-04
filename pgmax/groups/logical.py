"""Defines LogicalFactorGroup and its two children, ORFactorGroup and ANDFactorGroup."""

import collections
from dataclasses import dataclass, field
from typing import FrozenSet, OrderedDict, Type

from pgmax.factors import logical
from pgmax.fg import groups


@dataclass(frozen=True, eq=False)
class LogicalFactorGroup(groups.FactorGroup):
    """Class to represent a group of LogicalFactors.

    All factors in the group are assumed to have the same edge_states_offset.
    Consequently, the factors are all ORFactors or ANDFactors.

    Args:
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ORFactors the edge_states_offset is 1, for ANDFactors the edge_states_offset is -1.
    """

    edge_states_offset: int = field(init=False)

    def compile_wiring(self, vars_to_starts) -> logical.LogicalWiring:
        """Compile LogicalWiring for the LogicalFactorGroup

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
             LogicalWiring for the LogicalFactorGroup
        """
        return logical.compile_logical_wiring(
            factor_edges_num_states=self.factor_edges_num_states,
            variables_for_factors=self.variables_for_factors,
            vars_to_starts=vars_to_starts,
            edge_states_offset=self.edge_states_offset,
        )

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, logical.LogicalFactor]:
        """Function that generates a dictionary mapping set of connected variables to factors.
        This function is only called on demand when the user requires it.

        Returns:
            A dictionary mapping all possible set of connected variables to different factors.
        """
        variables_to_factors = collections.OrderedDict(
            [
                (
                    frozenset(self.variable_names_for_factors[ii]),
                    self.factor_type(
                        variables=tuple(
                            self.variable_group[self.variable_names_for_factors[ii]]
                        ),
                    ),
                )
                for ii in range(len(self.variable_names_for_factors))
            ]
        )
        return variables_to_factors


@dataclass(frozen=True, eq=False)
class ORFactorGroup(LogicalFactorGroup):
    """Class to represent a group of ORFactors.

    Args:
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ORFactors the edge_states_offset is 1.
    """

    edge_states_offset: int = field(init=False, default=1)
    factor_type: Type = field(init=False, default=logical.ORFactor)


@dataclass(frozen=True, eq=False)
class ANDFactorGroup(LogicalFactorGroup):
    """Class to represent a group of ANDFactors.

    Args:
        edge_states_offset: Offset to go from a variable's relevant state to its other state
            For ANDFactors the edge_states_offset is -1.
    """

    edge_states_offset: int = field(init=False, default=-1)
    factor_type: Type = field(init=False, default=logical.ANDFactor)
