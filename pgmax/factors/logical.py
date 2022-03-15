"""Defines an enumeration factor"""

from dataclasses import dataclass, field
from typing import Mapping, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax import utils
from pgmax.fg import nodes

ALLOWED_LOGICAL_TYPES = ["OR"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class LogicalWiring(nodes.Wiring):
    """Wiring for LogicalFactors.

    Args:
        parents_edge_states: Array of shape (num_parents, 2)
            parents_edge_states[ii, 0] contains the global factor index,
            which takes into account all the LogicalFactors of the same subtype (AND / OR).
            parents_edge_states[ii, 1] contains the message index of the parent variable's state 0,
            which takes into account all the EnumerationFactors and LogicalFactors.
            The parent variable's state 1 is parents_edge_states[ii, 2] + 1.
        children_edge_states: Array of shape (num_factors,)
            children_edge_states[ii] contains the message index of the child variable's state 0,
            which takes into account all the EnumerationFactors and LogicalFactors.
            The child variable's state 1 is children_edge_states[ii, 1] + 1.
    """

    parents_edge_states: Union[np.ndarray, jnp.ndarray]
    children_edge_states: Union[np.ndarray, jnp.ndarray]


@dataclass(frozen=True, eq=False)
class LogicalFactor(nodes.Factor):
    """A logical OR/AND factor
    Args:
        variables: List of connected variables.
            The last variable is assumed to be the child one.
        logical_type: The logical condition supported by the factor

    Raises:
        ValueError: If:
            (1) The variables are not all binary
            (2) There are less than 2 variables
    """

    logical_type: str
    log_potentials: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        if self.logical_type not in ALLOWED_LOGICAL_TYPES:
            raise ValueError(
                "Logical type {logical_type} is not one of the supported type {ALLOWED_LOGICAL_TYPES}"
            )

        if not np.all([variable.num_states == 2 for variable in self.variables]):
            raise ValueError("All variables should all be binary")

        if len(self.variables) < 2:
            raise ValueError(
                "At least one parent variable and one child variable is required"
            )

    @utils.cached_property
    def parents_edge_states(self) -> np.ndarray:
        """
        Returns:
            The parents variables edge states.
        """
        num_parents = len(self.variables) - 1

        parents_edge_states = np.vstack(
            [
                np.zeros(num_parents, dtype=int),
                np.arange(0, 2 * num_parents, 2, dtype=int),
            ],
        ).T
        return parents_edge_states

    @utils.cached_property
    def child_edge_state(self) -> np.ndarray:
        """
        Returns:
            The child variable edge states.
        """
        return np.array([2 * (len(self.variables) - 1)], dtype=int)

    def compile_wiring(
        self, vars_to_starts: Mapping[nodes.Variable, int]
    ) -> np.ndarray:
        """Compile LogicalWiring for the LogicalFactor

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
             LogicalWiring for the LogicalFactor
        """
        parents_var_states_for_edges = np.concatenate(
            [
                np.arange(variable.num_states) + vars_to_starts[variable]
                for variable in self.variables[:-1]
            ]
        )
        child_var_states_for_edges = (
            np.arange(self.variables[-1].num_states)
            + vars_to_starts[self.variables[-1]]
        )
        var_states_for_edges = np.concatenate(
            [parents_var_states_for_edges, child_var_states_for_edges]
        )
        return LogicalWiring(
            edges_num_states=self.edges_num_states,
            var_states_for_edges=var_states_for_edges,
            parents_edge_states=self.parents_edge_states,
            children_edge_states=self.child_edge_state,
        )
