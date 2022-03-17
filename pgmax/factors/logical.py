"""Defines a logical factor"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax import utils
from pgmax.fg import nodes


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class LogicalWiring(nodes.Wiring):
    """Wiring for LogicalFactors.

    Args:
        parents_edge_states: Array of shape (num_parents, 2)
            parents_edge_states[ii, 0] contains the global factor index,
            parents_edge_states[ii, 1] contains the message index of the parent variable's state 0,
            Both indices only take into account the LogicalFactors of the same subtype (OR/AND) of the FactorGraph.
            The parent variable's state 1 is parents_edge_states[ii, 2] + 1.
        children_edge_states: Array of shape (num_factors,)
            children_edge_states[ii] contains the message index of the child variable's state 0,
            which takes into account all the LogicalFactors of the same subtype (OR/AND) of the FactorGraph.
            The child variable's state 1 is children_edge_states[ii, 1] + 1.

    Raises:
        ValueError: If:
            (1) There is a factor index higher than num_logical_factors - 1
            (2) The are no num_logical_factors different factor indices
    """

    parents_edge_states: Union[np.ndarray, jnp.ndarray]
    children_edge_states: Union[np.ndarray, jnp.ndarray]

    def __post_init__(self):
        logical_factor_indices = self.parents_edge_states[:, 0]
        num_logical_factors = self.children_edge_states.shape[0]

        if logical_factor_indices.max() >= num_logical_factors:
            raise ValueError(
                f"The highest OR factor index must be {num_logical_factors - 1}"
            )

        if np.unique(logical_factor_indices).shape[0] != num_logical_factors:
            raise ValueError(
                f"There must be {num_logical_factors} different OR factor indices"
            )

        inference_arguments = {
            "parents_edge_states": self.parents_edge_states,
            "children_edge_states": self.children_edge_states,
        }
        object.__setattr__(self, "inference_arguments", inference_arguments)


@dataclass(frozen=True, eq=False)
class LogicalFactor(nodes.Factor):
    """A logical OR/AND factor
    See https://arxiv.org/pdf/1611.02252.pdf Appendix B

    Raises:
        ValueError: If:
            (1) The variables are not all binary
            (2) There are less than 2 variables
    """

    def __post_init__(self):
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
    ) -> LogicalWiring:
        """Compile LogicalWiring for the LogicalFactor

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
             LogicalWiring for the LogicalFactor
        """
        var_states_for_edges = np.concatenate(
            [
                np.arange(variable.num_states) + vars_to_starts[variable]
                for variable in self.variables
            ]
        )
        return LogicalWiring(
            edges_num_states=self.edges_num_states,
            var_states_for_edges=var_states_for_edges,
            parents_edge_states=self.parents_edge_states,
            children_edge_states=self.child_edge_state,
        )


def concatenate_logical_wirings(
    logical_wirings: Sequence[LogicalWiring],
) -> LogicalWiring:
    """Concatenate a list of LogicalWirings from individual LogicalFactors

    Args:
        logical_wirings: A list of LogicalWirings, one for each individual LogicalFactors.
            All these wirings must have the same type, which is a subclass of LogicalWiring

    Returns:
        Concatenated LogicalWirings

    Raises:
        ValueError: if the list of LogicalWirings is empty
    """
    if len(logical_wirings) == 0:
        raise ValueError("The list of LogicalWirings is empty")

    # Note: this correspomds to all the factor_to_msgs_starts for the EnumerationFactors
    num_edge_states_cumsum = np.insert(
        np.array(
            [wiring.edges_num_states.sum() for wiring in logical_wirings]
        ).cumsum(),
        0,
        0,
    )[:-1]

    parents_edge_states = []
    children_edge_states = []
    for ww, or_wiring in enumerate(logical_wirings):
        offsets = np.array([[ww, num_edge_states_cumsum[ww]]], dtype=int)
        parents_edge_states.append(or_wiring.parents_edge_states + offsets)
        children_edge_states.append(or_wiring.children_edge_states + offsets[:, 1])

    return LogicalWiring(
        edges_num_states=np.concatenate(
            [wiring.edges_num_states for wiring in logical_wirings]
        ),
        var_states_for_edges=np.concatenate(
            [wiring.var_states_for_edges for wiring in logical_wirings]
        ),
        parents_edge_states=np.concatenate(parents_edge_states, axis=0),
        children_edge_states=np.concatenate(children_edge_states, axis=0),
    )


@dataclass(frozen=True, eq=False)
class ORFactor(LogicalFactor):
    pass
