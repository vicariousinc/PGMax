"""A module containing classes that specify the components of a Factor Graph."""

from dataclasses import dataclass
from typing import Mapping, Tuple, Union

import jax.numpy as jnp
import numpy as np

from pgmax import utils


@dataclass(frozen=True, eq=False)
class Variable:
    """Base class for variables.
    Concrete variables can have additional associated meta information.
    """

    num_states: int


@utils.register_pytree_node_dataclass
@dataclass(frozen=True, eq=False)
class EnumerationWiring:
    """Wiring for enumeration factors.

    Args:
        edges_num_states: Array of shape (num_edges,)
            Number of states for the variables connected to each edge
        var_states_for_edges: Array of shape (num_edge_states,)
            Global variable state indices for each edge state
        factor_configs_edge_states: Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global factor_config and edge_state indices
            factor_configs_edge_states[ii, 0] contains the global factor config index
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index
    """

    edges_num_states: Union[np.ndarray, jnp.ndarray]
    var_states_for_edges: Union[np.ndarray, jnp.ndarray]
    factor_configs_edge_states: Union[np.ndarray, jnp.ndarray]

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            getattr(self, field).flags.writeable = False


@dataclass(frozen=True, eq=False)
class EnumerationFactor:
    """An enumeration factor

    Args:
        variables: List of involved variables
        configs: Array of shape (num_configs, num_variables)
            An array containing explicit enumeration of all valid configurations
    """

    variables: Tuple[Variable, ...]
    configs: np.ndarray

    def __post_init__(self):
        self.configs.flags.writeable = False
        if not np.issubdtype(self.configs.dtype, np.integer):
            raise ValueError(
                f"Configurations should be integers. Got {self.configs.dtype}."
            )

        if len(self.variables) != self.configs.shape[1]:
            raise ValueError(
                f"Number of variables {len(self.variables)} doesn't match given configurations {self.configs.shape}"
            )

        vars_num_states = np.array([variable.num_states for variable in self.variables])
        if not np.logical_and(
            self.configs >= 0, self.configs < vars_num_states[None]
        ).all():
            raise ValueError("Invalid configurations for given variables")

    @utils.cached_property
    def edges_num_states(self) -> np.ndarray:
        """Number of states for the variables connected to each edge

        Returns:
            Array of shape (num_edges,)
            Number of states for the variables connected to each edge
        """
        edge_num_states = np.array(
            [variable.num_states for variable in self.variables], dtype=int
        )
        return edge_num_states

    @utils.cached_property
    def factor_configs_edge_states(self) -> np.ndarray:
        """Array containing factor configs and edge states pairs

        Returns:
            Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global factor_config and edge_state indices
            factor_configs_edge_states[ii, 0] contains the global factor config index
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index
        """
        edges_starts = np.insert(self.edges_num_states.cumsum(), 0, 0)[:-1]
        factor_configs_edge_states = np.stack(
            [
                np.repeat(np.arange(self.configs.shape[0]), self.configs.shape[1]),
                (self.configs + edges_starts[None]).flatten(),
            ],
            axis=1,
        )
        return factor_configs_edge_states

    def compile_wiring(
        self, vars_to_starts: Mapping[Variable, int]
    ) -> EnumerationWiring:
        """Compile enumeration wiring for the enumeration factor

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
            Enumeration wiring for the enumeration factor
        """
        var_states_for_edges = np.concatenate(
            [
                np.arange(variable.num_states) + vars_to_starts[variable]
                for variable in self.variables
            ]
        )
        return EnumerationWiring(
            edges_num_states=self.edges_num_states,
            var_states_for_edges=var_states_for_edges,
            factor_configs_edge_states=self.factor_configs_edge_states,
        )
