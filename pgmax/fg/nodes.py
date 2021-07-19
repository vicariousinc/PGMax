from dataclasses import dataclass
from typing import Mapping, Sequence, Union

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
@dataclass(frozen=True)
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


@dataclass
class EnumerationFactor:
    """An enumeration factor

    Args:
        variables: List of involved variables
        configs: Array of shape (num_configs, num_variables)
            An array containing explicit enumeration of all valid configurations
    """

    variables: Sequence[Variable]
    configs: np.ndarray

    def __post_init__(self):
        if self.configs.flags.writeable:
            raise ValueError("Configurations need to be immutable.")

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
        if not hasattr(self, "_edges_num_states"):
            self._edges_num_states = np.array(
                [variable.num_states for variable in self.variables], dtype=int
            )

        var_states_for_edges = utils.concatenate_arrays(
            [
                np.arange(variable.num_states) + vars_to_starts[variable]
                for variable in self.variables
            ]
        )
        if not hasattr(self, "_factor_configs_edge_states"):
            configs = self.configs.copy()
            configs.flags.writeable = True
            edges_starts = np.insert(self._edges_num_states.cumsum(), 0, 0)[:-1]
            self._factor_configs_edge_states = np.stack(
                [
                    np.repeat(np.arange(configs.shape[0]), configs.shape[1]),
                    (configs + edges_starts[None]).flatten(),
                ],
                axis=1,
            )

        return EnumerationWiring(
            edges_num_states=self._edges_num_states,
            var_states_for_edges=var_states_for_edges,
            factor_configs_edge_states=self._factor_configs_edge_states,
        )
