from dataclasses import dataclass
from types import MappingProxyType
from typing import Hashable, Sequence, Union

import jax.numpy as jnp
import numpy as np

from pgmax import utils


@dataclass(frozen=True)
class Variable:
    num_states: int
    meta: Hashable


@utils.register_pytree_node_dataclass
@dataclass(frozen=True)
class EnumerationWiring:
    edges_num_states: Union[np.ndarray, jnp.ndarray]
    var_states_for_edges: Union[np.ndarray, jnp.ndarray]
    factor_configs_edge_states: Union[np.ndarray, jnp.ndarray]


@dataclass
class EnumerationFactor:
    """EnumerationFactor.

    Args:
        variables: List of involved variables
        configs: Array of shape (num_configs, num_variables)
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
        if not np.logical_and(self.configs >= 0, self.configs < vars_num_states[None]):
            raise ValueError("Invalid configurations for given variables")

    def compile_wiring(
        self, vars_to_starts: MappingProxyType[Variable, int]
    ) -> EnumerationWiring:
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
                    configs + edges_starts[None],
                ],
                axis=1,
            )

        return EnumerationWiring(
            edges_num_states=self._edges_num_states,
            var_states_for_edges=var_states_for_edges,
            factor_configs_edge_states=self._factor_configs_edge_states,
        )
