from dataclasses import dataclass
from types import MappingProxyType
from typing import Hashable, Sequence, Union

import jax.numpy as jnp
import numpy as np

from pgmax.utils import register_pytree_node_dataclass


@dataclass(frozen=True)
class Variable:
    num_states: int
    meta: Hashable


@register_pytree_node_dataclass
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

        var_states_for_edges = np.concatenate(
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


def concatenate_enumeration_wirings(
    wirings: Sequence[EnumerationWiring],
) -> EnumerationWiring:
    num_factor_configs_cumsum = np.insert(
        np.array(
            [np.max(wiring.factor_configs_edge_states[:, 0]) + 1 for wiring in wirings]
        ).cumsum(),
        0,
        0,
    )[:-1]
    num_edge_states_cumsum = np.insert(
        np.array(
            [wiring.factor_configs_edge_states.shape[0] + 1 for wiring in wirings]
        ).cumsum(),
        0,
        0,
    )[:-1]
    factor_configs_edge_states = []
    for ww, wiring in enumerate(wirings):
        factor_configs_edge_states.append(
            wiring.factor_configs_edge_states
            + np.array(
                [[num_factor_configs_cumsum[ww], num_edge_states_cumsum[ww]]], dtype=int
            )
        )

    return EnumerationWiring(
        edges_num_states=np.concatenate(
            [wiring.edges_num_states for wiring in wirings]
        ),
        var_states_for_edges=np.concatenate(
            [wiring.var_states_for_edges for wiring in wirings]
        ),
        factor_configs_edge_states=np.concatenate(factor_configs_edge_states, axis=0),
    )
