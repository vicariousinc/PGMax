from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from pgmax.fg import nodes


@dataclass
class FactorGraph:
    variables: Sequence[nodes.Variable]
    factors: Sequence[nodes.EnumerationFactor]

    def __post_init__(self):
        vars_num_states_cumsum = np.insert(
            np.array(
                [variable.num_states for variable in self.variables], dtype=int
            ).cumsum(),
            0,
            0,
        )
        self._vars_to_starts = MappingProxyType(
            {
                variable: vars_num_states_cumsum[vv]
                for vv, variable in enumerate(self.variables)
            }
        )
        self.num_var_states = vars_num_states_cumsum[-1]

    def compile_wiring(self) -> nodes.EnumerationWiring:
        wirings = [
            factor.compile_wiring(self._vars_to_starts) for factor in self.factors
        ]
        return nodes.concatenate_enumeration_wirings(wirings)

    def get_evidence(self, data: Any, context: Any) -> jnp.ndarray:
        raise NotImplementedError("get_evidence function needs to be implemented")

    def init_msgs(
        self, wiring: nodes.EnumerationWiring, context: Any = None
    ) -> jnp.ndarray:
        return jnp.zeros(wiring.var_states_for_edges.shape[0])
