from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from pgmax.fg import fg_utils, nodes


@dataclass
class FactorGraph:
    """Base class for factor graph
    Concrete factor graphs inherits from this class, and specifies get_evidence to generate
    the evidence array, and optionally init_msgs (default to initializing all messages to 0)

    Args:
        variables: List of involved variables
        factors: List of involved factors
    """

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

    def compile_wiring(self) -> None:
        """Compile wiring for belief propagation inference using JAX"""
        wirings = [
            factor.compile_wiring(self._vars_to_starts) for factor in self.factors
        ]
        self._wiring = fg_utils.concatenate_enumeration_wirings(wirings)

    def get_evidence(self, data: Any, context: Any) -> jnp.ndarray:
        """Function to generate evidence array. Need to be overwritten for concrete factor graphs

        Args:
            data: Data for generating evidence
            context: Optional context for generating evidence

        Returns:
            An evidence array of shape (num_var_states,)
        """
        raise NotImplementedError("get_evidence function needs to be implemented")

    def init_msgs(self, context: Any = None) -> jnp.ndarray:
        """Initialize messages. By default it initializes all messages to 0.
        Can be overwritten to support customized initialization schemes

        Args:
            context: Optional context for initializing messages

        Returns:
            Initialized messages
        """
        if not hasattr(self, "_wiring"):
            self.compile_wiring()

        return jnp.zeros(self._wiring.var_states_for_edges.shape[0])
