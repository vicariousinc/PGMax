from __future__ import annotations

"""A module containing classes that specify the basic components of a Factor Graph."""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax import utils


@dataclass(frozen=True, eq=False)
class Variable:
    """Base class for variables.
    If desired, this can be sub-classed to add additional concrete
    meta-information

    Args:
        num_states: an int representing the number of states this variable has.
    """

    num_states: int


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Wiring:
    """Wiring for factors.

    Args:
        edges_num_states: Array of shape (num_edges,)
            Number of states for the variables connected to each edge
        var_states_for_edges: Array of shape (num_edge_states,)
            Global variable state indices for each edge state
    """

    edges_num_states: Union[np.ndarray, jnp.ndarray]
    var_states_for_edges: Union[np.ndarray, jnp.ndarray]

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            if isinstance(getattr(self, field), np.ndarray):
                getattr(self, field).flags.writeable = False

    @property
    def inference_arguments(self) -> Mapping[str, Any]:
        """
        Returns:
            A dictionnary of elements used to run belief propagation.
        """
        raise NotImplementedError(
            "Please subclass the Wiring class and override this method."
        )

    def tree_flatten(self):
        return jax.tree_util.tree_flatten(asdict(self))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data.unflatten(children))


@dataclass(frozen=True, eq=False)
class Factor:
    """A factor

    Args:
        variables: List of connected variables
    """

    variables: Tuple[Variable, ...]
    log_potentials: np.ndarray

    @utils.cached_property
    def edges_num_states(self) -> np.ndarray:
        """Number of states for the variables connected to each edge

        Returns:
            Array of shape (num_edges,)
            Number of states for the variables connected to each edge
        """
        edges_num_states = np.array(
            [variable.num_states for variable in self.variables], dtype=int
        )
        return edges_num_states

    def compile_wiring(self, vars_to_starts: Mapping[Variable, int]) -> Wiring:
        """Compile wiring for the factor

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
            Wiring for the Factor
        """
        raise NotImplementedError(
            "Please subclass the Factor class and override this method"
        )

    @staticmethod
    def concatenate_wirings(wirings: Sequence) -> Wiring:
        """Concatenate a list of Wirings

        Args:
            wirings: A list of Wirings

        Returns:
            Concatenated Wiring
        """
        raise NotImplementedError(
            "Please subclass the Wiring class and override this method."
        )
