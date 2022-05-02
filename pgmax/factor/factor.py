"""A module containing classes that specify the basic components of a factor."""

from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


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

    def tree_flatten(self):
        return jax.tree_util.tree_flatten(asdict(self))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data.unflatten(children))


@dataclass(frozen=True, eq=False)
class Factor:
    """A factor

    Args:
        variables: List of variables connected by the Factor.
            Each variable is represented by a tuple (variable hash, variable num_states)

    Raises:
        NotImplementedError: If compile_wiring is not implemented
    """

    variables: List[Tuple[int, int]]
    log_potentials: np.ndarray

    def __post_init__(self):
        if not hasattr(self, "compile_wiring"):
            raise NotImplementedError(
                "Please implement compile_wiring in for your factor"
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
