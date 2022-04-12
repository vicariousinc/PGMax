"""A module containing classes that specify the basic components of a Factor Graph."""

from dataclasses import asdict, dataclass
from typing import OrderedDict, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax import utils


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
        vars_to_num_states: Dictionnary mapping the variables names, represented
            in the form of a hash, to the variables number of states.

    Raises:
        NotImplementedError: If compile_wiring is not implemented
    """

    vars_to_num_states: OrderedDict[int, int]
    log_potentials: np.ndarray

    def __post_init__(self):
        if not hasattr(self, "compile_wiring"):
            raise NotImplementedError(
                "Please implement compile_wiring in for your factor"
            )

    @utils.cached_property
    def edges_num_states(self) -> np.ndarray:
        """Number of states for the variables connected to each edge

        Returns:
            Array of shape (num_edges,)
            Number of states for the variables connected to each edge
        """
        return self.vars_to_num_states.values()

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
