"""A module containing the base class for variable groups in a Factor Graph."""

from dataclasses import dataclass
from functools import total_ordering
from typing import Any, List, Tuple, Union

import jax.numpy as jnp
import numpy as np

from pgmax.utils import cached_property

MAX_SIZE = 1e9


@total_ordering
@dataclass(frozen=True, eq=False)
class VarGroup:
    """Class to represent a group of variables.
    Each variable is represented via a tuple of the form (variable hash, variable num_states)

    Arguments:
        num_states: An integer or an array specifying the number of states of the variables
            in this VarGroup
    """

    num_states: Union[int, np.ndarray]

    def __post_init__(self):
        # Only compute the hash once, which is guaranteed to be an int64
        this_id = id(self) % 2**32
        _hash = this_id * int(MAX_SIZE)
        assert _hash < 2**63
        object.__setattr__(self, "_hash", _hash)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __getitem__(self, val: Any) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """Given a variable name, index, or a group of variable indices, retrieve the associated variable(s).
        Each variable is returned via a tuple of the form (variable hash, variable num_states)

        Args:
            val: a variable index, slice, or name

        Returns:
            A single variable or a list of variables
        """
        raise NotImplementedError(
            "Please subclass the VarGroup class and override this method"
        )

    @cached_property
    def variable_hashes(self) -> np.ndarray:
        """Function that generates a variable hash for each variable

        Returns:
            Array of variables hashes.
        """
        raise NotImplementedError(
            "Please subclass the VarGroup class and override this method"
        )

    @cached_property
    def variables(self) -> List[Tuple[int, int]]:
        """Function that returns the list of all variables in the VarGroup.
        Each variable is represented by a tuple of the form (variable hash, variable num_states)

        Returns:
            List of variables in the VarGroup
        """
        assert isinstance(self.variable_hashes, np.ndarray)
        assert isinstance(self.num_states, np.ndarray)
        vars_hashes = self.variable_hashes.flatten()
        vars_num_states = self.num_states.flatten()
        return list(zip(vars_hashes, vars_num_states))

    def flatten(self, data: Any) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data

        Returns:
            A flat jnp.array for internal use
        """
        raise NotImplementedError(
            "Please subclass the VarGroup class and override this method"
        )

    def unflatten(self, flat_data: Union[np.ndarray, jnp.ndarray]) -> Any:
        """Function that recovers meaningful structured data from internal flat data array

        Args:
            flat_data: Internal flat data array.

        Returns:
            Meaningful structured data
        """
        raise NotImplementedError(
            "Please subclass the VarGroup class and override this method"
        )
