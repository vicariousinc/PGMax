"""A module containing the base classes for variable and factor groups in a Factor Graph."""

import inspect
from dataclasses import dataclass, field
from functools import total_ordering
from typing import (
    Any,
    Collection,
    FrozenSet,
    List,
    Mapping,
    OrderedDict,
    Sequence,
    Tuple,
    Type,
    Union,
)

import jax.numpy as jnp
import numba as nb
import numpy as np

import pgmax.fg.nodes as nodes
from pgmax.utils import cached_property

MAX_SIZE = 1e10


@total_ordering
@dataclass(frozen=True, eq=False)
class VariableGroup:
    """Class to represent a group of variables.
    Each variable is represented via a tuple of the form (variable hash/name, number of states)

    Attributes:
        this_hash: Hash of the VariableGroup
    """

    def __hash__(self):
        return self.this_hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __getitem__(self, val: Any) -> Union[Tuple[Any, int], List[Tuple[Any, int]]]:
        """Given a variable name, index, or a group of variable indices, retrieve the associated variable(s).
        Each variable is returned via a tuple of the form (variable hash/name, number of states)

        Args:
            val: a variable index, slice, or name

        Returns:
            A single variable or a list of variables
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    @cached_property
    def variables(self) -> List[Tuple]:
        """Function that returns the list of all variables in the VariableGroup.
        Each variable is represented by a tuple of the form (variable hash/name, number of states)

        Returns:
            List of variables in the VariableGroup
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    def flatten(self, data: Any) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data

        Returns:
            A flat jnp.array for internal use
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    def unflatten(self, flat_data: Union[np.ndarray, jnp.ndarray]) -> Any:
        """Function that recovers meaningful structured data from internal flat data array

        Args:
            flat_data: Internal flat data array.

        Returns:
            Meaningful structured data
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )


@total_ordering
@dataclass(frozen=True, eq=False)
class FactorGroup:
    """Class to represent a group of Factors.

    Args:
        variables_for_factors: A list of list of variables. Each list within the outer list contains the
            variables connected to a Factor. The same variable can be connected to multiple Factors.
        factor_configs: Optional array containing an explicit enumeration of all valid configurations
        log_potentials: Array of log potentials.

    Attributes:
        factor_type: Factor type shared by all the Factors in the FactorGroup.
        factor_sizes: Array of the different factor sizes.
        factor_edges_num_states: Array concatenating the number of states for the variables connected to each Factor of
            the FactorGroup. Each variable will appear once for each Factor it connects to.

    Raises:
        ValueError: if the FactorGroup does not contain a Factor
    """

    variables_for_factors: Sequence[List]
    factor_configs: np.ndarray = field(init=False)
    log_potentials: np.ndarray = field(init=False, default=np.empty((0,)))
    factor_type: Type = field(init=False)
    factor_sizes: np.ndarray = field(init=False)
    factor_edges_num_states: np.ndarray = field(init=False)

    def __post_init__(self):
        if len(self.variables_for_factors) == 0:
            raise ValueError("Cannot create a FactorGroup with no Factor.")

        factor_sizes = np.array(
            [
                len(variables_for_factor)
                for variables_for_factor in self.variables_for_factors
            ]
        )
        object.__setattr__(self, "factor_sizes", factor_sizes)

        factor_edges_num_states = np.empty(shape=(self.factor_sizes.sum(),), dtype=int)
        _compile_edges_num_states_numba(
            factor_edges_num_states, self.variables_for_factors
        )
        object.__setattr__(self, "factor_edges_num_states", factor_edges_num_states)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __getitem__(self, variables: Union[Sequence, Collection]) -> Any:
        """Function to query individual factors in the factor group

        Args:
            variables: a set of variables, used to query an individual factor in the factor group
                involving this set of variables

        Returns:
            A queried individual factor

        Raises:
            ValueError: if the queried factor is not present in the factor group
        """
        variables = frozenset(variables)
        if variables not in self._variables_to_factors:
            raise ValueError(
                f"The queried factor connected to the set of variables {variables} is not present in the factor group."
            )
        return self._variables_to_factors[variables]

    @cached_property
    def _variables_to_factors(self) -> Mapping[FrozenSet, nodes.Factor]:
        """Function to compile potential array for the factor group.
        This function is only called on demand when the user requires it.

        Returns:
            A dictionnary mapping set of connected variables to the corresponding factors
        """
        return self._get_variables_to_factors()

    @cached_property
    def factor_group_log_potentials(self) -> np.ndarray:
        """Flattened array of log potentials"""
        return self.log_potentials.flatten()

    @cached_property
    def factors(self) -> Tuple[nodes.Factor, ...]:
        """Returns all factors in the factor group.
        This function is only called on demand when the user requires it."""
        return tuple(self._variables_to_factors.values())

    @cached_property
    def num_factors(self) -> int:
        """Returns the number of factors in the FactorGroup."""
        return len(self.variables_for_factors)

    def _get_variables_to_factors(self) -> OrderedDict[FrozenSet, Any]:
        """Function that generates a dictionary mapping names to factors.
        This function is only called on demand when the user requires it.

        Returns:
            A dictionary mapping all possible names to different factors.
        """
        raise NotImplementedError(
            "Please subclass the FactorGroup class and override this method"
        )

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data.

        Returns:
            A flat jnp.array for internal use
        """
        raise NotImplementedError(
            "Please subclass the FactorGroup class and override this method"
        )

    def unflatten(self, flat_data: Union[np.ndarray, jnp.ndarray]) -> Any:
        """Function that recovers meaningful structured data from internal flat data array

        Args:
            flat_data: Internal flat data array.

        Returns:
            Meaningful structured data.
        """
        raise NotImplementedError(
            "Please subclass the FactorGroup class and override this method"
        )

    def compile_wiring(self, vars_to_starts: Mapping[Tuple[Any, int], int]) -> Any:
        """Compile an efficient wiring for the FactorGroup.

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
            Wiring for the FactorGroup
        """
        compile_wiring_arguments = inspect.getfullargspec(
            self.factor_type.compile_wiring
        ).args
        compile_wiring_arguments.remove("vars_to_starts")
        compile_wiring_arguments = {
            key: getattr(self, key) for key in compile_wiring_arguments
        }

        wiring = self.factor_type.compile_wiring(
            vars_to_starts=vars_to_starts, **compile_wiring_arguments
        )
        return wiring


@dataclass(frozen=True, eq=False)
class SingleFactorGroup(FactorGroup):
    """Class to represent a FactorGroup with a single factor.
    For internal use only. Should not be directly used to add FactorGroups to a factor graph.

    Args:
        factor: the single factor in the SingleFactorGroup
    """

    factor: nodes.Factor

    def __post_init__(self):
        super().__post_init__()

        if not len(self.variables_for_factors) == 1:
            raise ValueError(
                f"SingleFactorGroup should only contain one factor. Got {len(self.variables_for_factors)}"
            )

        object.__setattr__(self, "factor_type", type(self.factor))
        compile_wiring_arguments = inspect.getfullargspec(
            self.factor_type.compile_wiring
        ).args
        compile_wiring_arguments.remove("vars_to_starts")
        for key in compile_wiring_arguments:
            if not hasattr(self, key):
                object.__setattr__(self, key, getattr(self.factor, key))

        object.__setattr__(
            self, "log_potentials", getattr(self.factor, "log_potentials")
        )

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, nodes.Factor]:
        """Function that generates a dictionary mapping names to factors.

        Returns:
            A dictionary mapping all possible names to different factors.
        """
        return OrderedDict([(frozenset(self.variables_for_factors[0]), self.factor)])

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data.

        Returns:
            A flat jnp.array for internal use
        """
        raise NotImplementedError(
            "SingleFactorGroup does not support vectorized factor operations."
        )

    def unflatten(self, flat_data: Union[np.ndarray, jnp.ndarray]) -> Any:
        """Function that recovers meaningful structured data from internal flat data array

        Args:
            flat_data: Internal flat data array.

        Returns:
            Meaningful structured data.
        """
        raise NotImplementedError(
            "SingleFactorGroup does not support vectorized factor operations."
        )


nb.jit(parallel=False, cache=True, fastmath=True, nopython=False)


def _compile_edges_num_states_numba(factor_edges_num_states, variables_for_factors):
    idx = 0
    for variables_for_factor in variables_for_factors:
        for variable in variables_for_factor:
            factor_edges_num_states[idx] = variable[1]
            idx += 1
