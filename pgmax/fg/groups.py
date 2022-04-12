"""A module containing the base classes for variable and factor groups in a Factor Graph."""

import collections
import inspect
import typing
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Collection,
    FrozenSet,
    Hashable,
    List,
    Mapping,
    OrderedDict,
    Sequence,
    Tuple,
    Type,
    Union,
)

import jax.numpy as jnp
import numpy as np

import pgmax.fg.nodes as nodes
from pgmax.utils import cached_property


@dataclass(frozen=True, eq=False)
class VariableGroup:
    """Class to represent a group of variables.

    All variables in the group are assumed to have the same size. Additionally, the
    variables are indexed by a variable name, and can be retrieved by direct indexing (even indexing
    a sequence of variable names) of the VariableGroup.
    """

    def __getitem__(self, val):
        """Given a name, retrieve the associated Variable.

        Args:
            val: a single name corresponding to a single variable, or a list of such names

        Returns:
            A single variable if the name is not a list. A list of variables if name is a list

        Raises:
            ValueError: if the name is not found in the group
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    @cached_property
    def variables_names(self) -> Any:
        """Function that generates a dictionary mapping names to variables.

        Returns:
            a dictionary mapping all possible names to different variables.
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


@dataclass(frozen=True, eq=False)
class FactorGroup:
    """Class to represent a group of Factors.

    Args:
        vars_to_num_states: TODO
        variable_names_for_factors: A list of list of variable names, where each innermost element is the
            name of a variable. Each list within the outer list is taken to contain the names of the
            variables connected to a Factor.
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

    vars_to_num_states: Mapping[int, int]
    variable_names_for_factors: Sequence[List]
    factor_configs: np.ndarray = field(init=False)
    log_potentials: np.ndarray = field(init=False, default=np.empty((0,)))
    factor_type: Type = field(init=False)
    factor_sizes: np.ndarray = field(init=False)
    variables_for_factors: Tuple[Tuple[int], ...] = field(init=False)
    factor_edges_num_states: np.ndarray = field(init=False)

    def __post_init__(self):
        if len(self.variable_names_for_factors) == 0:
            raise ValueError("Do not add a factor group with no factors.")

        # Note: variable_names_for_factors contains the HASHes
        # Note: this can probably be sped up by numba
        factor_sizes = []
        flat_var_names_for_factors = []
        factor_edges_num_states = []
        for variable_names_for_factor in self.variable_names_for_factors:
            for variable_name in variable_names_for_factor:
                factor_edges_num_states.append(self.vars_to_num_states[variable_name])
                flat_var_names_for_factors.append(variable_name)
            factor_sizes.append(len(variable_names_for_factor))

        object.__setattr__(self, "factor_sizes", np.array(factor_sizes))
        object.__setattr__(
            self, "variables_for_factors", np.array(flat_var_names_for_factors)
        )
        object.__setattr__(
            self, "factor_edges_num_states", np.array(factor_edges_num_states)
        )

    def __getitem__(self, variables: Sequence[int]) -> Any:
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
        return len(self.variable_names_for_factors)

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

    def compile_wiring(self, vars_to_starts: Mapping[int, int]) -> Any:
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

        if not len(self.variable_names_for_factors) == 1:
            raise ValueError(
                f"SingleFactorGroup should only contain one factor. Got {len(self.variable_names_for_factors)}"
            )

        object.__setattr__(self, "factor_type", type(self.factor))
        compile_wiring_arguments = inspect.getfullargspec(
            self.factor_type.compile_wiring
        ).args
        compile_wiring_arguments.remove("vars_to_starts")
        for key in compile_wiring_arguments:
            if not hasattr(self, key):
                object.__setattr__(self, key, getattr(self.factor, key))

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, nodes.Factor]:
        """Function that generates a dictionary mapping names to factors.

        Returns:
            A dictionary mapping all possible names to different factors.
        """
        return OrderedDict(
            [(frozenset(self.variable_names_for_factors[0]), self.factor)]
        )

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
