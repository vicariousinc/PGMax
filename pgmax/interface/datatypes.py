import itertools
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np

import pgmax.fg.nodes as nodes


@dataclass
class VariableGroup:
    """Base class to represent a group of variables.

    All variables in the group are assumed to have the same size. Additionally, the
    variables are indexed by a "key", and can be retrieved by direct indexing (even slicing)
    of the VariableGroup.

    Args:
        variable_size: the number of states that the variable can be in.
    """

    variable_size: int

    def __post_init__(self) -> None:
        self._key_to_var: Mapping[Any, nodes.Variable] = MappingProxyType(
            self._generate_vars()
        )

    def __getitem__(self, key) -> Union[nodes.Variable, List[nodes.Variable]]:
        if type(key) is slice:
            start, stop, step = key.indices(len(self._key_to_var.keys()))
            vars_list: List[nodes.Variable] = []
            for k in range(start, stop, step):
                var = self._key_to_var.get(k)
                if var is None:
                    raise ValueError(
                        f"The key {k} is not present in the VariableGroup {type(self)}; please ensure "
                        "it's been added to the VariableGroup before trying to query it."
                    )
                vars_list.append(var)
            return vars_list
        else:
            var = self._key_to_var.get(key)
            if var is None:
                raise ValueError(
                    f"The key {key} is not present in in the VariableGroup {type(self)}; please ensure "
                    "it's been added to the VariableGroup before trying to query it."
                )
            return var

    def _generate_vars(self) -> Dict[Any, nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        This function needs to be overriden by a concrete VariableGroup subclass.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    def get_all_vars(self) -> Tuple[nodes.Variable, ...]:
        """Function to return a tuple of all variables in the group.

        Returns:
            tuple of all variable that are part of this VariableGroup
        """
        return tuple(self._key_to_var.values())


@dataclass
class FactorGroup:
    """Base class to represent a group of factors.

    All factors in the group are assumed to have the same set of valid configurations and
    the same potential function.

    Args:
        factor_configs: Array of shape (num_configs, num_variables)
            An array containing explicit enumeration of all valid configurations

    Attributes:
        factors: a tuple of all the factors belonging to this group. These are constructed
            internally by invoking the _get_connected_var_keys_for_factors method.
    """

    factor_configs: np.ndarray

    def __post_init__(self) -> None:
        factors = []
        connected_var_keys_for_factors = self._get_connected_var_keys_for_factors()
        if len(connected_var_keys_for_factors) == 0:
            raise ValueError(
                "The list returned by _get_connected_var_keys_for_factors is empty"
            )
        for keys_list in connected_var_keys_for_factors:
            vars_list: List[nodes.Variable] = []
            for key_vargroup_tuple in keys_list:
                key, vargroup = key_vargroup_tuple
                var_query = vargroup[key]
                if type(var_query) is List[nodes.Variable]:
                    vars_list.extend(var_query)
                else:
                    vars_list.append(var_query)  # type: ignore
            if len(vars_list) == 0:
                raise ValueError(
                    "There was an empty sub-list in the output of _get_connected_var_keys_for_factors"
                )
            factors.append(
                nodes.EnumerationFactor(tuple(vars_list), self.factor_configs)
            )
        self.factors: Tuple[nodes.EnumerationFactor, ...] = tuple(factors)

    def _get_connected_var_keys_for_factors(
        self,
    ) -> List[List[Tuple[Any, VariableGroup]]]:
        """Fuction to generate indices of variables neighboring a factor.

        This function needs to be overridden by a concrete implementation of a factor group.

        Returns:
            A list of lists of length-2 tuples, where the 0th tuple element is a key and the 1st
                tuple element is a VariableGroup that contains that key. Each inner list represents
                a particular factor to be added.
        """
        raise NotImplementedError(
            "Please subclass the FactorGroup class and override this method"
        )


@dataclass
class NDVariableArray(VariableGroup):
    """Concrete subclass of VariableGroup for n-dimensional grids of variables.

    Args:
        shape: a tuple specifying the size of each dimension of the grid (similar to
            the notion of a NumPy ndarray shape)
    """

    shape: Tuple[int, ...]

    def _generate_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        key_to_var_mapping: Dict[Tuple[int, ...], nodes.Variable] = {}
        for key in itertools.product(*[list(range(k)) for k in self.shape]):
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        return key_to_var_mapping


@dataclass
class KeyTupleVariableGroup(VariableGroup):
    """Concrete subclass of VariableGroup for a group with explicitly-enumerated keys.

    Args:
        key_tuple: a tuple of any element, where each element represents a particular
            variable key.
    """

    key_tuple: Tuple[Any, ...]

    def _generate_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        key_to_var_mapping: Dict[Tuple[Any, ...], nodes.Variable] = {}
        for key in self.key_tuple:
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        return key_to_var_mapping
