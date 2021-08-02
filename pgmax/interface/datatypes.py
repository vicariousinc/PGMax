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
    variables are indexed by a "key", and can be retrieved by direct indexing (even indexing
    a list of keys) of the VariableGroup.

    Args:
        variable_size: the number of states that the variable can be in.
    """

    variable_size: int

    def __post_init__(self) -> None:
        self._key_to_var: Mapping[Any, nodes.Variable] = MappingProxyType(
            self._generate_vars()
        )

    def __getitem__(self, key) -> Union[nodes.Variable, List[nodes.Variable]]:
        if type(key) is list:
            vars_list: List[nodes.Variable] = []
            for k in key:
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
class CompositeVariableGroup:
    """A class to encapsulate a collection of instantiated VariableGroups.

    This class enables users to wrap various different VariableGroups and then index
    them in a straightforward manner. To index into a CompositeVariableGroup, simply
    provide the "key" of the VariableGroup within this CompositeVariableGroup followed
    by the key to be indexed within the VariableGroup.

    Args:
        key_vargroup_pairs: a tuple of tuples where each inner tuple is a pair of

    """

    key_vargroup_pairs: Tuple[Tuple[Any, VariableGroup], ...]

    def __post_init__(self):
        key_vargroup_dict: Dict[Any, VariableGroup] = {}
        for key_vargroup_tuple in self.key_vargroup_pairs:
            key, vargroup = key_vargroup_tuple
            key_vargroup_dict[key] = vargroup
        self._key_to_vargroup: Mapping[Any, VariableGroup] = MappingProxyType(
            key_vargroup_dict
        )

    def __getitem__(self, key) -> Union[nodes.Variable, List[nodes.Variable]]:
        if type(key) is list:
            vars_list: List[nodes.Variable] = []
            for k in key:
                var_group = self._key_to_vargroup.get(k[0])
                if var_group is None:
                    raise ValueError(
                        f"The key {key[0]} is not present in the CompositeVariableGroup {type(self)}; please ensure "
                        "it's been added to the VariableGroup before trying to query it."
                    )
                vars_list.append(var_group[k[1:]])  # type: ignore
            return vars_list
        else:
            var_group = self._key_to_vargroup.get(key[0])
            if var_group is None:
                raise ValueError(
                    f"The key {key[0]} is not present in the CompositeVariableGroup {type(self)}; please ensure "
                    "it's been added to the VariableGroup before trying to query it."
                )
            return var_group[key[1:]]

    def get_all_vars(self) -> Tuple[nodes.Variable, ...]:
        """Function to return a tuple of all variables from all VariableGroups in this group.

        Returns:
            tuple of all variable that are part of this VariableGroup
        """
        return sum(
            [var_group.get_all_vars() for var_group in self._key_to_vargroup.values()],
            (),
        )


@dataclass
class FactorGroup:
    """Base class to represent a group of factors.

    All factors in the group are assumed to have the same set of valid configurations and
    the same potential function. Additionally, all factors in a group are assumed to be
    connected to variables from VariableGroups within one CompositeVariableGroup

    Args:
        factor_configs: Array of shape (num_configs, num_variables)
            An array containing explicit enumeration of all valid configurations

    Attributes:
        factors: a tuple of all the factors belonging to this group. These are constructed
            internally by invoking the _get_connected_var_keys_for_factors method.
        vargroup: either a VariableGroup or - if the elements of more than one VariableGroup
            are connected to this FactorGroup - then a CompositeVariableGroup. This holds
            all the variables that are connected to this FactorGroup
    """

    factor_configs: np.ndarray
    var_group: Union[CompositeVariableGroup, VariableGroup]

    def __post_init__(self) -> None:
        connected_var_keys_for_factors = self.connected_variables()
        if len(connected_var_keys_for_factors) == 0:
            raise ValueError(
                "The list returned by _get_connected_var_keys_for_factors is empty"
            )
        self.factors: Tuple[nodes.EnumerationFactor, ...] = tuple(
            [
                nodes.EnumerationFactor(
                    tuple(self.var_group[keys_list]), self.factor_configs  # type: ignore
                )
                for keys_list in connected_var_keys_for_factors
            ]
        )

    def connected_variables(
        self,
    ) -> List[List[Tuple[Any, ...]]]:
        """Fuction to generate indices of variables neighboring a factor.

        This function needs to be overridden by a concrete implementation of a factor group.

        Returns:
            A list of lists of tuples, where each tuple contains a key into a CompositeVariableGroup.
                Each inner list represents a particular factor to be added.
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
        """Function that generates a dictionary mapping keys to variables.

        This is an overriden function from the parent class.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        key_to_var_mapping: Dict[Tuple[int, ...], nodes.Variable] = {}
        for key in itertools.product(*[list(range(k)) for k in self.shape]):
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        return key_to_var_mapping


@dataclass
class GenericVariableGroup(VariableGroup):
    """Function that generates a dictionary mapping keys to variables.

    This is an overriden function from the parent class.

    Returns:
        a dictionary mapping all possible keys to different variables.
    """

    key_tuple: Tuple[Any, ...]

    def _generate_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        key_to_var_mapping: Dict[Tuple[Any, ...], nodes.Variable] = {}
        for key in self.key_tuple:
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        return key_to_var_mapping
