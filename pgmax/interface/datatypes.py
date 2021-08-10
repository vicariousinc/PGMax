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
        """Initialize a private, immuable mapping from keys to Variables."""
        self._key_to_var: Mapping[Any, nodes.Variable] = MappingProxyType(
            self._generate_vars()
        )

    def __getitem__(self, key) -> Union[nodes.Variable, List[nodes.Variable]]:
        """Given a key, retrieve the associated Variable.

        Args:
            key: a single key corresponding to a single variable, or a list of such keys

        Returns:
            a single variable if the "key" argument is a single key. Otherwise, returns a list of
                variables corresponding to each key in the "key" argument.
        """

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
        key_vargroup_pairs: a tuple of tuples where each inner tuple is a (key, VariableGroup)
            pair

    """

    key_vargroup_pairs: Tuple[Tuple[Any, VariableGroup], ...]

    def __post_init__(self):
        """Initialize a private, immuable mapping from keys to VariableGroups."""
        key_vargroup_dict: Dict[Any, VariableGroup] = {}
        for key_vargroup_tuple in self.key_vargroup_pairs:
            key, vargroup = key_vargroup_tuple
            key_vargroup_dict[key] = vargroup
        self._key_to_vargroup: Mapping[Any, VariableGroup] = MappingProxyType(
            key_vargroup_dict
        )

    def __getitem__(self, key) -> Union[nodes.Variable, List[nodes.Variable]]:
        """Given a key, retrieve the associated Variable from the associated VariableGroup.

        Args:
            key: a single key corresponding to a single Variable within a VariableGroup, or a list
                of such keys

        Returns:
            a single variable if the "key" argument is a single key. Otherwise, returns a list of
                variables corresponding to each key in the "key" argument.
        """
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

    Args:
        var_group: either a VariableGroup or - if the elements of more than one VariableGroup
            are connected to this FactorGroup - then a CompositeVariableGroup. This holds
            all the variables that are connected to this FactorGroup

    Raises:
        ValueError: if the connected_variables() method returns an empty list
    """

    var_group: Union[CompositeVariableGroup, VariableGroup]

    def __post_init__(self) -> None:
        """Initializes a tuple of all the factors contained within this FactorGroup."""
        connected_var_keys_for_factors = self.connected_variables()
        if len(connected_var_keys_for_factors) == 0:
            raise ValueError("The list returned by self.connected_variables() is empty")

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
class EnumerationFactorGroup(FactorGroup):
    """Base class to represent a group of EnumerationFactors.

    All factors in the group are assumed to have the same set of valid configurations and
    the same potential function. Additionally, all factors in a group are assumed to be
    connected to variables from VariableGroups within one CompositeVariableGroup. Note that
    the log potential function is assumed to be uniform 0 unless the inheriting class
    includes a factor_configs_log_potentials argument.

    Args:
        factor_configs: Array of shape (num_val_configs, num_variables)
            An array containing explicit enumeration of all valid configurations

    Attributes:
        factors: a tuple of all the factors belonging to this group. These are constructed
            internally by invoking the _get_connected_var_keys_for_factors method.
        factor_configs_log_potentials: Can be specified by an inheriting class, or just left
            unspecified (equivalent to specifying None). If specified, must have (num_val_configs,).
            and contain the log of the potential value for every possible configuration.
            If none, it is assumed the log potential is uniform 0 and such an array is automatically
            initialized.

    Raises:
        ValueError: if the connected_variables() method returns an empty list
    """

    factor_configs: np.ndarray

    def __post_init__(self) -> None:
        """Initializes a tuple of all the factors contained within this FactorGroup."""
        connected_var_keys_for_factors = self.connected_variables()
        if len(connected_var_keys_for_factors) == 0:
            raise ValueError("The list returned by self.connected_variables() is empty")
        if (
            not hasattr(self, "factor_configs_log_potentials")
            or hasattr(self, "factor_configs_log_potentials")
            and self.factor_configs_log_potentials is None  # type: ignore
        ):
            factor_configs_log_potentials = np.zeros(
                self.factor_configs.shape[0], dtype=float
            )
        else:
            factor_configs_log_potentials = self.factor_configs_log_potentials  # type: ignore
        self.factors: Tuple[nodes.EnumerationFactor, ...] = tuple(
            [
                nodes.EnumerationFactor(
                    tuple(self.var_group[keys_list]), self.factor_configs, factor_configs_log_potentials  # type: ignore
                )
                for keys_list in connected_var_keys_for_factors
            ]
        )


@dataclass
class PairwiseFactorGroup(FactorGroup):
    """Base class to represent a group of EnumerationFactors where each factor connects to
    two different variables.

    All factors in the group are assumed to be such that all possible configuration of the two
    variable's states are valid. Additionally, all factors in the group are assumed to share
    the same potential function and to be connected to variables from VariableGroups within
    one CompositeVariableGroup.

    Args:
        log_potential_matrix: array of shape (var1.variable_size, var2.variable_size),
            where var1 and var2 are the 2 VariableGroups (that may refer to the same
            VariableGroup) whose keys are present in each sub-list of the list returned by
            the connected_variables() method.

    Attributes:
        factors: a tuple of all the factors belonging to this group. These are constructed
            internally by invoking the connected_variables() method.
        factor_configs_log_potentials: array of shape (num_val_configs,), where
            num_val_configs = var1.variable_size* var2.variable_size. This flattened array
            contains the log of the potential value for every possible configuration.

    Raises:
        ValueError: if the connected_variables() method returns an empty list or if every sub-list within the
            list returned by connected_variables() has len != 2, or if the shape of the log_potential_matrix
            is not the same as the variable sizes for each variable referenced in each sub-list of the list
            returned by connected_variables()
    """

    log_potential_matrix: np.ndarray

    def __post_init__(self) -> None:
        """Initializes a tuple of all the factors contained within this FactorGroup."""
        connected_var_keys_for_factors = self.connected_variables()
        if len(connected_var_keys_for_factors) == 0:
            raise ValueError("The list returned by self.connected_variables() is empty")

        for fac_list in connected_var_keys_for_factors:
            if len(fac_list) != 2:
                raise ValueError(
                    "All pairwise factors should connect to exactly 2 variables. Got a factor connecting to"
                    f" variables ({fac_list})."
                )

        var1_num_states = self.var_group[  # type: ignore
            connected_var_keys_for_factors[0][0]
        ].num_states
        var2_num_states = self.var_group[  # type: ignore
            connected_var_keys_for_factors[0][1]
        ].num_states
        if not (self.log_potential_matrix.shape == (var1_num_states, var2_num_states)):
            raise ValueError(
                f"self.log_potential_matrix must have shape {(var1_num_states, var2_num_states)} based "
                + f"on the return value of self.connected_variables(). Instead, it has shape {self.log_potential_matrix.shape}"
            )

        X, Y = np.mgrid[
            0 : self.log_potential_matrix.shape[0],
            0 : self.log_potential_matrix.shape[1],
        ]
        self.factor_configs = np.vstack([X.ravel(), Y.ravel()])
        self.factor_configs.swapaxes(0, 1)

        factor_configs_log_potentials = np.zeros(
            self.factor_configs.shape[0], dtype=float
        )
        for row_i in range(self.factor_configs.shape[0]):
            factor_configs_log_potentials[row_i] = self.log_potential_matrix[
                tuple(self.factor_configs[row_i])
            ]

        self.factors: Tuple[nodes.EnumerationFactor, ...] = tuple(
            [
                nodes.EnumerationFactor(
                    tuple(self.var_group[keys_list]), self.factor_configs, factor_configs_log_potentials  # type: ignore
                )
                for keys_list in connected_var_keys_for_factors
            ]
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

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        key_to_var_mapping: Dict[Tuple[int, ...], nodes.Variable] = {}
        for key in itertools.product(*[list(range(k)) for k in self.shape]):
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        return key_to_var_mapping


@dataclass
class GenericVariableGroup(VariableGroup):
    """A generic variable group that contains a set of variables of the same size

    This is an overriden function from the parent class.

    Returns:
        a dictionary mapping all possible keys to different variables.
    """

    key_tuple: Tuple[Any, ...]

    def _generate_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        key_to_var_mapping: Dict[Tuple[Any, ...], nodes.Variable] = {}
        for key in self.key_tuple:
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        return key_to_var_mapping
