import itertools
import typing
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Tuple, Union

import numpy as np

import pgmax.fg.nodes as nodes
from pgmax.utils import cached_property


@dataclass(frozen=True, eq=False)
class VariableGroup:
    """Base class to represent a group of variables.

    All variables in the group are assumed to have the same size. Additionally, the
    variables are indexed by a "key", and can be retrieved by direct indexing (even indexing
    a sequence of keys) of the VariableGroup.

    Attributes:
        _keys_to_vars: A private, immutable mapping from keys to variables
    """

    _keys_to_vars: Mapping[Any, nodes.Variable] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize a private, immutable mapping from keys to variables."""
        object.__setattr__(
            self, "_keys_to_vars", MappingProxyType(self._set_keys_to_vars())
        )

    @typing.overload
    def __getitem__(self, key: Hashable) -> nodes.Variable:
        pass

    @typing.overload
    def __getitem__(self, key: List) -> List[nodes.Variable]:
        pass

    def __getitem__(self, key):
        """Given a key, retrieve the associated Variable.

        Args:
            key: a single key corresponding to a single variable, or a list of such keys

        Returns:
            a single variable if the "key" argument is a single key. Otherwise, returns a list of
                variables corresponding to each key in the "key" argument.
        """

        if isinstance(key, List):
            keys_list = key
        else:
            keys_list = [key]

        vars_list = []
        for curr_key in keys_list:
            var = self._keys_to_vars.get(curr_key)
            if var is None:
                raise ValueError(
                    f"The key {curr_key} is not present in the VariableGroup {type(self)}; please ensure "
                    "it's been added to the VariableGroup before trying to query it."
                )

            vars_list.append(var)

        if isinstance(key, List):
            return vars_list
        else:
            return vars_list[0]

    def _set_keys_to_vars(self) -> Dict[Any, nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    def get_vars_to_evidence(self, evidence: Any) -> Dict[nodes.Variable, np.ndarray]:
        """Function that turns input evidence into a dictionary mapping variables to evidence.

        Returns:
            a dictionary mapping all possible variables to the corresponding evidence
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    @property
    def keys(self) -> Tuple[Any, ...]:
        """Function to return a tuple of all keys in the group.

        Returns:
            tuple of all keys that are part of this VariableGroup
        """
        return tuple(self._keys_to_vars.keys())

    @property
    def variables(self) -> Tuple[nodes.Variable, ...]:
        """Function to return a tuple of all variables in the group.

        Returns:
            tuple of all variable that are part of this VariableGroup
        """
        return tuple(self._keys_to_vars.values())


@dataclass(frozen=True, eq=False)
class CompositeVariableGroup(VariableGroup):
    """A class to encapsulate a collection of instantiated VariableGroups.

    This class enables users to wrap various different VariableGroups and then index
    them in a straightforward manner. To index into a CompositeVariableGroup, simply
    provide the "key" of the VariableGroup within this CompositeVariableGroup followed
    by the key to be indexed within the VariableGroup.

    Args:
        variable_group_container: A container containing multiple variable groups.
            Supported containers include mapping and sequence.
            For a mapping, the keys of the mapping are used to index the variable groups.
            For a sequence, the indices of the sequence are used to index the variable groups.

    Attributes:
        _keys_to_vars: A private, immutable mapping from keys to variables
    """

    variable_group_container: Union[
        Mapping[Any, VariableGroup], Sequence[VariableGroup]
    ]

    def __post_init__(self):
        if (not isinstance(self.variable_group_container, Mapping)) and (
            not isinstance(self.variable_group_container, Sequence)
        ):
            raise ValueError(
                f"variable_group_container needs to be a mapping or a sequence. Got {type(self.variable_group_container)}"
            )

        object.__setattr__(
            self, "_keys_to_vars", MappingProxyType(self._set_keys_to_vars())
        )

    @typing.overload
    def __getitem__(self, key: Hashable) -> nodes.Variable:
        pass

    @typing.overload
    def __getitem__(self, key: List) -> List[nodes.Variable]:
        pass

    def __getitem__(self, key):
        """Given a key, retrieve the associated Variable from the associated VariableGroup.

        Args:
            key: a single key corresponding to a single Variable within a VariableGroup, or a list
                of such keys

        Returns:
            a single variable if the "key" argument is a single key. Otherwise, returns a list of
                variables corresponding to each key in the "key" argument.
        """
        if isinstance(key, List):
            keys_list = key
        else:
            keys_list = [key]

        vars_list = []
        for curr_key in keys_list:
            if len(curr_key) < 2:
                raise ValueError(
                    "The key needs to have at least 2 elements to index from a composite variable group."
                )

            variable_group = self.variable_group_container[curr_key[0]]
            if len(curr_key) == 2:
                vars_list.append(variable_group[curr_key[1]])
            else:
                vars_list.append(variable_group[curr_key[1:]])

        if isinstance(key, List):
            return vars_list
        else:
            return vars_list[0]

    def _set_keys_to_vars(self) -> Dict[Any, nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        keys_to_vars = {}
        for container_key in self.container_keys:
            for variable_group_key in self.variable_group_container[container_key].keys:
                if isinstance(variable_group_key, tuple):
                    keys_to_vars[
                        (container_key,) + variable_group_key
                    ] = self.variable_group_container[container_key][variable_group_key]
                else:
                    keys_to_vars[
                        (container_key, variable_group_key)
                    ] = self.variable_group_container[container_key][variable_group_key]

        return keys_to_vars

    def get_vars_to_evidence(
        self, evidence: Union[Mapping, Sequence]
    ) -> Dict[nodes.Variable, np.ndarray]:
        """Function that turns input evidence into a dictionary mapping variables to evidence.

        Args:
            evidence: A mapping or a sequence of evidences.
                The type of evidence should match that of self.variable_group_container

        Returns:
            a dictionary mapping all possible variables to the corresponding evidence
        """
        vars_to_evidence: Dict[nodes.Variable, np.ndarray] = {}
        for key in self.container_keys:
            vars_to_evidence.update(
                self.variable_group_container[key].get_vars_to_evidence(evidence[key])
            )

        return vars_to_evidence

    @cached_property
    def container_keys(self) -> Tuple:
        if isinstance(self.variable_group_container, Mapping):
            container_keys = tuple(self.variable_group_container.keys())
        else:
            container_keys = tuple(range(len(self.variable_group_container)))

        return container_keys


@dataclass(frozen=True, eq=False)
class NDVariableArray(VariableGroup):
    """Subclass of VariableGroup for n-dimensional grids of variables.

    Args:
        variable_size: The size of the variables in this variable group
        shape: a tuple specifying the size of each dimension of the grid (similar to
            the notion of a NumPy ndarray shape)
    """

    variable_size: int
    shape: Tuple[int, ...]

    def _set_keys_to_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        keys_to_vars: Dict[Tuple[int, ...], nodes.Variable] = {}
        for key in itertools.product(*[list(range(k)) for k in self.shape]):
            keys_to_vars[key] = nodes.Variable(self.variable_size)
        return keys_to_vars

    def get_vars_to_evidence(
        self, evidence: np.ndarray
    ) -> Dict[nodes.Variable, np.ndarray]:
        """Function that turns input evidence into a dictionary mapping variables to evidence.

        Args:
            evidence: An array of shape self.shape + (variable_size,)
                An array containing evidence for all the variables

        Returns:
            a dictionary mapping all possible variables to the corresponding evidence

        Raises:
            ValueError: if input evidence array is of the wrong shape
        """
        expected_shape = self.shape + (self.variable_size,)
        if not evidence.shape == expected_shape:
            raise ValueError(
                f"Input evidence should be an array of shape {expected_shape}. "
                f"Got {evidence.shape}."
            )

        vars_to_evidence = {
            self._keys_to_vars[key]: evidence[key] for key in self._keys_to_vars
        }
        return vars_to_evidence


@dataclass(frozen=True, eq=False)
class GenericVariableGroup(VariableGroup):
    """A generic variable group that contains a set of variables of the same size

    Args:
        variable_size: The size of the variables in this variable group
        key_tuple: A tuple of all keys in this variable group

    """

    variable_size: int
    key_tuple: Tuple[Any, ...]

    def _set_keys_to_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        keys_to_vars: Dict[Tuple[Any, ...], nodes.Variable] = {}
        for key in self.key_tuple:
            keys_to_vars[key] = nodes.Variable(self.variable_size)
        return keys_to_vars

    def get_vars_to_evidence(
        self, evidence: Mapping[Any, np.ndarray]
    ) -> Dict[nodes.Variable, np.ndarray]:
        """Function that turns input evidence into a dictionary mapping variables to evidence.

        Args:
            evidence: A mapping from keys to np.ndarrays of evidence for that particular
                key

        Returns:
            a dictionary mapping all possible variables to the corresponding evidence

        Raises:
            ValueError: if a key has not previously been added to this VariableGroup, or
                if any evidence array is of the wrong shape.
        """
        vars_to_evidence = {}
        for key in evidence:
            if key not in self._keys_to_vars:
                raise ValueError(
                    f"The evidence is referring to a non-existent variable {key}."
                )

            if evidence[key].shape != (self.variable_size,):
                raise ValueError(
                    f"Variable {key} expect an evidence array of shape "
                    f"({(self.variable_size,)})."
                    f"Got {evidence[key].shape}."
                )

            vars_to_evidence[self._keys_to_vars[key]] = evidence[key]

        return vars_to_evidence


@dataclass(frozen=True, eq=False)
class FactorGroup:
    """Base class to represent a group of factors.

    Args:
        variable_group: either a VariableGroup or - if the elements of more than one VariableGroup
            are connected to this FactorGroup - then a CompositeVariableGroup. This holds
            all the variables that are connected to this FactorGroup
        connected_var_keys: A list of tuples of tuples, where each innermost tuple contains a
            key variable_group. Each list within the outer list is taken to contain the keys of variables
            neighboring a particular factor to be added.

    Raises:
        ValueError: if connected_var_keys is an empty list
    """

    variable_group: Union[CompositeVariableGroup, VariableGroup]
    connected_var_keys: List[List[Tuple[Any, ...]]]

    def __post_init__(self) -> None:
        """Initializes a tuple of all the factors contained within this FactorGroup."""
        if len(self.connected_var_keys) == 0:
            raise ValueError("self.connected_var_keys is empty")

    @cached_property
    def factors(self) -> Tuple[nodes.EnumerationFactor, ...]:
        raise NotImplementedError("Needs to be overriden by subclass")


@dataclass(frozen=True, eq=False)
class EnumerationFactorGroup(FactorGroup):
    """Base class to represent a group of EnumerationFactors.

    All factors in the group are assumed to have the same set of valid configurations and
    the same potential function. Note that the log potential function is assumed to be
    uniform 0 unless the inheriting class includes a factor_configs_log_potentials argument.

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

    """

    factor_configs: np.ndarray

    @cached_property
    def factors(self) -> Tuple[nodes.EnumerationFactor, ...]:
        """Returns a tuple of all the factors contained within this FactorGroup."""
        if getattr(self, "factor_configs_log_potentials", None) is None:
            factor_configs_log_potentials = np.zeros(
                self.factor_configs.shape[0], dtype=float
            )
        else:
            factor_configs_log_potentials = getattr(
                self, "factor_configs_log_potentials"
            )

        return tuple(
            [
                nodes.EnumerationFactor(
                    tuple(self.variable_group[keys_list]),
                    self.factor_configs,
                    factor_configs_log_potentials,
                )
                for keys_list in self.connected_var_keys
            ]
        )


@dataclass(frozen=True, eq=False)
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
            VariableGroup) whose keys are present in each sub-list from self.connected_var_keys.

    Attributes:
        factors: a tuple of all the factors belonging to this group. These are constructed
            internally using self.connected_var_keys
        factor_configs_log_potentials: array of shape (num_val_configs,), where
            num_val_configs = var1.variable_size* var2.variable_size. This flattened array
            contains the log of the potential value for every possible configuration.

    Raises:
        ValueError: if every sub-list within self.connected_var_keys has len != 2, or if the shape of the
            log_potential_matrix is not the same as the variable sizes for each variable referenced in
            each sub-list of self.connected_var_keys
    """

    log_potential_matrix: np.ndarray

    def __post_init__(self) -> None:
        for fac_list in self.connected_var_keys:
            if len(fac_list) != 2:
                raise ValueError(
                    "All pairwise factors should connect to exactly 2 variables. Got a factor connecting to"
                    f" more or less than 2 variables ({fac_list})."
                )
            if not (
                self.log_potential_matrix.shape
                == (
                    self.variable_group[fac_list[0]].num_states,
                    self.variable_group[fac_list[1]].num_states,
                )
            ):
                raise ValueError(
                    "self.log_potential_matrix must have shape"
                    + f"{(self.variable_group[fac_list[0]].num_states, self.variable_group[fac_list[1]].num_states)} "
                    + f"based on self.connected_var_keys. Instead, it has shape {self.log_potential_matrix.shape}"
                )

    @cached_property
    def factors(self) -> Tuple[nodes.EnumerationFactor, ...]:
        """Returns a tuple of all the factors contained within this FactorGroup."""
        factor_configs = np.array(
            np.meshgrid(
                np.arange(self.log_potential_matrix.shape[0]),
                np.arange(self.log_potential_matrix.shape[1]),
            )
        ).T.reshape((-1, 2))
        factor_configs_log_potentials = self.log_potential_matrix[
            factor_configs[:, 0], factor_configs[:, 1]
        ]
        return tuple(
            [
                nodes.EnumerationFactor(
                    tuple(self.variable_group[keys_list]),
                    factor_configs,
                    factor_configs_log_potentials,
                )
                for keys_list in self.connected_var_keys
            ]
        )
