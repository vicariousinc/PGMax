import itertools
import typing
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import pgmax.fg.nodes as nodes
from pgmax.fg import fg_utils
from pgmax.utils import cached_property


@dataclass(frozen=True, eq=False)
class VariableGroup:
    """Class to represent a group of variables.

    All variables in the group are assumed to have the same size. Additionally, the
    variables are indexed by a "key", and can be retrieved by direct indexing (even indexing
    a sequence of keys) of the VariableGroup.

    Attributes:
        _keys_to_vars: A private, immutable mapping from keys to variables
    """

    _keys_to_vars: Mapping[Hashable, nodes.Variable] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize a private, immutable mapping from keys to variables."""
        object.__setattr__(
            self, "_keys_to_vars", MappingProxyType(self._get_keys_to_vars())
        )

    @typing.overload
    def __getitem__(self, key: Hashable) -> nodes.Variable:
        """This function is a typing overload and is overwritten by the implemented __getitem__"""

    @typing.overload
    def __getitem__(self, key: List) -> List[nodes.Variable]:
        """This function is a typing overload and is overwritten by the implemented __getitem__"""

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

    def _get_keys_to_vars(self) -> Dict[Any, nodes.Variable]:
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

    @cached_property
    def keys(self) -> Tuple[Any, ...]:
        """Function to return a tuple of all keys in the group.

        Returns:
            tuple of all keys that are part of this VariableGroup
        """
        return tuple(self._keys_to_vars.keys())

    @cached_property
    def variables(self) -> Tuple[nodes.Variable, ...]:
        """Function to return a tuple of all variables in the group.

        Returns:
            tuple of all variable that are part of this VariableGroup
        """
        return tuple(self._keys_to_vars.values())

    @cached_property
    def container_keys(self) -> Tuple:
        """Placeholder function. Returns a tuple containing slice(None) for all variable groups
        other than a composite variable group
        """
        return (slice(None),)


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
        Mapping[Hashable, VariableGroup], Sequence[VariableGroup]
    ]

    def __post_init__(self):
        object.__setattr__(
            self, "_keys_to_vars", MappingProxyType(self._get_keys_to_vars())
        )

    @typing.overload
    def __getitem__(self, key: Hashable) -> nodes.Variable:
        """This function is a typing overload and is overwritten by the implemented __getitem__"""

    @typing.overload
    def __getitem__(self, key: List) -> List[nodes.Variable]:
        """This function is a typing overload and is overwritten by the implemented __getitem__"""

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

    def _get_keys_to_vars(self) -> Dict[Hashable, nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        keys_to_vars: Dict[Hashable, nodes.Variable] = {}
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
                The type of evidence should match that of self.variable_group_container.

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
        """Function to get keys referring to the variable groups within this
        CompositeVariableGroup.

        Returns:
            a tuple of the keys referring to the variable groups within this
            CompositeVariableGroup.
        """
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

    def _get_keys_to_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
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

    def _get_keys_to_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        keys_to_vars: Dict[Tuple[Any, ...], nodes.Variable] = {}
        for key in self.key_tuple:
            keys_to_vars[key] = nodes.Variable(self.variable_size)
        return keys_to_vars

    def get_vars_to_evidence(
        self, evidence: Mapping[Hashable, np.ndarray]
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
                    f"Variable {key} expects an evidence array of shape "
                    f"({(self.variable_size,)})."
                    f"Got {evidence[key].shape}."
                )

            vars_to_evidence[self._keys_to_vars[key]] = evidence[key]

        return vars_to_evidence


@dataclass(frozen=True, eq=False)
class FactorGroup:
    """Class to represent a group of factors.

    Args:
        variable_group: either a VariableGroup or - if the elements of more than one VariableGroup
            are connected to this FactorGroup - then a CompositeVariableGroup. This holds
            all the variables that are connected to this FactorGroup

    Attributes:
        _keys_to_factors: maps factor keys to the corresponding factors

    Raises:
        ValueError: if connected_var_keys is an empty list
    """

    variable_group: Union[CompositeVariableGroup, VariableGroup]
    _keys_to_factors: Mapping[Hashable, nodes.EnumerationFactor] = field(init=False)

    def __post_init__(self) -> None:
        """Initializes a tuple of all the factors contained within this FactorGroup."""
        object.__setattr__(
            self, "_keys_to_factors", MappingProxyType(self._get_keys_to_factors())
        )

    def __getitem__(self, key: Hashable) -> nodes.EnumerationFactor:
        """Function to query individual factors in the factor group

        Args:
            key: a key used to query an individual factor in the factor group

        Returns:
            A queried individual factor
        """
        if key not in self.keys:
            raise ValueError(
                f"The queried factor {key} is not present in the factor group"
            )

        return self._keys_to_factors[key]

    def compile_wiring(
        self, vars_to_starts: Mapping[nodes.Variable, int]
    ) -> nodes.EnumerationWiring:
        """Function to compile wiring for the factor group.

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
            compiled wiring for the factor group
        """
        wirings = [factor.compile_wiring(vars_to_starts) for factor in self.factors]
        wiring = fg_utils.concatenate_enumeration_wirings(wirings)
        return wiring

    @cached_property
    def factor_group_log_potentials(self) -> np.ndarray:
        """Function to compile potential array for the factor group

        Returns:
            a jnp array representing the log of the potential function for
            the factor group
        """
        return np.concatenate(
            [factor.factor_configs_log_potentials for factor in self.factors]
        )

    def _get_keys_to_factors(self) -> Dict[Hashable, nodes.EnumerationFactor]:
        """Function that generates a dictionary mapping keys to factors.

        Returns:
            a dictionary mapping all possible keys to different factors.
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    @cached_property
    def keys(self) -> Tuple[Hashable, ...]:
        """Returns all keys in the factor group."""
        return tuple(self._keys_to_factors.keys())

    @cached_property
    def factors(self) -> Tuple[nodes.EnumerationFactor, ...]:
        """Returns all factors in the factor group."""
        return tuple(self._keys_to_factors.values())

    @cached_property
    def factor_num_states(self) -> np.ndarray:
        """Returns the list of total number of edge states for factors in the factor group."""
        factor_num_states = np.array(
            [np.sum(factor.edges_num_states) for factor in self.factors], dtype=int
        )
        return factor_num_states


@dataclass(frozen=True, eq=False)
class EnumerationFactorGroup(FactorGroup):
    """Class to represent a group of EnumerationFactors.

    All factors in the group are assumed to have the same set of valid configurations and
    the same potential function. Note that the log potential function is assumed to be
    uniform 0 unless the inheriting class includes a factor_configs_log_potentials argument.

    Args:
        connected_var_keys: A list of list of tuples, where each innermost tuple contains a
            key into variable_group. Each list within the outer list is taken to contain the keys of variables
            neighboring a particular factor to be added.
        factor_configs: Array of shape (num_val_configs, num_variables)
            An array containing explicit enumeration of all valid configurations
        factor_configs_log_potentials: Optional array of shape (num_val_configs,).
            If specified, it contains the log of the potential value for every possible configuration.
            If none, it is assumed the log potential is uniform 0 and such an array is automatically
            initialized.
    """

    connected_var_keys: Union[
        Sequence[List[Tuple[Hashable, ...]]],
        Mapping[Any, List[Tuple[Hashable, ...]]],
    ]
    factor_configs: np.ndarray
    factor_configs_log_potentials: Optional[np.ndarray] = None

    def _get_keys_to_factors(self) -> Dict[Hashable, nodes.EnumerationFactor]:
        """Function that generates a dictionary mapping keys to factors.

        Returns:
            a dictionary mapping all possible keys to different factors.
        """
        if self.factor_configs_log_potentials is None:
            factor_configs_log_potentials = np.zeros(
                self.factor_configs.shape[0], dtype=float
            )
        else:
            factor_configs_log_potentials = self.factor_configs_log_potentials

        if isinstance(self.connected_var_keys, Sequence):
            keys_to_factors: Dict[Hashable, nodes.EnumerationFactor] = {
                frozenset(self.connected_var_keys[ii]): nodes.EnumerationFactor(
                    tuple(self.variable_group[self.connected_var_keys[ii]]),
                    self.factor_configs,
                    factor_configs_log_potentials,
                )
                for ii in range(len(self.connected_var_keys))
            }
        else:
            keys_to_factors = {
                key: nodes.EnumerationFactor(
                    tuple(self.variable_group[self.connected_var_keys[key]]),
                    self.factor_configs,
                    factor_configs_log_potentials,
                )
                for key in self.connected_var_keys
            }

        return keys_to_factors


@dataclass(frozen=True, eq=False)
class PairwiseFactorGroup(FactorGroup):
    """Class to represent a group of EnumerationFactors where each factor connects to
    two different variables.

    All factors in the group are assumed to be such that all possible configuration of the two
    variable's states are valid. Additionally, all factors in the group are assumed to share
    the same potential function and to be connected to variables from VariableGroups within
    one CompositeVariableGroup.

    Args:
        connected_var_keys: A list of list of tuples, where each innermost tuple contains a
            key into variable_group. Each list within the outer list is taken to contain the keys of variables
            neighboring a particular factor to be added.
        log_potential_matrix: array of shape (var1.variable_size, var2.variable_size),
            where var1 and var2 are the 2 VariableGroups (that may refer to the same
            VariableGroup) whose keys are present in each sub-list from self.connected_var_keys.
    """

    connected_var_keys: Union[
        Sequence[List[Tuple[Hashable, ...]]],
        Mapping[Any, List[Tuple[Hashable, ...]]],
    ]
    log_potential_matrix: np.ndarray

    def _get_keys_to_factors(self) -> Dict[Hashable, nodes.EnumerationFactor]:
        """Function that generates a dictionary mapping keys to factors.

        Returns:
            a dictionary mapping all possible keys to different factors.

        Raises:
            ValueError: if every sub-list within self.connected_var_keys has len != 2, or if the shape of the
                log_potential_matrix is not the same as the variable sizes for each variable referenced in
                each sub-list of self.connected_var_keys
        """
        if isinstance(self.connected_var_keys, Sequence):
            connected_var_keys = self.connected_var_keys
        else:
            connected_var_keys = tuple(self.connected_var_keys.values())

        for fac_list in connected_var_keys:
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

        factor_configs = np.array(
            np.meshgrid(
                np.arange(self.log_potential_matrix.shape[0]),
                np.arange(self.log_potential_matrix.shape[1]),
            )
        ).T.reshape((-1, 2))
        factor_configs_log_potentials = self.log_potential_matrix[
            factor_configs[:, 0], factor_configs[:, 1]
        ]
        if isinstance(self.connected_var_keys, Sequence):
            keys_to_factors: Dict[Hashable, nodes.EnumerationFactor] = {
                frozenset(self.connected_var_keys[ii]): nodes.EnumerationFactor(
                    tuple(self.variable_group[self.connected_var_keys[ii]]),
                    factor_configs,
                    factor_configs_log_potentials,
                )
                for ii in range(len(self.connected_var_keys))
            }
        else:
            keys_to_factors = {
                key: nodes.EnumerationFactor(
                    tuple(self.variable_group[self.connected_var_keys[key]]),
                    factor_configs,
                    factor_configs_log_potentials,
                )
                for key in self.connected_var_keys
            }

        return keys_to_factors
