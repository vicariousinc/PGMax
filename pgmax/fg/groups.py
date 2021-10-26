import collections
import itertools
import typing
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Collection,
    Dict,
    FrozenSet,
    Hashable,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
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

    def _get_keys_to_vars(self) -> OrderedDict[Any, nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
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
        """Placeholder function. Returns a tuple containing None for all variable groups
        other than a composite variable group
        """
        return (None,)

    def flatten(self, data: Any) -> jnp.ndarray:
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    def unflatten(self, flat_data: Union[np.ndarray, jnp.ndarray]) -> Any:
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )


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

    def _get_keys_to_vars(self) -> OrderedDict[Hashable, nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        keys_to_vars: OrderedDict[Hashable, nodes.Variable] = collections.OrderedDict()
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

    def flatten(self, data: Union[Mapping, Sequence]) -> jnp.ndarray:
        flat_data = jnp.concatenate(
            [
                self.variable_group_container[key].flatten(data[key])
                for key in self.container_keys
            ]
        )
        return flat_data

    def unflatten(
        self, flat_data: Union[np.ndarray, jnp.ndarray]
    ) -> Union[Mapping, Sequence]:
        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        num_variables = 0
        num_variable_states = 0
        for key in self.container_keys:
            variable_group = self.variable_group_container[key]
            num_variables += len(variable_group.variables)
            num_variable_states += (
                len(variable_group.variables) * variable_group.variables[0].num_states
            )

        if flat_data.shape[0] == num_variables:
            use_num_states = False
        elif flat_data.shape[0] == num_variable_states:
            use_num_states = True
        else:
            raise ValueError(
                f"flat_data should either be of shape (num_variables={len(self.variables)},), "
                f"or (num_variable_states={num_variable_states},). "
                f"Got {flat_data.shape}"
            )

        data: List[np.ndarray] = []
        start = 0
        for key in self.container_keys:
            variable_group = self.variable_group_container[key]
            length = len(variable_group.variables)
            if use_num_states:
                length *= variable_group.variables[0].num_states

            data.append(variable_group.unflatten(flat_data[start : start + length]))
            start += length
        if isinstance(self.variable_group_container, Mapping):
            return dict([(key, data[kk]) for kk, key in enumerate(self.container_keys)])
        else:
            return data

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

    def _get_keys_to_vars(
        self,
    ) -> OrderedDict[Union[int, Tuple[int, ...]], nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        keys_to_vars: OrderedDict[
            Union[int, Tuple[int, ...]], nodes.Variable
        ] = collections.OrderedDict()
        for key in itertools.product(*[list(range(k)) for k in self.shape]):
            if len(key) == 1:
                keys_to_vars[key[0]] = nodes.Variable(self.variable_size)
            else:
                keys_to_vars[key] = nodes.Variable(self.variable_size)

        return keys_to_vars

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        if data.shape != self.shape and data.shape != self.shape + (
            self.variable_size,
        ):
            raise ValueError(
                f"data should be of shape {self.shape} or {self.shape + (self.variable_size,)}. "
                f"Got {data.shape}."
            )

        return jax.device_put(data).flatten()

    def unflatten(
        self, flat_data: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        if flat_data.size == np.product(self.shape):
            data = flat_data.reshape(self.shape)
        elif flat_data.size == np.product(self.shape) * self.variable_size:
            data = flat_data.reshape(self.shape + (self.variable_size,))
        else:
            raise ValueError(
                f"flat_data should be compatible with shape {self.shape} or {self.shape + (self.variable_size,)}. "
                f"Got {flat_data.shape}."
            )

        return data


@dataclass(frozen=True, eq=False)
class VariableDict(VariableGroup):
    """A variable dictionary that contains a set of variables of the same size

    Args:
        variable_size: The size of the variables in this variable group
        variable_names: A tuple of all names of the variables in this variable group

    """

    variable_size: int
    variable_names: Tuple[Any, ...]

    def _get_keys_to_vars(self) -> OrderedDict[Tuple[int, ...], nodes.Variable]:
        """Function that generates a dictionary mapping keys to variables.

        Returns:
            a dictionary mapping all possible keys to different variables.
        """
        keys_to_vars: OrderedDict[
            Tuple[Any, ...], nodes.Variable
        ] = collections.OrderedDict()
        for key in self.variable_names:
            keys_to_vars[key] = nodes.Variable(self.variable_size)

        return keys_to_vars

    def flatten(
        self, data: Mapping[Hashable, Union[np.ndarray, jnp.ndarray]]
    ) -> jnp.ndarray:
        for key in data:
            if key not in self._keys_to_vars:
                raise ValueError(f"data is referring to a non-existent variable {key}.")

            if data[key].shape != (self.variable_size,):
                raise ValueError(
                    f"Variable {key} expects an data array of shape "
                    f"({(self.variable_size,)})."
                    f"Got {data[key].shape}."
                )

        flat_data = jnp.concatenate([data[key].flatten() for key in self.keys])
        return flat_data

    def unflatten(
        self, flat_data: Union[np.ndarray, jnp.ndarray]
    ) -> Dict[Hashable, Union[np.ndarray, jnp.ndarray]]:
        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        num_variables = len(self.variable_names)
        num_variable_states = len(self.variable_names) * self.variable_size
        if flat_data.shape[0] == num_variables:
            use_num_states = False
        elif flat_data.shape[0] == num_variable_states:
            use_num_states = True
        else:
            raise ValueError(
                f"flat_data should either be of shape (num_variables={len(self.variables)},), "
                f"or (num_variable_states={num_variable_states},). "
                f"Got {flat_data.shape}"
            )

        start = 0
        data = {}
        for key in self.variable_names:
            if use_num_states:
                data[key] = flat_data[start : start + self.variable_size]
                start += self.variable_size
            else:
                data[key] = flat_data[start]
                start += 1

        return data


@dataclass(frozen=True, eq=False)
class FactorGroup:
    """Class to represent a group of factors.

    Args:
        variable_group: either a VariableGroup or - if the elements of more than one VariableGroup
            are connected to this FactorGroup - then a CompositeVariableGroup. This holds
            all the variables that are connected to this FactorGroup

    Attributes:
        _variables_to_factors: maps set of involved variables to the corresponding factors

    Raises:
        ValueError: if connected_var_keys is an empty list
    """

    variable_group: Union[CompositeVariableGroup, VariableGroup]
    _variables_to_factors: Mapping[FrozenSet, nodes.EnumerationFactor] = field(
        init=False
    )

    def __post_init__(self) -> None:
        """Initializes a tuple of all the factors contained within this FactorGroup."""
        object.__setattr__(
            self,
            "_variables_to_factors",
            MappingProxyType(self._get_variables_to_factors()),
        )

    def __getitem__(
        self,
        variables: Union[Sequence, Collection],
    ) -> nodes.EnumerationFactor:
        """Function to query individual factors in the factor group

        Args:
            variables: a set of variables, used to query an individual factor in the factor group
                involving this set of variables

        Returns:
            A queried individual factor
        """
        variables = frozenset(variables)
        if variables not in self._variables_to_factors:
            raise ValueError(
                f"The queried factor {variables} is not present in the factor group."
            )

        return self._variables_to_factors[variables]

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
        return np.concatenate([factor.log_potentials for factor in self.factors])

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, nodes.EnumerationFactor]:
        """Function that generates a dictionary mapping keys to factors.

        Returns:
            a dictionary mapping all possible keys to different factors.
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    @cached_property
    def factors(self) -> Tuple[nodes.EnumerationFactor, ...]:
        """Returns all factors in the factor group."""
        return tuple(self._variables_to_factors.values())

    @cached_property
    def factor_num_states(self) -> np.ndarray:
        """Returns the list of total number of edge states for factors in the factor group."""
        factor_num_states = np.array(
            [np.sum(factor.edges_num_states) for factor in self.factors], dtype=int
        )
        return factor_num_states

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        raise NotImplementedError(
            "Please subclass the FactorGroup class and override this method"
        )

    def unflatten(self, flat_data: Union[np.ndarray, jnp.ndarray]) -> Any:
        raise NotImplementedError(
            "Please subclass the FactorGroup class and override this method"
        )


@dataclass(frozen=True, eq=False)
class EnumerationFactorGroup(FactorGroup):
    """Class to represent a group of EnumerationFactors.

    All factors in the group are assumed to have the same set of valid configurations and
    the same potential function. Note that the log potential function is assumed to be
    uniform 0 unless the inheriting class includes a log_potentials argument.

    Args:
        connected_var_keys: A list of list of tuples, where each innermost tuple contains a
            key into variable_group. Each list within the outer list is taken to contain the keys of variables
            neighboring a particular factor to be added.
        factor_configs: Array of shape (num_val_configs, num_variables)
            An array containing explicit enumeration of all valid configurations
        log_potentials: Optional array of shape (num_val_configs,) or (num_factors, num_val_configs).
            If specified, it contains the log of the potential value for every possible configuration.
            If none, it is assumed the log potential is uniform 0 and such an array is automatically
            initialized.
    """

    connected_var_keys: Sequence[List[Tuple[Hashable, ...]]]
    factor_configs: np.ndarray
    log_potentials: Optional[np.ndarray] = None

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, nodes.EnumerationFactor]:
        """Function that generates a dictionary mapping set of involved variables to factors.

        Returns:
            a dictionary mapping all possible set of involved variables to different factors.
        """
        num_factors = len(self.connected_var_keys)
        num_val_configs = self.factor_configs.shape[0]
        if self.log_potentials is None:
            log_potentials = np.zeros((num_factors, num_val_configs), dtype=float)
        else:
            if self.log_potentials.shape != (
                num_val_configs,
            ) and self.log_potentials.shape != (
                num_factors,
                num_val_configs,
            ):
                raise ValueError(
                    f"Expected log potentials shape: {(num_val_configs,)} or {(num_factors, num_val_configs)}. "
                    f"Got {self.log_potentials.shape}."
                )

            log_potentials = np.broadcast_to(
                self.log_potentials, (num_factors, self.factor_configs.shape[0])
            )

        variables_to_factors = collections.OrderedDict(
            [
                (
                    frozenset(self.connected_var_keys[ii]),
                    nodes.EnumerationFactor(
                        tuple(self.variable_group[self.connected_var_keys[ii]]),
                        self.factor_configs,
                        log_potentials[ii],
                    ),
                )
                for ii in range(len(self.connected_var_keys))
            ]
        )
        return variables_to_factors

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        num_factors = len(self.factors)
        if (
            data.shape != (num_factors, self.factor_configs.shape[0])
            and data.shape
            != (
                num_factors,
                np.sum(self.factors[0].edges_num_states),
            )
            and data.shape != (self.factor_configs.shape[0],)
        ):
            raise ValueError(
                f"data should be of shape {(num_factors, self.factor_configs.shape[0])} or "
                f"{(num_factors, np.sum(self.factors[0].edges_num_states))} or "
                f"{(self.factor_configs.shape[0],)}. Got {data.shape}."
            )

        if data.shape == (self.factor_configs.shape[0],):
            flat_data = jnp.tile(data, num_factors)
        else:
            flat_data = jax.device_put(data).flatten()

        return flat_data

    def unflatten(
        self, flat_data: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        num_factors = len(self.factors)
        if flat_data.size == num_factors * self.factor_configs.shape[0]:
            data = flat_data.reshape(
                (num_factors, self.factor_configs.shape[0]),
            )
        elif flat_data.size == num_factors * np.sum(self.factors[0].edges_num_states):
            data = flat_data.reshape(
                (num_factors, np.sum(self.factors[0].edges_num_states))
            )
        else:
            raise ValueError(
                f"flat_data should be compatible with shape {(num_factors, self.factor_configs.shape[0])} "
                f"or {(num_factors, np.sum(self.factors[0].edges_num_states))}. Got {flat_data.shape}."
            )

        return data


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

    connected_var_keys: Sequence[List[Tuple[Hashable, ...]]]
    log_potential_matrix: np.ndarray

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, nodes.EnumerationFactor]:
        """Function that generates a dictionary mapping set of involved variables to factors.

        Returns:
            a dictionary mapping all possible set of involved variables to different factors.

        Raises:
            ValueError: if every sub-list within self.connected_var_keys has len != 2, or if the shape of the
                log_potential_matrix is not the same as the variable sizes for each variable referenced in
                each sub-list of self.connected_var_keys
        """
        if not (
            self.log_potential_matrix.ndim == 2 or self.log_potential_matrix.ndim == 3
        ):
            raise ValueError(
                "log_potential_matrix should be either a 2D array, specifying shared parameters for all "
                "pairwise factors, or 3D array, specifying parameters for individual pairwise factors. "
                f"Got a {self.log_potential_matrix.ndim}D log_potential_matrix array."
            )

        if self.log_potential_matrix.ndim == 3 and self.log_potential_matrix.shape[
            0
        ] != len(self.connected_var_keys):
            raise ValueError(
                f"Expected log_potential_matrix for {len(self.connected_var_keys)} factors. "
                f"Got log_potential_matrix for {self.log_potential_matrix.shape[0]} factors."
            )

        for fac_list in self.connected_var_keys:
            if len(fac_list) != 2:
                raise ValueError(
                    "All pairwise factors should connect to exactly 2 variables. Got a factor connecting to"
                    f" {len(fac_list)} variables ({fac_list})."
                )

            if not (
                self.log_potential_matrix.shape[-2:]
                == (
                    self.variable_group[fac_list[0]].num_states,
                    self.variable_group[fac_list[1]].num_states,
                )
            ):
                raise ValueError(
                    f"The specified pairwise factor {fac_list} (with "
                    f"{(self.variable_group[fac_list[0]].num_states, self.variable_group[fac_list[1]].num_states)} "
                    f"configurations) does not match the specified log_potential_matrix "
                    f"(with {self.log_potential_matrix.shape[-2:]} configurations)."
                )

        factor_configs = (
            np.mgrid[
                : self.log_potential_matrix.shape[0],
                : self.log_potential_matrix.shape[1],
            ]
            .transpose((1, 2, 0))
            .reshape((-1, 2))
        )
        log_potential_matrix = np.broadcast_to(
            self.log_potential_matrix,
            (len(self.connected_var_keys),) + self.log_potential_matrix.shape[-2:],
        )
        variables_to_factors = collections.OrderedDict(
            [
                (
                    frozenset(self.connected_var_keys[ii]),
                    nodes.EnumerationFactor(
                        tuple(self.variable_group[self.connected_var_keys[ii]]),
                        factor_configs,
                        log_potential_matrix[
                            ii, factor_configs[:, 0], factor_configs[:, 1]
                        ],
                    ),
                )
                for ii in range(len(self.connected_var_keys))
            ]
        )
        return variables_to_factors

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        num_factors = len(self.factors)
        if (
            data.shape != (num_factors,) + self.log_potential_matrix.shape[-2:]
            and data.shape
            != (num_factors, np.sum(self.log_potential_matrix.shape[-2:]))
            and data.shape != self.log_potential_matrix.shape[-2:]
        ):
            raise ValueError(
                f"data should be of shape {(num_factors,) + self.log_potential_matrix.shape[-2:]} or "
                f"{(num_factors, np.sum(self.log_potential_matrix.shape[-2:]))} or "
                f"{self.log_potential_matrix.shape[-2:]}. Got {data.shape}."
            )

        if data.shape == self.log_potential_matrix.shape[-2:]:
            flat_data = jnp.tile(jax.device_put(data).flatten(), num_factors)
        else:
            flat_data = jax.device_put(data).flatten()

        return flat_data

    def unflatten(
        self, flat_data: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        num_factors = len(self.factors)
        if flat_data.size == num_factors * np.product(
            self.log_potential_matrix.shape[-2:]
        ):
            data = flat_data.reshape(
                (num_factors,) + self.log_potential_matrix.shape[-2:]
            )
        elif flat_data.size == num_factors * np.sum(
            self.log_potential_matrix.shape[-2:]
        ):
            data = flat_data.reshape(
                (num_factors, np.sum(self.log_potential_matrix.shape[-2:]))
            )
        else:
            raise ValueError(
                f"flat_data should be compatible with shape {(num_factors,) + self.log_potential_matrix.shape[-2:]} "
                f"or (num_factors, np.sum(self.log_potential_matrix.shape[-2:])). Got {flat_data.shape}."
            )

        return data
