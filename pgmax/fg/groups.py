"""A module containing the classes for variable and factor groups in a Factor Graph."""

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
from pgmax.utils import cached_property


@dataclass(frozen=True, eq=False)
class VariableGroup:
    """Class to represent a group of variables.

    All variables in the group are assumed to have the same size. Additionally, the
    variables are indexed by a variable name, and can be retrieved by direct indexing (even indexing
    a sequence of variable names) of the VariableGroup.

    Attributes:
        _names_to_variables: A private, immutable mapping from variable names to variables
    """

    _names_to_variables: Mapping[Hashable, nodes.Variable] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize a private, immutable mapping from variable names to variables."""
        object.__setattr__(
            self,
            "_names_to_variables",
            MappingProxyType(self._get_names_to_variables()),
        )

    @typing.overload
    def __getitem__(self, name: Hashable) -> nodes.Variable:
        """This function is a typing overload and is overwritten by the implemented __getitem__"""

    @typing.overload
    def __getitem__(self, name: List) -> List[nodes.Variable]:
        """This function is a typing overload and is overwritten by the implemented __getitem__"""

    def __getitem__(self, name):
        """Given a name, retrieve the associated Variable.

        Args:
            name: a single name corresponding to a single variable, or a list of such names

        Returns:
            A single variable if the name is not a list. A list of variables if name is a list

        Raises:
            ValueError: if the name is not found in the group
        """

        if isinstance(name, List):
            names_list = name
        else:
            names_list = [name]

        vars_list = []
        for curr_name in names_list:
            var = self._names_to_variables.get(curr_name)
            if var is None:
                raise ValueError(
                    f"The name {curr_name} is not present in the VariableGroup {type(self)}; please ensure "
                    "it's been added to the VariableGroup before trying to query it."
                )

            vars_list.append(var)

        if isinstance(name, List):
            return vars_list
        else:
            return vars_list[0]

    def _get_names_to_variables(self) -> OrderedDict[Any, nodes.Variable]:
        """Function that generates a dictionary mapping names to variables.

        Returns:
            a dictionary mapping all possible names to different variables.
        """
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    @cached_property
    def names(self) -> Tuple[Any, ...]:
        """Function to return a tuple of all names in the group.

        Returns:
            tuple of all names that are part of this VariableGroup
        """
        return tuple(self._names_to_variables.keys())

    @cached_property
    def variables(self) -> Tuple[nodes.Variable, ...]:
        """Function to return a tuple of all variables in the group.

        Returns:
            tuple of all variable that are part of this VariableGroup
        """
        return tuple(self._names_to_variables.values())

    @cached_property
    def container_names(self) -> Tuple:
        """Placeholder function. Returns a tuple containing None for all variable groups
        other than a composite variable group
        """
        return (None,)

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
class CompositeVariableGroup(VariableGroup):
    """A class to encapsulate a collection of instantiated VariableGroups.

    This class enables users to wrap various different VariableGroups and then index
    them in a straightforward manner. To index into a CompositeVariableGroup, simply
    provide the name of the VariableGroup within this CompositeVariableGroup followed
    by the name to be indexed within the VariableGroup.

    Args:
        variable_group_container: A container containing multiple variable groups.
            Supported containers include mapping and sequence.
            For a mapping, the names of the mapping are used to index the variable groups.
            For a sequence, the indices of the sequence are used to index the variable groups.

    Attributes:
        _names_to_variables: A private, immutable mapping from names to variables
    """

    variable_group_container: Union[
        Mapping[Hashable, VariableGroup], Sequence[VariableGroup]
    ]

    def __post_init__(self):
        object.__setattr__(
            self,
            "_names_to_variables",
            MappingProxyType(self._get_names_to_variables()),
        )

    @typing.overload
    def __getitem__(self, name: Hashable) -> nodes.Variable:
        """This function is a typing overload and is overwritten by the implemented __getitem__"""

    @typing.overload
    def __getitem__(self, name: List) -> List[nodes.Variable]:
        """This function is a typing overload and is overwritten by the implemented __getitem__"""

    def __getitem__(self, name):
        """Given a name, retrieve the associated Variable from the associated VariableGroup.

        Args:
            name: a single name corresponding to a single Variable within a VariableGroup, or a list
                of such names

        Returns:
            A single variable if the name is not a list. A list of variables if name is a list

        Raises:
            ValueError: if the name does not have the right format (tuples with at least two elements).
        """
        if isinstance(name, List):
            names_list = name
        else:
            names_list = [name]

        vars_list = []
        for curr_name in names_list:
            if len(curr_name) < 2:
                raise ValueError(
                    "The name needs to have at least 2 elements to index from a composite variable group."
                )

            variable_group = self.variable_group_container[curr_name[0]]
            if len(curr_name) == 2:
                vars_list.append(variable_group[curr_name[1]])
            else:
                vars_list.append(variable_group[curr_name[1:]])

        if isinstance(name, List):
            return vars_list
        else:
            return vars_list[0]

    def _get_names_to_variables(self) -> OrderedDict[Hashable, nodes.Variable]:
        """Function that generates a dictionary mapping names to variables.

        Returns:
            a dictionary mapping all possible names to different variables.
        """
        names_to_variables: OrderedDict[
            Hashable, nodes.Variable
        ] = collections.OrderedDict()
        for container_name in self.container_names:
            for variable_group_name in self.variable_group_container[
                container_name
            ].names:
                if isinstance(variable_group_name, tuple):
                    names_to_variables[
                        (container_name,) + variable_group_name
                    ] = self.variable_group_container[container_name][
                        variable_group_name
                    ]
                else:
                    names_to_variables[
                        (container_name, variable_group_name)
                    ] = self.variable_group_container[container_name][
                        variable_group_name
                    ]

        return names_to_variables

    def flatten(self, data: Union[Mapping, Sequence]) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data.
                The structure of data should match self.variable_group_container.


        Returns:
            A flat jnp.array for internal use
        """
        flat_data = jnp.concatenate(
            [
                self.variable_group_container[name].flatten(data[name])
                for name in self.container_names
            ]
        )
        return flat_data

    def unflatten(
        self, flat_data: Union[np.ndarray, jnp.ndarray]
    ) -> Union[Mapping, Sequence]:
        """Function that recovers meaningful structured data from internal flat data array

        Args:
            flat_data: Internal flat data array.

        Returns:
            Meaningful structured data, with structure matching that of self.variable_group_container.

        Raises:
            ValueError if:
                (1) flat_data is not a 1D array
                (2) flat_data is not of the right shape
        """
        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        num_variables = 0
        num_variable_states = 0
        for name in self.container_names:
            variable_group = self.variable_group_container[name]
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
                f"flat_data should be either of shape (num_variables(={len(self.variables)}),), "
                f"or (num_variable_states(={num_variable_states}),). "
                f"Got {flat_data.shape}"
            )

        data: List[np.ndarray] = []
        start = 0
        for name in self.container_names:
            variable_group = self.variable_group_container[name]
            length = len(variable_group.variables)
            if use_num_states:
                length *= variable_group.variables[0].num_states

            data.append(variable_group.unflatten(flat_data[start : start + length]))
            start += length
        if isinstance(self.variable_group_container, Mapping):
            return dict(
                [(name, data[kk]) for kk, name in enumerate(self.container_names)]
            )
        else:
            return data

    @cached_property
    def container_names(self) -> Tuple:
        """Function to get names referring to the variable groups within this
        CompositeVariableGroup.

        Returns:
            a tuple of the names referring to the variable groups within this
            CompositeVariableGroup.
        """
        if isinstance(self.variable_group_container, Mapping):
            container_names = tuple(self.variable_group_container.keys())
        else:
            container_names = tuple(range(len(self.variable_group_container)))

        return container_names


@dataclass(frozen=True, eq=False)
class NDVariableArray(VariableGroup):
    """Subclass of VariableGroup for n-dimensional grids of variables.

    Args:
        num_states: The size of the variables in this variable group
        shape: a tuple specifying the size of each dimension of the grid (similar to
            the notion of a NumPy ndarray shape)
    """

    num_states: int
    shape: Tuple[int, ...]

    def _get_names_to_variables(
        self,
    ) -> OrderedDict[Union[int, Tuple[int, ...]], nodes.Variable]:
        """Function that generates a dictionary mapping names to variables.

        Returns:
            a dictionary mapping all possible names to different variables.
        """
        names_to_variables: OrderedDict[
            Union[int, Tuple[int, ...]], nodes.Variable
        ] = collections.OrderedDict()
        for name in itertools.product(*[list(range(k)) for k in self.shape]):
            if len(name) == 1:
                names_to_variables[name[0]] = nodes.Variable(self.num_states)
            else:
                names_to_variables[name] = nodes.Variable(self.num_states)

        return names_to_variables

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data. Should be an array of shape self.shape (for e.g. MAP decodings)
                or self.shape + (self.num_states,) (for e.g. evidence, beliefs).

        Returns:
            A flat jnp.array for internal use

        Raises:
            ValueError: If the data is not of the correct shape.
        """
        if data.shape != self.shape and data.shape != self.shape + (self.num_states,):
            raise ValueError(
                f"data should be of shape {self.shape} or {self.shape + (self.num_states,)}. "
                f"Got {data.shape}."
            )

        return jax.device_put(data).flatten()

    def unflatten(
        self, flat_data: Union[np.ndarray, jnp.ndarray]
    ) -> Union[np.ndarray, jnp.ndarray]:
        """Function that recovers meaningful structured data from internal flat data array

        Args:
            flat_data: Internal flat data array.

        Returns:
            Meaningful structured data. An array of shape self.shape (for e.g. MAP decodings)
                or an array of shape self.shape + (self.num_states,) (for e.g. evidence, beliefs).

        Raises:
            ValueError if:
                (1) flat_data is not a 1D array
                (2) flat_data is not of the right shape
        """
        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        if flat_data.size == np.product(self.shape):
            data = flat_data.reshape(self.shape)
        elif flat_data.size == np.product(self.shape) * self.num_states:
            data = flat_data.reshape(self.shape + (self.num_states,))
        else:
            raise ValueError(
                f"flat_data should be compatible with shape {self.shape} or {self.shape + (self.num_states,)}. "
                f"Got {flat_data.shape}."
            )

        return data


@dataclass(frozen=True, eq=False)
class VariableDict(VariableGroup):
    """A variable dictionary that contains a set of variables of the same size

    Args:
        num_states: The size of the variables in this variable group
        variable_names: A tuple of all names of the variables in this variable group

    """

    num_states: int
    variable_names: Tuple[Any, ...]

    def _get_names_to_variables(self) -> OrderedDict[Tuple[int, ...], nodes.Variable]:
        """Function that generates a dictionary mapping names to variables.

        Returns:
            a dictionary mapping all possible names to different variables.
        """
        names_to_variables: OrderedDict[
            Tuple[Any, ...], nodes.Variable
        ] = collections.OrderedDict()
        for name in self.variable_names:
            names_to_variables[name] = nodes.Variable(self.num_states)

        return names_to_variables

    def flatten(
        self, data: Mapping[Hashable, Union[np.ndarray, jnp.ndarray]]
    ) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data. Should be a mapping with names from self.variable_names.
                Each value should be an array of shape (1,) (for e.g. MAP decodings) or
                (self.num_states,) (for e.g. evidence, beliefs).

        Returns:
            A flat jnp.array for internal use

        Raises:
            ValueError if:
                (1) data is referring to a non-existing variable
                (2) data is not of the correct shape
        """
        for name in data:
            if name not in self._names_to_variables:
                raise ValueError(
                    f"data is referring to a non-existent variable {name}."
                )

            if data[name].shape != (self.num_states,) and data[name].shape != (1,):
                raise ValueError(
                    f"Variable {name} expects a data array of shape "
                    f"{(self.num_states,)} or (1,). Got {data[name].shape}."
                )

        flat_data = jnp.concatenate([data[name].flatten() for name in self.names])
        return flat_data

    def unflatten(
        self, flat_data: Union[np.ndarray, jnp.ndarray]
    ) -> Dict[Hashable, Union[np.ndarray, jnp.ndarray]]:
        """Function that recovers meaningful structured data from internal flat data array

        Args:
            flat_data: Internal flat data array.

        Returns:
            Meaningful structured data. Should be a mapping with names from self.variable_names.
                Each value should be an array of shape (1,) (for e.g. MAP decodings) or
                (self.num_states,) (for e.g. evidence, beliefs).

        Raises:
            ValueError if:
                (1) flat_data is not a 1D array
                (2) flat_data is not of the right shape
        """
        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        num_variables = len(self.variable_names)
        num_variable_states = len(self.variable_names) * self.num_states
        if flat_data.shape[0] == num_variables:
            use_num_states = False
        elif flat_data.shape[0] == num_variable_states:
            use_num_states = True
        else:
            raise ValueError(
                f"flat_data should be either of shape (num_variables(={len(self.variables)}),), "
                f"or (num_variable_states(={num_variable_states}),). "
                f"Got {flat_data.shape}"
            )

        start = 0
        data = {}
        for name in self.variable_names:
            if use_num_states:
                data[name] = flat_data[start : start + self.num_states]
                start += self.num_states
            else:
                data[name] = flat_data[np.array([start])]
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
        _variables_to_factors: maps set of connected variables to the corresponding factors

    Raises:
        ValueError: if variable_names_for_factors is an empty list
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
        """Function that generates a dictionary mapping names to factors.

        Returns:
            a dictionary mapping all possible names to different factors.
        """
        raise NotImplementedError(
            "Please subclass the FactorGroup class and override this method"
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


@dataclass(frozen=True, eq=False)
class EnumerationFactorGroup(FactorGroup):
    """Class to represent a group of EnumerationFactors.

    All factors in the group are assumed to have the same set of valid configurations and
    the same potential function. Note that the log potential function is assumed to be
    uniform 0 unless the inheriting class includes a log_potentials argument.

    Args:
        variable_names_for_factors: A list of list of variable names, where each innermost element is the
            name of a variable in variable_group. Each list within the outer list is taken to contain
            the names of the variables connected to a factor.
        factor_configs: Array of shape (num_val_configs, num_variables)
            An array containing explicit enumeration of all valid configurations
        log_potentials: Optional array of shape (num_val_configs,) or (num_factors, num_val_configs).
            If specified, it contains the log of the potential value for every possible configuration.
            If none, it is assumed the log potential is uniform 0 and such an array is automatically
            initialized.
    """

    variable_names_for_factors: Sequence[List]
    factor_configs: np.ndarray
    log_potentials: Optional[np.ndarray] = None

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, nodes.EnumerationFactor]:
        """Function that generates a dictionary mapping set of connected variables to factors.

        Returns:
            a dictionary mapping all possible set of connected variables to different factors.

        Raises:
            ValueError: if the specified log_potentials is not of the right shape
        """
        num_factors = len(self.variable_names_for_factors)
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
                    frozenset(self.variable_names_for_factors[ii]),
                    nodes.EnumerationFactor(
                        tuple(self.variable_group[self.variable_names_for_factors[ii]]),
                        self.factor_configs,
                        log_potentials[ii],
                    ),
                )
                for ii in range(len(self.variable_names_for_factors))
            ]
        )
        return variables_to_factors

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data. Should be an array of shape (num_val_configs,)
                (for shared log potentials) or (num_factors, num_val_configs) (for log potentials)
                or (num_factors, num_edge_states) (for ftov messages).

        Returns:
            A flat jnp.array for internal use

        Raises:
            ValueError: if data is not of the right shape.
        """
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
        """Function that recovers meaningful structured data from internal flat data array

        Args:
            flat_data: Internal flat data array.

        Returns:
            Meaningful structured data. Should be an array of shape (num_val_configs,)
                (for shared log potentials) or (num_factors, num_val_configs) (for log potentials)
                or (num_factors, num_edge_states) (for ftov messages).

        Raises:
            ValueError if:
                (1) flat_data is not a 1D array
                (2) flat_data is not of the right shape
        """
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
        variable_names_for_factors: A list of list of tuples, where each innermost tuple contains a
            name into variable_group. Each list within the outer list is taken to contain the names of variables
            neighboring a particular factor to be added.
        log_potential_matrix: array of shape (var1.num_states, var2.num_states),
            where var1 and var2 are the 2 VariableGroups (that may refer to the same
            VariableGroup) whose names are present in each sub-list from self.variable_names_for_factors.
    """

    variable_names_for_factors: Sequence[List]
    log_potential_matrix: Optional[np.ndarray] = None

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, nodes.EnumerationFactor]:
        """Function that generates a dictionary mapping set of connected variables to factors.

        Returns:
            a dictionary mapping all possible set of connected variables to different factors.

        Raises:
            ValueError if:
                (1) The specified log_potential_matrix is not a 2D or 3D array.
                (2) Some pairwise factors connect to less or more than 2 variables.
                (3) The specified log_potential_matrix does not match the number of factors.
                (4) The specified log_potential_matrix does not match the number of variable states of the
                    variables in the factors.
        """
        if self.log_potential_matrix is None:
            log_potential_matrix = np.zeros(
                (
                    self.variable_group[
                        self.variable_names_for_factors[0][0]
                    ].num_states,
                    self.variable_group[
                        self.variable_names_for_factors[0][1]
                    ].num_states,
                )
            )
        else:
            log_potential_matrix = self.log_potential_matrix

        if not (log_potential_matrix.ndim == 2 or log_potential_matrix.ndim == 3):
            raise ValueError(
                "log_potential_matrix should be either a 2D array, specifying shared parameters for all "
                "pairwise factors, or 3D array, specifying parameters for individual pairwise factors. "
                f"Got a {log_potential_matrix.ndim}D log_potential_matrix array."
            )

        if log_potential_matrix.ndim == 3 and log_potential_matrix.shape[0] != len(
            self.variable_names_for_factors
        ):
            raise ValueError(
                f"Expected log_potential_matrix for {len(self.variable_names_for_factors)} factors. "
                f"Got log_potential_matrix for {log_potential_matrix.shape[0]} factors."
            )

        for fac_list in self.variable_names_for_factors:
            if len(fac_list) != 2:
                raise ValueError(
                    "All pairwise factors should connect to exactly 2 variables. Got a factor connecting to"
                    f" {len(fac_list)} variables ({fac_list})."
                )

            if not (
                log_potential_matrix.shape[-2:]
                == (
                    self.variable_group[fac_list[0]].num_states,
                    self.variable_group[fac_list[1]].num_states,
                )
            ):
                raise ValueError(
                    f"The specified pairwise factor {fac_list} (with "
                    f"{(self.variable_group[fac_list[0]].num_states, self.variable_group[fac_list[1]].num_states)} "
                    f"configurations) does not match the specified log_potential_matrix "
                    f"(with {log_potential_matrix.shape[-2:]} configurations)."
                )

        factor_configs = (
            np.mgrid[
                : log_potential_matrix.shape[-2],
                : log_potential_matrix.shape[-1],
            ]
            .transpose((1, 2, 0))
            .reshape((-1, 2))
        )
        object.__setattr__(self, "log_potential_matrix", log_potential_matrix)
        log_potential_matrix = np.broadcast_to(
            log_potential_matrix,
            (len(self.variable_names_for_factors),) + log_potential_matrix.shape[-2:],
        )
        variables_to_factors = collections.OrderedDict(
            [
                (
                    frozenset(self.variable_names_for_factors[ii]),
                    nodes.EnumerationFactor(
                        tuple(self.variable_group[self.variable_names_for_factors[ii]]),
                        factor_configs,
                        log_potential_matrix[
                            ii, factor_configs[:, 0], factor_configs[:, 1]
                        ],
                    ),
                )
                for ii in range(len(self.variable_names_for_factors))
            ]
        )
        return variables_to_factors

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data. Should be an array of shape
                (num_factors, var0_num_states, var1_num_states) (for log potential matrices)
                or (num_factors, var0_num_states + var1_num_states) (for ftov messages)
                or (var0_num_states, var1_num_states) (for shared log potential matrix).

        Returns:
            A flat jnp.array for internal use
        """
        assert isinstance(self.log_potential_matrix, np.ndarray)
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
        """Function that recovers meaningful structured data from internal flat data array

        Args:
            flat_data: Internal flat data array.

        Returns:
            Meaningful structured data. Should be an array of shape
                (num_factors, var0_num_states, var1_num_states) (for log potential matrices)
                or (num_factors, var0_num_states + var1_num_states) (for ftov messages)
                or (var0_num_states, var1_num_states) (for shared log potential matrix).

        Raises:
            ValueError if:
                (1) flat_data is not a 1D array
                (2) flat_data is not of the right shape
        """
        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        assert isinstance(self.log_potential_matrix, np.ndarray)
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
                f"or {(num_factors, np.sum(self.log_potential_matrix.shape[-2:]))}. Got {flat_data.shape}."
            )

        return data
