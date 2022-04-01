"""A module containing the base classes for variable and factor groups in a Factor Graph."""

import collections
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
class FactorGroup:
    """Class to represent a group of Factors.

    Args:
        variable_group: either a VariableGroup or - if the elements of more than one VariableGroup
            are connected to this FactorGroup - then a CompositeVariableGroup. This holds
            all the variables that are connected to this FactorGroup
        variable_names_for_factors: A list of list of variable names, where each innermost element is the
            name of a variable in variable_group. Each list within the outer list is taken to contain
            the names of the variables connected to a Factor.
        num_factors: Number of Factors in the FactorGroup.
        factor_edges_num_states: An array concatenating the number of states for the variables connected to each
            Factor of the FactorGroup. Each variable will appear once for each Factor it connects to.
        variables_for_factors: A tuple concatenating the variables (with repetition) connected to each
            Factor of the FactorGroup. Each variable will appear once for each Factor it connects to.
        factor_type: Factor type shared by all the Factors in the FactorGroup.
        log_potentials: array of log potentials.
    """

    variable_group: Union[CompositeVariableGroup, VariableGroup]
    variable_names_for_factors: Sequence[List]
    num_factors: int = field(init=False)
    factor_edges_num_states: np.ndarray = field(init=False)
    variables_for_factors: Tuple[nodes.Variable, ...] = field(init=False)
    factor_type: Type = field(init=False)
    factor_configs: np.ndarray = field(init=False, default=None)
    log_potentials: np.ndarray = field(init=False, default=np.empty((0,)))

    def __post_init__(self):
        if len(self.variable_names_for_factors) == 0:
            raise ValueError("Do not add a factor group with no factors.")

        object.__setattr__(self, "num_factors", len(self.variable_names_for_factors))

        factor_edges_num_states = []
        variables_for_factors = []

        for variable_names_for_factor in self.variable_names_for_factors:
            for variable_name in variable_names_for_factor:
                variable = self.variable_group._names_to_variables[variable_name]
                num_states = variable.num_states
                factor_edges_num_states.append(num_states)
                variables_for_factors.append(variable)

        object.__setattr__(
            self, "factor_edges_num_states", np.array(factor_edges_num_states)
        )
        object.__setattr__(self, "variables_for_factors", tuple(variables_for_factors))

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

    def compile_wiring(self, vars_to_starts: Mapping[nodes.Variable, int]) -> Any:
        """Compile wiring for the FactorGroup.
        In pratice, this function is overwritten to implement an efficient wiring at the
        FactorGroup level.
        When this does not happen, the slower fallback concatenates wiring at the Factor level.

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
            Wiring for the FactorGroup
        """

        wirings = [factor.compile_wiring(vars_to_starts) for factor in self.factors]
        wiring = self.factor_type.concatenate_wirings(wirings)
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
