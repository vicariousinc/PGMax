"""A module containing the variables group classes inheriting from the base VariableGroup."""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Mapping, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax.fg import groups
from pgmax.utils import cached_property


@dataclass(frozen=True, eq=False)
class NDVariableArray(groups.VariableGroup):
    """Subclass of VariableGroup for n-dimensional grids of variables.

    Args:
        shape: Tuple specifying the size of each dimension of the grid (similar to
            the notion of a NumPy ndarray shape)
        num_states: An integer or an array specifying the number of states of the
            variables in this VariableGroup
    """

    shape: Tuple[int, ...]
    num_states: Union[int, np.ndarray]

    def __post_init__(self):
        if np.prod(self.shape) > int(groups.MAX_SIZE):
            raise ValueError(
                f"Currently only support NDVariableArray of size smaller than {int(groups.MAX_SIZE)}. Got {np.prod(self.shape)}"
            )

        if np.isscalar(self.num_states):
            num_states = np.full(self.shape, fill_value=self.num_states, dtype=np.int32)
            object.__setattr__(self, "num_states", num_states)
        elif isinstance(self.num_states, np.ndarray) and np.issubdtype(
            self.num_states.dtype, int
        ):
            if self.num_states.shape != self.shape:
                raise ValueError(
                    f"Expected num_states shape {self.shape}. Got {self.num_states.shape}."
                )
        else:
            raise ValueError(
                "num_states should be an integer or a NumPy array of dtype int"
            )

    def __getitem__(
        self, val: Union[int, slice, Tuple]
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """Given an index or a slice, retrieve the associated variable(s).
        Each variable is returned via a tuple of the form (variable hash, number of states)

        Note: Relies on numpy indexation to throw IndexError if val is out-of-bounds

        Args:
            val: a variable index or slice

        Returns:
            A single variable or a list of variables
        """
        assert isinstance(self.num_states, np.ndarray)
        if np.isscalar(self.variable_names[val]):
            return (self.variable_names[val], self.num_states[val])
        else:
            vars_names = self.variable_names[val].flatten()
            vars_num_states = self.num_states[val].flatten()
            return list(zip(vars_names, vars_num_states))

    @cached_property
    def variables(self) -> List[Tuple]:
        """Function that returns the list of all variables in the VariableGroup.
        Each variable is represented by a tuple of the form (variable hash, number of states)

        Returns:
            List of variables in the VariableGroup
        """
        assert isinstance(self.num_states, np.ndarray)
        vars_names = self.variable_names.flatten()
        vars_num_states = self.num_states.flatten()
        return list(zip(vars_names, vars_num_states))

    @cached_property
    def variable_names(self) -> np.ndarray:
        """Function that generates all the variables names, in the form of hashes

        Returns:
            Array of variables names.
        """
        indices = np.reshape(np.arange(np.product(self.shape)), self.shape)
        return self.__hash__() + indices

    def flatten(self, data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """Function that turns meaningful structured data into a flat data array for internal use.

        Args:
            data: Meaningful structured data. Should be an array of shape self.shape (for e.g. MAP decodings)
                or self.shape + (self.num_states.max(),) (for e.g. evidence, beliefs).

        Returns:
            A flat jnp.array for internal use

        Raises:
            ValueError: If the data is not of the correct shape.
        """
        assert isinstance(self.num_states, np.ndarray)

        if data.shape == self.shape:
            return jax.device_put(data).flatten()
        elif data.shape == self.shape + (self.num_states.max(),):
            return jax.device_put(
                data[np.arange(data.shape[-1]) < self.num_states[..., None]]
            )
        else:
            raise ValueError(
                f"data should be of shape {self.shape} or {self.shape + (self.num_states.max(),)}. "
                f"Got {data.shape}."
            )

    def unflatten(self, flat_data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
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
        assert isinstance(self.num_states, np.ndarray)

        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        if flat_data.size == np.product(self.shape):
            data = flat_data.reshape(self.shape)
        elif flat_data.size == self.num_states.sum():
            data = jnp.zeros(self.shape + (self.num_states.max(),))
            data = data.at[np.arange(data.shape[-1]) < self.num_states[..., None]].set(
                flat_data
            )
        else:
            raise ValueError(
                f"flat_data should be compatible with shape {self.shape} or {self.shape + (self.num_states.max(),)}. "
                f"Got {flat_data.shape}."
            )

        return data


@dataclass(frozen=True, eq=False)
class VariableDict(groups.VariableGroup):
    """A variable dictionary that contains a set of variables of the same size

    Args:
        num_states: The size of the variables in this variable group
        variable_names: A tuple of all names of the variables in this variable group.
            Note that we overwrite variable_names to add the hash of the VariableDict
    """

    variable_names: Tuple[Any, ...]
    num_states: int

    def __post_init__(self):
        num_states = np.full(
            (len(self.variable_names),), fill_value=self.num_states, dtype=np.int32
        )
        object.__setattr__(self, "num_states", num_states)

        hash_and_names = tuple(
            (self.__hash__(), var_name) for var_name in self.variable_names
        )
        object.__setattr__(self, "variable_names", hash_and_names)

    @cached_property
    def variables(self) -> List[Tuple[Tuple[Any, int], int]]:
        """Function that returns the list of all variables in the VariableGroup.
        Each variable is represented by a tuple of the form (variable name, number of states)

        Returns:
            List of variables in the VariableGroup
        """
        assert isinstance(self.num_states, np.ndarray)
        vars_names = list(self.variable_names)
        vars_num_states = self.num_states.flatten()
        return list(zip(vars_names, vars_num_states))

    def __getitem__(self, val: Any) -> Tuple[Tuple[Any, int], int]:
        """Given a variable name retrieve the associated variable, returned via a tuple of the form
        (variable name, number of states)

        Args:
            val: a variable name

        Returns:
            The queried variable
        """
        assert isinstance(self.num_states, np.ndarray)
        if (self.__hash__(), val) not in self.variable_names:
            raise ValueError(f"Variable {val} is not in VariableDict")
        return ((self.__hash__(), val), self.num_states[0])

    def flatten(
        self, data: Mapping[Tuple[Tuple[int, int], int], Union[np.ndarray, jnp.ndarray]]
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
        assert isinstance(self.num_states, np.ndarray)

        for variable in data:
            if variable not in self.variables:
                raise ValueError(
                    f"data is referring to a non-existent variable {variable}."
                )

            if data[variable].shape != (variable[1],) and data[variable].shape != (1,):
                raise ValueError(
                    f"Variable {variable} expects a data array of shape "
                    f"{(variable[1],)} or (1,). Got {data[variable].shape}."
                )

        flat_data = jnp.concatenate(
            [data[variable].flatten() for variable in self.variables]
        )
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
        assert isinstance(self.num_states, np.ndarray)

        if flat_data.ndim != 1:
            raise ValueError(
                f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
            )

        num_variables = len(self.variable_names)
        num_variable_states = self.num_states.sum()
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
        for variable in self.variables:
            if use_num_states:
                data[variable] = flat_data[start : start + variable[1]]
                start += variable[1]
            else:
                data[variable] = flat_data[np.array([start])]
                start += 1

        return data
