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
        num_states: An integer or an array specifying the number of states of the
            variables in this VariableGroup
        shape: Tuple specifying the size of each dimension of the grid (similar to
            the notion of a NumPy ndarray shape)
    """

    num_states: Union[int, np.ndarray]
    shape: Tuple[int, ...]

    def __post_init__(self):
        super().__post_init__()

        max_size = int(groups.MAX_SIZE)
        if np.prod(self.shape) > max_size:
            raise ValueError(
                f"Currently only support NDVariableArray of size smaller than {max_size}. Got {np.prod(self.shape)}"
            )

        if np.isscalar(self.num_states):
            num_states = np.full(self.shape, fill_value=self.num_states, dtype=np.int64)
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

        if isinstance(val, slice) or (
            isinstance(val, tuple) and isinstance(val[0], slice)
        ):
            assert isinstance(self.num_states, np.ndarray)
            vars_names = self.variable_hashes[val].flatten()
            vars_num_states = self.num_states[val].flatten()
            return list(zip(vars_names, vars_num_states))

        return (self.variable_hashes[val], self.num_states[val])

    @cached_property
    def variable_hashes(self) -> np.ndarray:
        """Function that generates a variable hash for each variable

        Returns:
            Array of variables hashes.
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
            data = jnp.full(
                shape=self.shape + (self.num_states.max(),), fill_value=jnp.nan
            )
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
    """A variable dictionary that contains a set of variables

    Args:
        num_states: The size of the variables in this VariableGroup
        variable_names: A tuple of all the names of the variables in this VariableGroup.
    """

    num_states: Union[int, np.ndarray]
    variable_names: Tuple[Any, ...]

    def __post_init__(self):
        super().__post_init__()

        if np.isscalar(self.num_states):
            num_states = np.full(
                (len(self.variable_names),), fill_value=self.num_states, dtype=np.int64
            )
            object.__setattr__(self, "num_states", num_states)
        elif isinstance(self.num_states, np.ndarray) and np.issubdtype(
            self.num_states.dtype, int
        ):
            if self.num_states.shape != len(self.variable_names):
                raise ValueError(
                    f"Expected num_states shape ({len(self.variable_names)},). Got {self.num_states.shape}."
                )
        else:
            raise ValueError(
                "num_states should be an integer or a NumPy array of dtype int"
            )

    @cached_property
    def variable_hashes(self) -> np.ndarray:
        """Function that generates a variable hash for each variable

        Returns:
            Array of variables hashes.
        """
        indices = np.arange(len(self.variable_names))
        return self.__hash__() + indices

    def __getitem__(self, var_name: Any) -> Tuple[int, int]:
        """Given a variable name retrieve the associated variable, returned via a tuple of the form
        (variable hash, number of states)

        Args:
            val: a variable name

        Returns:
            The queried variable
        """
        assert isinstance(self.num_states, np.ndarray)
        if var_name not in self.variable_names:
            raise ValueError(f"Variable {var_name} is not in VariableDict")

        var_idx = self.variable_names.index(var_name)
        return (self.variable_hashes[var_idx], self.num_states[var_idx])

    def flatten(
        self, data: Mapping[Any, Union[np.ndarray, jnp.ndarray]]
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

        for var_name in data:
            if var_name not in self.variable_names:
                raise ValueError(
                    f"data is referring to a non-existent variable {var_name}."
                )

            var_idx = self.variable_names.index(var_name)
            if data[var_name].shape != (self.num_states[var_idx],) and data[
                var_name
            ].shape != (1,):
                raise ValueError(
                    f"Variable {var_name} expects a data array of shape "
                    f"{(self.num_states[var_idx],)} or (1,). Got {data[var_name].shape}."
                )

        flat_data = jnp.concatenate(
            [data[var_name].flatten() for var_name in self.variable_names]
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
        for var_name in self.variable_names:
            if use_num_states:
                var_idx = self.variable_names.index(var_name)
                var_num_states = self.num_states[var_idx]
                data[var_name] = flat_data[start : start + var_num_states]
                start += var_num_states
            else:
                data[var_name] = flat_data[np.array([start])]
                start += 1

        return data
