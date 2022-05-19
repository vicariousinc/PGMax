"""A module containing a variable dictionnary class inheriting from the base VarGroup."""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Tuple, Union

import jax.numpy as jnp
import numpy as np

from pgmax.utils import cached_property

from . import vgroup


@dataclass(frozen=True, eq=False)
class VarDict(vgroup.VarGroup):
    """A variable dictionary that contains a set of variables

    Args:
        num_states: The size of the variables in this VarGroup
        variable_names: A tuple of all the names of the variables in this VarGroup.
    """

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
            if self.num_states.shape != (len(self.variable_names),):
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
            raise ValueError(f"Variable {var_name} is not in VarDict")

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
