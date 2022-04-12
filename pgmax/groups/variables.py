"""A module containing the variables group classes inheriting from the base VariableGroup."""

import itertools
import random
from dataclasses import dataclass
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax.utils import cached_property


@dataclass(frozen=True, eq=False)
class NDVariableArray:
    """Subclass of VariableGroup for n-dimensional grids of variables.

    Args:
        num_states: The size of the variables in this variable group
        shape: a tuple specifying the size of each dimension of the grid (similar to
            the notion of a NumPy ndarray shape)
    """

    shape: Tuple[int, ...]
    num_states: Union[int, np.ndarray]

    def __post_init__(self):
        # super().__post_init__()

        if isinstance(self.num_states, int):
            num_states = np.full(self.shape, fill_value=self.num_states)
            object.__setattr__(self, "num_states", num_states)
        elif isinstance(self.num_states, np.ndarray):
            if self.num_states.shape != self.shape:
                raise ValueError("Should be same shape")

    def __getitem__(self, val):
        # Numpy indexation will throw IndexError for us if out-of-bounds
        return self.variable_names[val]

    @cached_property
    def variable_names(self) -> np.ndarray:
        """Function that generates a dictionary mapping names to variables.

        Returns:
            a dictionary mapping all possible names to different variables.
        """
        # Overwite default hash as it does not give enough spacing across consecutive objects
        this_hash = random.randint(0, 2**63)
        indices = np.reshape(np.arange(np.product(self.shape)), self.shape)
        return this_hash + indices

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
        # TODO: what should we do for different number of states
        if data.shape != self.shape and data.shape != self.shape + (
            self.num_states.max(),
        ):
            raise ValueError(
                f"data should be of shape {self.shape} or {self.shape + (self.num_states.max(),)}. "
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
        elif flat_data.size == self.num_states.sum():
            # TODO: what should we dot for different number of states
            data = flat_data.reshape(self.shape + (self.num_states.max(),))
        else:
            raise ValueError(
                f"flat_data should be compatible with shape {self.shape} or {self.shape + (self.num_states,)}. "
                f"Got {flat_data.shape}."
            )

        return data


# TODO: delete?
# @dataclass(frozen=True, eq=False)
# class VariableDict():
#     """A variable dictionary that contains a set of variables of the same size

#     Args:
#         num_states: The size of the variables in this variable group
#         num_variables: The number of variables

#     """

#     num_states: int
#     num_variables: int

#     @cached_property
#     def variable_names(self) -> np.ndarray:
#         """Function that generates a dictionary mapping names to variables.

#         Returns:
#             a dictionary mapping all possible names to different variables.
#         """
#         return self.__hash__() + np.arange(self.num_states)


#     def flatten(
#         self, data: Mapping[Hashable, Union[np.ndarray, jnp.ndarray]]
#     ) -> jnp.ndarray:
#         """Function that turns meaningful structured data into a flat data array for internal use.

#         Args:
#             data: Meaningful structured data. Should be a mapping with names from self.variable_names.
#                 Each value should be an array of shape (1,) (for e.g. MAP decodings) or
#                 (self.num_states,) (for e.g. evidence, beliefs).

#         Returns:
#             A flat jnp.array for internal use

#         Raises:
#             ValueError if:
#                 (1) data is referring to a non-existing variable
#                 (2) data is not of the correct shape
#         """
#         for name in data:
#             if name not in self._names_to_variables:
#                 raise ValueError(
#                     f"data is referring to a non-existent variable {name}."
#                 )

#             if data[name].shape != (self.num_states,) and data[name].shape != (1,):
#                 raise ValueError(
#                     f"Variable {name} expects a data array of shape "
#                     f"{(self.num_states,)} or (1,). Got {data[name].shape}."
#                 )

#         flat_data = jnp.concatenate([data[name].flatten() for name in self.names])
#         return flat_data

#     def unflatten(
#         self, flat_data: Union[np.ndarray, jnp.ndarray]
#     ) -> Dict[Hashable, Union[np.ndarray, jnp.ndarray]]:
#         """Function that recovers meaningful structured data from internal flat data array

#         Args:
#             flat_data: Internal flat data array.

#         Returns:
#             Meaningful structured data. Should be a mapping with names from self.variable_names.
#                 Each value should be an array of shape (1,) (for e.g. MAP decodings) or
#                 (self.num_states,) (for e.g. evidence, beliefs).

#         Raises:
#             ValueError if:
#                 (1) flat_data is not a 1D array
#                 (2) flat_data is not of the right shape
#         """
#         if flat_data.ndim != 1:
#             raise ValueError(
#                 f"Can only unflatten 1D array. Got a {flat_data.ndim}D array."
#             )

#         num_variables = len(self.variable_names)
#         num_variable_states = len(self.variable_names) * self.num_states
#         if flat_data.shape[0] == num_variables:
#             use_num_states = False
#         elif flat_data.shape[0] == num_variable_states:
#             use_num_states = True
#         else:
#             raise ValueError(
#                 f"flat_data should be either of shape (num_variables(={len(self.variables)}),), "
#                 f"or (num_variable_states(={num_variable_states}),). "
#                 f"Got {flat_data.shape}"
#             )

#         start = 0
#         data = {}
#         for name in self.variable_names:
#             if use_num_states:
#                 data[name] = flat_data[start : start + self.num_states]
#                 start += self.num_states
#             else:
#                 data[name] = flat_data[np.array([start])]
#                 start += 1

#         return data
