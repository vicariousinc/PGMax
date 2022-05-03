"""Defines EnumFactorGroup and PairwiseFactorGroup."""

import collections
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, OrderedDict, Type, Union

import jax
import jax.numpy as jnp
import numba as nb
import numpy as np

from pgmax.factor import enum

from .fgroup import FactorGroup


@dataclass(frozen=True, eq=False)
class EnumFactorGroup(FactorGroup):
    """Class to represent a group of EnumFactors.

    All factors in the group are assumed to have the same set of valid configurations and
    the same potential function. Note that the log potential function is assumed to be
    uniform 0 unless the inheriting class includes a log_potentials argument.

    Args:
        factor_configs: Array of shape (num_val_configs, num_variables)
            An array containing explicit enumeration of all valid configurations
        log_potentials: Optional array of shape (num_val_configs,) or (num_factors, num_val_configs).
            If specified, it contains the log of the potential value for every possible configuration.
            If none, it is assumed the log potential is uniform 0 and such an array is automatically
            initialized.
        factor_type: Factor type shared by all the Factors in the FactorGroup.

    Raises:
        ValueError if:
            (1) The specified log_potentials is not of the expected shape.
            (2) The dtype of the potential array is not float
    """

    factor_configs: np.ndarray
    log_potentials: Optional[np.ndarray] = None
    factor_type: Type = field(init=False, default=enum.EnumFactor)

    def __post_init__(self):
        super().__post_init__()

        num_val_configs = self.factor_configs.shape[0]
        if self.log_potentials is None:
            log_potentials = np.zeros((self.num_factors, num_val_configs), dtype=float)
        else:
            if self.log_potentials.shape != (
                num_val_configs,
            ) and self.log_potentials.shape != (
                self.num_factors,
                num_val_configs,
            ):
                raise ValueError(
                    f"Expected log potentials shape: {(num_val_configs,)} or {(self.num_factors, num_val_configs)}. "
                    f"Got {self.log_potentials.shape}."
                )
            log_potentials = np.broadcast_to(
                self.log_potentials, (self.num_factors, num_val_configs)
            )

        if not np.issubdtype(log_potentials.dtype, np.floating):
            raise ValueError(
                f"Potentials should be floats. Got {log_potentials.dtype}."
            )
        object.__setattr__(self, "log_potentials", log_potentials)

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, enum.EnumFactor]:
        """Function that generates a dictionary mapping set of connected variables to factors.
        This function is only called on demand when the user requires it.

        Returns:
            A dictionary mapping all possible set of connected variables to different factors.
        """
        variables_to_factors = collections.OrderedDict(
            [
                (
                    frozenset(variables_for_factor),
                    enum.EnumFactor(
                        variables=variables_for_factor,
                        factor_configs=self.factor_configs,
                        log_potentials=np.array(self.log_potentials)[ii],
                    ),
                )
                for ii, variables_for_factor in enumerate(self.variables_for_factors)
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
        factor_edges_num_states = sum(
            [variable[1] for variable in self.variables_for_factors[0]]
        )
        if (
            data.shape != (num_factors, self.factor_configs.shape[0])
            and data.shape != (num_factors, factor_edges_num_states)
            and data.shape != (self.factor_configs.shape[0],)
        ):
            raise ValueError(
                f"data should be of shape {(num_factors, self.factor_configs.shape[0])} or "
                f"{(num_factors, factor_edges_num_states)} or "
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
        factor_edges_num_states = sum(
            [variable[1] for variable in self.variables_for_factors[0]]
        )
        if flat_data.size == num_factors * self.factor_configs.shape[0]:
            data = flat_data.reshape(
                (num_factors, self.factor_configs.shape[0]),
            )
        elif flat_data.size == num_factors * np.sum(factor_edges_num_states):
            data = flat_data.reshape((num_factors, np.sum(factor_edges_num_states)))
        else:
            raise ValueError(
                f"flat_data should be compatible with shape {(num_factors, self.factor_configs.shape[0])} "
                f"or {(num_factors, np.sum(factor_edges_num_states))}. Got {flat_data.shape}."
            )

        return data


@dataclass(frozen=True, eq=False)
class PairwiseFactorGroup(FactorGroup):
    """Class to represent a group of EnumFactors where each factor connects to
    two different variables.

    All factors in the group are assumed to be such that all possible configuration of the two
    variable's states are valid. Additionally, all factors in the group are assumed to share
    the same potential function and to be connected to variables from VarGroups within
    one CompositeVarGroup.

    Args:
        log_potential_matrix: array of shape (var1.num_states, var2.num_states),
            where var1 and var2 are the 2 VarGroups (that may refer to the same
            VarGroup) whose names are present in each sub-list from self.variables_for_factors.
        factor_type: Factor type shared by all the Factors in the FactorGroup.

    Raises:
        ValueError if:
            (1) The specified log_potential_matrix is not a 2D or 3D array.
            (2) The dtype of the potential array is not float
            (3) Some pairwise factors connect to less or more than 2 variables.
            (4) The specified log_potential_matrix does not match the number of factors.
            (5) The specified log_potential_matrix does not match the number of variable states of the
                variables in the factors.
    """

    log_potential_matrix: Optional[np.ndarray] = None
    factor_type: Type = field(init=False, default=enum.EnumFactor)

    def __post_init__(self):
        super().__post_init__()

        if self.log_potential_matrix is None:
            log_potential_matrix = np.zeros(
                (
                    self.variables_for_factors[0][0][1],
                    self.variables_for_factors[0][1][1],
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

        if not np.issubdtype(log_potential_matrix.dtype, np.floating):
            raise ValueError(
                f"Potential matrix should be floats. Got {self.log_potential_matrix.dtype}."
            )

        if log_potential_matrix.ndim == 3 and log_potential_matrix.shape[0] != len(
            self.variables_for_factors
        ):
            raise ValueError(
                f"Expected log_potential_matrix for {len(self.variables_for_factors)} factors. "
                f"Got log_potential_matrix for {log_potential_matrix.shape[0]} factors."
            )

        log_potential_shape = log_potential_matrix.shape[-2:]
        for variables_for_factor in self.variables_for_factors:
            if len(variables_for_factor) != 2:
                raise ValueError(
                    "All pairwise factors should connect to exactly 2 variables. Got a factor connecting to"
                    f" {len(variables_for_factor)} variables ({variables_for_factor})."
                )

            factor_num_configs = (
                variables_for_factor[0][1],
                variables_for_factor[1][1],
            )
            if log_potential_shape != factor_num_configs:
                raise ValueError(
                    f"The specified pairwise factor {variables_for_factor} (with {factor_num_configs}"
                    f"configurations) does not match the specified log_potential_matrix "
                    f"(with {log_potential_shape} configurations)."
                )
        object.__setattr__(self, "log_potential_matrix", log_potential_matrix)

        factor_configs = (
            np.mgrid[
                : log_potential_matrix.shape[-2],
                : log_potential_matrix.shape[-1],
            ]
            .transpose((1, 2, 0))
            .reshape((-1, 2))
        )
        object.__setattr__(self, "factor_configs", factor_configs)

        log_potential_matrix = np.broadcast_to(
            log_potential_matrix,
            (len(self.variables_for_factors),) + log_potential_matrix.shape[-2:],
        )
        log_potentials = np.empty(
            shape=(self.num_factors, self.factor_configs.shape[0])
        )
        _compute_log_potentials(
            log_potentials, log_potential_matrix, self.factor_configs
        )
        object.__setattr__(self, "log_potentials", log_potentials)

    def _get_variables_to_factors(
        self,
    ) -> OrderedDict[FrozenSet, enum.EnumFactor]:
        """Function that generates a dictionary mapping set of connected variables to factors.
        This function is only called on demand when the user requires it.

        Returns:
            A dictionary mapping all possible set of connected variables to different factors.
        """
        variables_to_factors = collections.OrderedDict(
            [
                (
                    frozenset(variable_for_factor),
                    enum.EnumFactor(
                        variables=variable_for_factor,
                        factor_configs=self.factor_configs,
                        log_potentials=self.log_potentials[ii],
                    ),
                )
                for ii, variable_for_factor in enumerate(self.variables_for_factors)
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


@nb.jit(parallel=False, cache=True, fastmath=True, nopython=True)
def _compute_log_potentials(
    log_potentials: np.ndarray,
    log_potential_matrix: np.ndarray,
    factor_configs: np.ndarray,
) -> np.ndarray:
    """Fast numba computation of the log_potentials of a PairwiseFactorGroup.
    log_potentials is updated in-place.
    """

    for config_idx in range(factor_configs.shape[0]):
        log_potentials[:, config_idx] = log_potential_matrix[
            :, factor_configs[config_idx, 0], factor_configs[config_idx, 1]
        ]
