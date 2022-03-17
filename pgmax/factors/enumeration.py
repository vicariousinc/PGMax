"""Defines an enumeration factor"""

from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax import utils
from pgmax.fg import nodes


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class EnumerationWiring(nodes.Wiring):
    """Wiring for EnumerationFactors.

    Args:
        factor_configs_edge_states: Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global enumeration factor_config and global edge_state indices
            factor_configs_edge_states[ii, 0] contains the global EnumerationFactor config index,
            which takes into account all the EnumerationFactors and LogicalFactors.
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index
            which takes into account all the EnumerationFactors.
    """

    factor_configs_edge_states: Union[np.ndarray, jnp.ndarray]

    def __post_init__(self):
        inference_arguments = {
            "factor_configs_edge_states": self.factor_configs_edge_states,
            "num_val_configs": int(self.factor_configs_edge_states[-1, 0]) + 1,
        }
        object.__setattr__(self, "inference_arguments", inference_arguments)


@dataclass(frozen=True, eq=False)
class EnumerationFactor(nodes.Factor):
    """An enumeration factor

    Args:
        configs: Array of shape (num_val_configs, num_variables)
            An array containing an explicit enumeration of all valid configurations

    Raises:
        ValueError: If:
            (1) The dtype of the configs array is not int
            (2) The dtype of the potential array is not float
            (3) Configs does not have the correct shape
            (4) The potential array does not have the correct shape
            (5) The configs array contains invalid values
    """

    configs: np.ndarray

    def __post_init__(self):
        self.configs.flags.writeable = False
        if not np.issubdtype(self.configs.dtype, np.integer):
            raise ValueError(
                f"Configurations should be integers. Got {self.configs.dtype}."
            )

        if not np.issubdtype(self.log_potentials.dtype, np.floating):
            raise ValueError(
                f"Potential should be floats. Got {self.log_potentials.dtype}."
            )

        if self.configs.ndim != 2:
            raise ValueError(
                "configs should be a 2D array containing a list of valid configurations for "
                f"EnumerationFactor. Got a configs array of shape {self.configs.shape}."
            )

        if len(self.variables) != self.configs.shape[1]:
            raise ValueError(
                f"Number of variables {len(self.variables)} doesn't match given configurations {self.configs.shape}"
            )

        if self.log_potentials.shape != (self.configs.shape[0],):
            raise ValueError(
                f"Expected log potentials of shape {(self.configs.shape[0],)} for "
                f"({self.configs.shape[0]}) valid configurations. Got log potentials of "
                f"shape {self.log_potentials.shape}."
            )

        vars_num_states = np.array([variable.num_states for variable in self.variables])
        if not np.logical_and(
            self.configs >= 0, self.configs < vars_num_states[None]
        ).all():
            raise ValueError("Invalid configurations for given variables")

    @utils.cached_property
    def factor_configs_edge_states(self) -> np.ndarray:
        """Array containing factor configs and edge states pairs

        Returns:
            Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global factor_config and edge_state indices
            factor_configs_edge_states[ii, 0] contains the global factor config index,
            which takes into account all the EnumerationFactors
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index,
            which takes into account all the EnumerationFactors and LogicalFactors
        """
        edges_starts = np.insert(self.edges_num_states.cumsum(), 0, 0)[:-1]
        factor_configs_edge_states = np.stack(
            [
                np.repeat(np.arange(self.configs.shape[0]), self.configs.shape[1]),
                (self.configs + edges_starts[None]).flatten(),
            ],
            axis=1,
        )
        return factor_configs_edge_states

    def compile_wiring(
        self, vars_to_starts: Mapping[nodes.Variable, int]
    ) -> EnumerationWiring:
        """Compile EnumerationWiring for the EnumerationFactor

        Args:
            vars_to_starts: A dictionary that maps variables to their global starting indices
                For an n-state variable, a global start index of m means the global indices
                of its n variable states are m, m + 1, ..., m + n - 1

        Returns:
            EnumerationWiring for the EnumerationFactor
        """
        var_states_for_edges = np.concatenate(
            [
                np.arange(variable.num_states) + vars_to_starts[variable]
                for variable in self.variables
            ]
        )
        return EnumerationWiring(
            edges_num_states=self.edges_num_states,
            var_states_for_edges=var_states_for_edges,
            factor_configs_edge_states=self.factor_configs_edge_states,
        )


def concatenate_enumeration_wirings(
    enum_wirings: Sequence[EnumerationWiring],
) -> EnumerationWiring:
    """Concatenate a list of EnumerationWirings from individual EnumerationFactors

    Args:
        enum_wirings: A list of EnumerationWirings, one for each individual EnumerationFactor

    Returns:
        Concatenated EnumerationWirings

    Raises:
        ValueError: if the list of EnumerationWirings is empty
    """
    if len(enum_wirings) == 0:
        raise ValueError("No EnumerationWiring in the graph")

    factor_configs_cumsum = np.insert(
        np.array(
            [wiring.factor_configs_edge_states[-1, 0] + 1 for wiring in enum_wirings]
        ).cumsum(),
        0,
        0,
    )[:-1]

    # Note: this correspomds to all the factor_to_msgs_starts for the EnumerationFactors
    num_edge_states_cumsum = np.insert(
        np.array([wiring.edges_num_states.sum() for wiring in enum_wirings]).cumsum(),
        0,
        0,
    )[:-1]

    factor_configs_edge_states = []
    for ww, wiring in enumerate(enum_wirings):
        factor_configs_edge_states.append(
            wiring.factor_configs_edge_states
            + np.array(
                [[factor_configs_cumsum[ww], num_edge_states_cumsum[ww]]], dtype=int
            )
        )

    return EnumerationWiring(
        edges_num_states=np.concatenate(
            [wiring.edges_num_states for wiring in enum_wirings]
        ),
        var_states_for_edges=np.concatenate(
            [wiring.var_states_for_edges for wiring in enum_wirings]
        ),
        factor_configs_edge_states=np.concatenate(factor_configs_edge_states, axis=0),
    )
