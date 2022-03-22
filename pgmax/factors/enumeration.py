from __future__ import annotations

"""Defines an enumeration factor"""

import functools
from dataclasses import dataclass
from typing import Mapping, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from pgmax import utils
from pgmax.bp import bp_utils
from pgmax.fg import nodes


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class EnumerationWiring(nodes.Wiring):
    """Wiring for EnumerationFactors.

    Args:
        factor_configs_edge_states: Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global enumeration factor_config and global edge_state indices
            factor_configs_edge_states[ii, 0] contains the global EnumerationFactor config index,
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index.
            Both indices only take into account the EnumerationFactors of the FactorGraph
    """

    factor_configs_edge_states: Union[np.ndarray, jnp.ndarray]

    @property
    def inference_arguments(self) -> Mapping[str, Union[np.ndarray, int]]:
        """
        Returns:
            A dictionnary of elements used to run belief propagation.
        """
        if self.factor_configs_edge_states.shape[0] == 0:
            num_val_configs = 0
        else:
            num_val_configs = int(self.factor_configs_edge_states[-1, 0]) + 1

        return {
            "factor_configs_edge_states": self.factor_configs_edge_states,
            "num_val_configs": num_val_configs,
        }


@dataclass(frozen=True, eq=False)
class EnumerationFactor(nodes.Factor):
    """An enumeration factor

    Args:
        configs: Array of shape (num_val_configs, num_variables)
            An array containing an explicit enumeration of all valid configurations
        log_potentials: Array of shape (num_val_configs,)
            An array containing the log of the potential value for each valid configuration

    Raises:
        ValueError: If:
            (1) The dtype of the configs array is not int
            (2) The dtype of the potential array is not float
            (3) Configs does not have the correct shape
            (4) The potential array does not have the correct shape
            (5) The configs array contains invalid values
    """

    configs: np.ndarray
    log_potentials: np.ndarray

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
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index.
            Both indices only take into account the EnumerationFactors of the FactorGraph
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

    @staticmethod
    def concatenate_wirings(wirings: Sequence[EnumerationWiring]) -> EnumerationWiring:
        """Concatenate a list of EnumerationWirings

        Args:
            wirings: A list of EnumerationWirings

        Returns:
            Concatenated EnumerationWiring
        """
        if len(wirings) == 0:
            return EnumerationWiring(
                edges_num_states=np.empty((0,), dtype=int),
                var_states_for_edges=np.empty((0,), dtype=int),
                factor_configs_edge_states=np.empty((0, 2), dtype=int),
            )

        factor_configs_cumsum = np.insert(
            np.array(
                [wiring.factor_configs_edge_states[-1, 0] + 1 for wiring in wirings]
            ).cumsum(),
            0,
            0,
        )[:-1]

        # Note: this correspomds to all the factor_to_msgs_starts of the EnumerationFactors
        num_edge_states_cumsum = np.insert(
            np.array([wiring.edges_num_states.sum() for wiring in wirings]).cumsum(),
            0,
            0,
        )[:-1]

        factor_configs_edge_states = []
        for ww, wiring in enumerate(wirings):
            factor_configs_edge_states.append(
                wiring.factor_configs_edge_states
                + np.array(
                    [[factor_configs_cumsum[ww], num_edge_states_cumsum[ww]]], dtype=int
                )
            )

        return EnumerationWiring(
            edges_num_states=np.concatenate(
                [wiring.edges_num_states for wiring in wirings]
            ),
            var_states_for_edges=np.concatenate(
                [wiring.var_states_for_edges for wiring in wirings]
            ),
            factor_configs_edge_states=np.concatenate(
                factor_configs_edge_states, axis=0
            ),
        )


@functools.partial(jax.jit, static_argnames=("num_val_configs", "temperature"))
def pass_enum_fac_to_var_messages(
    vtof_msgs: jnp.ndarray,
    factor_configs_edge_states: jnp.ndarray,
    log_potentials: jnp.ndarray,
    num_val_configs: int,
    temperature: float,
) -> jnp.ndarray:

    """Passes messages from EnumerationFactors to Variables.

    The update is performed in two steps. First, a "summary" array is generated that has an entry for every valid
    configuration for every EnumerationFactor. The elements of this array are simply the sums of messages across
    each valid config. Then, the info from factor_configs_edge_states is used to apply the scattering operation and
    generate a flat set of output messages.

    Args:
        vtof_msgs: Array of shape (num_edge_state,). This holds all the flattened variable
            to all the EnumerationFactors messages
        factor_configs_edge_states: Array of shape (num_factor_configs, 2)
            factor_configs_edge_states[ii] contains a pair of global factor_config and edge_state indices
            factor_configs_edge_states[ii, 0] contains the global EnumerationFactor config index,
            factor_configs_edge_states[ii, 1] contains the corresponding global edge_state index.
            Both indices only take into account the EnumerationFactors of the FactorGraph
        log_potentials: Array of shape (num_val_configs, ). An entry at index i is the log potential
            function value for the configuration with global EnumerationFactor config index i.
        num_val_configs: the total number of valid configurations for all the EnumerationFactors
            in the factor graph.
        temperature: Temperature for loopy belief propagation.
            1.0 corresponds to sum-product, 0.0 corresponds to max-product.

    Returns:
        Array of shape (num_edge_state,). This holds all the flattened EnumerationFactors to variable messages.
    """
    fac_config_summary_sum = (
        jnp.zeros(shape=(num_val_configs,))
        .at[factor_configs_edge_states[..., 0]]
        .add(vtof_msgs[factor_configs_edge_states[..., 1]])
    ) + log_potentials
    max_factor_config_summary_for_edge_states = (
        jnp.full(shape=(vtof_msgs.shape[0],), fill_value=bp_utils.NEG_INF)
        .at[factor_configs_edge_states[..., 1]]
        .max(fac_config_summary_sum[factor_configs_edge_states[..., 0]])
    )
    ftov_msgs = max_factor_config_summary_for_edge_states - vtof_msgs
    if temperature != 0.0:
        ftov_msgs = ftov_msgs + (
            temperature
            * jnp.log(
                jnp.full(
                    shape=(vtof_msgs.shape[0],), fill_value=jnp.exp(bp_utils.NEG_INF)
                )
                .at[factor_configs_edge_states[..., 1]]
                .add(
                    jnp.exp(
                        (
                            fac_config_summary_sum[factor_configs_edge_states[..., 0]]
                            - max_factor_config_summary_for_edge_states[
                                factor_configs_edge_states[..., 1]
                            ]
                        )
                        / temperature
                    )
                )
            )
        )
    return ftov_msgs
