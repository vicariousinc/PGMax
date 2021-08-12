from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from pgmax.fg import groups, nodes


def create_enum_factor(
    var_group: groups.VariableGroup,
    var_keys: Sequence[Tuple[Any, ...]],
    configs: np.ndarray,
    factor_configs_log_potentials: Optional[np.ndarray] = None,
):
    """Helper function to create EnumerationFactors.

    Given a VariableGroup, a set of keys into this VariableGroup representing the
    EnumerationFactor's neighbors, an array of valid configurations of these
    Variables (represented by the keys), and optionally a potential function,
    instantiates and returns an EnumerationFactor.

    Args:
        var_group: A VariableGroup containing all the Variables that var_keys will
            index.
        var_keys: A sequence of tuples, where each tuple represents a key into
            var_group. The sequence is taken to contain keys for all neighbors the
            factor may have.
        configs: Array of shape (num_val_configs, num_variables)
            An array containing an explicit enumeration of all valid configurations
        factor_configs_log_potentials: Optional array of shape (num_val_configs,).
            If this is not specified, then the log potentials are assumed to be
            uniform 0. If specified, then interpreted as an array containing
            the log of the potential value for every possible configuration

    Returns:
        EnumerationFactor constructed from these inputs

    """

    if factor_configs_log_potentials is None:
        factor_configs_log_potentials = np.zeros((configs.shape[0],))

    var_neighbors: List[nodes.Variable] = []
    for key in var_keys:
        var_neighbors.append(var_group[key])

    var_neighbors = tuple(var_neighbors)

    return nodes.EnumerationFactor(
        var_neighbors, configs, factor_configs_log_potentials
    )
