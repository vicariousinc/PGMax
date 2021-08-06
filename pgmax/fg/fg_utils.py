"""A module containing helper functions for constructing Factor Graphs."""

from typing import Sequence

import numpy as np

from pgmax.fg.nodes import EnumerationWiring


def concatenate_enumeration_wirings(
    wirings: Sequence[EnumerationWiring],
) -> EnumerationWiring:
    """Concatenate a list of enumeration wirings from individual enumeration factors

    Args:
        wirings: A list of enumeration wirings, one for each individual enumeration factor

    Returns:
        Concatenated enumeration wiring
    """
    num_factor_configs_cumsum = np.insert(
        np.array(
            [np.max(wiring.factor_configs_edge_states[:, 0]) + 1 for wiring in wirings]
        ).cumsum(),
        0,
        0,
    )[:-1]
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
                [[num_factor_configs_cumsum[ww], num_edge_states_cumsum[ww]]], dtype=int
            )
        )

    return EnumerationWiring(
        edges_num_states=np.concatenate(
            [wiring.edges_num_states for wiring in wirings]
        ),
        var_states_for_edges=np.concatenate(
            [wiring.var_states_for_edges for wiring in wirings]
        ),
        factor_configs_edge_states=np.concatenate(factor_configs_edge_states, axis=0),
    )
