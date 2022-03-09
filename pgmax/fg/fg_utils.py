"""A module containing helper functions for constructing Factor Graphs."""

from typing import Sequence

import numpy as np

from pgmax.fg.nodes import EnumerationWiring, ORWiring


def concatenate_enumeration_wirings(
    enum_wirings: Sequence[EnumerationWiring], start_edge_states: Sequence[int]
) -> None or EnumerationWiring:
    """Concatenate a list of enumeration wirings from individual enumeration factors

    Args:
        wirings: A list of enumeration wirings, one for each individual enumeration factor
        start_edge_states: A list of offsets indices for the edge_states indices of each
            factor. The offsets take into account all the different types of factors in the graph

    Returns:
        Concatenated enumeration wiring
    """
    assert start_edge_states.shape[0] == len(enum_wirings)
    if len(enum_wirings) == 0:
        return None

    num_factor_configs_cumsum = np.insert(
        np.array(
            [
                np.max(wiring.factor_configs_edge_states[:, 0]) + 1
                for wiring in enum_wirings
            ]
        ).cumsum(),
        0,
        0,
    )[:-1]

    factor_configs_edge_states = []
    for ww, wiring in enumerate(enum_wirings):
        factor_configs_edge_states.append(
            wiring.factor_configs_edge_states
            + np.array(
                [[num_factor_configs_cumsum[ww], start_edge_states[ww]]], dtype=int
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


def concatenate_or_wirings(
    or_wirings: Sequence[ORWiring], start_edge_states: Sequence[int]
) -> ORWiring:
    """Concatenate a list of OR wirings from individual OR factors

    Args:
        wirings: A list of OR wirings, one for each individual OR factor
        start_edge_states: A list of offsets indices for the edge_states indices of each
            factor. The offsets take into account all the different types of factors in the graph

    Returns:
        Concatenated OR wiring
    """
    assert start_edge_states.shape[0] == len(or_wirings)
    if len(or_wirings) == 0:
        return None

    parents_edge_states = []
    children_edge_states = []
    for ww, or_wiring in enumerate(or_wirings):
        offsets = np.array([[ww, start_edge_states[ww]]], dtype=int)
        parents_edge_states.append(or_wiring.parents_edge_states + offsets)
        children_edge_states.append(or_wiring.children_edge_states + offsets[:, 1])

    return ORWiring(
        edges_num_states=np.concatenate(
            [wiring.edges_num_states for wiring in or_wirings]
        ),
        var_states_for_edges=np.concatenate(
            [wiring.var_states_for_edges for wiring in or_wirings]
        ),
        parents_edge_states=np.concatenate(parents_edge_states, axis=0),
        children_edge_states=np.concatenate(children_edge_states, axis=0),
    )
