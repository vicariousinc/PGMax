"""A module containing helper functions for constructing Factor Graphs."""

from typing import Sequence, Union

import numpy as np

from pgmax.fg.nodes import EnumerationWiring, ORWiring


def concatenate_enumeration_wirings(
    enum_wirings: Sequence[EnumerationWiring], enum_factor_to_msgs_starts: np.ndarray
) -> Union[None, EnumerationWiring]:
    """Concatenate a list of enumeration wirings from individual enumeration factors

    Args:
        enum_wirings: A list of enumeration wirings, one for each individual enumeration factor
        enum_factor_to_msgs_starts: List of offsets indices for the edge_states of each enumeration factor.
            The offsets take into account all the factors in the graph.

    Returns:
        Concatenated enumeration wiring
    """
    assert len(enum_wirings) == len(enum_factor_to_msgs_starts)
    if len(enum_wirings) == 0:
        return None

    factor_configs_cumsum = np.insert(
        np.array(
            [wiring.factor_configs_edge_states[-1, 0] + 1 for wiring in enum_wirings]
        ).cumsum(),
        0,
        0,
    )[:-1]

    factor_configs_edge_states = []
    for ww, wiring in enumerate(enum_wirings):
        factor_configs_edge_states.append(
            wiring.factor_configs_edge_states
            + np.array(
                [[factor_configs_cumsum[ww], enum_factor_to_msgs_starts[ww]]], dtype=int
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
    or_wirings: Sequence[ORWiring], or_factor_to_msgs_starts: np.ndarray
) -> Union[None, ORWiring]:
    """Concatenate a list of OR wirings from individual OR factors

    Args:
        or_wirings: A list of OR wirings, one for each individual OR factor
        or_factor_to_msgs_starts: List of offsets indices for the edge_states of each OR factor.
            The offsets take into account all the factors in the graph.

    Returns:
        Concatenated OR wiring
    """
    assert len(or_wirings) == len(or_factor_to_msgs_starts)
    if len(or_wirings) == 0:
        return None

    parents_edge_states = []
    children_edge_states = []
    for ww, or_wiring in enumerate(or_wirings):
        offsets = np.array([[ww, or_factor_to_msgs_starts[ww]]], dtype=int)
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
