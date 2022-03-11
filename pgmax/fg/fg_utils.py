"""A module containing helper functions for constructing Factor Graphs."""

from typing import Sequence, Union

import numpy as np

from pgmax.fg.nodes import EnumerationWiring, LogicalWiring, ORWiring


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


def concatenate_logical_wirings(
    logical_wirings: Sequence[Union[LogicalWiring, ORWiring]],
    logical_factor_to_msgs_starts: np.ndarray,
) -> Union[None, LogicalWiring, ORWiring]:
    """Concatenate a list of logical wirings from individual logical factors

    Args:
        logical_wirings: A list of logical wirings, one for each individual logical factor.
            All these wirings must have the same type, which is a subclass of nodes.LogicalWiring
        logical_factor_to_msgs_starts: List of offsets indices for the edge_states of each logical factor.
            The offsets take into account all the enumeration and logical factors in the graph.

    Returns:
        Concatenated logical wiring.
    """
    assert len(logical_wirings) == len(logical_factor_to_msgs_starts)
    if len(logical_wirings) == 0:
        return None

    logical_type = type(logical_wirings[0])
    for wiring_idx in range(1, len(logical_wirings)):
        if not isinstance(logical_wirings[wiring_idx], logical_type):
            raise ValueError(f"Wiring {wiring_idx} is not of type {logical_type}")

    parents_edge_states = []
    children_edge_states = []
    for ww, or_wiring in enumerate(logical_wirings):
        offsets = np.array([[ww, logical_factor_to_msgs_starts[ww]]], dtype=int)
        parents_edge_states.append(or_wiring.parents_edge_states + offsets)
        children_edge_states.append(or_wiring.children_edge_states + offsets[:, 1])

    return logical_type(
        edges_num_states=np.concatenate(
            [wiring.edges_num_states for wiring in logical_wirings]
        ),
        var_states_for_edges=np.concatenate(
            [wiring.var_states_for_edges for wiring in logical_wirings]
        ),
        parents_edge_states=np.concatenate(parents_edge_states, axis=0),
        children_edge_states=np.concatenate(children_edge_states, axis=0),
    )
