from typing import Sequence

import numpy as np

from pgmax import utils
from pgmax.fg.nodes import EnumerationWiring


def concatenate_enumeration_wirings(
    wirings: Sequence[EnumerationWiring],
) -> EnumerationWiring:
    num_factor_configs_cumsum = np.insert(
        np.array(
            [np.max(wiring.factor_configs_edge_states[:, 0]) + 1 for wiring in wirings]
        ).cumsum(),
        0,
        0,
    )[:-1]
    num_edge_states_cumsum = np.insert(
        np.array(
            [wiring.factor_configs_edge_states.shape[0] + 1 for wiring in wirings]
        ).cumsum(),
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
        edges_num_states=utils.concatenate_arrays(
            [wiring.edges_num_states for wiring in wirings]
        ),
        var_states_for_edges=utils.concatenate_arrays(
            [wiring.var_states_for_edges for wiring in wirings]
        ),
        factor_configs_edge_states=utils.concatenate_arrays(factor_configs_edge_states),
    )
