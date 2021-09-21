import numpy as np
from typing import Dict, List, Tuple

import pgmax.fg.graph as graph
import pgmax.fg.groups as groups

# Prob not good to use rw_common as introduces dependency on realworld
from rw_common.utils.io import resolve_path
from rw_common.utils.pickle import PickleHound


def _compute_valid_configs(
    perturb_radius: int, pool_size: int = 5
) -> Dict[int, List[List[Tuple[int, int], Tuple[int, int]], ...]]:
    """
    Helper function to compute valid configurations given pool_size and perturb_radius. (WIP)

    Args
        perturb_radius: Perturb radius of edge factor.
        pool_size: Pool size of the model.

    Returns
        Dict with perturb radius as key and list of pairwise valid configurations in tuples.
        E.g {1: [[(0,0), (0,0)], [(0,0), (1,0)]...]}
    """
    valid_config = {}
    list_of_configs = []

    # For every element, check the neighbors given by perturb_radius
    for row in range(pool_size):
        for col in range(pool_size):
            min_row = max(0, row - perturb_radius)
            max_row = min(pool_size - 1, row + perturb_radius)
            min_col = max(0, col - perturb_radius)
            max_col = min(pool_size - 1, col + perturb_radius)

            for i in range(min_row, max_row):
                for j in range(min_col, max_col):
                    euclidean_dist = ((row - i) ** 2 + (col - j)) ** 0.5

                    if euclidean_dist <= perturb_radius:
                        # TODO: Handle/Check for duplicates
                        list_of_configs.append([(row, col), (i, j)])

    valid_config[perturb_radius] = list_of_configs

    return valid_config


model_path = resolve_path(path_spec="pkg_data://rcn/config/base/model.pkl")

model = PickleHound.load(model_path)

# Load graph
static_graph = model.static_graphs_per_layer[0][0]

frcs = np.asarray([vertex[1] for vertex in static_graph.vertices])
num_of_variables = frcs.shape[0]
pool_shape = model.layer_models[-1].master_pools.get_shape(0)
pool_size = pool_shape[0]
variable_size = pool_size * pool_size


generic_variable_group = groups.GenericVariableGroup(
    variable_size=variable_size, key_tuple=tuple(range(num_of_variables))
)
fg = graph.FactorGraph(variables=dict(pool_vars=generic_variable_group))

valid_configs_dict = {}

for edge in static_graph.edges:
    perturb_radius = edge[-1]

    if perturb_radius in valid_configs_dict:
        continue

    valid_config = _compute_valid_configs(
        perturb_radius=perturb_radius, pool_size=pool_size
    )

    valid_configs_dict.update(valid_config)


# From edges, add factors to the graph one by one.
for edge in static_graph.edges:

    curr_keys = [edge[0][0], edge[0][1]]

    fg.add_factor(
        curr_keys,
        valid_configs_dict[edge[-1]],
        np.zeros(valid_configs_dict[edge[-1]], dtype=float),
    )
