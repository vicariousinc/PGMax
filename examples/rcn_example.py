import numpy as np

import pgmax.fg.graph as graph
import pgmax.fg.groups as groups

# Prob not good to use rw_common as introduces dependency on realworld
from rw_common.utils.io import resolve_path
from rw_common.utils.pickle import PickleHound


def _compute_valid_configs(perturb_radius: int, pool_size: int = 5):
    """
    Helper function to compute valid configurations given perturb radius. (WIP)

    Args
        perturb_radius: Perturb radius of edge factor.
        pool_size: Pool size of the model.

    Returns
        Dict
    """
    pass


model_path = resolve_path(path_spec="pkg_data://rcn/config/base/model.pkl")

model = PickleHound.load(model_path)

# Load graph
static_graph = model.static_graphs_per_layer[0][0]

frcs = np.asarray([vertex[1] for vertex in static_graph.vertices])
num_of_variables = frcs.shape[0]
pool_size = 5
variable_size = pool_size * pool_size


generic_variable_group = groups.GenericVariableGroup(
    variable_size=variable_size, key_tuple=tuple(range(num_of_variables))
)
fg = graph.FactorGraph(variables=dict(pool_vars=generic_variable_group))

valid_configs_dict = None

for edge in static_graph.edges:
    perturb_radius = edge[-1]

    valid_config = _compute_valid_configs(
        perturb_radius=perturb_radius, pool_size=pool_size
    )

    valid_configs_dict.update(valid_config)

valid_configs_size = sum(
    [len(list_of_valid_nodes) for list_of_valid_nodes in valid_configs_dict.keys()]
)

# From edges, add factors to the graph one by one.
for edge in static_graph.edges:

    curr_keys = [edge[0][0], edge[0][1]]

    fg.add_factor(
        curr_keys,
        valid_configs_dict,
        np.zeros(valid_configs_dict.shape[0], dtype=float),
    )
