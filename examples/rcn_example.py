import numpy as np

import pgmax.fg.graph as graph
import pgmax.fg.groups as groups

# Prob not good to use rw_common as introduces dependency on realworld
from rw_common.utils.io import resolve_path
from rw_common.utils.pickle import PickleHound


def _compute_valid_configs(perturb_radius, pool_size=5):
    """
    Helper function to compute valid configurations given perturb radius.

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
fg = graph.FactorGraph(variables=generic_variable_group)


# From edges, add factors to the graph one by onw
for edge in static_graph.edges:

    curr_keys = [
        ("pool_vars", edge[0][0]),
        ("pool_vars", edge[0][1]),
    ]

    perturb_radius = edge[-1]
    valid_configs_non_supp = _compute_valid_configs(
        perturb_radius=perturb_radius, pool_size=pool_size
    )

    fg.add_factor(
        curr_keys,
        valid_configs_non_supp,
        np.zeros(valid_configs_non_supp.shape[0], dtype=float),
    )
