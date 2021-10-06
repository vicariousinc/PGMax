import numpy as np

import pgmax.fg.graph as graph
import pgmax.fg.groups as groups

# Prob not good to use rw_common as introduces dependency on realworld
# from rw_common.utils.io import resolve_path
# from rw_common.utils.pickle import PickleHound


def _generate_mapping(pool_size: int):
    """
    Generate a mapping from a 2D grid into a range of integer values
    """
    index = 0
    mapping = {}
    for row in range(pool_size):
        for col in range(pool_size):
            mapping[f"{row},{col}"] = index
            index += 1

    return mapping


def _compute_valid_configs(
    perturb_radius: int,
    grid_2d_to_config_map,
    pool_size: int = 5,
) -> np.ndarray:
    """
    Helper function to compute valid configurations given pool_size and perturb_radius.

    Note: The configurations map to 2D grid configurations in the pool shape.

    Args
        perturb_radius: Perturb radius of edge factor.
        pool_size: Pool size of the model.

    Returns
        Dict with perturb radius as key and list of pairwise valid configurations in tuples.
        E.g [[0,0], [0,1], [1,0], [1,1]]...]
    """
    valid_configs = []

    curr_configs = set()

    # For every element, check the neighbors given by perturb_radius
    for row in range(pool_size):
        for col in range(pool_size):
            min_row = max(0, row - perturb_radius)
            max_row = min(pool_size, row + perturb_radius)
            min_col = max(0, col - perturb_radius)
            max_col = min(pool_size, col + perturb_radius)

            for i in range(min_row, max_row):
                for j in range(min_col, max_col):
                    euclidean_dist = ((row - i) ** 2 + (col - j) ** 2) ** 0.5

                    if euclidean_dist <= perturb_radius:
                        # Lookup config based on 2D grid row, col mapping
                        config_1 = grid_2d_to_config_map[f"{row},{col}"]
                        config_2 = grid_2d_to_config_map[f"{i},{j}"]

                        # TODO: Handle/Check for duplicates
                        config_pair = f"{config_1},{config_2}"
                        if config_pair not in curr_configs:
                            valid_configs.append([config_1, config_2])
                            curr_configs.add(config_pair)

            # Handle this special case explicitly for now
            if perturb_radius == 0:
                config_1 = grid_2d_to_config_map[f"{row},{col}"]
                config_2 = config_1

                config_pair = f"{config_1},{config_2}"

                if config_pair not in curr_configs:
                    valid_configs.append([config_1, config_2])
                    curr_configs.add(config_pair)

    valid_configs = np.asarray(valid_configs)

    return valid_configs


"""

Needs realworld dependency

# model_path = resolve_path(path_spec="pkg_data://rcn/config/base/model.pkl")

# model = PickleHound.load(model_path)

# Load graph
# static_graph = model.static_graphs_per_layer[0][0]

# frcs = np.asarray([vertex[1] for vertex in static_graph.vertices])
# num_of_variables = frcs.shape[0]
# pool_shape = model.layer_models[-1].master_pools.get_shape(0)
# pool_size = pool_shape[0]
# variable_size = pool_size * pool_size
"""

# Load frcs
with open("pool_centers.npy", "rb") as f:
    template_frcs = np.load(f)

frcs = template_frcs

num_of_variables = frcs.shape[0]
pool_shape = (5, 5)
pool_size = pool_shape[0]
variable_size = pool_size * pool_size

# Load edge factors
with open("edge_factors.npy", "rb") as f:
    edge_factors = np.load(f)


generic_variable_group = groups.GenericVariableGroup(
    variable_size=variable_size, key_tuple=tuple(range(num_of_variables))
)
fg = graph.FactorGraph(variables=generic_variable_group)

valid_configs_dict = {}

grid_2d_to_config_map = _generate_mapping(pool_size)

# for edge in static_graph.edges:
for edge in edge_factors:
    perturb_radius = edge[-1]

    if perturb_radius in valid_configs_dict:
        continue

    valid_config = _compute_valid_configs(
        perturb_radius=perturb_radius,
        pool_size=pool_size,
        grid_2d_to_config_map=grid_2d_to_config_map,
    )

    valid_configs_dict[perturb_radius] = valid_config

    # From edges, add factors to the graph one by one.
    # for edge in edge_factors:

    curr_keys = [edge[0], edge[1]]

    fg.add_factor(
        curr_keys,
        valid_configs_dict[edge[-1]],
        np.zeros(valid_configs_dict[edge[-1]].shape[0], dtype=float),
    )


# Setup Evidence

# Load the extracted bu_evidence
with open("bu_evidence_extracted.npy", "rb") as f:
    bu_evidence_extracted = np.load(f)

init_msgs = fg.get_init_msgs()
evidence = np.reshape(bu_evidence_extracted, (bu_evidence_extracted.shape[0], -1))


for i in range(evidence.shape[0]):
    init_msgs.evidence[i] = evidence[i]

msgs = fg.run_bp(num_iters=3000, damping_factor=0.5, init_msgs=init_msgs)

from IPython import embed

embed()
