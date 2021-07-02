import numpy as np


class FactorType(object):
    def __init__(self, name: str, neighbor_configs: np.array) -> None:
        # NOTE: neighbor_configs must be of shape (num_valid_configs x num_neighbors)
        self.name = name
        self.neighbor_configs_arr = neighbor_configs.astype(int)

    def num_neighbors(self) -> int:
        return self.neighbor_configs_arr.shape[1]  # type: ignore


class VariableNode(object):
    def __init__(self, name, num_states):
        self.name = name
        self.neighbors = []
        self.num_states = num_states

    def __lt__(self, other):
        return self.name < other.name

    def add_neighbor(self, neighbor):
        if not isinstance(neighbor, FactorNode):
            raise ValueError(
                "Error! A Variable Node can only be connected to a Factor Node"
            )
        self.neighbors.append(neighbor)

    def num_neighbors(self):
        return len(self.neighbors)


class FactorNode(object):
    def __init__(self, name, factor_type):
        self.name = name
        self.neighbors = []
        self.neighbor_to_index_mapping = {}
        self.factor_type = factor_type

    def __lt__(self, other):
        return self.name < other.name

    def set_neighbors(self, neighbor_list):
        for index, neighbor in enumerate(neighbor_list):
            if not isinstance(neighbor, VariableNode):
                raise ValueError(
                    "Error! A Factor Node can only be connected to a Variable Node"
                )
            self.neighbor_to_index_mapping[neighbor] = index
        self.neighbors = neighbor_list

    def num_neighbors(self):
        return len(self.neighbors)


class FactorGraph(object):
    def __init__(self, name, factor_nodes, variable_nodes, factor_types):
        self.name = name
        self.check_factor_nodes(factor_nodes)
        self.factor_nodes = factor_nodes
        self.check_variable_nodes(variable_nodes)
        self.variable_nodes = variable_nodes
        self.check_factor_types(factor_nodes, factor_types)
        self.factor_types = factor_types
        self.factor_type_to_index_dict = self.construct_factor_type_to_index_dict()

    def construct_factor_type_to_index_dict(self):
        fac_type_to_index_dict = {}
        for i, fac_type in enumerate(self.factor_types):
            fac_type_to_index_dict[fac_type] = i
        return fac_type_to_index_dict

    def count_num_edges(self):
        return sum(fac_node.num_neighbors() for fac_node in self.factor_nodes)

    def count_total_num_valid_configs(self):
        return sum(
            fac_node.factor_type.neighbor_configs_arr.shape[0]
            for fac_node in self.factor_nodes
        )

    def find_max_num_factor_neighbors(self):
        return max(
            (len(fac_node.neighbors) for fac_node in self.factor_nodes), default=0
        )

    def find_max_msg_size(self):
        return max((x.num_states for x in self.variable_nodes), default=0)

    def find_max_num_valid_configs(self):
        return max(
            (fac_type.neighbor_configs_arr.shape[0] for fac_type in self.factor_types),
            default=0,
        )

    def find_max_valid_config_size(self):
        return max(
            (fac_type.neighbor_configs_arr.shape[1] for fac_type in self.factor_types),
            default=0,
        )

    def count_num_factor_nodes(self):
        return len(self.factor_nodes)

    def count_num_variable_nodes(self):
        return len(self.variable_nodes)

    def count_num_factor_types(self):
        return len(self.factor_types)

    def check_factor_nodes(self, factor_nodes):
        for fac_node in factor_nodes:
            if not isinstance(fac_node, FactorNode):
                raise ValueError(
                    f"Error! The 'factor_nodes' argument of a Factor Graph contained a {type(fac_node)}"
                )
            elif fac_node.num_neighbors() == 0:
                raise ValueError(
                    "Error! A FactorNode that's part of a FactorGraph must have at least 1 neighbor"
                )
            elif not (fac_node.num_neighbors() == fac_node.factor_type.num_neighbors()):
                raise ValueError(
                    f"Error! FactorNode {fac_node.name} has {fac_node.num_neighbors()} neighbors, but it's factor_type has {fac_node.factor_type.num_neighbors()} neighbors!"
                )

    def check_variable_nodes(self, variable_nodes):
        for var_node in variable_nodes:
            if not isinstance(var_node, VariableNode):
                raise ValueError(
                    f"Error! The 'variable_nodes' argument of a Factor Graph contained a {type(var_node)}"
                )
            elif var_node.num_neighbors() == 0:
                raise ValueError(
                    "Error! A VariableNode that's part of a FactorGraph must have at least 1 neighbor"
                )

    def check_factor_types(self, factor_nodes, factor_types):
        for fac_node in factor_nodes:
            if fac_node.factor_type not in factor_types:
                raise ValueError(
                    f"Error! The FactorNode {fac_node.name} has factor_type {fac_node.factor_type.name}, which was not passed to the FactorGraph!"
                )
