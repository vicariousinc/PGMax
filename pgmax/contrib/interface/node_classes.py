import numpy as np


class FactorNode(object):
    def __init__(self, name):
        self.name = name
        self.neighbors = []
        self.neighbor_to_index_mapping = {}
        self.neighbor_config_list = np.array([[]])

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

    def set_valid_configs(self, config_arr):
        """
        Sets allowed configurations for all neighboring variable nodes

        Args:
            config_list (np.array([np.array([int])])): Each sub-list indicates a valid configuration
                of the neighbors' states.
        """
        if not isinstance(config_arr, np.ndarray):
            raise ValueError("The array of valid configurations must be an np.array")
        num_neighbors = self.num_neighbors()
        if config_arr.shape[1] != num_neighbors:
            raise ValueError(
                f"The valid configuration array is not the right size for the number of neighbors of FactorNode {self.name}"
            )
        self.neighbor_config_list = config_arr

    def num_neighbors(self):
        return len(self.neighbors)


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


class FactorGraph(object):
    def __init__(self, name, factor_nodes, variable_nodes):
        self.name = name
        self.check_factor_nodes(factor_nodes)
        self.factor_nodes = factor_nodes
        self.check_variable_nodes(variable_nodes)
        self.variable_nodes = variable_nodes

    def count_num_edges(self):
        num_edges = 0
        for fac_node in self.factor_nodes:
            num_edges += fac_node.num_neighbors()
        return num_edges

    def find_max_msg_size(self):
        return max((x.num_states for x in self.variable_nodes), default=0)

    def find_max_num_valid_configs(self):
        return max(
            (fac_node.neighbor_config_list.shape[0] for fac_node in self.factor_nodes),
            default=0,
        )

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
