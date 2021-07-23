import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import pgmax.fg.graph as graph
import pgmax.fg.nodes as nodes


@dataclass
class FactorSubGraph:
    factors: Sequence[nodes.EnumerationFactor]


class GridFactorGraph2D:
    def __init__(
        self,
        top_left_subgraph: FactorSubGraph,
        col_ext_del_connect_idx_mapping: Dict[nodes.EnumerationFactor, Tuple[int, int]],
        row_ext_del_connect_idx_mapping: Dict[nodes.EnumerationFactor, Tuple[int, int]],
        num_cols: int,
        num_rows: int,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.factor_grid = np.full((num_rows, num_cols), None)
        first_row = self.extend_first_row_from_elem(
            top_left_subgraph, col_ext_del_connect_idx_mapping
        )
        self.factor_grid[0] = np.array(first_row)
        other_rows = self.extend_all_rows_from_first(
            top_left_subgraph, first_row, row_ext_del_connect_idx_mapping
        )
        for row_idx in range(1, self.num_rows):
            self.factor_grid[row_idx] = other_rows[row_idx - 1]

    # TODO: Run some checks on the input via __post_init__ to make sure it's actually valid

    def extend_first_row_from_elem(
        self,
        top_left_subgraph: FactorSubGraph,
        col_ext_del_connect_idx_mapping: Dict[nodes.EnumerationFactor, Tuple[int, int]],
    ) -> Sequence[FactorSubGraph]:
        # factor_to_del_connect_idxs_mapping is a dict from a factor to a pair of ints representing the indices of the variables in factor.variables
        # to be deleted and connected to on copy respectively

        # Build up a dictionary mapping every factor to be deleted/connected to an index in the top_left_subgraph.factors list. This will be useful in the below loops
        facs_list_idx_to_del_connect_idxs_mapping = {}
        for factor, idx_tuple in col_ext_del_connect_idx_mapping.items():
            facs_list_idx_to_del_connect_idxs_mapping[
                top_left_subgraph.factors.index(factor)
            ] = idx_tuple

        new_subgraphs = [copy.deepcopy(top_left_subgraph) for _ in range(self.num_cols)]
        fsg_to_connect_to = top_left_subgraph

        for extension in range(self.num_cols):
            fsg_to_be_connected = new_subgraphs[extension]
            for (
                fac_index,
                connection_tuple,
            ) in facs_list_idx_to_del_connect_idxs_mapping.items():
                fsg_to_be_connected.factors[fac_index].variables[  # type: ignore
                    connection_tuple[0]
                ] = fsg_to_connect_to.factors[fac_index].variables[connection_tuple[1]]
            fsg_to_connect_to = new_subgraphs[extension]

        return new_subgraphs

    def extend_all_rows_from_first(
        self,
        top_left_subgraph: FactorSubGraph,
        first_row_subgraphs: Sequence[FactorSubGraph],
        row_ext_del_connect_idx_mapping: Dict[nodes.EnumerationFactor, Tuple[int, int]],
    ) -> List[Sequence[FactorSubGraph]]:
        subgraph_fac_idxs_to_del_connect_idxs_mapping = {}
        for factor, idx_tuple in row_ext_del_connect_idx_mapping.items():
            subgraph_fac_idxs_to_del_connect_idxs_mapping[
                top_left_subgraph.factors.index(factor)
            ] = idx_tuple

        new_rows = [
            copy.deepcopy(first_row_subgraphs) for _ in range(self.num_rows - 1)
        ]
        row_to_connect_to = first_row_subgraphs

        for extension in range(self.num_rows - 1):
            row_to_be_connected = new_rows[extension]
            for fsg_idx in range(len(row_to_be_connected)):
                for (
                    fac_index,
                    connection_tuple,
                ) in subgraph_fac_idxs_to_del_connect_idxs_mapping.items():
                    row_to_be_connected[fsg_idx].factors[fac_index].variables[  # type: ignore
                        connection_tuple[0]
                    ] = (
                        row_to_connect_to[fsg_idx]
                        .factors[fac_index]
                        .variables[connection_tuple[1]]
                    )
            row_to_connect_to = new_rows[extension]

        return new_rows

    def convert_to_factor_graph(self) -> graph.FactorGraph:
        fg_factors = []
        fg_vars = set()

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if self.factor_grid[row, col] is not None:
                    fg_factors.extend(self.factor_grid[row, col].factors)
                    for factor in self.factor_grid[row, col].factors:
                        for variable in factor.variables:
                            fg_vars.add(variable)

        return graph.FactorGraph(tuple(fg_vars), tuple(fg_factors))
