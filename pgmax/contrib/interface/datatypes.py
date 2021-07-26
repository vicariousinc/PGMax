import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

import pgmax.fg.nodes as nodes


@dataclass
class FactorSubGraph:
    factors: Sequence[nodes.EnumerationFactor]

    def __repr__(self):
        return f"FactorSubGraph with {len(self.factors)} factors"


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
        self.non_grid_factors: List[nodes.EnumerationFactor] = []

    # TODO: Run some checks on the input via __post_init__ to make sure it's actually valid

    def extend_first_row_from_elem(
        self,
        top_left_subgraph: FactorSubGraph,
        col_ext_del_connect_idx_mapping: Dict[nodes.EnumerationFactor, Tuple[int, int]],
    ) -> Sequence[FactorSubGraph]:
        """Creates a row by duplicating a particular FactorSubGraph and connecting the new copies.

        This method performs extension by duplicating top_left_subgraph num_columns - 1 times. For each
        duplicate element, it deletes the variables corresponding to those specified in
        col_ext_del_connect_idx_mapping and then connects factors from the duplicate to other variables
        specified in col_ext_del_connect_idx_mapping. Note that this assumes that the row can be constructed
        by replicating one particular subgraph multiple times, then deleting and linking particular factors and
        variables for each replication.
        """

        # factor_to_del_connect_idxs_mapping is a dict from a factor to a pair of ints representing the indices of the variables in factor.variables
        # to be deleted and connected to on copy respectively

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
                fsg_to_be_connected.factors[fac_index].variables[
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
        """Creates multiple rows by duplicating a particular sequence of FactorSubGraphs and connecting all the new copies.

        This method performs extension by duplicating first_row_subgraphs num_rows - 1 times. For each
        duplicate element, it deletes the variables corresponding to those specified in
        row_ext_del_connect_idx_mapping and then connects factors from the duplicate to other variables
        specified in row_ext_del_connect_idx_mapping. Note that this assumes that rows can be constructed
        by replicating one particular row of subgraph multiple times, then deleting and linking particular factors and
        variables for each replication.
        """

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
                    row_to_be_connected[fsg_idx].factors[fac_index].variables[
                        connection_tuple[0]
                    ] = (
                        row_to_connect_to[fsg_idx]
                        .factors[fac_index]
                        .variables[connection_tuple[1]]
                    )
            row_to_connect_to = new_rows[extension]

        return new_rows

    def apply_and_modify_along_axis(
        self, func, axis, row_idx, elem_start_idx, elem_end_idx
    ) -> None:
        """Applies a function to certain elements of self.factor_grid along an axis and modifies those
        elements to be the function output.
        """
        mod_idxs = (
            slice(
                None,
            ),
        ) * axis + (row_idx,)
        elems_to_mod = self.factor_grid[mod_idxs][elem_start_idx:elem_end_idx]
        modded_elems = func(elems_to_mod, row_idx, elem_start_idx, elem_end_idx)
        self.factor_grid[mod_idxs][elem_start_idx:elem_end_idx] = modded_elems

    def slide_apply_and_modify_row(
        self, func, axis, row_idx, slice_size, start=0, stop=-1, step=1
    ) -> None:
        """Applies a function 'convolutionally' across a specified 1D row of an axis."""
        # TODO: Throw exceptions for faulty input?
        if stop == -1 or stop >= self.factor_grid.shape[axis]:
            stop = self.factor_grid.shape[axis]
        while start + slice_size <= stop:
            self.apply_and_modify_along_axis(
                func, axis, row_idx, start, start + slice_size
            )
            start += step

    def slide_apply_and_modify_axis(
        self,
        func,
        axis,
        row_slice_size,
        row_idxs=None,
        row_elem_start=0,
        row_elem_stop=-1,
        row_elem_step=1,
    ) -> None:
        """Applies a function 'convolutionally' across specified 1D rows of a particular axis."""
        if row_idxs is None:
            row_idxs = range(self.factor_grid.shape[axis])
        for row_idx in row_idxs:
            self.slide_apply_and_modify_row(
                func,
                axis,
                row_idx,
                row_slice_size,
                row_elem_start,
                row_elem_stop,
                row_elem_step,
            )

    def output_vars_and_facs(
        self,
    ) -> Tuple[Tuple[nodes.Variable, ...], Tuple[nodes.EnumerationFactor, ...]]:
        """Outputs the variables and factors that make up this specified factor graph."""
        fg_factors = self.non_grid_factors[:]
        fg_vars: Set[nodes.Variable] = set()
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if self.factor_grid[row, col] is not None:
                    fg_factors.extend(self.factor_grid[row, col].factors)
                    for factor in self.factor_grid[row, col].factors:
                        for variable in factor.variables:
                            fg_vars.add(variable)
        return (tuple(fg_vars), tuple(fg_factors))
