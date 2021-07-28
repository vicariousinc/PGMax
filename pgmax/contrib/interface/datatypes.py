import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

import pgmax.fg.nodes as nodes


@dataclass
class VariableGroup:
    variable_size: int  # NOTE: all variables in a VariableGroup are assumed to have the same size

    def __post_init__(self) -> None:
        self._key_to_var_mapping: Dict[Any, nodes.Variable] = {}

    def query_var(self, key) -> nodes.Variable:
        ret_var = self._key_to_var_mapping.get(key)
        if ret_var is None:
            raise ValueError(
                f"The key {key} is not present in this VariableGroup; please ensure "
                + "it's been added to the VariableGroup before trying to query it."
            )
        return ret_var

    def add_var(self, key) -> None:
        if self._key_to_var_mapping.get(key) is not None:
            raise ValueError(
                f"The key {key} is already present in this VariableGroup; please ensure "
                + "it hasn't previously been added before adding it."
            )
        self._key_to_var_mapping[key] = nodes.Variable(self.variable_size)

    def get_all_vars(self) -> List[nodes.Variable]:
        all_vars = []
        for var in self._key_to_var_mapping.values():
            all_vars.append(var)
        return all_vars


# NOTE: A thought: maybe we should make the factors within a factorgroup optionally accessible via some form of
# indexing similar to VariableGroups? This could potentially make it easier to build hierarchical graphs.
@dataclass
class FactorGroup:
    factor_configs: np.ndarray

    def __post_init__(self) -> None:
        self._factor_list: List[nodes.EnumerationFactor] = []

    def add_factor(self, var_neighbor_keys_and_groups: List[Tuple]) -> None:
        # var_neighbor_keys_and_groups is a list of size-2 tuples.
        # In each tuple, the 0th elem is a key, and the 1st elem is a VariableGroup
        # that contains the key
        neighbor_vars = []
        for key_and_group in var_neighbor_keys_and_groups:
            key, group = key_and_group
            neighbor_vars.append(group.query_var(key))
        self._factor_list.append(
            nodes.EnumerationFactor(tuple(neighbor_vars), self.factor_configs)
        )

    def get_all_factors(self) -> List[nodes.EnumerationFactor]:
        return self._factor_list


@dataclass
class GridVariableGroup(VariableGroup):
    # this is a tuple that should have the same length as each key_tuple, but
    # contain the lengths of each dimension as each of the elements. E.g.
    # to instantiate a 3D grid with shape (3,3,2), key_tuple_dim_sizes must be
    # (3,3,2)
    key_tuple_dim_sizes: Tuple[int]

    def __post_init__(self) -> None:
        self._key_to_var_mapping: Dict[Tuple[int], nodes.Variable] = {}
        for key in itertools.product(
            *[list(range(k)) for k in self.key_tuple_dim_sizes]
        ):
            self.add_var(key)
