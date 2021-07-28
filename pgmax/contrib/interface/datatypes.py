import itertools
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

import pgmax.fg.nodes as nodes


@dataclass
class VariableGroup:
    variable_size: int  # NOTE: all variables in a VariableGroup are assumed to have the same size

    def __post_init__(self) -> None:
        self._key_to_var_mapping: Mapping[Any, nodes.Variable] = MappingProxyType(
            self._generate_vars()
        )

    def __getitem__(self, key) -> nodes.Variable:
        ret_var = self._key_to_var_mapping.get(key)
        if ret_var is None:
            raise ValueError(
                f"The key {key} is not present in this VariableGroup; please ensure "
                + "it's been added to the VariableGroup before trying to query it."
            )
        return ret_var

    def _generate_vars(self) -> Dict[Any, nodes.Variable]:
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    def get_all_vars(self) -> Tuple[nodes.Variable, ...]:
        return tuple(self._key_to_var_mapping.values())


@dataclass
class FactorGroup:
    factor_configs: np.ndarray

    def __post_init__(self) -> None:
        factors_list = []
        nested_keys_list = self._get_connected_var_keys_for_factors()
        if len(nested_keys_list) == 0:
            raise ValueError(
                "The list returned by _get_connected_var_keys_for_factors is empty"
            )
        for keys_list in nested_keys_list:
            vars_list = []
            for key_vargroup_tuple in keys_list:
                key, vargroup = key_vargroup_tuple
                vars_list.append(vargroup[key])
            if len(vars_list) == 0:
                raise ValueError(
                    "There was an empty sub-list in the output of _get_connected_var_keys_for_factors"
                )
            factors_list.append(
                nodes.EnumerationFactor(tuple(vars_list), self.factor_configs)
            )
        self._factors: Tuple[nodes.EnumerationFactor, ...] = tuple(factors_list)

    def _get_connected_var_keys_for_factors(
        self,
    ) -> List[List[Tuple[Any, VariableGroup]]]:
        raise NotImplementedError(
            "Please subclass the FactorGroup class and override this method"
        )

    def get_all_factors(self) -> Tuple[nodes.EnumerationFactor, ...]:
        return self._factors


@dataclass
class GridVariableGroup(VariableGroup):
    # this is a tuple that should have the same length as each key_tuple, but
    # contain the lengths of each dimension as each of the elements. E.g.
    # to instantiate a 3D grid with shape (3,3,2), key_tuple_dim_sizes must be
    # (3,3,2)
    key_tuple_dim_sizes: Tuple[int, ...]
    additional_keys: List[Tuple[int, ...]]

    def _generate_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        key_to_var_mapping: Dict[Tuple[int, ...], nodes.Variable] = {}
        for key in itertools.product(
            *[list(range(k)) for k in self.key_tuple_dim_sizes]
        ):
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        for key in self.additional_keys:
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        return key_to_var_mapping
