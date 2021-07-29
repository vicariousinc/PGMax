import itertools
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np

import pgmax.fg.nodes as nodes


@dataclass
class VariableGroup:
    variable_size: int  # NOTE: all variables in a VariableGroup are assumed to have the same size

    def __post_init__(self) -> None:
        self._key_to_var: Mapping[Any, nodes.Variable] = MappingProxyType(
            self._generate_vars()
        )

    def __getitem__(self, key) -> Union[nodes.Variable, List[nodes.Variable]]:
        if type(key) is slice:
            start, stop, step = key.indices(len(self._key_to_var.keys()))
            vars_list: List[nodes.Variable] = []
            for k in range(start, stop, step):
                var = self._key_to_var.get(k)
                if var is None:
                    raise ValueError(
                        f"The key {k} is not present in the VariableGroup {type(self)}; please ensure "
                        "it's been added to the VariableGroup before trying to query it."
                    )
                vars_list.append(var)
            return vars_list
        else:
            var = self._key_to_var.get(key)
            if var is None:
                raise ValueError(
                    f"The key {key} is not present in in the VariableGroup {type(self)}; please ensure "
                    "it's been added to the VariableGroup before trying to query it."
                )
            return var

    def _generate_vars(self) -> Dict[Any, nodes.Variable]:
        raise NotImplementedError(
            "Please subclass the VariableGroup class and override this method"
        )

    def get_all_vars(self) -> Tuple[nodes.Variable, ...]:
        return tuple(self._key_to_var.values())


@dataclass
class FactorGroup:
    factor_configs: np.ndarray

    def __post_init__(self) -> None:
        factors = []
        connected_var_keys_for_factors = self._get_connected_var_keys_for_factors()
        if len(connected_var_keys_for_factors) == 0:
            raise ValueError(
                "The list returned by _get_connected_var_keys_for_factors is empty"
            )
        for keys_list in connected_var_keys_for_factors:
            vars_list: List[nodes.Variable] = []
            for key_vargroup_tuple in keys_list:
                key, vargroup = key_vargroup_tuple
                var_query = vargroup[key]
                if type(var_query) is List[nodes.Variable]:
                    vars_list.extend(var_query)
                else:
                    vars_list.append(var_query)  # type: ignore
            if len(vars_list) == 0:
                raise ValueError(
                    "There was an empty sub-list in the output of _get_connected_var_keys_for_factors"
                )
            factors.append(
                nodes.EnumerationFactor(tuple(vars_list), self.factor_configs)
            )
        self.factors: Tuple[nodes.EnumerationFactor, ...] = tuple(factors)

    def _get_connected_var_keys_for_factors(
        self,
    ) -> List[List[Tuple[Any, VariableGroup]]]:
        raise NotImplementedError(
            "Please subclass the FactorGroup class and override this method"
        )


@dataclass
class NDVariableArray(VariableGroup):
    # this is a tuple that should have the same length as each key_tuple, but
    # contain the lengths of each dimension as each of the elements. E.g.
    # to instantiate a 3D grid with shape (3,3,2), shape must be
    # (3,3,2)
    shape: Tuple[int, ...]

    def _generate_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        key_to_var_mapping: Dict[Tuple[int, ...], nodes.Variable] = {}
        for key in itertools.product(*[list(range(k)) for k in self.shape]):
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        return key_to_var_mapping


@dataclass
class KeyTupleVariableGroup(VariableGroup):
    key_tuple: Tuple[Tuple[Any, ...], ...]

    def _generate_vars(self) -> Dict[Tuple[int, ...], nodes.Variable]:
        key_to_var_mapping: Dict[Tuple[Any, ...], nodes.Variable] = {}
        for key in self.key_tuple:
            key_to_var_mapping[key] = nodes.Variable(self.variable_size)
        return key_to_var_mapping
