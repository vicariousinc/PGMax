import collections
from typing import Callable, OrderedDict, Type

import jax.numpy as jnp

from pgmax.factors import enumeration, logical
from pgmax.fg import groups, nodes

REGISTERED_FACTOR_TYPES = [enumeration.EnumerationFactor, logical.ORFactor]
FACTOR_GROUP_FACTORY: OrderedDict[Type, Type] = collections.OrderedDict(
    [
        (enumeration.EnumerationFactor, groups.EnumerationFactorGroup),
        (logical.ORFactor, groups.ORFactorGroup),
    ]
)
CONCATENATE_WIRING: OrderedDict[
    Type, Callable[..., nodes.Wiring]
] = collections.OrderedDict(
    [
        (enumeration.EnumerationFactor, enumeration.concatenate_enumeration_wirings),
        (logical.ORFactor, logical.concatenate_logical_wirings),
    ]
)
FAC_TO_VAR_UPDATES: OrderedDict[
    Type, Callable[..., jnp.ndarray]
] = collections.OrderedDict(
    [
        (enumeration.EnumerationFactor, enumeration.pass_enum_fac_to_var_messages),
        (logical.ORFactor, logical.pass_OR_fac_to_var_messages),
    ]
)
