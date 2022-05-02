"""A sub-package containing different types of factors."""

import collections
from typing import Callable, OrderedDict, Type

import jax.numpy as jnp

from . import enum, logical
from .enum import EnumerationFactor
from .factor import Factor, Wiring
from .logical import LogicalFactor

FAC_TO_VAR_UPDATES: OrderedDict[
    Type, Callable[..., jnp.ndarray]
] = collections.OrderedDict(
    [
        (enum.EnumerationFactor, enum.pass_enum_fac_to_var_messages),
        (logical.ORFactor, logical.pass_logical_fac_to_var_messages),
        (logical.ANDFactor, logical.pass_logical_fac_to_var_messages),
    ]
)
