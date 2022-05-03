"""A module containing helper functions."""

import functools
from typing import Callable

NEG_INF = (
    -100000.0
)  # A large negative value to use as -inf for numerical stability reasons


def cached_property(func: Callable) -> property:
    """Customized cached property decorator

    Args:
        func: Member function to be decorated

    Returns:
        Decorated cached property
    """
    return property(functools.lru_cache(None)(func))
