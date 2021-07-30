"""A module containing helper functions useful while constructing Factor Graphs."""

import functools
from typing import Callable


def cached_property(func: Callable) -> property:
    """Customized cached property decorator

    Args:
        func: Member function to be decorated

    Returns:
        Decorated cached property
    """
    return property(functools.lru_cache(None)(func))
