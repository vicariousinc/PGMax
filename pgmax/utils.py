"""A module containing helper functions useful while constructing Factor Graphs."""

import dataclasses
import functools
from typing import Any, Callable

import jax


def cached_property(func: Callable) -> property:
    """Customized cached property decorator

    Args:
        func: Member function to be decorated

    Returns:
        Decorated cached property
    """
    return property(functools.lru_cache(None)(func))


def register_pytree_node_dataclass(cls: Any) -> Any:
    """Decorator to register a dataclass as a pytree

    Args:
        cls: A dataclass to be registered as a pytree

    Returns:
        The registered dataclass
    """

    def _flatten(obj):
        jax.tree_flatten(dataclasses.asdict(obj))

    def _unflatten(d, children):
        cls(**d.unflatten(children))

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls
