import dataclasses
from typing import Any, Sequence

import jax
import numpy as np


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


def concatenate_arrays(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Convenience function to concatenate a list of arrays along the 0th axis

    Args:
        arrays: A list of numpy arrays to be concatenated

    Returns:
        The concatenated array
    """
    lengths = np.array([array.shape[0] for array in arrays], dtype=int)
    lengths_cumsum = np.insert(lengths.cumsum(), 0, 0)
    starts, total_length = lengths_cumsum[:-1], lengths_cumsum[-1]
    concatenated_array = np.zeros(
        (total_length,) + arrays[0].shape[1:], dtype=arrays[0].dtype
    )
    for start, length, array in zip(starts, lengths, arrays):
        concatenated_array[start : start + length] = array

    return concatenated_array


class cached_property(object):
    """Descriptor (non-data) for building an attribute on-demand on first use."""

    def __init__(self, factory):
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self._factory(instance)
        # Cache the value; hide ourselves.
        setattr(instance, self._attr_name, attr)
        return attr
