from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np


# Generic padding function, taken from https://codereview.stackexchange.com/questions/222623/pad-a-ragged-multidimensional-array-to-rectangular-shape
def get_dimensions(array, level=0):
    yield level, len(array)
    try:
        for row in array:
            yield from get_dimensions(row, level + 1)
    except TypeError:  # not an iterable
        pass


def get_max_shape(array):
    dimensions = defaultdict(int)
    for level, length in get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


def iterate_nested_array(array, index=()):
    try:
        for idx, row in enumerate(array):
            yield from iterate_nested_array(row, (*index, idx))
    except TypeError:  # final level
        yield (*index, slice(len(array))), array


def pad(array, fill_value):
    dimensions = get_max_shape(array)
    result = np.full(dimensions, fill_value)
    for index, value in iterate_nested_array(array):
        result[index] = value
    return result


@jax.partial(jax.jit, static_argnames="max_segment_length")
def segment_sum_opt(data, segments_lengths, max_segment_length):
    @jax.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
    def get_sum(data, start_index, segment_length):
        return jnp.sum(
            jnp.where(
                jnp.arange(max_segment_length) < segment_length,
                jax.lax.dynamic_slice(
                    data, jnp.array([start_index]), [max_segment_length]
                ),
                0,
            )
        )

    start_indices = jnp.concatenate(
        [jnp.zeros(1, dtype=int), jnp.cumsum(segments_lengths)]
    )[:-1]
    expanded_data = jnp.concatenate([data, jnp.zeros(max_segment_length)])
    return get_sum(expanded_data, start_indices, segments_lengths)
