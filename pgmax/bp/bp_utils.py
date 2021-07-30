"""A module containing helper functions used for belief propagation."""

import jax
import jax.numpy as jnp

NEG_INF = (
    -100000.0
)  # A large negative value to use as -inf for numerical stability reasons


@jax.partial(jax.jit, static_argnames="max_segment_length")
def segment_max_opt(
    data: jnp.ndarray, segments_lengths: jnp.ndarray, max_segment_length: int
) -> jnp.ndarray:
    """
    Computes the max of every segment of data, where segments_lengths
    specifies the segments

    Args:
        data: Array of shape (a_len,) where a_len is an arbitrary integer.
        segments_lengths: Array of shape (num_segments,), where 0 < num_segments <= a_len.
            segments_lengths.sum() should yield a_len, and all elements must be > 0.
        max_segment_length: The maximum value in segments_lengths.

    Returns:
        An array of shape (num_segments,) that contains the maximum value from data of
            every segment sepcified by segments_lengths

    """

    @jax.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
    def get_max(data, start_index, segment_length):
        return jnp.max(
            jnp.where(
                jnp.arange(max_segment_length) < segment_length,
                jax.lax.dynamic_slice(
                    data, jnp.array([start_index]), [max_segment_length]
                ),
                NEG_INF,
            )
        )

    start_indices = jnp.concatenate(
        [
            jnp.full(shape=(1,), fill_value=int(NEG_INF), dtype=int),
            jnp.cumsum(segments_lengths),
        ]
    )[:-1]
    expanded_data = jnp.concatenate([data, jnp.zeros(max_segment_length)])
    return get_max(expanded_data, start_indices, segments_lengths)
