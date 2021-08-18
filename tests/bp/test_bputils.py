import jax
import jax.numpy as jnp
import numpy as np

import pgmax.bp.bp_utils as bp_utils


def test_with_zero():
    assert jnp.allclose(
        bp_utils.segment_max_opt(jnp.zeros(1, dtype=int), jnp.ones(1, dtype=int), 1),
        jnp.zeros(1),
    )


def test_with_ones():
    assert jnp.allclose(
        bp_utils.segment_max_opt(
            jnp.ones(10, dtype=int), jnp.ones(1, dtype=int) * 10, 10
        ),
        jnp.ones(1),
    )


def test_common_case():
    assert jnp.allclose(
        bp_utils.segment_max_opt(
            jax.device_put(np.array([1.7, 2.3, 4.3, 3.8, 9.2, 111.3])),
            jax.device_put(np.array([2, 3, 1])),
            3,
        ),
        jax.device_put(np.array([2.3, 9.2, 111.3])),
    )
