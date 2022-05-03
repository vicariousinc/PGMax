import re

import jax.numpy as jnp
import numpy as np
import pytest

from pgmax import fgroup, vgroup


def test_single_factor():
    with pytest.raises(ValueError, match="Cannot create a FactorGroup with no Factor."):
        fgroup.ORFactorGroup(variables_for_factors=[])

    A = vgroup.NDVariableArray(num_states=2, shape=(10,))
    B = vgroup.NDVariableArray(num_states=2, shape=(10,))

    variables0 = (A[0], B[0])
    variables1 = (A[1], B[1])
    ORFactor0 = fgroup.ORFactorGroup(variables_for_factors=[variables0])
    with pytest.raises(
        ValueError, match="SingleFactorGroup should only contain one factor. Got 2"
    ):
        fgroup.SingleFactorGroup(
            variables_for_factors=[variables0, variables1],
            factor=ORFactor0,
        )
    ORFactor1 = fgroup.ORFactorGroup(variables_for_factors=[variables1])
    ORFactor0 < ORFactor1


def test_enumeration_factor_group():
    vg = vgroup.NDVariableArray(shape=(2, 2), num_states=3)
    with pytest.raises(
        ValueError,
        match=re.escape("Expected log potentials shape: (1,) or (2, 1). Got (3, 2)"),
    ):
        enumeration_factor_group = fgroup.EnumerationFactorGroup(
            variables_for_factors=[
                [vg[0, 0], vg[0, 1], vg[1, 1]],
                [vg[0, 1], vg[1, 0], vg[1, 1]],
            ],
            factor_configs=np.zeros((1, 3), dtype=int),
            log_potentials=np.zeros((3, 2)),
        )

    with pytest.raises(ValueError, match=re.escape("Potentials should be floats")):
        enumeration_factor_group = fgroup.EnumerationFactorGroup(
            variables_for_factors=[
                [vg[0, 0], vg[0, 1], vg[1, 1]],
                [vg[0, 1], vg[1, 0], vg[1, 1]],
            ],
            factor_configs=np.zeros((1, 3), dtype=int),
            log_potentials=np.zeros((2, 1), dtype=int),
        )

    enumeration_factor_group = fgroup.EnumerationFactorGroup(
        variables_for_factors=[
            [vg[0, 0], vg[0, 1], vg[1, 1]],
            [vg[0, 1], vg[1, 0], vg[1, 1]],
        ],
        factor_configs=np.zeros((1, 3), dtype=int),
    )
    name = [vg[0, 0], vg[1, 1]]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The queried factor connected to the set of variables {frozenset(name)} is not present in the factor group."
        ),
    ):
        enumeration_factor_group[name]

    assert (
        enumeration_factor_group[[vg[0, 1], vg[1, 0], vg[1, 1]]]
        == enumeration_factor_group.factors[1]
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "data should be of shape (2, 1) or (2, 9) or (1,). Got (4, 5)."
        ),
    ):
        enumeration_factor_group.flatten(np.zeros((4, 5)))

    assert jnp.all(enumeration_factor_group.flatten(np.ones(1)) == jnp.ones(2))
    assert jnp.all(enumeration_factor_group.flatten(np.ones((2, 9))) == jnp.ones(18))
    with pytest.raises(
        ValueError, match=re.escape("Can only unflatten 1D array. Got a 3D array.")
    ):
        enumeration_factor_group.unflatten(jnp.ones((1, 2, 3)))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "flat_data should be compatible with shape (2, 1) or (2, 9). Got (30,)"
        ),
    ):
        enumeration_factor_group.unflatten(jnp.zeros(30))

    assert jnp.all(
        enumeration_factor_group.unflatten(jnp.arange(2)) == jnp.array([[0], [1]])
    )
    assert jnp.all(enumeration_factor_group.unflatten(jnp.ones(18)) == jnp.ones((2, 9)))


def test_pairwise_factor_group():
    vg = vgroup.NDVariableArray(shape=(2, 2), num_states=3)

    with pytest.raises(
        ValueError, match=re.escape("log_potential_matrix should be either a 2D array")
    ):
        fgroup.PairwiseFactorGroup([[vg[0, 0], vg[1, 1]]], np.zeros((1,), dtype=float))

    with pytest.raises(
        ValueError, match=re.escape("Potential matrix should be floats")
    ):
        fgroup.PairwiseFactorGroup([[vg[0, 0], vg[1, 1]]], np.zeros((3, 3), dtype=int))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected log_potential_matrix for 1 factors. Got log_potential_matrix for 2 factors."
        ),
    ):
        fgroup.PairwiseFactorGroup(
            [[vg[0, 0], vg[1, 1]]], np.zeros((2, 3, 3), dtype=float)
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "All pairwise factors should connect to exactly 2 variables. Got a factor connecting to 3 variables"
        ),
    ):
        fgroup.PairwiseFactorGroup(
            [[vg[0, 0], vg[1, 1], vg[0, 1]]], np.zeros((3, 3), dtype=float)
        )

    name = [vg[0, 0], vg[1, 1]]
    with pytest.raises(
        ValueError,
        match=re.escape(f"The specified pairwise factor {name}"),
    ):
        fgroup.PairwiseFactorGroup([name], np.zeros((4, 4), dtype=float))

    pairwise_factor_group = fgroup.PairwiseFactorGroup(
        [[vg[0, 0], vg[1, 1]], [vg[1, 0], vg[0, 1]]],
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "data should be of shape (2, 3, 3) or (2, 6) or (3, 3). Got (4, 4)."
        ),
    ):
        pairwise_factor_group.flatten(np.zeros((4, 4)))

    assert jnp.all(
        pairwise_factor_group.flatten(np.zeros((3, 3))) == jnp.zeros(2 * 3 * 3)
    )
    assert jnp.all(pairwise_factor_group.flatten(np.zeros((2, 6))) == jnp.zeros(12))
    with pytest.raises(ValueError, match="Can only unflatten 1D array. Got a 2D array"):
        pairwise_factor_group.unflatten(np.zeros((10, 20)))

    assert jnp.all(
        pairwise_factor_group.unflatten(np.zeros(2 * 3 * 3)) == jnp.zeros((2, 3, 3))
    )
    assert jnp.all(
        pairwise_factor_group.unflatten(np.zeros(2 * 6)) == jnp.zeros((2, 6))
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "flat_data should be compatible with shape (2, 3, 3) or (2, 6). Got (10,)."
        ),
    ):
        pairwise_factor_group.unflatten(np.zeros(10))
