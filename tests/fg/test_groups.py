import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgmax.groups import enumeration
from pgmax.groups import variables as vgroup


def test_variable_dict():
    variable_dict = vgroup.VariableDict(variable_names=tuple([0, 1, 2]), num_states=15)
    with pytest.raises(
        ValueError, match="data is referring to a non-existent variable 3"
    ):
        variable_dict.flatten({3: np.zeros(10)})

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Variable (2, 15) expects a data array of shape (15,) or (1,). Got (10,)"
        ),
    ):
        variable_dict.flatten({(2, 15): np.zeros(10)})

    with pytest.raises(
        ValueError, match="Can only unflatten 1D array. Got a 2D array."
    ):
        variable_dict.unflatten(jnp.zeros((10, 20)))

    assert jnp.all(
        jnp.array(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_multimap(
                    lambda x, y: jnp.all(x == y),
                    variable_dict.unflatten(jnp.zeros(3)),
                    {(name, 15): np.zeros(1) for name in range(3)},
                )
            )
        )
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "flat_data should be either of shape (num_variables(=3),), or (num_variable_states(=45),)"
        ),
    ):
        variable_dict.unflatten(jnp.zeros((100)))


def test_nd_variable_array():
    num_states = np.full((2, 3), fill_value=2)
    with pytest.raises(
        ValueError, match=re.escape("Expected num_states shape (2, 2). Got (2, 3).")
    ):
        vgroup.NDVariableArray(shape=(2, 2), num_states=num_states)

    num_states = np.full((2, 3), fill_value=2, dtype=np.float32)
    with pytest.raises(
        ValueError, match=re.escape("num_states entries should be of type np.int")
    ):
        vgroup.NDVariableArray(shape=(2, 2), num_states=num_states)

    variable_group = vgroup.NDVariableArray(shape=(5, 5), num_states=2)
    assert len(variable_group[:3, :3]) == 9

    variable_group = vgroup.NDVariableArray(shape=(2, 2), num_states=3)
    with pytest.raises(
        ValueError,
        match=re.escape("data should be of shape (2, 2) or (2, 2, 3). Got (3, 3)."),
    ):
        variable_group.flatten(np.zeros((3, 3)))

    assert jnp.all(
        variable_group.flatten(np.array([[1, 2], [3, 4]])) == jnp.array([1, 2, 3, 4])
    )
    with pytest.raises(
        ValueError, match="Can only unflatten 1D array. Got a 2D array."
    ):
        variable_group.unflatten(np.zeros((10, 20)))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "flat_data should be compatible with shape (2, 2) or (2, 2, 3). Got (10,)."
        ),
    ):
        variable_group.unflatten(np.zeros((10,)))

    assert jnp.all(variable_group.unflatten(np.zeros(4)) == jnp.zeros((2, 2)))
    assert jnp.all(variable_group.unflatten(np.zeros(12)) == jnp.zeros((2, 2, 3)))


def test_enumeration_factor_group():
    vg = vgroup.NDVariableArray(shape=(2, 2), num_states=3)
    with pytest.raises(
        ValueError,
        match=re.escape("Expected log potentials shape: (1,) or (2, 1). Got (3, 2)"),
    ):
        enumeration_factor_group = enumeration.EnumerationFactorGroup(
            variables_for_factors=[
                [vg[0, 0], vg[0, 1], vg[1, 1]],
                [vg[0, 1], vg[1, 0], vg[1, 1]],
            ],
            factor_configs=np.zeros((1, 3), dtype=int),
            log_potentials=np.zeros((3, 2)),
        )

    with pytest.raises(ValueError, match=re.escape("Potentials should be floats")):
        enumeration_factor_group = enumeration.EnumerationFactorGroup(
            variables_for_factors=[
                [vg[0, 0], vg[0, 1], vg[1, 1]],
                [vg[0, 1], vg[1, 0], vg[1, 1]],
            ],
            factor_configs=np.zeros((1, 3), dtype=int),
            log_potentials=np.zeros((2, 1), dtype=int),
        )

    enumeration_factor_group = enumeration.EnumerationFactorGroup(
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
        enumeration.PairwiseFactorGroup(
            [[vg[0, 0], vg[1, 1]]], np.zeros((1,), dtype=float)
        )

    with pytest.raises(
        ValueError, match=re.escape("Potential matrix should be floats")
    ):
        enumeration.PairwiseFactorGroup(
            [[vg[0, 0], vg[1, 1]]], np.zeros((3, 3), dtype=int)
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected log_potential_matrix for 1 factors. Got log_potential_matrix for 2 factors."
        ),
    ):
        enumeration.PairwiseFactorGroup(
            [[vg[0, 0], vg[1, 1]]], np.zeros((2, 3, 3), dtype=float)
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "All pairwise factors should connect to exactly 2 variables. Got a factor connecting to 3 variables"
        ),
    ):
        enumeration.PairwiseFactorGroup(
            [[vg[0, 0], vg[1, 1], vg[0, 1]]], np.zeros((3, 3), dtype=float)
        )

    name = [vg[0, 0], vg[1, 1]]
    with pytest.raises(
        ValueError,
        match=re.escape(f"The specified pairwise factor {name}"),
    ):
        enumeration.PairwiseFactorGroup([name], np.zeros((4, 4), dtype=float))

    pairwise_factor_group = enumeration.PairwiseFactorGroup(
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
