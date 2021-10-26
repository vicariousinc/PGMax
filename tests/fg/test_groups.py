import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pgmax.fg import groups, nodes


def test_composite_variable_group():
    variable_dict1 = groups.VariableDict(15, tuple([0, 1, 2]))
    variable_dict2 = groups.VariableDict(15, tuple([0, 1, 2]))
    composite_variable_sequence = groups.CompositeVariableGroup(
        [variable_dict1, variable_dict2]
    )
    composite_variable_dict = groups.CompositeVariableGroup(
        {(0, 1): variable_dict1, (2, 3): variable_dict2}
    )
    with pytest.raises(ValueError, match="The key needs to have at least 2 elements"):
        composite_variable_sequence[(0,)]

    assert composite_variable_sequence[0, 1] == variable_dict1[1]
    assert (
        composite_variable_sequence[[(0, 1), (1, 2)]]
        == composite_variable_dict[[((0, 1), 1), ((2, 3), 2)]]
    )
    assert composite_variable_dict[(0, 1), 0] == variable_dict1[0]
    assert composite_variable_dict[[((0, 1), 1), ((2, 3), 2)]] == [
        variable_dict1[1],
        variable_dict2[2],
    ]
    assert jnp.all(
        composite_variable_sequence.flatten(
            [{key: np.zeros(15) for key in range(3)} for _ in range(2)]
        )
        == composite_variable_dict.flatten(
            {
                (0, 1): {key: np.zeros(15) for key in range(3)},
                (2, 3): {key: np.zeros(15) for key in range(3)},
            }
        )
    )
    assert jnp.all(
        jnp.array(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_multimap(
                    lambda x, y: jnp.all(x == y),
                    composite_variable_sequence.unflatten(jnp.zeros(15 * 3 * 2)),
                    [{key: jnp.zeros(15) for key in range(3)} for _ in range(2)],
                )
            )
        )
    )
    assert jnp.all(
        jnp.array(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_multimap(
                    lambda x, y: jnp.all(x == y),
                    composite_variable_dict.unflatten(jnp.zeros(3 * 2)),
                    {
                        (0, 1): {key: np.zeros(1) for key in range(3)},
                        (2, 3): {key: np.zeros(1) for key in range(3)},
                    },
                )
            )
        )
    )
    with pytest.raises(
        ValueError, match="Can only unflatten 1D array. Got a 2D array."
    ):
        composite_variable_dict.unflatten(jnp.zeros((10, 20)))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "flat_data should be either of shape (num_variables(=6),), or (num_variable_states(=90),)"
        ),
    ):
        composite_variable_dict.unflatten(jnp.zeros((100)))


def test_variable_dict():
    variable_dict = groups.VariableDict(15, tuple([0, 1, 2]))
    with pytest.raises(
        ValueError, match="data is referring to a non-existent variable 3"
    ):
        variable_dict.flatten({3: np.zeros(10)})

    with pytest.raises(
        ValueError,
        match=re.escape("Variable 2 expects a data array of shape (15,). Got (10,)"),
    ):
        variable_dict.flatten({2: np.zeros(10)})

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
                    {key: np.zeros(1) for key in range(3)},
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
    variable_group = groups.NDVariableArray(2, (1,))
    assert isinstance(variable_group[0], nodes.Variable)
    variable_group = groups.NDVariableArray(3, (2, 2))
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
    variable_group = groups.NDVariableArray(3, (2, 2))
    with pytest.raises(
        ValueError,
        match=re.escape("Expected log potentials shape: (1,) or (2, 1). Got (3, 2)"),
    ):
        enumeration_factor_group = groups.EnumerationFactorGroup(
            variable_group=variable_group,
            connected_var_keys=[[(0, 0), (0, 1), (1, 1)], [(0, 1), (1, 0), (1, 1)]],
            factor_configs=np.zeros((1, 3), dtype=int),
            log_potentials=np.zeros((3, 2)),
        )

    enumeration_factor_group = groups.EnumerationFactorGroup(
        variable_group=variable_group,
        connected_var_keys=[[(0, 0), (0, 1), (1, 1)], [(0, 1), (1, 0), (1, 1)]],
        factor_configs=np.zeros((1, 3), dtype=int),
    )
    key = [(0, 0), (1, 1)]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The queried factor {frozenset(key)} is not present in the factor group."
        ),
    ):
        enumeration_factor_group[key]

    assert (
        enumeration_factor_group[[(0, 1), (1, 0), (1, 1)]]
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
    variable_group = groups.NDVariableArray(3, (2, 2))
    with pytest.raises(
        ValueError, match=re.escape("log_potential_matrix should be either a 2D array")
    ):
        groups.PairwiseFactorGroup(
            variable_group, [[(0, 0), (1, 1)]], np.zeros((1,), dtype=float)
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected log_potential_matrix for 1 factors. Got log_potential_matrix for 2 factors."
        ),
    ):
        groups.PairwiseFactorGroup(
            variable_group, [[(0, 0), (1, 1)]], np.zeros((2, 3, 3), dtype=float)
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "All pairwise factors should connect to exactly 2 variables. Got a factor connecting to 3 variables"
        ),
    ):
        groups.PairwiseFactorGroup(
            variable_group, [[(0, 0), (1, 1), (0, 1)]], np.zeros((3, 3), dtype=float)
        )

    with pytest.raises(
        ValueError,
        match=re.escape("The specified pairwise factor [(0, 0), (1, 1)]"),
    ):
        groups.PairwiseFactorGroup(
            variable_group, [[(0, 0), (1, 1)]], np.zeros((4, 4), dtype=float)
        )

    pairwise_factor_group = groups.PairwiseFactorGroup(
        variable_group,
        [[(0, 0), (1, 1)], [(1, 0), (0, 1)]],
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
