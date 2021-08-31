import numpy as np
import pytest

from pgmax.fg import groups


def test_vargroup_list_idx():
    v_group = groups.GenericVariableGroup(15, tuple([0, 1, 2]))
    assert v_group[[0, 1, 2]][0].num_states == 15


def test_composite_vargroup_valueerror():
    v_group1 = groups.GenericVariableGroup(15, tuple([0, 1, 2]))
    v_group2 = groups.GenericVariableGroup(15, tuple([0, 1, 2]))
    comp_var_group = groups.CompositeVariableGroup(tuple([v_group1, v_group2]))
    with pytest.raises(ValueError) as verror:
        comp_var_group[tuple([0])]
    assert "The key needs" in str(verror.value)


def test_composite_vargroup_evidence():
    v_group1 = groups.GenericVariableGroup(3, tuple([0, 1, 2]))
    v_group2 = groups.GenericVariableGroup(3, tuple([0, 1, 2]))
    comp_var_group = groups.CompositeVariableGroup(tuple([v_group1, v_group2]))
    vars_to_evidence = comp_var_group.get_vars_to_evidence(
        [{0: np.zeros(3)}, {0: np.zeros(3)}]
    )
    assert set(vars_to_evidence.keys()) == set(
        [comp_var_group[0, 0], comp_var_group[1, 0]]
    )
    for arr in vars_to_evidence.values():
        assert (arr == np.zeros(3, dtype=float)).all()


def test_ndvararray_evidence_error():
    v_group = groups.NDVariableArray(3, (2, 2))
    with pytest.raises(ValueError) as verror:
        v_group.get_vars_to_evidence(np.zeros((1, 1)))
    assert "Input evidence" in str(verror.value)


def test_pairwisefacgroup_errors():
    v_group = groups.NDVariableArray(3, (2, 2))
    with pytest.raises(ValueError) as verror:
        groups.PairwiseFactorGroup(
            v_group, [[(0, 0), (1, 1), (0, 1)]], np.zeros((1,), dtype=float)
        )
    assert "All pairwise factors" in str(verror.value)

    with pytest.raises(ValueError) as verror:
        groups.PairwiseFactorGroup(
            v_group, [[(0, 0), (1, 1)]], np.zeros((1,), dtype=float)
        )
    assert "self.log_potential_matrix must" in str(verror.value)
    factor_group = groups.PairwiseFactorGroup(
        v_group, {0: [(0, 0), (1, 1)]}, np.zeros((3, 3), dtype=float)
    )
    with pytest.raises(ValueError) as verror:
        factor_group[1]
    assert "The queried factor" in str(verror.value)


def test_generic_evidence_errors():
    v_group = groups.GenericVariableGroup(3, tuple([0]))
    with pytest.raises(ValueError) as verror:
        v_group.get_vars_to_evidence({1: np.zeros((1, 1))})
    assert "The evidence is referring" in str(verror.value)
    with pytest.raises(ValueError) as verror:
        v_group.get_vars_to_evidence({0: np.zeros((1, 1))})
    assert "expects an evidence array" in str(verror.value)
