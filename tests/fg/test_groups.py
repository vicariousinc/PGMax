import pytest

from pgmax.fg import groups


def test_vargroup_list_idx():
    v_group = groups.GenericVariableGroup(15, tuple([0, 1, 2]))
    assert v_group[[0, 1, 2]][0].num_states == 15


def test_vargroup_set_keys_to_vars():
    with pytest.raises(NotImplementedError) as nerror:
        groups.VariableGroup()
    assert "Please subclass" in str(nerror.value)


def test_composite_vargroup_valueerror():
    v_group1 = groups.GenericVariableGroup(15, tuple([0, 1, 2]))
    v_group2 = groups.GenericVariableGroup(15, tuple([0, 1, 2]))
    comp_var_group = groups.CompositeVariableGroup(tuple([v_group1, v_group2]))
    with pytest.raises(ValueError) as verror:
        comp_var_group[tuple([0])]
    assert "The key needs" in str(verror.value)
