import pytest

from pgmax.fg import graph, groups


def test_onevar_graph():
    v_group = groups.GenericVariableGroup(15, tuple([0]))
    fg = graph.FactorGraph(v_group)
    assert fg._composite_variable_group[0, 0].num_states == 15


def test_set_evidence_Error():
    v_group = groups.GenericVariableGroup(15, tuple([0]))
    fg = graph.FactorGraph(v_group)
    fg.evidence_default_mode = "test"
    with pytest.raises(NotImplementedError) as nerror:
        fg.evidence

    assert "evidence_default_mode" in str(nerror.value)
