from pgmax.fg import graph, groups


def test_onevar_graph():
    v_group = groups.GenericVariableGroup(15, tuple([0]))
    fg = graph.FactorGraph(v_group)
    assert fg._composite_variable_group[0, 0].num_states == 15
