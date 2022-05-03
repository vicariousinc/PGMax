import re

import numpy as np
import pytest

from pgmax import factor, fgraph, fgroup, vgroup


def test_wiring_with_PairwiseFactorGroup():
    """
    Test the equivalence of the wiring compiled at the PairwiseFactorGroup level
    vs at the individual EnumerationFactor level (which is called from SingleFactorGroup)
    """
    A = vgroup.NDVariableArray(num_states=2, shape=(10,))
    B = vgroup.NDVariableArray(num_states=2, shape=(10,))

    # First test that compile_wiring enforces the correct factor_edges_num_states shape
    fg = fgraph.FactorGraph(variable_groups=[A, B])
    factor_group = fgroup.PairwiseFactorGroup(
        variables_for_factors=[[A[idx], B[idx]] for idx in range(10)]
    )
    fg.add_factors(factor_group)

    factor_group = fg.factor_groups[factor.EnumerationFactor][0]
    object.__setattr__(
        factor_group, "factor_configs", factor_group.factor_configs[:, :1]
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Expected factor_edges_num_states shape is (10,). Got (20,)."),
    ):
        factor_group.compile_wiring(fg._vars_to_starts)

    # FactorGraph with a single PairwiseFactorGroup
    fg1 = fgraph.FactorGraph(variable_groups=[A, B])
    factor_group = fgroup.PairwiseFactorGroup(
        variables_for_factors=[[A[idx], B[idx]] for idx in range(10)]
    )
    fg1.add_factors(factor_group)
    assert len(fg1.factor_groups[factor.EnumerationFactor]) == 1

    # FactorGraph with multiple PairwiseFactorGroup
    fg2 = fgraph.FactorGraph(variable_groups=[A, B])
    for idx in range(10):
        factor_group = fgroup.PairwiseFactorGroup(
            variables_for_factors=[[A[idx], B[idx]]]
        )
        fg2.add_factors(factor_group)
    assert len(fg2.factor_groups[factor.EnumerationFactor]) == 10

    # FactorGraph with multiple SingleFactorGroup
    fg3 = fgraph.FactorGraph(variable_groups=[A, B])
    factors = []
    for idx in range(10):
        enum_factor = factor.EnumerationFactor(
            variables=[A[idx], B[idx]],
            factor_configs=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            log_potentials=np.zeros((4,)),
        )
        factors.append(enum_factor)
    fg3.add_factors(factors)
    assert len(fg3.factor_groups[factor.EnumerationFactor]) == 10

    assert len(fg1.factors) == len(fg2.factors) == len(fg3.factors)

    # Compile wiring via factor_group.compile_wiring
    wiring1 = fg1.wiring[factor.EnumerationFactor]
    wiring2 = fg2.wiring[factor.EnumerationFactor]

    # Compile wiring via factor.compile_wiring
    wiring3 = fg3.wiring[factor.EnumerationFactor]

    assert np.all(wiring1.edges_num_states == wiring2.edges_num_states)
    assert np.all(wiring1.var_states_for_edges == wiring2.var_states_for_edges)
    assert np.all(
        wiring1.factor_configs_edge_states == wiring2.factor_configs_edge_states
    )

    assert np.all(wiring1.edges_num_states == wiring3.edges_num_states)
    assert np.all(wiring1.var_states_for_edges == wiring3.var_states_for_edges)
    assert np.all(
        wiring1.factor_configs_edge_states == wiring3.factor_configs_edge_states
    )


def test_wiring_with_ORFactorGroup():
    """
    Test the equivalence of the wiring compiled at the ORFactorGroup level
    vs at the individual ORFactor level (which is called from SingleFactorGroup)
    """
    A = vgroup.NDVariableArray(num_states=2, shape=(10,))
    B = vgroup.NDVariableArray(num_states=2, shape=(10,))
    C = vgroup.NDVariableArray(num_states=2, shape=(10,))

    # FactorGraph with a single ORFactorGroup
    fg1 = fgraph.FactorGraph(variable_groups=[A, B, C])
    factor_group = fgroup.ORFactorGroup(
        variables_for_factors=[[A[idx], B[idx], C[idx]] for idx in range(10)],
    )
    fg1.add_factors(factor_group)
    assert len(fg1.factor_groups[factor.ORFactor]) == 1

    # FactorGraph with multiple ORFactorGroup
    fg2 = fgraph.FactorGraph(variable_groups=[A, B, C])
    for idx in range(10):
        factor_group = fgroup.ORFactorGroup(
            variables_for_factors=[[A[idx], B[idx], C[idx]]],
        )
        fg2.add_factors(factor_group)
    assert len(fg2.factor_groups[factor.ORFactor]) == 10

    # FactorGraph with multiple SingleFactorGroup
    fg3 = fgraph.FactorGraph(variable_groups=[A, B, C])
    for idx in range(10):
        or_factor = factor.ORFactor(
            variables=[A[idx], B[idx], C[idx]],
        )
        fg3.add_factors(or_factor)
    assert len(fg3.factor_groups[factor.ORFactor]) == 10

    assert len(fg1.factors) == len(fg2.factors) == len(fg3.factors)

    # Compile wiring via factor_group.compile_wiring
    wiring1 = fg1.wiring[factor.ORFactor]
    wiring2 = fg2.wiring[factor.ORFactor]

    # Compile wiring via factor.compile_wiring
    wiring3 = fg3.wiring[factor.ORFactor]

    assert np.all(wiring1.edges_num_states == wiring2.edges_num_states)
    assert np.all(wiring1.var_states_for_edges == wiring2.var_states_for_edges)
    assert np.all(wiring1.parents_edge_states == wiring2.parents_edge_states)
    assert np.all(wiring1.children_edge_states == wiring2.children_edge_states)

    assert np.all(wiring1.edges_num_states == wiring3.edges_num_states)
    assert np.all(wiring1.var_states_for_edges == wiring3.var_states_for_edges)
    assert np.all(wiring1.parents_edge_states == wiring3.parents_edge_states)
    assert np.all(wiring1.children_edge_states == wiring3.children_edge_states)


def test_wiring_with_ANDFactorGroup():
    """
    Test the equivalence of the wiring compiled at the ANDFactorGroup level
    vs at the individual ANDFactor level (which is called from SingleFactorGroup)
    """
    A = vgroup.NDVariableArray(num_states=2, shape=(10,))
    B = vgroup.NDVariableArray(num_states=2, shape=(10,))
    C = vgroup.NDVariableArray(num_states=2, shape=(10,))

    # FactorGraph with a single ANDFactorGroup
    fg1 = fgraph.FactorGraph(variable_groups=[A, B, C])
    factor_group = fgroup.ANDFactorGroup(
        variables_for_factors=[[A[idx], B[idx], C[idx]] for idx in range(10)],
    )
    fg1.add_factors(factor_group)
    assert len(fg1.factor_groups[factor.ANDFactor]) == 1

    # FactorGraph with multiple ANDFactorGroup
    fg2 = fgraph.FactorGraph(variable_groups=[A, B, C])
    for idx in range(10):
        factor_group = fgroup.ANDFactorGroup(
            variables_for_factors=[[A[idx], B[idx], C[idx]]],
        )
        fg2.add_factors(factor_group)
    assert len(fg2.factor_groups[factor.ANDFactor]) == 10

    # FactorGraph with multiple SingleFactorGroup
    fg3 = fgraph.FactorGraph(variable_groups=[A, B, C])
    for idx in range(10):
        and_factor = factor.ANDFactor(
            variables=[A[idx], B[idx], C[idx]],
        )
        fg3.add_factors(and_factor)
    assert len(fg3.factor_groups[factor.ANDFactor]) == 10

    assert len(fg1.factors) == len(fg2.factors) == len(fg3.factors)

    # Compile wiring via factor_group.compile_wiring
    wiring1 = fg1.wiring[factor.ANDFactor]
    wiring2 = fg2.wiring[factor.ANDFactor]

    # Compile wiring via factor.compile_wiring
    wiring3 = fg3.wiring[factor.ANDFactor]

    assert np.all(wiring1.edges_num_states == wiring2.edges_num_states)
    assert np.all(wiring1.var_states_for_edges == wiring2.var_states_for_edges)
    assert np.all(wiring1.parents_edge_states == wiring2.parents_edge_states)
    assert np.all(wiring1.children_edge_states == wiring2.children_edge_states)

    assert np.all(wiring1.edges_num_states == wiring3.edges_num_states)
    assert np.all(wiring1.var_states_for_edges == wiring3.var_states_for_edges)
    assert np.all(wiring1.parents_edge_states == wiring3.parents_edge_states)
    assert np.all(wiring1.children_edge_states == wiring3.children_edge_states)
