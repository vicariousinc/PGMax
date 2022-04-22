import re

import numpy as np
import pytest

from pgmax.factors import enumeration as enumeration_factor
from pgmax.factors import logical as logical_factor
from pgmax.fg import graph
from pgmax.groups import enumeration, logical
from pgmax.groups import variables as vgroup


def test_wiring_with_PairwiseFactorGroup():
    """
    Test the equivalence of the wiring compiled at the PairwiseFactorGroup level
    vs at the individual EnumerationFactor level (which is called from SingleFactorGroup)
    """
    A = vgroup.NDVariableArray(num_states=2, shape=(10,))
    B = vgroup.NDVariableArray(num_states=2, shape=(10,))

    # First test that compile_wiring enforces the correct factor_edges_num_states shape
    fg = graph.FactorGraph(variables=[A, B])
    factor_group = enumeration.PairwiseFactorGroup(
        variables_for_factors=[[A[idx], B[idx]] for idx in range(10)]
    )
    fg.add_factor_group(factor_group)

    factor_group = fg.factor_groups[enumeration_factor.EnumerationFactor][0]
    object.__setattr__(
        factor_group, "factor_configs", factor_group.factor_configs[:, :1]
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Expected factor_edges_num_states shape is (10,). Got (20,)."),
    ):
        factor_group.compile_wiring(fg._vars_to_starts)

    # FactorGraph with a single PairwiseFactorGroup
    fg1 = graph.FactorGraph(variables=[A, B])
    factor_group = enumeration.PairwiseFactorGroup(
        variables_for_factors=[[A[idx], B[idx]] for idx in range(10)]
    )
    fg1.add_factor_group(factor_group)
    assert len(fg1.factor_groups[enumeration_factor.EnumerationFactor]) == 1

    # FactorGraph with multiple PairwiseFactorGroup
    fg2 = graph.FactorGraph(variables=[A, B])
    for idx in range(10):
        factor_group = enumeration.PairwiseFactorGroup(
            variables_for_factors=[[A[idx], B[idx]]]
        )
        fg2.add_factor_group(factor_group)
    assert len(fg2.factor_groups[enumeration_factor.EnumerationFactor]) == 10

    # FactorGraph with multiple SingleFactorGroup
    fg3 = graph.FactorGraph(variables=[A, B])
    for idx in range(10):
        factor = enumeration_factor.EnumerationFactor(
            variables=[A[idx], B[idx]],
            factor_configs=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            log_potentials=np.zeros((4,)),
        )
        fg3.add_factor(factor)
    assert len(fg3.factor_groups[enumeration_factor.EnumerationFactor]) == 10

    assert len(fg1.factors) == len(fg2.factors) == len(fg3.factors)

    # Compile wiring via factor_group.compile_wiring
    wiring1 = fg1.wiring[enumeration_factor.EnumerationFactor]
    wiring2 = fg2.wiring[enumeration_factor.EnumerationFactor]

    # Compile wiring via factor.compile_wiring
    wiring3 = fg3.wiring[enumeration_factor.EnumerationFactor]

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
    fg1 = graph.FactorGraph(variables=[A, B, C])
    factor_group = logical.ORFactorGroup(
        variables_for_factors=[[A[idx], B[idx], C[idx]] for idx in range(10)],
    )
    fg1.add_factor_group(factor_group)
    assert len(fg1.factor_groups[logical_factor.ORFactor]) == 1

    # FactorGraph with multiple ORFactorGroup
    fg2 = graph.FactorGraph(variables=[A, B, C])
    for idx in range(10):
        factor_group = logical.ORFactorGroup(
            variables_for_factors=[[A[idx], B[idx], C[idx]]],
        )
        fg2.add_factor_group(factor_group)
    assert len(fg2.factor_groups[logical_factor.ORFactor]) == 10

    # FactorGraph with multiple SingleFactorGroup
    fg3 = graph.FactorGraph(variables=[A, B, C])
    for idx in range(10):
        factor = logical_factor.ORFactor(
            variables=[A[idx], B[idx], C[idx]],
        )
        fg3.add_factor(factor)
    assert len(fg3.factor_groups[logical_factor.ORFactor]) == 10

    assert len(fg1.factors) == len(fg2.factors) == len(fg3.factors)

    # Compile wiring via factor_group.compile_wiring
    wiring1 = fg1.wiring[logical_factor.ORFactor]
    wiring2 = fg2.wiring[logical_factor.ORFactor]

    # Compile wiring via factor.compile_wiring
    wiring3 = fg3.wiring[logical_factor.ORFactor]

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
    fg1 = graph.FactorGraph(variables=[A, B, C])
    factor_group = logical.ANDFactorGroup(
        variables_for_factors=[[A[idx], B[idx], C[idx]] for idx in range(10)],
    )
    fg1.add_factor_group(factor_group)
    assert len(fg1.factor_groups[logical_factor.ANDFactor]) == 1

    # FactorGraph with multiple ANDFactorGroup
    fg2 = graph.FactorGraph(variables=[A, B, C])
    for idx in range(10):
        factor_group = logical.ANDFactorGroup(
            variables_for_factors=[[A[idx], B[idx], C[idx]]],
        )
        fg2.add_factor_group(factor_group)
    assert len(fg2.factor_groups[logical_factor.ANDFactor]) == 10

    # FactorGraph with multiple SingleFactorGroup
    fg3 = graph.FactorGraph(variables=[A, B, C])
    for idx in range(10):
        factor = logical_factor.ANDFactor(
            variables=[A[idx], B[idx], C[idx]],
        )
        fg3.add_factor(factor)
    assert len(fg3.factor_groups[logical_factor.ANDFactor]) == 10

    assert len(fg1.factors) == len(fg2.factors) == len(fg3.factors)

    # Compile wiring via factor_group.compile_wiring
    wiring1 = fg1.wiring[logical_factor.ANDFactor]
    wiring2 = fg2.wiring[logical_factor.ANDFactor]

    # Compile wiring via factor.compile_wiring
    wiring3 = fg3.wiring[logical_factor.ANDFactor]

    assert np.all(wiring1.edges_num_states == wiring2.edges_num_states)
    assert np.all(wiring1.var_states_for_edges == wiring2.var_states_for_edges)
    assert np.all(wiring1.parents_edge_states == wiring2.parents_edge_states)
    assert np.all(wiring1.children_edge_states == wiring2.children_edge_states)

    assert np.all(wiring1.edges_num_states == wiring3.edges_num_states)
    assert np.all(wiring1.var_states_for_edges == wiring3.var_states_for_edges)
    assert np.all(wiring1.parents_edge_states == wiring3.parents_edge_states)
    assert np.all(wiring1.children_edge_states == wiring3.children_edge_states)
