import re

import numpy as np
import pytest

from pgmax import factor, vgroup


def test_enumeration_factor():
    variables = vgroup.NDVariableArray(num_states=3, shape=(1,))

    with pytest.raises(
        NotImplementedError, match="Please implement compile_wiring in for your factor"
    ):
        factor.Factor(
            variables=[variables[0]],
            log_potentials=np.array([0.0]),
        )

    with pytest.raises(ValueError, match="Configurations should be integers. Got"):
        factor.EnumerationFactor(
            variables=[variables[0]],
            factor_configs=np.array([[1.0]]),
            log_potentials=np.array([0.0]),
        )

    with pytest.raises(ValueError, match="Potential should be floats. Got"):
        factor.EnumerationFactor(
            variables=[variables[0]],
            factor_configs=np.array([[1]]),
            log_potentials=np.array([0]),
        )

    with pytest.raises(ValueError, match="factor_configs should be a 2D array"):
        factor.EnumerationFactor(
            variables=[variables[0]],
            factor_configs=np.array([1]),
            log_potentials=np.array([0.0]),
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Number of variables 1 doesn't match given configurations (1, 2)"
        ),
    ):
        factor.EnumerationFactor(
            variables=[variables[0]],
            factor_configs=np.array([[1, 2]]),
            log_potentials=np.array([0.0]),
        )

    with pytest.raises(
        ValueError, match=re.escape("Expected log potentials of shape (1,)")
    ):
        factor.EnumerationFactor(
            variables=[variables[0]],
            factor_configs=np.array([[1]]),
            log_potentials=np.array([0.0, 1.0]),
        )

    with pytest.raises(ValueError, match="Invalid configurations for given variables"):
        factor.EnumerationFactor(
            variables=[variables[0]],
            factor_configs=np.array([[10]]),
            log_potentials=np.array([0.0]),
        )


def test_logical_factor():
    child = vgroup.NDVariableArray(num_states=2, shape=(1,))[0]
    wrong_parent = vgroup.NDVariableArray(num_states=3, shape=(1,))[0]
    parent = vgroup.NDVariableArray(num_states=2, shape=(1,))[0]

    with pytest.raises(
        ValueError,
        match="A LogicalFactor requires at least one parent variable and one child variable",
    ):
        factor.logical.LogicalFactor(
            variables=(child,),
        )

    with pytest.raises(ValueError, match="All variables should all be binary"):
        factor.logical.LogicalFactor(
            variables=(wrong_parent, child),
        )

    logical_factor = factor.logical.LogicalFactor(
        variables=(parent, child),
    )
    num_parents = len(logical_factor.variables) - 1
    parents_edge_states = np.vstack(
        [
            np.zeros(num_parents, dtype=int),
            np.arange(0, 2 * num_parents, 2, dtype=int),
        ],
    ).T
    child_edge_state = np.array([2 * num_parents], dtype=int)

    with pytest.raises(ValueError, match="The highest LogicalFactor index must be 0"):
        factor.logical.LogicalWiring(
            edges_num_states=[2, 2],
            var_states_for_edges=None,
            parents_edge_states=parents_edge_states + np.array([[1, 0]]),
            children_edge_states=child_edge_state,
            edge_states_offset=1,
        )

    with pytest.raises(
        ValueError,
        match="The LogicalWiring must have 1 different LogicalFactor indices",
    ):
        factor.logical.LogicalWiring(
            edges_num_states=[2, 2],
            var_states_for_edges=None,
            parents_edge_states=parents_edge_states + np.array([[0], [1]]),
            children_edge_states=child_edge_state,
            edge_states_offset=1,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The LogicalWiring's edge_states_offset must be 1 (for OR) and -1 (for AND), but is 0"
        ),
    ):
        factor.logical.LogicalWiring(
            edges_num_states=[2, 2],
            var_states_for_edges=None,
            parents_edge_states=parents_edge_states,
            children_edge_states=child_edge_state,
            edge_states_offset=0,
        )
