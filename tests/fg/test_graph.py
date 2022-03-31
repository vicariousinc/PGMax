import re
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from pgmax.factors import FAC_TO_VAR_UPDATES
from pgmax.factors import enumeration as enumeration_factor
from pgmax.factors import logical as logical_factor
from pgmax.fg import graph, groups
from pgmax.groups import logical, variables


def test_factor_graph():
    variable_group = variables.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor_by_type(
        factor_type=enumeration_factor.EnumerationFactor,
        variable_names=[0],
        configs=np.arange(15)[:, None],
        log_potentials=np.zeros(15),
        name="test",
    )
    with pytest.raises(
        ValueError,
        match="A factor group with the name test already exists. Please choose a different name",
    ):
        fg.add_factor(
            variable_names=[0],
            factor_configs=np.arange(15)[:, None],
            name="test",
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"A {enumeration_factor.EnumerationFactor} involving variables {tuple([0])} already exists."
        ),
    ):
        fg.add_factor(
            variable_names=[0],
            factor_configs=np.arange(10)[:, None],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Type {groups.FactorGroup} is not one of the supported factor types {FAC_TO_VAR_UPDATES.keys()}"
        ),
    ):
        fg.add_factor_by_type(variable_names=[0], factor_type=groups.FactorGroup)


def test_factor_adding():
    A = variables.NDVariableArray(num_states=2, shape=(10,))
    B = variables.NDVariableArray(num_states=2, shape=(10,))
    fg = graph.FactorGraph(variables=dict(A=A, B=B))

    with pytest.raises(ValueError, match="Do not add a factor group with no factors."):
        fg.add_factor_group(
            factory=logical.LogicalFactorGroup,
            variable_names_for_factors=[],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Type {logical_factor.LogicalFactor} is not one of the supported factor types {FAC_TO_VAR_UPDATES.keys()}"
        ),
    ):
        fg.add_factor_group(
            factory=logical.LogicalFactorGroup,
            variable_names_for_factors=[[("A", 0), ("B", 0)]],
        )

    variables0 = [("A", 0), ("B", 0)]
    variables1 = [("A", 1), ("B", 1)]
    LogicalFactor = logical_factor.LogicalFactor(fg._variable_group[variables0])
    with pytest.raises(
        ValueError, match="SingleFactorGroup should only contain one factor. Got 2"
    ):
        groups.SingleFactorGroup(
            variable_group=fg._variable_group,
            variable_names_for_factors=[variables0, variables1],
            factor=LogicalFactor,
        )


def test_bp_state():
    variable_group = variables.VariableDict(15, (0,))
    fg0 = graph.FactorGraph(variable_group)
    fg0.add_factor(
        variable_names=[0],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    fg1 = graph.FactorGraph(variable_group)
    fg1.add_factor(
        variable_names=[0],
        factor_configs=np.arange(15)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError,
        match="log_potentials, ftov_msgs and evidence should be derived from the same fg_state",
    ):
        graph.BPState(
            log_potentials=fg0.bp_state.log_potentials,
            ftov_msgs=fg1.bp_state.ftov_msgs,
            evidence=fg1.bp_state.evidence,
        )


def test_log_potentials():
    variable_group = variables.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor(
        variable_names=[0],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Expected log potentials shape (10,) for factor group test."),
    ):
        fg.bp_state.log_potentials["test"] = jnp.zeros((1, 15))

    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid name {frozenset([0])} for log potentials updates."),
    ):
        fg.bp_state.log_potentials[[0]] = np.zeros(10)

    with pytest.raises(
        ValueError, match=re.escape("Expected log potentials shape (10,). Got (15,)")
    ):
        graph.LogPotentials(fg_state=fg.fg_state, value=np.zeros(15))

    log_potentials = graph.LogPotentials(fg_state=fg.fg_state, value=np.zeros(10))
    assert jnp.all(log_potentials["test"] == jnp.zeros(10))
    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid name {frozenset([1])} for log potentials updates."),
    ):
        fg.bp_state.log_potentials[[1]]


def test_ftov_msgs():
    variable_group = variables.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor(
        variable_names=[0],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid names for setting messages"),
    ):
        fg.bp_state.ftov_msgs[[0], 0] = np.ones(10)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Given belief shape (10,) does not match expected shape (15,) for variable 0"
        ),
    ):
        fg.bp_state.ftov_msgs[0] = np.ones(10)

    with pytest.raises(
        ValueError, match=re.escape("Expected messages shape (15,). Got (10,)")
    ):
        graph.FToVMessages(fg_state=fg.fg_state, value=np.zeros(10))

    ftov_msgs = graph.FToVMessages(fg_state=fg.fg_state, value=np.zeros(15))
    with pytest.raises(
        TypeError, match=re.escape("'FToVMessages' object is not subscriptable")
    ):
        ftov_msgs[(10,)]


def test_evidence():
    variable_group = variables.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor(
        variable_names=[0],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError, match=re.escape("Expected evidence shape (15,). Got (10,).")
    ):
        graph.Evidence(fg_state=fg.fg_state, value=np.zeros(10))

    evidence = graph.Evidence(fg_state=fg.fg_state, value=np.zeros(15))
    assert jnp.all(evidence.value == jnp.zeros(15))


def test_bp():
    variable_group = variables.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor(
        variable_names=[0],
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    run_bp, get_bp_state, get_beliefs = graph.BP(fg.bp_state, 1)
    bp_arrays = replace(
        run_bp(ftov_msgs_updates={0: np.zeros(15)}),
        log_potentials=jnp.zeros((10)),
    )
    bp_state = get_bp_state(bp_arrays)
    assert bp_state.fg_state == fg.fg_state
