import re
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from pgmax.fg import graph, groups


def test_factor_graph():
    variable_group = groups.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor(
        variable_names=[0],
        factor_type="EnumerationFactor",
        factor_configs=np.arange(15)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError,
        match="A factor group with the name test already exists. Please choose a different name",
    ):
        fg.add_factor(
            variable_names=[0],
            factor_type="EnumerationFactor",
            factor_configs=np.arange(15)[:, None],
            name="test",
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"A factor involving variables {frozenset([0])} already exists."
        ),
    ):
        fg.add_factor(
            variable_names=[0],
            factor_type="EnumerationFactor",
            factor_configs=np.arange(10)[:, None],
        )


def test_bp_state():
    variable_group = groups.VariableDict(15, (0,))
    fg0 = graph.FactorGraph(variable_group)
    fg0.add_factor(
        variable_names=[0],
        factor_type="EnumerationFactor",
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    fg1 = graph.FactorGraph(variable_group)
    fg1.add_factor(
        variable_names=[0],
        factor_type="EnumerationFactor",
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
    variable_group = groups.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor(
        variable_names=[0],
        factor_type="EnumerationFactor",
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Expected log potentials shape (10,) for factor group test."),
    ):
        fg.bp_state.log_potentials["test"] = jnp.zeros((1, 15))

    fg.bp_state.log_potentials[[0]] = np.zeros(10)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Expected log potentials shape (10,) for factor {frozenset([0])}. Got (15,)"
        ),
    ):
        fg.bp_state.log_potentials[[0]] = np.zeros(15)

    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid name {frozenset([1])} for log potentials updates."),
    ):
        fg.bp_state.log_potentials[frozenset([1])] = np.zeros(10)

    with pytest.raises(
        ValueError, match=re.escape("Expected log potentials shape (10,). Got (15,)")
    ):
        graph.LogPotentials(fg_state=fg.fg_state, value=np.zeros(15))

    log_potentials = graph.LogPotentials(fg_state=fg.fg_state, value=np.zeros(10))
    assert jnp.all(log_potentials["test"] == jnp.zeros(10))
    assert jnp.all(log_potentials[[0]] == jnp.zeros(10))
    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid name {frozenset([1])} for log potentials updates."),
    ):
        fg.bp_state.log_potentials[[1]]


def test_ftov_msgs():
    variable_group = groups.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor(
        variable_names=[0],
        factor_type="EnumerationFactor",
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Given message shape (10,) does not match expected shape (15,) from factor {frozenset([0])} to variable 0"
        ),
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
        ValueError,
        match=re.escape("Invalid names for setting messages"),
    ):
        fg.bp_state.ftov_msgs[1] = np.ones(10)

    with pytest.raises(
        ValueError, match=re.escape("Expected messages shape (15,). Got (10,)")
    ):
        graph.FToVMessages(fg_state=fg.fg_state, value=np.zeros(10))

    ftov_msgs = graph.FToVMessages(fg_state=fg.fg_state, value=np.zeros(15))
    with pytest.raises(ValueError, match=re.escape("Invalid names (10,)")):
        ftov_msgs[(10,)]


def test_evidence():
    variable_group = groups.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor(
        variable_names=[0],
        factor_type="EnumerationFactor",
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
    variable_group = groups.VariableDict(15, (0,))
    fg = graph.FactorGraph(variable_group)
    fg.add_factor(
        variable_names=[0],
        factor_type="EnumerationFactor",
        factor_configs=np.arange(10)[:, None],
        name="test",
    )
    run_bp, get_bp_state, get_beliefs = graph.BP(fg.bp_state, 1)
    bp_arrays = replace(
        run_bp(ftov_msgs_updates={(frozenset([0]), 0): np.zeros(15)}),
        log_potentials=np.zeros(10),
    )
    bp_state = get_bp_state(bp_arrays)
    assert bp_state.fg_state == fg.fg_state
